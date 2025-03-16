import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup, 
    DataCollatorForSeq2Seq,
    T5Tokenizer, 
    T5ForConditionalGeneration
)
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset


class BaseModelTrainer:
    """
    Base class for transformer model training and evaluation.
    """
    def __init__(
        self,
        model,
        tokenizer,
        device='cuda',
        save_path='./model_checkpoints'
    ):
        """
        Initialize the trainer with model and tokenizer.
        
        Args:
            model: The model to train
            tokenizer: Tokenizer for the model
            device: Device to train on ('cuda' or 'cpu')
            save_path: Path to save model checkpoints
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.save_path = save_path
        
        # Create directory for saving model checkpoints
        os.makedirs(save_path, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize training statistics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Initialize data loaders
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
    
    def prepare_dataloaders(
        self,
        tokenized_dataset_train,
        tokenized_dataset_validation,
        tokenized_dataset_test,
        batch_size=16,
        shuffle_train=True,
        collate_fn=None
    ):
        """
        Prepare PyTorch DataLoaders from tokenized datasets.
        
        Args:
            tokenized_dataset_train: Tokenized training dataset
            tokenized_dataset_validation: Tokenized validation dataset
            tokenized_dataset_test: Tokenized test dataset
            batch_size: Batch size for DataLoaders
            shuffle_train: Whether to shuffle training data
            collate_fn: Optional collate function for DataLoader
        """
        self.train_dataloader = DataLoader(
            tokenized_dataset_train,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=collate_fn
        )
        
        self.val_dataloader = DataLoader(
            tokenized_dataset_validation,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        self.test_dataloader = DataLoader(
            tokenized_dataset_test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    def setup_optimizer(self, learning_rate=5e-5, weight_decay=0.01):
        """
        Set up optimizer for training.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
    
    def setup_scheduler(self, warmup_steps=500, num_epochs=3):
        """
        Set up learning rate scheduler for training.
        
        Args:
            warmup_steps: Number of warmup steps
            num_epochs: Number of training epochs
        """
        total_steps = len(self.train_dataloader) * num_epochs
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def save_checkpoint(self, path, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint_path = os.path.join(self.save_path, path)
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        if is_best:
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
        else:
            print(f"Checkpoint saved at {checkpoint_path}")
    
    def train_epoch(self, max_grad_norm=1.0):
        """
        Train the model for one epoch.
        
        Args:
            max_grad_norm: Maximum gradient norm for gradient clipping
        
        Returns:
            Average loss for the epoch
        """
        # Set model to training mode
        self.model.train()
        epoch_loss = 0
        
        # Create progress bar for tracking
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Update tracking variables
            self.global_step += 1
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Save model checkpoint periodically
            if self.global_step % 1000 == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / len(self.train_dataloader)
        return avg_loss
    
    def validate(self):
        """
        Validate the model on validation data.
        
        Returns:
            Average validation loss
        """
        # Set model to evaluation mode
        self.model.eval()
        val_loss = 0
        
        try:
            # Create progress bar for tracking
            progress_bar = tqdm(self.val_dataloader, desc="Validation")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass (no gradient calculation needed)
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                # Update tracking variables
                val_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
            
            # Calculate average loss for this epoch
            avg_loss = val_loss / len(self.val_dataloader)
            return avg_loss
            
        except Exception as e:
            print(f"Error during validation: {e}")
            print("Continuing training...")
            return float('inf')
    
    def train(self, num_epochs=3, learning_rate=5e-5, weight_decay=0.01, 
              warmup_steps=500, max_grad_norm=1.0):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for scheduler
            max_grad_norm: Maximum gradient norm for gradient clipping
        
        Returns:
            Dictionary of training statistics
        """
        # Setup optimizer and scheduler
        self.setup_optimizer(learning_rate, weight_decay)
        self.setup_scheduler(warmup_steps, num_epochs)
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            avg_train_loss = self.train_epoch(max_grad_norm)
            self.train_losses.append(avg_train_loss)
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            avg_val_loss = self.validate()
            self.val_losses.append(avg_val_loss)
            print(f"Average validation loss: {avg_val_loss:.4f}")
            
            # Save model if validation loss improved
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint("best_model", is_best=True)
        
        # Save the final model
        self.save_checkpoint("final_model")
        print(f"Final model saved at {os.path.join(self.save_path, 'final_model')}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "global_steps": self.global_step
        }
    
    def evaluate(self, dataloader=None, **kwargs):
        """
        Evaluate model on test data.
        
        Args:
            dataloader: DataLoader for evaluation (defaults to test_dataloader)
            **kwargs: Additional parameters for evaluation
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Use test_dataloader if no dataloader is provided
        if dataloader is None:
            if self.test_dataloader is None:
                raise ValueError("No test dataloader provided for evaluation")
            dataloader = self.test_dataloader
        
        # Implement in subclass
        raise NotImplementedError("Evaluate method must be implemented by subclass")
    
    def plot_training_history(self):
        """
        Plot training and validation loss history.
        
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plot_path = os.path.join(self.save_path, 'loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path


class T5ModelTrainer(BaseModelTrainer):
    """
    Specialized trainer for T5 models.
    """
    def __init__(
        self,
        model,
        tokenizer,
        device='cuda',
        save_path='./model_checkpoints'
    ):
        """
        Initialize T5 model trainer.
        
        Args:
            model: T5 model
            tokenizer: T5 tokenizer
            device: Device to train on ('cuda' or 'cpu')
            save_path: Path to save model checkpoints
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            save_path=save_path
        )
    
    def evaluate(self, dataloader=None, generate_params=None):
        """
        Evaluate T5 model on test data.
        
        Args:
            dataloader: DataLoader for evaluation (defaults to test_dataloader)
            generate_params: Parameters for model.generate()
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Use test_dataloader if no dataloader is provided
        if dataloader is None:
            if self.test_dataloader is None:
                raise ValueError("No test dataloader provided for evaluation")
            dataloader = self.test_dataloader
        
        # Default generation parameters
        if generate_params is None:
            generate_params = {
                "max_length": 128,
                "num_beams": 4,
                "early_stopping": True,
                "no_repeat_ngram_size": 3
            }
        
        # Set model to evaluation mode
        self.model.eval()
        
        all_preds = []
        all_labels = []
        test_loss = 0
        
        try:
            progress_bar = tqdm(dataloader, desc="Evaluating")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass for loss calculation
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    test_loss += loss.item()
                
                # Generate predictions
                try:
                    # Get actual label text (replacing -100s with pad token id)
                    label_ids = torch.where(
                        labels != -100, 
                        labels, 
                        self.tokenizer.pad_token_id
                    )
                    
                    # Generate predicted text
                    gen_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generate_params
                    )
                    
                    # Decode predictions and labels
                    decoded_preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
                    
                    # Store for evaluation
                    all_preds.extend(decoded_preds)
                    all_labels.extend(decoded_labels)
                    
                except Exception as e:
                    print(f"Error during generation: {e}")
                    # Continue with next batch
                    continue
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {"eval_loss": float('nan')}
        
        # Calculate average loss
        avg_loss = test_loss / len(dataloader)
        
        # Calculate BLEU scores
        bleu_scores = []
        smoothie = SmoothingFunction().method1
        for pred, label in zip(all_preds, all_labels):
            try:
                # Split into words for BLEU calculation
                pred_tokens = pred.split()
                label_tokens = label.split()
                
                # Calculate BLEU score
                score = sentence_bleu(
                    [label_tokens], 
                    pred_tokens,
                    smoothing_function=smoothie
                )
                bleu_scores.append(score)
            except Exception as e:
                print(f"Error calculating BLEU: {e}")
                bleu_scores.append(0)
        
        # Calculate average BLEU score
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        
        # Calculate exact match accuracy
        exact_matches = sum(1 for p, l in zip(all_preds, all_labels) if p.strip() == l.strip())
        exact_match_accuracy = exact_matches / len(all_preds) if all_preds else 0
        
        # Return evaluation metrics
        return {
            "eval_loss": avg_loss,
            "bleu_score": avg_bleu,
            "exact_match": exact_match_accuracy,
            "predictions": all_preds,
            "references": all_labels
        }

    def predict(self, input_text, **generate_params):
        """
        Generate prediction for a single input text.
        
        Args:
            input_text: Input text to generate from
            generate_params: Parameters for model.generate()
        
        Returns:
            Generated text prediction
        """
        # Default generation parameters
        if not generate_params:
            generate_params = {
                "max_length": 128,
                "num_beams": 4,
                "early_stopping": True,
                "no_repeat_ngram_size": 3
            }
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generate_params
            )
        
        # Decode prediction
        prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return prediction


def load_t5_model(model_name_or_path, device='cuda'):
    """
    Load T5 model and tokenizer.
    
    Args:
        model_name_or_path: Name or path of the T5 model
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading T5 model: {model_name_or_path}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    
    return model, tokenizer


def prepare_dataset(
    dataset_name,
    tokenizer,
    input_column,
    target_column,
    prefix="",
    max_input_length=512,
    max_target_length=128
):
    """
    Load and tokenize a dataset from Hugging Face's datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer to use for tokenization
        input_column: Column to use as input
        target_column: Column to use as target
        prefix: Prefix to add to the input text
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Tokenization function
    def tokenize_function(examples):
        inputs = [prefix + text for text in examples[input_column]]
        targets = examples[target_column]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=max_input_length,
            padding="max_length",
            truncation=True
        )
        
        # Tokenize targets with special handling for T5
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True
        ).input_ids
        
        # Replace padding token id with -100 for loss calculation
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels
        ]
        
        return model_inputs
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        tokenized_datasets["test"] if "test" in tokenized_datasets else tokenized_datasets["validation"]
    )


def main():
    """
    Main function to run T5 model fine-tuning.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune a T5 model")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Name or path of T5 model")
    parser.add_argument("--dataset", type=str, default="cnn_dailymail", help="Dataset name")
    parser.add_argument("--input_column", type=str, default="article", help="Column name for input text")
    parser.add_argument("--target_column", type=str, default="highlights", help="Column name for target text")
    parser.add_argument("--prefix", type=str, default="summarize: ", help="Prefix to add to input text")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="./t5_model", help="Path to save model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    # Check for CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load model and tokenizer
    model, tokenizer = load_t5_model(args.model_name, device=args.device)
    
    # Create trainer
    trainer = T5ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        save_path=args.save_path
    )
    
    # Prepare dataset
    print(f"Loading and tokenizing dataset: {args.dataset}")
    train_dataset, val_dataset, test_dataset = prepare_dataset(
        args.dataset,
        tokenizer,
        args.input_column,
        args.target_column,
        prefix=args.prefix
    )
    
    # Set up data loaders
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer.prepare_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collator
    )
    
    # Start training
    print(f"Starting training process with {args.epochs} epochs")
    training_stats = trainer.train(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Plot training history
    plot_path = trainer.plot_training_history()
    print(f"Training history plot saved to {plot_path}")
    
    # Evaluate on test set
    print("Evaluating model on test set")
    evaluation_results = trainer.evaluate()
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Test Loss: {evaluation_results['eval_loss']:.4f}")
    print(f"BLEU Score: {evaluation_results['bleu_score']:.4f}")
    print(f"Exact Match Accuracy: {evaluation_results['exact_match']:.4f}")
    
    # Show some example predictions
    print("\nExample Predictions:")
    for i in range(min(3, len(evaluation_results['predictions']))):
        print(f"\nInput: {test_dataset[i][args.input_column]}")
        print(f"Prediction: {evaluation_results['predictions'][i]}")
        print(f"Reference: {evaluation_results['references'][i]}")

if __name__ == "__main__":
    main()