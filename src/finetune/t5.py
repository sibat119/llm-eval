"""
This file contains code for k-fold cross validation finetuning of a T5 model as a surrogate model.
The model is trained on input-output pairs to mimic the behavior of another model (e.g., LLaMA).
"""

# external imports
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

# local imports
from src.utils.files import get_project_root, get_full_path, path_exists
from src.utils.strings import now, green, yellow, red

class T5SurrogateDataset(Dataset):
    """
    Dataset for T5 surrogate model training
    """
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        tokenizer: T5Tokenizer,
        max_input_length: int = 512,
        max_output_length: int = 128
    ):
        """
        Initialize the dataset
        
        :param inputs: List of input strings
        :param outputs: List of output strings
        :param tokenizer: T5 tokenizer
        :param max_input_length: Maximum input sequence length
        :param max_output_length: Maximum output sequence length
        """
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        
        # Tokenize input and output
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create model inputs
        input_ids = input_encoding.input_ids.squeeze()
        attention_mask = input_encoding.attention_mask.squeeze()
        labels = output_encoding.input_ids.squeeze()
        
        # Replace padding token id with -100 in labels so it's ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def split_data(inputs: List[str], outputs: List[str], num_folds: int, seed: int = 42) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Split data into train, validation, and test sets based on number of folds
    
    :param inputs: List of input strings
    :param outputs: List of output strings
    :param num_folds: Number of folds (2, 3, or 4)
    :param seed: Random seed
    :return: Dictionary containing train, val, and test splits
    """
    assert num_folds in [2, 3, 4], "num_folds must be 2, 3, or 4"
    
    # Set random seed
    np.random.seed(seed)
    
    # Create indices and shuffle
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    
    # Calculate split points
    test_ratio = 1/num_folds
    test_size = int(len(indices) * test_ratio)
    
    # Split into train and test
    test_indices = indices[:test_size]
    train_val_indices = indices[test_size:]
    
    # Split training into train and validation (10% of training data)
    val_size = int(len(train_val_indices) * 0.1)
    val_indices = train_val_indices[:val_size]
    train_indices = train_val_indices[val_size:]
    
    # Create the splits
    splits = {
        "train": ([inputs[i] for i in train_indices], [outputs[i] for i in train_indices]),
        "val": ([inputs[i] for i in val_indices], [outputs[i] for i in val_indices]),
        "test": ([inputs[i] for i in test_indices], [outputs[i] for i in test_indices])
    }
    
    return splits

def evaluate_model(model_path: str, data_splits: Dict[str, Tuple[List[str], List[str]]], 
                  max_input_length: int = 512, max_output_length: int = 128, 
                  batch_size: int = 8) -> Dict[str, float]:
    """
    Evaluate model on train, validation, and test sets
    
    :param model_path: Path to the trained model
    :param data_splits: Dictionary containing data splits
    :param max_input_length: Maximum input sequence length
    :param max_output_length: Maximum output sequence length
    :param batch_size: Batch size for inference
    :return: Dictionary containing evaluation metrics
    """
    metrics = {}
    
    for split_name, (inputs, outputs) in data_splits.items():
        predictions = predict_with_t5_surrogate(
            model_path=model_path,
            inputs=inputs,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            batch_size=batch_size
        )
        
        # Calculate accuracy
        accuracy = sum(1 for pred, true in zip(predictions, outputs) 
                      if pred.strip() == true.strip()) / len(outputs)
        
        metrics[f"{split_name}_accuracy"] = accuracy
        metrics[f"{split_name}_size"] = len(inputs)
    
    return metrics

def plot_training_results(results: Dict, output_dir: str):
    """
    Plot training results
    
    :param results: Dictionary containing training results
    :param output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    
    # Create accuracy plot
    plt.figure(figsize=(10, 6))
    splits = ['train', 'val', 'test']
    for split in splits:
        accuracies = [fold[f'{split}_accuracy'] for fold in results['fold_results']]
        plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', label=f'{split} accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Across Splits')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()

def train_t5_surrogate(
    inputs: List[str],
    outputs: List[str],
    model_name: str = "t5-base",
    output_dir: str = "models/t5_surrogate",
    num_folds: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    max_input_length: int = 512,
    max_output_length: int = 128,
    seed: int = 42,
    save_model: bool = True
) -> Dict:
    """
    Train a T5 surrogate model with train/val/test splits
    """
    assert len(inputs) == len(outputs), "Inputs and outputs must have the same length"
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Split data
    data_splits = split_data(inputs, outputs, num_folds, seed)
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Prepare output directory
    output_dir = get_full_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "fold_results": [],
        "model_name": model_name,
        "num_folds": num_folds,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs
    }
    
    # Create datasets
    train_dataset = T5SurrogateDataset(
        inputs=data_splits["train"][0],
        outputs=data_splits["train"][1],
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length
    )
    
    val_dataset = T5SurrogateDataset(
        inputs=data_splits["val"][0],
        outputs=data_splits["val"][1],
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length
    )
    
    # Initialize model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        learning_rate=learning_rate,
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train model
    train_result = trainer.train()
    
    # Save model and tokenizer
    if save_model:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Evaluate model on all splits
    metrics = evaluate_model(
        model_path=output_dir,
        data_splits=data_splits,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        batch_size=batch_size
    )
    
    # Update results
    results.update(metrics)
    results["training_loss"] = train_result.training_loss
    
    # Generate plots
    plot_training_results(results, output_dir)
    
    # Print results
    print(green("Training completed!"))
    print(f"Training loss: {train_result.training_loss:.4f}")
    for split in ['train', 'val', 'test']:
        print(f"{split.capitalize()} accuracy: {metrics[f'{split}_accuracy']:.4f}")
    
    return results

def predict_with_t5_surrogate(
    model_path: str,
    inputs: List[str],
    max_input_length: int = 512,
    max_output_length: int = 128,
    batch_size: int = 8,
    device: str = None
) -> List[str]:
    """
    Generate predictions using a trained T5 surrogate model
    
    :param model_path: Path to the trained model
    :param inputs: List of input strings
    :param max_input_length: Maximum input sequence length
    :param max_output_length: Maximum output sequence length
    :param batch_size: Batch size for inference
    :param device: Device to use for inference (None for auto-detection)
    
    :return: List of predicted outputs
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    # Generate predictions in batches
    predictions = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="Generating predictions"):
        batch_inputs = inputs[i:i+batch_size]
        
        # Tokenize inputs
        input_encodings = tokenizer(
            batch_inputs,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate outputs
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask,
                max_length=max_output_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode outputs
        batch_predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(batch_predictions)
    
    return predictions


