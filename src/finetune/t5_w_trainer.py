"""
This file contains code for k-fold cross validation finetuning of a T5 model as a surrogate model.
The model is trained on input-output pairs to mimic the behavior of another model (e.g., LLaMA).
"""

# external imports
import os
import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq, 
    AdamW,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import load_dataset, DownloadMode, DatasetDict
from evaluate import load
import numpy as np
import nltk

# local imports
# from src.utils.files import get_project_root, get_full_path, path_exists
# from src.utils.strings import now, green, yellow, red

import os
import matplotlib.pyplot as plt
from datasets import load_dataset

def preprocess_function(examples, tokenizer):
    """
    Format the input prompt for T5 and tokenize both input and target.
    Improved to handle potential issues better.
    """
    inputs = [
        f"question: {q} choices: {', '.join(choices)} answer:" 
        for q, choices in zip(examples["question"], examples["choices"])
    ]
    targets = [str(c[a]) for c, a in zip(examples["choices"], examples["answer"])]
    
    # Tokenize inputs with padding='max_length' for consistency
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True, 
        padding='max_length',
    )
    
    # Tokenize targets with padding='max_length' for consistency
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=128, 
            truncation=True, 
            padding='max_length',
        )
    
    # Replace pad token id with -100 for proper loss calculation
    labels_with_ignore = np.array(labels["input_ids"])
    labels_with_ignore = np.where(
        labels_with_ignore != tokenizer.pad_token_id, 
        labels_with_ignore, 
        -100
    )
    
    model_inputs["labels"] = labels_with_ignore.tolist()
    return model_inputs

def preprocess_function_custom_mmlu(examples, tokenizer):
    """
    Format the input prompt for T5 and tokenize both input and target.
    Improved to handle potential issues better.
    """
    inputs = examples["prompt"]
    targets = examples["response"]
    
    # Tokenize inputs with padding='max_length' for consistency
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True, 
        padding='max_length',
    )
    
    # Tokenize targets with padding='max_length' for consistency
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=128, 
            truncation=True, 
            padding='max_length',
        )
    
    # Replace pad token id with -100 for proper loss calculation
    labels_with_ignore = np.array(labels["input_ids"])
    labels_with_ignore = np.where(
        labels_with_ignore != tokenizer.pad_token_id, 
        labels_with_ignore, 
        -100
    )
    
    model_inputs["labels"] = labels_with_ignore.tolist()
    return model_inputs

def get_tokenized_dataset(dataset_name, tokenizer):
    """
    Load the dataset and apply tokenization.
    Replace 'dataset_name' with your actual dataset identifier or local script.
    """
    dataset_train = load_dataset(dataset_name, "all", split="auxiliary_train", download_mode=DownloadMode.FORCE_REDOWNLOAD)
    dataset_val = load_dataset(dataset_name, "all", split="validation", download_mode=DownloadMode.FORCE_REDOWNLOAD)
    dataset_test = load_dataset(dataset_name, "all", split="test", download_mode=DownloadMode.FORCE_REDOWNLOAD)
    tokenized_dataset_train = dataset_train.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, batch_size=1000)
    tokenized_dataset_validation = dataset_val.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    tokenized_dataset_test = dataset_test.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    return tokenized_dataset_train, tokenized_dataset_validation, tokenized_dataset_test

def get_custom_dataset(dataset_path, tokenizer):
    """
    Load the dataset and apply tokenization.
    Replace 'dataset_name' with your actual dataset identifier or local script.
    """
    dataset_path = "data/dataset/meta-llama_Llama-3.2-3B-Instruct_meta-llama_Llama-3.2-3B-Instruct-evals_results.csv"
    dataset = load_dataset("csv", data_files=dataset_path)
    # Split the dataset into train, validation, and test sets with 70:20:10 ratio
    dataset = dataset["train"]  # Get the train split from the loaded CSV dataset
    
    # Shuffle the dataset to ensure random distribution
    dataset = dataset.shuffle(seed=42)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Create the splits
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, dataset_size))
    
    # Create a new DatasetDict with the splits
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Tokenize each split separately
    tokenized_dataset_train = dataset['train'].map(lambda examples: preprocess_function_custom_mmlu(examples, tokenizer), batched=True)
    tokenized_dataset_validation = dataset['validation'].map(lambda examples: preprocess_function_custom_mmlu(examples, tokenizer), batched=True)
    tokenized_dataset_test = dataset['test'].map(lambda examples: preprocess_function_custom_mmlu(examples, tokenizer), batched=True)
    
    # Return all three tokenized datasets
    return tokenized_dataset_train, tokenized_dataset_validation, tokenized_dataset_test
    
class EpochEvalCallback(TrainerCallback):
    """
    Custom callback to capture evaluation metrics after each epoch.
    The eval_history list stores tuples of (epoch, eval_accuracy).
    """
    def __init__(self):
        self.eval_history = []

    def on_evaluate(self, args: Seq2SeqTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Assume the last logged dictionary contains the evaluation metrics.
        logs = state.log_history[-1]
        # The key is "eval_accuracy" if our compute_metrics returns {"accuracy": ...}
        if "eval_accuracy" in logs and "epoch" in logs:
            self.eval_history.append((logs["epoch"], logs["eval_accuracy"]))
        return control



def setup_trainer(model, tokenized_dataset_train, tokenized_dataset_validation, tokenizer, eval_callback):
    """
    Set up TrainingArguments and the Trainer.
    We use evaluation_strategy "epoch" so validation is run after every epoch.
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Simple exact-match accuracy
        correct = sum([p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)])
        return {'accuracy': correct/len(decoded_labels)}
    
    def compute_metrics_rouge(eval_pred):
        predictions, labels = eval_pred
        # Add safety check for predictions - filter out any invalid token IDs
        predictions = np.where(
            (predictions >= 0) & (predictions < tokenizer.vocab_size),
            predictions,
            tokenizer.pad_token_id
        )
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        try:
            # Note that other metrics may not have a `use_aggregator` parameter
            # and thus will return a list, computing a metric for each sentence
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
            # Extract a few results
            result = {key: value * 100 for key, value in result.items()}
            
            # Add mean generated length
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)
            
            return {k: round(v, 4) for k, v in result.items()}
        except Exception as e:
            print(f"Error computing metrics: {e}")
            # Return dummy metrics to allow training to continue
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0, "gen_len": 0.0}

    metric = load("rouge")
    
    training_args = Seq2SeqTrainingArguments(
        f"t5-base-finetuned-mmlu",
        evaluation_strategy="epoch",
        learning_rate=1e-5,  # Reduced learning rate for stability
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=False,  # Disable fp16 to improve stability
        max_grad_norm=1.0,  # Add gradient clipping
        generation_max_length=128,  # Set reasonable generation length
        generation_num_beams=4,  # Use beam search for better predictions
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Initialize optimizer with epsilon parameter to avoid division by zero
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, eps=1e-8)
    
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_validation,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
        # callbacks=[eval_callback],  # Add the callback
        optimizers=(optimizer, None),  # Use our custom optimizer
    )
    return trainer

def plot_accuracy(eval_history):
    """
    Plot the validation accuracy over epochs.
    """
    # Sort by epoch number
    eval_history.sort(key=lambda x: x[0])
    epochs = [entry[0] for entry in eval_history]
    accuracies = [entry[1] for entry in eval_history]
    
    plt.plot(epochs, accuracies, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.grid(True)
    plt.show()

def main():
    # Set model and dataset parameters
    model_name = "t5-base"
    dataset_name = "cais/mmlu"  # Replace with your actual dataset identifier
    
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    
    # Prepare the tokenized dataset
    tokenized_dataset_train, tokenized_dataset_validation, tokenized_dataset_test = get_custom_dataset(dataset_path="", tokenizer=tokenizer)
    
    # Initialize custom callback for tracking evaluation metrics
    eval_callback = EpochEvalCallback()
    
    # Setup the Trainer
    trainer = setup_trainer(model, tokenized_dataset_train, tokenized_dataset_validation, tokenizer, eval_callback)
    
    # breakpoint()
    # Start training (validation will run after each epoch)
    trainer.train()
    
    # Optionally, save the final model
    trainer.save_model(os.path.join("./results", "final_model"))
    
    # Plot the collected validation accuracy over epochs
    # plot_accuracy(eval_callback.eval_history)

    # Test the model
    test_results = trainer.evaluate(tokenized_dataset_test)
    print(test_results)

if __name__ == "__main__":
    main()