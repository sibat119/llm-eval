"""
This file contains code for k-fold cross validation finetuning of a T5 model as a surrogate model.
The model is trained on input-output pairs to mimic the behavior of another model (e.g., LLaMA).
"""

# external imports
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# local imports
# from src.utils.files import get_project_root, get_full_path, path_exists
# from src.utils.strings import now, green, yellow, red

import os
import matplotlib.pyplot as plt
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import load_dataset

def preprocess_function(examples, tokenizer):
    """
    Format the input prompt for T5 and tokenize both input and target.
    Assumes examples has keys: "question", "choices", and "answer".
    """
    inputs = [
        f"question: {q} choices: {', '.join(choices)} answer:" 
        for q, choices in zip(examples["question"], examples["choices"])
    ]
    targets = [str(c[a]) for c, a in zip(examples["choices"], examples["answer"])]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True,)
    
    # Tokenize targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def get_tokenized_dataset(dataset_name, tokenizer):
    """
    Load the dataset and apply tokenization.
    Replace 'dataset_name' with your actual dataset identifier or local script.
    """
    dataset_train = load_dataset(dataset_name, "all", split="auxiliary_train[:1000]")
    dataset_val = load_dataset(dataset_name, "all", split="validation")
    dataset_test = load_dataset(dataset_name, "all", split="test")
    tokenized_dataset_train = dataset_train.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, batch_size=1000)
    tokenized_dataset_validation = dataset_val.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    tokenized_dataset_test = dataset_test.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    return tokenized_dataset_train, tokenized_dataset_validation, tokenized_dataset_test

class EpochEvalCallback(TrainerCallback):
    """
    Custom callback to capture evaluation metrics after each epoch.
    The eval_history list stores tuples of (epoch, eval_accuracy).
    """
    def __init__(self):
        self.eval_history = []

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Assume the last logged dictionary contains the evaluation metrics.
        logs = state.log_history[-1]
        # The key is "eval_accuracy" if our compute_metrics returns {"accuracy": ...}
        if "eval_accuracy" in logs and "epoch" in logs:
            self.eval_history.append((logs["epoch"], logs["eval_accuracy"]))
        return control

def compute_metrics(eval_pred, tokenizer):
    """
    Compute accuracy by comparing the decoded predictions with the labels.
    Adjust the comparison as necessary for your task.
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    correct = sum(
        1 for pred, label in zip(decoded_preds, decoded_labels)
        if pred.strip().lower() == label.strip().lower()
    )
    accuracy = correct / len(decoded_preds) if decoded_preds else 0
    return {"accuracy": accuracy}

def setup_trainer(model, tokenized_dataset_train, tokenized_dataset_validation, tokenizer, eval_callback):
    """
    Set up TrainingArguments and the Trainer.
    We use evaluation_strategy "epoch" so validation is run after every epoch.
    """
    # Wrap compute_metrics to include the tokenizer
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",      # Validate at the end of each epoch
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        load_best_model_at_end=True,
        report_to="none", 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_validation,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[eval_callback],
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
    tokenized_dataset_train, tokenized_dataset_validation, tokenized_dataset_test = get_tokenized_dataset(dataset_name, tokenizer)
    
    # Initialize custom callback for tracking evaluation metrics
    eval_callback = EpochEvalCallback()
    
    # Setup the Trainer
    trainer = setup_trainer(model, tokenized_dataset_train, tokenized_dataset_validation, tokenizer, eval_callback)
    
    breakpoint()
    # Start training (validation will run after each epoch)
    trainer.train()
    
    # Optionally, save the final model
    trainer.save_model(os.path.join("./results", "final_model"))
    
    # Plot the collected validation accuracy over epochs
    plot_accuracy(eval_callback.eval_history)

    # Test the model
    test_results = trainer.evaluate(tokenized_dataset_test)
    print(test_results)

if __name__ == "__main__":
    main()