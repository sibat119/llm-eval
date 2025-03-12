from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset

# Load the BoolQ dataset
dataset = load_dataset("boolq")

# Display the first few rows of the dataset
print(dataset['train'].to_pandas().head())
 

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
def preprocess_function(examples):
    inputs = [f"Question: {question}  Passage: {passage}" for question, passage in zip(examples['question'], examples['passage'])]
    targets = ['true' if answer else 'false' for answer in examples['answer']]
    
    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=10, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Preprocess the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the validation dataset
eval_results = trainer.evaluate()

# Print the evaluation results
print(f"Evaluation results: {eval_results}")


model = T5ForConditionalGeneration.from_pretrained("./results")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Prepare a new input
input_text = "question: Is the sky blue? context: The sky is blue on a clear day."

# Tokenize the input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate the answer using the model
output_ids = model.generate(input_ids)

# Decode the generated tokens to get the predicted answer
predicted_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the predicted answer
print(f"Predicted answer: {predicted_answer}")  # Predicted answer: yes