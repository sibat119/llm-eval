from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def load_data(dataset_name="shawhin/youtube-titles-dpo"):
    """Load and return the dataset."""
    return load_dataset(dataset_name)


def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Load and return the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # set pad token
    return model, tokenizer, model_name


def format_chat_prompt(user_input, system_message="You are a helpful assistant."):
    """
    Formats user input into the chat template format with <|im_start|> and <|im_end|> tags.

    Args:
        user_input (str): The input text from the user.

    Returns:
        str: Formatted prompt for the model.
    """
    
    # Format user message
    user_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n"
    
    # Start assistant's turn
    assistant_prompt = "<|im_start|>assistant\n"
    
    # Combine prompts
    formatted_prompt = user_prompt + assistant_prompt
    
    return formatted_prompt


def setup_generator(model, tokenizer, device='cuda'):
    """Set up and return a text generation pipeline."""
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)


def test_generation(generator, prompt, max_length=100, temperature=0.7):
    """Generate and print output from the model."""
    outputs = generator(prompt, max_length=max_length, truncation=True, 
                       num_return_sequences=1, temperature=temperature)
    print(outputs[0]['generated_text'])
    return outputs


def setup_training_config(model_name):
    """Set up and return the training configuration."""
    ft_model_name = model_name.split('/')[1].replace("Instruct", "DPO")
    
    training_args = DPOConfig(
        output_dir=ft_model_name, 
        logging_steps=25,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        eval_strategy="epoch",
        eval_steps=1,
    )
    
    return training_args, ft_model_name


def train_model(model, tokenizer, training_args, dataset):
    """Train and return the DPO model."""
    trainer = DPOTrainer(
        model=model, 
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
    )
    trainer.train()
    return trainer


def save_model(trainer, path="./dpo/"):
    """Save the trained model."""
    trainer.save_model(path)
    # Uncomment to push to hub
    # model_id = f"shawhin/{ft_model_name}"
    # trainer.push_to_hub(model_id)

def load_and_test_local_model(model_path="./dpo/", example_prompt=None):
    """Load locally saved model and test generation."""
    print("\nTesting locally loaded model:")
    
    # Load model and tokenizer from local path
    model, tokenizer, _ = load_model_and_tokenizer(model_path=model_path)
    device = torch.device('cuda')
    
    # Set up generator with loaded model
    generator = setup_generator(model, tokenizer, device)
    
    # Test generation
    if example_prompt is None:
        example_prompt = "What is machine learning?"
    test_generation(generator, example_prompt)
    
    return model, tokenizer


def main():
    # Load dataset
    dataset = load_data()
    
    # Load model and tokenizer
    model, tokenizer, model_name = load_model_and_tokenizer()
    
    # Set up device
    device = torch.device('cuda')
    
    # Create generator
    generator = setup_generator(model, tokenizer, device)
    
    # Get example prompt
    example_prompt = format_chat_prompt(dataset['valid']['prompt'][0][0]['content'])
    
    # Test generation before training
    print("Output before training:")
    test_generation(generator, example_prompt)
    
    # Set up training configuration
    training_args, ft_model_name = setup_training_config(model_name)
    
    # Train model
    trainer = train_model(model, tokenizer, training_args, dataset)
    
    # Get fine-tuned model
    ft_model = trainer.model
    
    # Set up generator with fine-tuned model
    generator = setup_generator(ft_model, tokenizer, device)
    
    # Test generation after training
    print("\nOutput after training:")
    test_generation(generator, example_prompt)
    
    # Save the model
    save_model(trainer)    
    

    # Test loading and generating from local model
    load_and_test_local_model(example_prompt=example_prompt)


if __name__ == "__main__":
    main()