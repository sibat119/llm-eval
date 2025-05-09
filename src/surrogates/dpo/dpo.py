import os
import yaml
import json
from datasets import load_dataset, Dataset, DatasetDict
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from tqdm import tqdm
import pandas as pd
import csv
import re
from src.utils.model_loader import load_model, load_model_vllm, load_model_pipeline
from src.utils import metrics

def create_dataset(candidate_model="meta-llama/Llama-3.1-8B-Instruct", surrogate_model="Qwen/Qwen2.5-7B-Instruct"):
    data_map = {
        "allenai/OLMo-2-1124-7B-Instruct": "",
        "Qwen/Qwen2.5-7B-Instruct": "data/dataset/full/custom_Qwen_Qwen2.5-7B-Instruct_mmlu_results.csv",
        "meta-llama/Llama-3.1-8B-Instruct": "data/dataset/full/custom_meta-llama_Llama-3.1-8B-Instruct_mmlu_results.csv",
    }
    surrogate_ds = Dataset.from_csv(data_map[surrogate_model])    
    candidate_ds = Dataset.from_csv(data_map[candidate_model]) 
       
    if len(surrogate_ds) != len(candidate_ds):
        min_len = min(len(surrogate_ds), len(candidate_ds))
        surrogate_ds = surrogate_ds.select(range(min_len))
        candidate_ds = candidate_ds.select(range(min_len))
    
    # Create new dataset in DPO format
    dpo_data = []
    for i, _ in enumerate(tqdm(range(len(candidate_ds)), desc="Creating DPO dataset")):
        prompt_str = candidate_ds['prompt'][i]
        prompt_list = eval(prompt_str)
        user_prompt = prompt_list[-1]
        dpo_example = {
            'prompt': [user_prompt],
            'chosen': [{'content': candidate_ds['model_output'][i], 'role': 'assistant'}],
            'rejected': [{'content': surrogate_ds['model_output'][i], 'role': 'assistant'}]
        }
        dpo_data.append(dpo_example)
    
    # Convert to dataset
    dpo_dataset = Dataset.from_list(dpo_data)
    
    # Create a safe filename by removing problematic characters
    def create_safe_filename(name):
        # Replace special characters with underscores
        safe_name = re.sub(r'[<>:/\\|?*.-]', '_', name)
        # Remove any duplicate underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        return safe_name
    
    candidate_name = create_safe_filename(candidate_model.split('/')[-1])
    surrogate_name = create_safe_filename(surrogate_model.split('/')[-1])
    
    # Create a safe output path
    output_path = f"data/dataset/dpo/dpo_{candidate_name}_vs_{surrogate_name}.csv"
    
    # Convert to pandas and save with careful quoting settings
    dpo_df = dpo_dataset.to_pandas()
    dpo_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"DPO dataset created and saved to {output_path}")
    return output_path

def apply_chat_template(tokenizer, prompts):
    model_name = tokenizer.name_or_path
    if 'allenai/OLMo-2-1124-7B-Instruct' in model_name or 'Qwen/Qwen2.5-7B-Instruct' in model_name:
        text = tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = tokenizer.apply_chat_template(
            prompts,
            tokenize=False
        )
    
    return text

def load_data(dataset_name="shawhin/youtube-titles-dpo"):
    """Load and return the dataset."""
    if dataset_name =="shawhin/youtube-titles-dpo":
        # This is a HuggingFace dataset path (e.g., "shawhin/youtube-titles-dpo")
        return load_dataset(dataset_name)
    elif dataset_name.endswith('.csv'):
        # This is a local CSV file path
        # Load full dataset
        dataset = load_dataset("csv", data_files=dataset_name)
        
        # Split into train/validation/test (80-10-10)
        splits = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
        train_valid = splits["train"]
        test = splits["test"]
        
        # Split the remaining 80% into train and validation
        splits = train_valid.train_test_split(test_size=0.125, shuffle=True, seed=42) # 0.125 of 80 is 10% of total
        
        return DatasetDict({
            "train": splits["train"],
            "valid": splits["test"], 
            "test": test
        })
    else:
        raise ValueError(f"Invalid dataset name format: {dataset_name}. "
                        "Must be either a HuggingFace dataset path or a CSV file path")


def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Load and return the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name,)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # set pad token
    return model, tokenizer, model_name


def format_chat_prompt(prompt_str, tokenizer):
    """
    Formats user input into the chat template format with <|im_start|> and <|im_end|> tags.

    Args:
        user_input (str): The input text from the user.

    Returns:
        str: Formatted prompt for the model.
    """
    messages = []
    sys_msg = "You are a helpful AI assistant answering questions. Provide only accurate answers in natural language."
    messages.append({"role": "system", "content": sys_msg})
    messages.append(prompt_str)
    
    # Combine prompts
    formatted_prompt = apply_chat_template(tokenizer=tokenizer, prompts=messages)
    
    return formatted_prompt


def setup_generator(model, tokenizer, device='cuda'):
    """Set up and return a text generation pipeline."""
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


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
    model, tokenizer, _ = load_model_and_tokenizer(model_name=model_path)
    device = torch.device('cuda')
    
    # Set up generator with loaded model
    generator = setup_generator(model, tokenizer, device)
    
    # Test generation
    if example_prompt is None:
        example_prompt = "What is machine learning?"
    test_generation(generator, example_prompt)
    
    return model, tokenizer


def get_surrogate_responses(model_name, dataset_path, use_vllm=True, surrogate_datapath="", batch_size=4):
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    ds = Dataset.from_csv(dataset_path).select(range(100))
    
    if use_vllm:
        model, tokenizer, sampling_params = load_model_vllm(model_name, config)
    else:
        model, tokenizer, pipe = load_model_pipeline(model_name, config)
    
    response_list = []
    # Process examples in batches
    
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i:i + min(batch_size, len(ds) - i)]
        
        # Create all prompts for the batch
        batch_prompts = [eval(example)[-1] for example in batch['prompt']]
        
        
        batch_prompts = [format_chat_prompt(prompt_str=p, tokenizer=tokenizer) for p in batch_prompts]
        
        # Process the batch with vLLM or pipeline
        if use_vllm:
            # vLLM handles batching efficiently
            seqs = model.generate(
                batch_prompts,
                sampling_params=sampling_params,
            )
            batch_answers = [seq.outputs[0].text for seq in seqs]
        else:
            # Process non-vllm batches one at a time
            batch_answers = []
            for prompt in batch_prompts:
                outputs = pipe(
                    prompt, 
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    do_sample=True,
                )
                answer = outputs[0][0]['generated_text'][len(prompt):].strip()
                batch_answers.append(answer)
        
        # Create response dictionaries for all examples in the batch
        for question, options, model_output, prompt, answer, ground_truth in zip(batch['question'], batch['options'], batch['model_output'], batch_prompts, batch_answers, batch['ground_truth']):
            if '<|start_header_id|>assistant<|end_header_id|>' in answer:
                answer = answer.replace("<|start_header_id|>assistant<|end_header_id|>", "")
            response_dict = {
                'question': question,
                'options': options,
                'prompt': prompt,
                'blackbox_output': model_output,
                'surrogate_output': answer,
                'ground_truth': ground_truth
            }
            response_list.append(response_dict)
        
    ds = Dataset.from_list(response_list)
    ds.to_csv(surrogate_datapath)
    
    return response_list


def main():
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)
    data_path = create_dataset()
    # Load dataset
    # data_path = "data/dataset/dpo/dpo_Llama_3_1_8B_Instruct_vs_Qwen2_5_7B_Instruct.csv"
    dataset = load_data(dataset_name=data_path)
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model, tokenizer = load_model(model_name=model_name, config=config)
    # model, tokenizer, model_name = load_model_and_tokenizer(model_name=surrogate_model)
    
    # Set up device
    device = torch.device('cuda')
    
    # Create generator
    generator = setup_generator(model, tokenizer, device)
    
    # Get example prompt
    example_prompt = format_chat_prompt(tokenizer=tokenizer, prompt_str=dataset['valid']['prompt'][0])
    
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

def compute_dual_metrics_from_csv(csv_filename):
    """
    Reads the CSV file and computes all metrics with error handling
    """
    try:
        results = {}
        df = pd.read_csv(csv_filename)
        
        # Handle missing values
        df['blackbox_output'] = df['blackbox_output'].fillna('')
        df['ground_truth'] = df['ground_truth'].fillna('')
        df['surrogate_output'] = df['surrogate_output'].fillna('')
        
        predictions = df['surrogate_output'].tolist()
        ground_truths = df['ground_truth'].tolist()
        
        results['wrt_gt'] = metrics.compute_all_metrics(predictions, ground_truths)
        
        ground_truths = df['blackbox_output'].tolist()
        results['wrt_blackbox'] = metrics.compute_all_metrics(predictions, ground_truths)
        
        return results
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return {
            'exact_match': 0.0,
            'f1_score': 0.0,
            'rouge_scores': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
            'bleu_score': 0.0,
            'sbert_similarity': 0.0
        }

if __name__ == "__main__":
    # main()
    surrogate_response_path = "data/dataset/surrogate/dpo_qwen.csv"
    get_surrogate_responses(
        model_name="./dpo",
        dataset_path="data/dataset/full/custom_meta-llama_Llama-3.1-8B-Instruct_mmlu_results.csv",
        surrogate_datapath=surrogate_response_path,
        batch_size=16
    )
    results = compute_dual_metrics_from_csv(surrogate_response_path)
    print(results)
    with open(f"data/dataset/surrogate/dpo_qwen_result.json", "w") as f:
        json.dump(results, f, indent=4)
    # 