import pandas as pd
import csv
from datasets import load_dataset, DownloadMode, Dataset
import yaml
from tqdm import tqdm
import argparse
import random
import os
from src.utils import metrics
from src.inference import selector

def format_prompt(examples, dataset_name):
    """
    Format prompts for a batch of examples.
    
    Args:
        examples: Either a single example dict or a batch of examples (list or dataset batch)
        dataset_name: Name of the dataset being used
        
    Returns:
        A single prompt string if input is a single example, or a list of prompts if input is a batch
    """
    
    # Batch case
    prompts = []
    system_messages = []
    # Handle batch from HuggingFace dataset
    if dataset_name == "cais/mmlu":
        questions = examples["question"]
        options = examples["choices"]
        for q, opts in zip(questions, options):
            option_text = "\n".join([f"- {opt}" for i, opt in enumerate(opts)])
            prompt = f"Given the following question and candidate answers, choose the best answer. Do not include option labels (A, B, C, D); respond only with natural answer text. Provide only the answer text in your response. \nQuestion: {q}\nOptions: {option_text}."
            system_message = "You are a helpful AI assistant answering questions. Provide only accurate answers in natural language."
            
            prompts.append(prompt)
            system_messages.append(system_message)
    elif dataset_name == "meta-llama/Llama-3.2-3B-evals" or dataset_name == "meta-llama/Llama-3.2-3B-Instruct-evals":
        prompts.extend(examples["input_final_prompts"])
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return prompts, system_messages

def get_mmlu_example(examples, dataset_name):
    """
    Get the question, options, and ground truth from a batch of examples.
    
    Args:
        examples: List of examples/batch from dataset
        dataset_name: Name of dataset being used
        
    Returns:
        Tuple of lists containing questions, options, and ground truths
    """
    questions = []
    options_list = []
    ground_truths = []
    
    # Handle batch from HuggingFace dataset
    if dataset_name == "cais/mmlu":
        questions.extend(examples["question"])
        options_list.extend(examples["choices"])
        ground_truths.extend([choices[answer] for choices, answer in zip(examples["choices"], examples["answer"])])
    elif dataset_name == "meta-llama/Llama-3.2-3B-evals" or dataset_name == "meta-llama/Llama-3.2-3B-Instruct-evals":
        questions.extend(examples["input_question"])
        options_list.extend(examples["input_choice_list"]) 
        ground_truths.extend(examples["input_correct_responses"])
            
    return questions, options_list, ground_truths

def get_custom_mmlu_response(
    model_name, csv_filename, data_sub_field, cfg, batch_size
):
    dataset_name = "cais/mmlu"
    # data_sub_field = "high_school_computer_science"
    dataset = load_dataset(dataset_name, data_sub_field, split="test")    
    session = selector.select_chat_model(cfg=cfg, model_name=model_name)
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # CSV header
        writer.writerow(["question", "options", "prompt", "model_output", "ground_truth"])
        # Process examples in batches
        # batch_size = 2
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]
            system_messages = None
            # Process batch of examples
            if dataset_name == "cais/mmlu":
                questions, options_list, ground_truths = get_mmlu_example(batch, dataset_name)
                # Use format_prompt with the constructed example dictionaries
                prompts, system_messages = format_prompt(batch, dataset_name)
            else:
                # For other dataset formats, use the existing function
                questions, options_list, ground_truths = zip(*[get_mmlu_example(ex, dataset_name) for ex in batch])
                
                prompts = format_prompt(batch, dataset_name)
            
            ground_truths = [gt[0] if isinstance(gt, list) else gt for gt in ground_truths]
            
            answer_text = session.get_response(user_message=prompts, system_message=system_messages, clean_output=True)
            
            # breakpoint()
            # Write batch of rows
            for question, options, prompt, answer, ground_truth in zip(questions, options_list, prompts, answer_text, ground_truths):
                if '<|start_header_id|>assistant<|end_header_id|>\n\n' in answer:
                    answer = answer.replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '')
                writer.writerow([question, options, prompt, answer, ground_truth])
            
    
    print(f"Saved evaluation results to {csv_filename}")

if __name__ == "__main__":
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Run LLM benchmark evaluation")
        parser.add_argument("--shot", type=int, default=3,
                            help="prompt shot count")
        parser.add_argument("--model_name",type=str, default="Qwen/Qwen2.5-7B-Instruct",
                            help="provide model name")
        parser.add_argument("--sub_field",type=str, default="high_school_computer_science",
                            help="provide field name")
        parser.add_argument("--batch_size",type=int, default=16,
                            help="provide batch size")
        
        return parser.parse_args()
    
    args = parse_args()
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)
        
    model_name = args.model_name if args.model_name else config['model_name']
    
    csv_dir = f"{config['data_path']}/{args.sub_field}/{args.shot}"
    os.makedirs(csv_dir, exist_ok=True)
    csv_file_name = f"{csv_dir}/custom_{model_name.replace('/', '_')}_{config['dataset_name'].replace('/', '_')}_results.csv"
    get_custom_mmlu_response(
        model_name=model_name,
        csv_filename=csv_file_name,
        data_sub_field=args.sub_field,
        cfg=config,
        batch_size=args.batch_size
    )
    metrics = metrics.compute_metrics_from_csv(csv_file_name)
    print(f"Result: {metrics}")
    # Write metrics to results file
    results_file = os.path.join(csv_dir, "results.txt")
    with open(results_file, "a") as f:
        f.write(f"{model_name}: {metrics}\n")
    print(f"Saved metrics to {results_file}")
    