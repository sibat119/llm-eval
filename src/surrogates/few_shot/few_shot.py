import os
import json
import argparse
import yaml
from datasets import Dataset
from src.utils import model_loader, metrics
from tqdm import tqdm
import random
import pandas as pd
from src.inference import selector

def get_few_shot_prompt(shot, ds, question):
    prompt = """You are a mind reader tasked with predicting how someone will answer a given question. To simplify this process, here are some examples of how the person previously responded to questions on similar topics:
"""
    # Your goal is to predict what the black-box model will answer to a given question. Below are examples of inputs (questions) and the black-box model's actual responses:
    
    # Add examples based on the shot parameter
    # Skip examples that match the target question
    examples_added = 0
    i = 0
    # Create list of valid example indices (excluding target question)
    valid_indices = [i for i in range(len(ds)) if ds[i]['question'] != question]
    
    # Randomly sample shot number of examples
    selected_indices = random.sample(valid_indices, min(shot, len(valid_indices)))
    
    for i, idx in enumerate(selected_indices):
        example = ds[idx]
        prompt += f"Example {i+1}:\n"
        prompt += f"Question: \"{example['question']}\"\n" 
        prompt += f"Person's response:: \"{example['model_output']}\"\n\n"
        examples_added += 1
    
    # Add the target question
    prompt += "Now predict how the person will respond to this question. Your response should only be the persons predicted response.\n\n"
    prompt += f"Question: \"{question}\"\n"
    prompt += "Person's response:"
    
    return prompt

def get_few_shot_surrogate(model_name, dataset_path, shot=3, surrogate_datapath="", batch_size=4, cfg=None):
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # batch_size = 2  # Adjust batch size as needed
    ds = Dataset.from_csv(dataset_path)
    # ds = ds.select(range(batch_size))
    session = selector.select_chat_model(model_name=model_name, cfg=cfg)
    
    response_list = []
    # Process examples in batches
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i:i + min(batch_size, len(ds) - i)]
        
        # Create all prompts for the batch
        batch_questions = batch['question']
        batch_prompts = [get_few_shot_prompt(shot=shot, ds=ds, question=q) for q in batch_questions]
        
        batch_answers = session.get_response(user_message=batch_prompts) 
        
        # Create response dictionaries for all examples in the batch
        # Process each item in the batch with clearer variable names
        for idx in range(len(next(iter(batch.values())))):
            # Create response dict directly from batch data
            response_dict = {
                'question': batch['question'][idx],
                'options': batch['options'][idx],
                'blackbox_output': batch['model_output'][idx],
                'surrogate_output': batch_answers[idx],
                'ground_truth': batch['ground_truth'][idx],
                'prompt': batch_prompts[idx]
            }
            response_list.append(response_dict)
        
    ds = Dataset.from_list(response_list)
    ds.to_csv(surrogate_datapath)
    
    return response_list


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
        options = df['options'].tolist()
        
        results['wrt_gt'] = metrics.compute_all_metrics(predictions, ground_truths, options)
        
        ground_truths = df['blackbox_output'].tolist()
        results['wrt_blackbox'] = metrics.compute_all_metrics_wo_rank(predictions, ground_truths)
        
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
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Run LLM benchmark evaluation")
        parser.add_argument("--sub_field",type=str, default="high_school_computer_science",
                            help="provide field name")
        parser.add_argument("--candidate",type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                            help="Candidate model name")
        parser.add_argument("--surrogate",type=str, default="Qwen/Qwen2.5-7B-Instruct",
                            help="Surrogate model name")
        parser.add_argument("--shot", type=int, default=3,
                            help="prompt shot count")
        parser.add_argument("--batch_size",type=int, default=16,
                            help="provide batch size")
        
        return parser.parse_args()
    
    args = parse_args()
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)

    surrogate_llm = args.surrogate
    candidate_llm  = args.candidate
    surrogate_dir = os.path.join(config['data_path'], 'surrogate', args.sub_field, f"{args.shot}-shot")
    os.makedirs(surrogate_dir, exist_ok=True)
    ds_file_name = f"{surrogate_dir}/candidate_{candidate_llm.replace('/', '_')}_surrogate_{surrogate_llm.replace('/', '_')}_responses.csv"
    data_folder = f"{config['data_path']}/{args.sub_field}/0"
    candidate_model_data_path = f"{data_folder}/custom_{candidate_llm.replace('/', '_')}_{config['dataset_name'].replace('/', '_')}_results.csv"
    
    get_few_shot_surrogate(
        model_name=surrogate_llm, 
        dataset_path=candidate_model_data_path, 
        shot=args.shot, 
        surrogate_datapath=ds_file_name,
        batch_size=args.batch_size,
        cfg=config
        )
    results = compute_dual_metrics_from_csv(ds_file_name)
    with open(f"{surrogate_dir}/candidate-{candidate_llm.replace('/', '-')}-surrogate-{surrogate_llm.replace('/', '-')}.json", "w") as f:
        json.dump(results, f, indent=4)