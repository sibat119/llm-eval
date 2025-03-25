import json
import argparse
import yaml
from datasets import Dataset
from src.utils import model_loader, metrics
from tqdm import tqdm
import random
import pandas as pd

def get_few_shot_prompt(shot, ds, question):
    prompt = """Your goal is to predict what the black-box model will answer to a given question.
Below are examples of inputs (questions) and the black-box model's actual responses:

"""
    
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
        prompt += f"Black-box response: \"{example['model_output']}\"\n\n"
        examples_added += 1
    
    # Add the target question
    prompt += "Now predict the black-box model's response:\n\n"
    prompt += f"Question: \"{question}\"\n"
    prompt += "Black-box response:"
    
    return prompt

def get_few_shot_surrogate(model_name, dataset_path, use_vllm=True, shot=3):
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    if use_vllm:
        model, tokenizer, sampling_params = model_loader.load_model_vllm(model_name, config)
    else:
        model, tokenizer, pipe = model_loader.load_model_pipeline(model_name, config)
    
    # ds = Dataset.from_csv(dataset_path).select(range(20))
    ds = Dataset.from_csv(dataset_path)
    response_list = []
    for idx, example in tqdm(enumerate(ds)):
        response_dict = {}
        prompt = get_few_shot_prompt(shot=shot, ds=ds, question=example['question'])
        
        if use_vllm:
            seqs = model.generate(
                prompt,
                sampling_params=sampling_params,
            )
            answer_text = [seq.outputs[0].text for seq in seqs][0].strip()
        else:
            outputs = pipe(
                prompt, 
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                do_sample=True,
                )
            answer_text = outputs[0][0]['generated_text'][len(prompt[0]):].strip()
        
        response_dict['question'] = example['question']
        response_dict['options'] = example['options']
        response_dict['blackbox_output'] = example['model_output']
        response_dict['surrogate_output'] = answer_text
        response_dict['ground_truth'] = example['ground_truth']
        
        response_list.append(response_dict)
        
    # Write results to JSON file
    
    json_filename = f"data/dataset/qwen_surrogate_{shot}.json"
    with open(json_filename, 'w') as f:
        json.dump(response_list, f, indent=4)
    
    return response_list


def compute_dual_metrics_from_csv(csv_filename):
    """
    Reads the CSV file and computes all metrics with error handling
    """
    try:
        results = {}
        df = pd.read_csv(csv_filename)
        
        # Handle missing values
        df['model_output'] = df['model_output'].fillna('')
        df['ground_truth'] = df['ground_truth'].fillna('')
        df['ground_truth'] = df['ground_truth'].fillna('')
        
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
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Run LLM benchmark evaluation")
        parser.add_argument("--use_vllm", action="store_true", default=False, 
                            help="Use VLLM for inference")
        parser.add_argument("--use_pipeline", action="store_true", default=False,
                            help="Use HuggingFace pipeline for inference")
        parser.add_argument("--shot", type=int, default=3,
                            help="prompt shot count")
        
        return parser.parse_args()
    
    args = parse_args()
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)

    llms = config.get("model_list", [])
    surrogate_llm = "Qwen/Qwen2.5-7B-Instruct"
    candidate_model_data_path = "data/dataset/custom_Qwen_Qwen2.5-7B-Instruct_mmlu_results.csv"
    for llm in llms:
        ds_file_name = config["data_path"] + f"/{llm.replace('/', '_')}_surrogate_responses.csv"
        
        get_few_shot_surrogate(model_name=surrogate_llm, dataset_path=candidate_model_data_path, shot=args.shot)
        results = compute_dual_metrics_from_csv(ds_file_name)
        with open(f"data/surrogate/candidate-{llm.replace('/', '-')}-surrogate{surrogate_llm.replace('/', '-')}.json", "w") as f:
            json.dump(results, f, indent=4)