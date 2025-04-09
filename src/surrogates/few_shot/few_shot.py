import os
import json
import argparse
import yaml
from datasets import Dataset
from src.utils import metrics
from tqdm import tqdm
import random
import pandas as pd
from src.inference import selector
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def generate_and_select_paraphrase(question: str, 
                                 paraphrase_model: T5ForConditionalGeneration,
                                 paraphrase_tokenizer: T5Tokenizer,
                                 sbert_model: SentenceTransformer,
                                 device: torch.device,
                                 num_paraphrases: int = 5,
                                 similarity_threshold: float = 0.8) -> Optional[str]:
    """
    Generate multiple paraphrases and select the most similar one above a threshold.
    
    Args:
        question: Original question to paraphrase
        paraphrase_model: T5 model for paraphrasing
        paraphrase_tokenizer: T5 tokenizer
        sbert_model: SentenceTransformer model for similarity calculation
        device: Device to run models on
        num_paraphrases: Number of paraphrases to generate
        similarity_threshold: Minimum similarity score required
        
    Returns:
        str or None: Selected paraphrase if similarity threshold is met, None otherwise
    """
    # Prepare input for T5
    input_text = f"paraphrase: {question}"
    input_ids = paraphrase_tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(device)
    
    # Generate multiple paraphrases
    outputs = paraphrase_model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        num_return_sequences=num_paraphrases,
        temperature=0.7,
        do_sample=True,
        no_repeat_ngram_size=2
    )
    
    # Decode all paraphrases
    paraphrases = [paraphrase_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    # Calculate similarities using SBERT
    question_embedding = sbert_model.encode([question], convert_to_numpy=True)
    paraphrase_embeddings = sbert_model.encode(paraphrases, convert_to_numpy=True)
    similarities = cosine_similarity(question_embedding, paraphrase_embeddings)[0]
    
    # Find the best paraphrase above threshold
    best_idx = np.argmax(similarities)
    if similarities[best_idx] >= similarity_threshold:
        return paraphrases[best_idx]
    return None

def select_few_shot_examples(ds, shot: int, question: str, selection_strategy: str = "random") -> List[Dict]:
    """
    Select example questions for few-shot prompting using different strategies.
    
    Args:
        ds (Dataset): The dataset containing questions and responses
        shot (int): Number of examples to select
        question (str): The target question to exclude from selection
        selection_strategy (str): Strategy to select examples. Options:
            - "random": Random sampling
            - "similarity": Select based on semantic similarity
            - "similarity_with_paraphrase": Select based on similarity and paraphrase
            
    Returns:
        list: List of dictionaries containing selected example questions and their responses
    """
    # Create list of valid example indices (excluding target question)
    valid_indices = [i for i in range(len(ds)) if ds[i]['question'] != question]
    
    if selection_strategy == "random":
        # Random sampling strategy
        selected_indices = random.sample(valid_indices, min(shot, len(valid_indices)))
        selected_examples = [ds[idx] for idx in selected_indices]
        
    elif selection_strategy in ["similarity", "similarity_with_paraphrase"]:
        # Load SBERT model for similarity calculation
        sbert_model = SentenceTransformer('all-mpnet-base-v2')
            
        # Get embeddings for target question and all valid questions
        target_embedding = sbert_model.encode([question], convert_to_numpy=True)
        valid_questions = [ds[i]['question'] for i in valid_indices]
        valid_embeddings = sbert_model.encode(valid_questions, convert_to_numpy=True)
        
        # Calculate similarities
        similarities = cosine_similarity(target_embedding, valid_embeddings)[0]
        
        # Get indices of top K most similar questions
        top_k_indices = np.argsort(similarities)[-shot:][::-1]
        selected_indices = [valid_indices[i] for i in top_k_indices]
        selected_examples = [ds[idx] for idx in selected_indices]
        
        if selection_strategy == "similarity_with_paraphrase":
            # Load T5 model for paraphrasing
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_name = 't5-base'  # Can be changed to larger models for better quality
            paraphrase_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
            paraphrase_tokenizer = T5Tokenizer.from_pretrained(model_name)
            
            paraphrased_examples = []
            for example in selected_examples:
                # Generate and select best paraphrase
                best_paraphrase = generate_and_select_paraphrase(
                    question=example['question'],
                    paraphrase_model=paraphrase_model,
                    paraphrase_tokenizer=paraphrase_tokenizer,
                    sbert_model=sbert_model,
                    device=device
                )
                
                # If a good paraphrase was found, use it; otherwise keep original
                if best_paraphrase is not None:
                    paraphrased_examples.append({
                        'question': best_paraphrase,
                        'model_output': example['model_output']
                    })
                else:
                    paraphrased_examples.append(example)
            
            selected_examples = paraphrased_examples
            
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")
    
    return selected_examples

def get_few_shot_prompt(shot, ds, question, selection_strategy="random", prompt_id="black_box"):
    
    with open('data/config/surrogate_prompts.yaml', "r") as file:
        prompts = yaml.safe_load(file)
    
    prompt = prompts.get(prompt_id)
    # Get selected examples using the new function
    selected_examples = select_few_shot_examples(ds, shot, question, selection_strategy)
    
    # Format the examples in the prompt
    example_str = ""
    for i, example in enumerate(selected_examples):
        example_str += f"Example {i+1}:\n"
        example_str += f"Question: \"{example['question']}\"\n" 
        example_str += f"Response: \"{example['model_output']}\"\n\n"
    
    prompt = prompt.format(examples=example_str, question=question)
    
    return prompt

def get_few_shot_surrogate(model_name, dataset_path, shot=3, surrogate_datapath="", batch_size=4, cfg=None, selection_strategy="random", prompt_variation="black_box"):
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # batch_size = 2  # Adjust batch size as needed
    ds = Dataset.from_csv(dataset_path)
    # ds = ds.select(range(batch_size))
    # session = selector.select_chat_model(model_name=model_name, cfg=cfg)
    session = None
    
    response_list = []
    # Process examples in batches
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i:i + min(batch_size, len(ds) - i)]
        
        # Create all prompts for the batch
        batch_questions = batch['question']
        batch_prompts = [get_few_shot_prompt(shot=shot, ds=ds, question=q, selection_strategy=selection_strategy, prompt_id=prompt_variation) for q in batch_questions]
        
        batch_answers = session.get_response(user_message=batch_prompts) 
        
        # Create response dictionaries for all examples in the batch
        # Process each item in the batch with clearer variable names
        for idx in range(len(next(iter(batch.values())))):
            # Create response dict directly from batch data
            response_dict = {
                'question': batch['question'][idx],
                'options': batch['options'][idx],
                'blackbox_output': batch['model_output'][idx].replace('<|start_header_id|>assistant\n\n', ''),
                'surrogate_output': batch_answers[idx].replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '').replace('<|start_header_id|>assistant\n\n', ''),
                'ground_truth': batch['ground_truth'][idx],
                'prompt': batch_prompts[idx]
            }
            # breakpoint()
            response_list.append(response_dict)
        
    ds = Dataset.from_list(response_list)
    ds.to_csv(surrogate_datapath)
    
    return response_list


def compute_dual_metrics_from_csv(csv_filename, surrogate_wo_prior_ds_path):
    """
    Reads the CSV file and computes all metrics with error handling
    """
    try:
        results = {}
        df = pd.read_csv(csv_filename)
        surrogate_df_wo_prior = pd.read_csv(surrogate_wo_prior_ds_path)
        
        # Handle missing values
        df['blackbox_output'] = df['blackbox_output'].fillna('')
        df['ground_truth'] = df['ground_truth'].fillna('')
        df['surrogate_output'] = df['surrogate_output'].fillna('')
        df['surrogate_output_wo_prior'] = surrogate_df_wo_prior['model_output'].fillna('').apply(lambda x: x.replace('<|start_header_id|>assistant\n\n', ''))
        
        predictions = df['surrogate_output'].tolist()
        ground_truths = df['ground_truth'].tolist()
        options = df['options'].tolist()
        blackbox_output = df['blackbox_output'].tolist()
        surrogate_output_wo_prior = df['surrogate_output_wo_prior'].tolist()
        
        results['wrt_gt'] = metrics.compute_all_metrics(predictions, ground_truths, options, blackbox_output, surrogate_output_wo_prior)
        
        results['wrt_blackbox'] = metrics.compute_all_metrics_wo_rank(predictions, blackbox_output)
        
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
        parser.add_argument("--candidate",type=str, default="Qwen/Qwen2.5-7B-Instruct",
                            help="Candidate model name")
        parser.add_argument("--surrogate",type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                            help="Surrogate model name")
        parser.add_argument("--selection_strategy",type=str, default="random",
                            help="prompt examples selection strategy")
        parser.add_argument("--prompt_variation",type=str, default="black_box",
                            help="pick the variation of prompt")
        parser.add_argument("--shot", type=int, default=3,
                            help="prompt shot count")
        parser.add_argument("--batch_size",type=int, default=16,
                            help="provide batch size")
        parser.add_argument("--eval",action="store_true", default=False,
                            help="run evaluation script")
        
        
        
        return parser.parse_args()
    
    args = parse_args()
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)

    surrogate_llm = args.surrogate
    candidate_llm  = args.candidate
    surrogate_dir = os.path.join(config['data_path'], 'surrogate', args.sub_field, args.prompt_variation, f"{args.shot}-shot-{args.selection_strategy}-selection")
    os.makedirs(surrogate_dir, exist_ok=True)
    print(surrogate_dir)
    ds_file_name = f"{surrogate_dir}/candidate_{candidate_llm.replace('/', '_')}_surrogate_{surrogate_llm.replace('/', '_')}_responses.csv"
    data_folder = f"{config['data_path']}/{args.sub_field}/0"
    candidate_model_data_path = f"{data_folder}/custom_{candidate_llm.replace('/', '_')}_{config['dataset_name'].replace('/', '_')}_results.csv"
    surrogate_wo_prior_ds_path = f"{data_folder}/custom_{surrogate_llm.replace('/', '_')}_{config['dataset_name'].replace('/', '_')}_results.csv"
    if not args.eval:
        get_few_shot_surrogate(
            model_name=surrogate_llm, 
            dataset_path=candidate_model_data_path, 
            shot=args.shot, 
            surrogate_datapath=ds_file_name,
            batch_size=args.batch_size,
            selection_strategy=args.selection_strategy,
            prompt_variation=args.prompt_variation,
            cfg=config
            )
    results = compute_dual_metrics_from_csv(ds_file_name, surrogate_wo_prior_ds_path)
    with open(f"{surrogate_dir}/candidate-{candidate_llm.replace('/', '-')}-surrogate-{surrogate_llm.replace('/', '-')}.json", "w") as f:
        json.dump(results, f, indent=4)
