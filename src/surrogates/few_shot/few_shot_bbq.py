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
from typing import List, Dict, Optional, Tuple
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

def get_similar_questions(
    target_question: str, 
    target_context: str,
    all_questions: List[str],
    all_contexts: List[str],
    current_ds: Dataset,
    top_k: int = 10,
    selection_strategy="similarity") -> List[Tuple[float, str, str, str]]:
    """
    Get top k similar questions based on semantic similarity with different selection strategies.
    Uses efficient precomputed embeddings for faster calculation.
    
    Args:
        target_question: The question to find similar questions for
        target_context: The context for the target question
        all_questions: List of all questions to compare against
        all_contexts: List of all contexts to compare against
        current_ds: Current dataset to get responses from
        sbert_model: SentenceTransformer model for similarity calculation
        top_k: Number of similar questions to return
        selection_strategy: Strategy for selecting examples
        
    Returns:
        List of tuples containing (similarity_score, question, response, context)
    """
    # Get the target item with context
    target_item = next((i for i in current_ds if i['question'] == target_question), None)
    if not target_item:
        return []
    
    # Prepare mapping from questions to items for efficient lookup
    q2item = {i['question']: i for i in current_ds}
    
    if selection_strategy == "select_by_context":
        # Precompute all context embeddings
        # Initialize SBERT model for similarity calculation
        sbert_model = SentenceTransformer('all-mpnet-base-v2')
        context_embeddings = sbert_model.encode(all_contexts, convert_to_numpy=True)
        ctx2emb = dict(zip(all_contexts, context_embeddings))
        
        # Create a mapping from context to item
        ctx2item = {}
        for item in current_ds:
            if 'context' in item and item['context']:
                ctx2item[item['context']] = item
        
        # Get target context embedding
        if target_context not in ctx2emb:
            target_ctx_emb = sbert_model.encode([target_context], convert_to_numpy=True)[0]
        else:
            target_ctx_emb = ctx2emb[target_context]
        
        # Find similar contexts
        target_ctx_emb = target_ctx_emb.reshape(1, -1)
        
        # Build a list of other contexts and their embeddings
        others = []
        for ctx in all_contexts:
            if ctx != target_context and ctx in ctx2item:
                others.append((ctx, ctx2emb[ctx]))
        
        if not others:
            return []
            
        ctx_list, embs = zip(*others)
        sims = cosine_similarity(target_ctx_emb, np.vstack(embs))[0]
        
        # Pair and sort
        paired = sorted(zip(sims, ctx_list), key=lambda x: x[0], reverse=True)[:top_k]
        
        # Attach the rest of the info
        return [(sim, ctx2item[ctx]['question'], ctx2item[ctx]['model_output'], ctx) 
                for sim, ctx in paired]
    
    elif selection_strategy == "select_by_question":
        # Precompute all question embeddings
        # Initialize SBERT model for similarity calculation
        sbert_model = SentenceTransformer('all-mpnet-base-v2')
        question_embeddings = sbert_model.encode(all_questions, convert_to_numpy=True)
        q2emb = dict(zip(all_questions, question_embeddings))
        
        def top_k_similar(target_q, k):
            # Get target embedding
            if target_q not in q2emb:
                target_emb = sbert_model.encode([target_q], convert_to_numpy=True)[0]
            else:
                target_emb = q2emb[target_q]
            
            target_emb = target_emb.reshape(1, -1)
            
            # Build a list of other embeddings and questions
            others = [(q, q2emb[q]) for q in all_questions if q != target_q and q in q2item]
            
            if not others:
                return []
                
            qs, embs = zip(*others)
            sims = cosine_similarity(target_emb, np.vstack(embs))[0]
            
            # Pair and sort
            paired = sorted(zip(sims, qs), key=lambda x: x[0], reverse=True)[:k]
            
            # Attach the rest of the info
            return [(sim, q, q2item[q]['model_output'], q2item[q].get('context', ''))
                    for sim, q in paired]
        
        return top_k_similar(target_question, top_k)
    
    elif selection_strategy == "select_by_both":
        # Create combined texts
        # Initialize SBERT model for similarity calculation
        sbert_model = SentenceTransformer('all-mpnet-base-v2')
        combined_texts = [f"{ctx} {q}" for q, ctx in zip(all_questions, all_contexts)]
        
        # Precompute all combined embeddings
        combined_embeddings = sbert_model.encode(combined_texts, convert_to_numpy=True)
        idx2emb = dict(zip(range(len(combined_texts)), combined_embeddings))
        
        # Create mapping from index to question and context
        idx2q = {i: q for i, q in enumerate(all_questions)}
        idx2ctx = {i: ctx for i, ctx in enumerate(all_contexts)}
        
        # Get target combined embedding
        target_combined = f"{target_context} {target_question}"
        target_emb = sbert_model.encode([target_combined], convert_to_numpy=True)[0].reshape(1, -1)
        
        # Build a list of other embeddings and their indices
        others = []
        for idx, q in idx2q.items():
            if q != target_question and q in q2item:
                others.append((idx, idx2emb[idx]))
        
        if not others:
            return []
            
        indices, embs = zip(*others)
        sims = cosine_similarity(target_emb, np.vstack(embs))[0]
        
        # Pair and sort
        paired = sorted(zip(sims, indices), key=lambda x: x[0], reverse=True)[:top_k]
        
        # Attach the rest of the info
        return [(sim, idx2q[idx], q2item[idx2q[idx]]['model_output'], idx2ctx[idx]) 
                for sim, idx in paired]
    
    elif selection_strategy == "generate_by_surrogate":
        # This would involve using a surrogate model to generate examples
        print("Warning: 'generate_by_surrogate' strategy not fully implemented. Using 'select_by_question' instead.")
        
        pass
        
    else:  # Default random selection
        filtered_questions = [q for q in all_questions if q != target_question]
        selected_questions = random.sample(filtered_questions, min(top_k, len(filtered_questions)))
        result = []
        for q in selected_questions:
            # Find response and context for this question
            item = next((i for i in current_ds if i['question'] == q), None)
            if item:
                result.append((0.0, q, item['model_output'], item.get('context', '')))
        return result

def collect_responses_for_questions(similar_questions: List[Tuple[float, str, str, str]],
                                  all_ds: List[Tuple[str, Dataset]]) -> List[Tuple[str, List[Dict], str]]:
    """
    Collect responses for similar questions from all models.
    
    Args:
        similar_questions: List of similar questions with their scores, responses, and contexts
        all_ds: List of (model_name, dataset) tuples
        
    Returns:
        List of tuples containing (question, responses, context) where responses is a list of dicts
        with model_name, response, and options
    """
    example_responses = []
    for _, question, _, context in similar_questions:
        responses = []
        for model_name, model_ds in all_ds:
            item = next((i for i in model_ds if i['question'] == question and i['context'] == context), None)
            if item is not None:
                responses.append({
                    'llm_name': model_name,
                    'response': item['model_output'],
                    'options': item['options']
                })
        example_responses.append((question, responses, context))
    return example_responses

def format_prompt_examples(examples: List[Tuple[str, List[Dict], str]],
                         current_model: str) -> str:
    """
    Format examples into a string for the prompt.
    
    Args:
        examples: List of tuples containing (question, responses, context)
        current_model: Name of the current model to get responses from
        
    Returns:
        Formatted string of examples
    """
    example_str = ""
    for i, (question, responses, context) in enumerate(examples):
        # Use the response from the current model
        model_response = next((r['response'] for r in responses if r['llm_name'] == current_model), None)
        if model_response is None:
            continue
            
        options = next((r['options'] for r in responses if r['llm_name'] == current_model), None)
        if options is None:
            continue
            
        option_text = "\n".join([f"- {opt}" for opt in eval(options)])
        response_text = model_response.replace('<|start_header_id|>assistant\n\n', '')
        
        example_str += f"Example {i+1}:\n"
        if context:
            example_str += f"Context: \"{context}\"\n" 
        example_str += f"Question: \"{question}\"\n" 
        example_str += f"Options: \"{option_text}\"\n" 
        example_str += f"Response: \"{response_text}\"\n\n"
    return example_str

def create_few_shot_prompt(llm_response_path_pairs: List[Tuple[str, str]], 
                         shot: int, 
                         selection_strategy: str = "random", 
                         prompt_variation: str = "black_box",
                         prompt_datapaths: List[Tuple[str, str]] = None,
                         llms: List[str] = None):
    """
    Create prompts for few-shot learning and save them to files.
    
    Args:
        llm_response_path_pairs: List of tuples containing (llm_name, response_path) pairs
        shot: Number of examples to use in few-shot prompting
        selection_strategy: Strategy for selecting examples ("random", "similarity", etc.)
        prompt_variation: The type of prompt template to use
        prompt_datapaths: List of tuples containing (llm_name, prompt_path) pairs
        shared_examples: Whether to use the same example set for all models
    """
    # Load candidate model responses for all models
    all_ds = []
    for llm_name, response_path in llm_response_path_pairs:
        ds = Dataset.from_csv(response_path)
        all_ds.append((llm_name, ds))
    
    # Create prompts for each model
    prompt_lists = {llm_name: [] for llm_name, _ in llm_response_path_pairs}
    
    # Get all questions from the first dataset
    base_ds = all_ds[0][1]
    base_ds = base_ds.select(range(min(512, len(base_ds))))
    
    all_questions = [item['question'] for item in base_ds]
    all_contexts = [item.get('context', '') for item in base_ds]
    
    # Get the prompt template
    with open('data/config/surrogate_prompts_bbq.yaml', "r") as file:
        prompts = yaml.safe_load(file)
    prompt_template = prompts.get(prompt_variation)
    
    # Precompute similar questions for all questions in base_ds
    similar_questions_lookup = {}
    
    print("Precomputing similar questions for all questions...")
    for idx, item in enumerate(tqdm(base_ds, desc="Building similarity index")):
        question_id = idx  # Use index as ID for simplicity
        question = item['question']
        context = item.get('context', '')
        
        # Get similar questions for this question
        similar_q = get_similar_questions(
            target_question=question,
            target_context=context,
            all_questions=all_questions,
            all_contexts=all_contexts,
            current_ds=base_ds,  # Use base_ds for similarity calculation
            top_k=shot*2,  # Get twice as many as needed
            selection_strategy=selection_strategy,
        )
        
        # Store in lookup dictionary using both ID and question as composite key
        similar_questions_lookup[(question_id, question, context)] = similar_q
    
    # Process each model's dataset
    for llm_name, ds in all_ds:
        print(f"Creating prompts for {llm_name}...")
        ds = ds.select(range(min(512, len(ds))))
        for idx, item in enumerate(tqdm(ds, desc=f"Processing {llm_name}")):
            question = item['question']
            context = item['context']
                
            # Get precomputed similar questions
            similar_questions = similar_questions_lookup[(idx, question, context)]
            
            # Collect responses for these questions from all models
            example_responses = collect_responses_for_questions(similar_questions, all_ds)
            
            # Take top questions for the prompt
            top_examples = example_responses[:shot]
            
            # Format the examples in the prompt
            example_str = format_prompt_examples(top_examples, llm_name)
            
            # Format the final prompt
            option_text = "\n".join([f"- {opt}" for opt in eval(item['options'])])
            
            # Include context if available, otherwise use empty string
            context_text = ""
            if 'context' in item and item['context']:
                context_text = f"{item['context']}\n"
            
            # Prepare prompt components
            prompt_components = {
                'examples': example_str,
                'question': item['question'],
                'options': option_text
            }
            
            # Add context if available in the template
            if '{context}' in prompt_template:
                prompt_components['context'] = context_text
                
            prompt = prompt_template.format(**prompt_components)
            
            # Create response dict
            prompt_dict = {
                'question': item['question'],
                'options': item['options'],
                'model_output': item['model_output'],
                'ground_truth': item['ground_truth'],
                'prompt': prompt
            }
            if 'context' in item:
                prompt_dict['context'] = item['context']
                
            prompt_lists[llm_name].append(prompt_dict)
    
    # Save prompts to CSV for each model
    for llm_name, prompt_list in prompt_lists.items():
        prompt_ds = Dataset.from_list(prompt_list)
        # Find the corresponding prompt path for this model
        prompt_path = next(path for name, path in prompt_datapaths if name == llm_name)
        prompt_ds.to_csv(prompt_path)
    
    return prompt_lists

def get_few_shot_surrogate(model_name, dataset_path, surrogate_datapath="", batch_size=4, cfg=None):
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # batch_size = 2  # Adjust batch size as needed
    breakpoint()
    ds = Dataset.from_csv(dataset_path)
    ds = ds.select(range(batch_size))
    session = selector.select_chat_model(model_name=model_name, cfg=cfg)
    # session = None
    
    response_list = []
    # Process examples in batches
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i:i + min(batch_size, len(ds) - i)]
        
        # Create all prompts for the batch
        # batch_questions = batch['question']
        batch_prompts = batch['prompt']
        
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
        df['prompt'] = df['prompt'].fillna('')
        
        
        predictions = df['surrogate_output'].tolist()
        ground_truths = df['ground_truth'].tolist()
        options = df['options'].tolist()
        blackbox_output = df['blackbox_output'].tolist()
        surrogate_output_wo_prior = df['surrogate_output_wo_prior'].tolist()
        prompts = df['prompt'].tolist()
        
        results['wrt_gt'] = metrics.compute_all_metrics(predictions, ground_truths, options, blackbox_output, surrogate_output_wo_prior, prompts)
        
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
        parser.add_argument("--dataset_name",type=str, default="heegyu/bbq",
                            help="provide field name")
        parser.add_argument("--sub_field",type=str, default="Gender_identity",
                            help="provide field name")
        parser.add_argument("--pair",type=str, default="Qwen/Qwen2.5-7B-Instruct;meta-llama/Llama-3.1-8B-Instruct",
                            help="Candidate model name")
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
        parser.add_argument("--create_prompt",action="store_true", default=False,
                            help="Create only few shot prompt")
        
        
        
        return parser.parse_args()
    
    args = parse_args()
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)

    surrogate_llm = args.surrogate
    candidate_llm  = args.candidate
    dataset_name = args.dataset_name
    surrogate_dir = os.path.join(config['data_path'], dataset_name.replace('/', '_'), 'surrogate', args.sub_field, args.prompt_variation, f"{args.shot}-shot-{args.selection_strategy}-selection")
    os.makedirs(surrogate_dir, exist_ok=True)
    print(surrogate_dir)
    ds_file_name = f"{surrogate_dir}/candidate_{candidate_llm.replace('/', '_')}_surrogate_{surrogate_llm.replace('/', '_')}_responses.csv"
    data_folder = f"{config['data_path']}/{dataset_name.replace('/', '_')}/{args.sub_field}"
    prompt_ds_file_name = f"{data_folder}/candidate-{candidate_llm.replace('/', '_')}-shot-{args.shot}-selection-strategy-{args.selection_strategy}-prompt-variation-{args.prompt_variation}-prompt.csv"
    surrogate_wo_prior_ds_path = f"{data_folder}/custom_{surrogate_llm.replace('/', '_')}_{dataset_name.replace('/', '_')}_results.csv"
    if not args.eval:
        if args.create_prompt:
            llms = args.pair.split(";")
            candidate_model_data_paths = []
            prompt_ds_file_names = []
            # Create a list of tuples containing (llm_name, data_path) pairs
            for llm in llms:
                candidate_model_data_path = f"{data_folder}/custom_{llm.replace('/', '_')}_{dataset_name.replace('/', '_')}_results.csv"
                prompt_ds_file_name = f"{data_folder}/candidate-{llm.replace('/', '_')}-shot-{args.shot}-selection-strategy-{args.selection_strategy}-prompt-variation-{args.prompt_variation}-prompt.csv"
                # Append tuples to the lists
                candidate_model_data_paths.append((llm, candidate_model_data_path))
                prompt_ds_file_names.append((llm, prompt_ds_file_name))
                
                # Check if the data path exists
                if not os.path.exists(candidate_model_data_path):
                    print(f"Warning: Data path not found for {llm}: {candidate_model_data_path}")
            
            # Ensure we have data for all models
            if all(os.path.exists(path) for _, path in candidate_model_data_paths):
                create_few_shot_prompt(
                    llm_response_path_pairs=candidate_model_data_paths,
                    shot=args.shot,
                    selection_strategy=args.selection_strategy,
                    prompt_variation=args.prompt_variation,
                    prompt_datapaths=prompt_ds_file_names,
                    llms=llms  # Ensure all models use the same example set
                )
            else:
                print("Error: Some model data paths are missing. Please run the response generation for all models first.")
        else:
            if not os.path.exists(prompt_ds_file_name):
                raise ValueError("Path not found for the dataset.")
            get_few_shot_surrogate(
                model_name=surrogate_llm, 
                dataset_path=prompt_ds_file_name, 
                surrogate_datapath=ds_file_name,
                batch_size=args.batch_size,
                cfg=config
                )
    results = compute_dual_metrics_from_csv(ds_file_name, surrogate_wo_prior_ds_path)
    
    agree_to_disagree_cases = results["wrt_gt"]['transition_metrics']['agreement_transitions']['agree_to_disagree_samples']
    disagree_to_agree_cases = results["wrt_gt"]['transition_metrics']['agreement_transitions']['disagree_to_agree_samples']
    agree_to_disagree_cases_ds = Dataset.from_list(agree_to_disagree_cases)
    disagree_to_agree_cases_ds = Dataset.from_list(disagree_to_agree_cases)
    agree_to_disagree_cases_ds.to_csv(f"{surrogate_dir}/agreement-to-disagreement-candidate-{candidate_llm.replace('/', '-')}-surrogate-{surrogate_llm.replace('/', '-')}.csv")
    disagree_to_agree_cases_ds.to_csv(f"{surrogate_dir}/disagreement-to-agreement-candidate-{candidate_llm.replace('/', '-')}-surrogate-{surrogate_llm.replace('/', '-')}.csv")
    with open(f"{surrogate_dir}/candidate-{candidate_llm.replace('/', '-')}-surrogate-{surrogate_llm.replace('/', '-')}.json", "w") as f:
        json.dump(results, f, indent=4)
