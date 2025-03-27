import pandas as pd
import csv
from datasets import load_dataset, DownloadMode, Dataset
from ..utils.model_loader import load_model, load_model_pipeline, load_model_vllm, get_response_from_hub
import yaml
from tqdm import tqdm
import argparse
import random
from src.utils import metrics

def format_prompt(example, dataset_name):
    # Format: "Question: <question text>\nOptions: A. <opt1> B. <opt2> ...\nAnswer:"
    if dataset_name == "cais/mmlu":
        question = example["question"]
        options = example["choices"]
        option_text = "\n".join([f"- {opt}" for i, opt in enumerate(options)])
        prompt = f"Given the following question and candidate answers, choose the best answer. Do not include option labels (A, B, C, D); respond only with natural answer text. Provide only the answer text in your response. \nQuestion: {question}\nOptions: {option_text}."
    elif dataset_name == "meta-llama/Llama-3.2-3B-evals" or dataset_name == "meta-llama/Llama-3.2-3B-Instruct-evals":
        prompt = example["input_final_prompts"]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return prompt



def get_mmlu_example(example, dataset_name):
    """
    Get the question, options, and ground truth from the example.
    """
    if dataset_name == "cais/mmlu":
        question = example["question"]
        options = example["choices"]
        ground_truth = example['choices'][example['answer']]
    elif dataset_name == "meta-llama/Llama-3.2-3B-evals" or dataset_name == "meta-llama/Llama-3.2-3B-Instruct-evals":
        question = example["input_question"]
        options = example["input_choice_list"]
        ground_truth = example['input_correct_responses']
    return question, options, ground_truth

def recreate_llama_benchmark(
    model_name: str, 
    dataset_name: str, 
    config: dict, 
    csv_filename: str = None,
    use_vllm: bool = False, 
    use_pipeline: bool = False, 
    use_hub: bool = False):
    if dataset_name == "cais/mmlu":
        dataset = load_dataset(dataset_name, "all", split="test", download_mode=DownloadMode.FORCE_REDOWNLOAD)
        # print(dataset)
    elif dataset_name == "meta-llama/Llama-3.2-3B-evals":
        dataset = load_dataset(dataset_name, "Llama-3.2-3B-evals__mmlu__details", split="latest", download_mode=DownloadMode.FORCE_REDOWNLOAD)
    elif dataset_name == "meta-llama/Llama-3.2-3B-Instruct-evals":
        dataset = load_dataset(dataset_name, "Llama-3.2-3B-Instruct-evals__mmlu__details", split="latest", download_mode=DownloadMode.FORCE_REDOWNLOAD)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    if use_vllm:
        model, tokenizer, sampling_params = load_model_vllm(model_name, config)
    elif use_pipeline:
        model, tokenizer, pipe = load_model_pipeline(model_name, config)
    elif use_hub:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_model(model_name, config)
        
    # csv_filename = config["data_path"] + f"/{model_name.replace('/', '_')}_{dataset_name.replace('/', '_')}_results.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # CSV header
        writer.writerow(["question", "options", "prompt", "model_output", "ground_truth", "response"])
        
        for example in tqdm(dataset):
            # Extract fields; adjust field names if needed.
            question, options, ground_truth = get_mmlu_example(example, dataset_name)
            
            ground_truth = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
            prompt = format_prompt(example, dataset_name)
            # breakpoint()
            # Generate output using greedy decoding
            if use_vllm:
                seqs = model.generate(
                    prompt,
                    sampling_params=sampling_params,
                )
                answer_text = [seq.outputs[0].text for seq in seqs][0].strip()
            elif use_pipeline:
                outputs = pipe(
                    prompt, 
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    do_sample=True,
                    )
                generated_text = outputs[0][0]['generated_text']
                answer_text = outputs[0][0]['generated_text'][len(prompt[0]):].strip()
            elif use_hub:
                answer_text = get_response_from_hub(model_name, prompt)
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
                # Extract the answer: assume answer appears immediately after the prompt.
                answer_text = generated_text[len(prompt):].strip()
            
            predicted = answer_text[0].upper() if answer_text else ""
            
            writer.writerow([question, " | ".join(options), prompt, predicted, ground_truth, answer_text])
    
    print(f"Saved evaluation results to {csv_filename}")

    
    
def get_custom_mmlu_response(
    model_name, csv_filename, use_vllm, config
):
    dataset_name = "cais/mmlu"
    # data_sub_field = "high_school_computer_science"
    data_sub_field = "all"
    dataset = load_dataset(dataset_name, data_sub_field, split="test", download_mode=DownloadMode.FORCE_REDOWNLOAD)
    if use_vllm:
        model, tokenizer, sampling_params = load_model_vllm(model_name, config)
    else:
        model, tokenizer, pipe = load_model_pipeline(model_name, config)
        
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # CSV header
        writer.writerow(["question", "options", "prompt", "model_output", "ground_truth"])
        
        for example in tqdm(dataset):
            # Extract fields; adjust field names if needed.
            question, options, ground_truth = get_mmlu_example(example, dataset_name)
            ground_truth = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
            prompt = format_prompt(example, dataset_name)
            
            print(ground_truth)
            
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
                generated_text = outputs[0][0]['generated_text']
                answer_text = outputs[0][0]['generated_text'][len(prompt[0]):].strip()
                
            # predicted = answer_text[0].upper() if answer_text else ""
            
            writer.writerow([question, options, prompt, answer_text, ground_truth])
    
    print(f"Saved evaluation results to {csv_filename}")
    

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

    
def get_few_shot_surrogate(dataset_path, use_vllm=True, shot=3):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    if use_vllm:
        model, tokenizer, sampling_params = load_model_vllm(model_name, config)
    else:
        model, tokenizer, pipe = load_model_pipeline(model_name, config)
    ds = Dataset.from_csv(dataset_path).select(range(20))
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
            generated_text = outputs[0][0]['generated_text']
            answer_text = outputs[0][0]['generated_text'][len(prompt[0]):].strip()
        
        response_dict['question'] = example['question']
        response_dict['options'] = example['options']
        response_dict['model_output'] = example['model_output']
        response_dict['surrogate_output'] = answer_text
        response_dict['ground_truth'] = example['ground_truth']
        
        response_list.append(response_dict)
        
    # Write results to JSON file
    import json
    json_filename = f"data/dataset/qwen_surrogate_{shot}.json"
    with open(json_filename, 'w') as f:
        json.dump(response_list, f, indent=4)
    
    return response_list



if __name__ == "__main__":
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Run LLM benchmark evaluation")
        parser.add_argument("--use_vllm", action="store_true", default=False, 
                            help="Use VLLM for inference")
        parser.add_argument("--use_pipeline", action="store_true", default=False,
                            help="Use HuggingFace pipeline for inference")
        parser.add_argument("--use_hub", action="store_true", default=False,
                            help="Use HuggingFace hub for inference")
        parser.add_argument("--custom_resp", action="store_true", default=False,
                            help="use custom responses")
        parser.add_argument("--multiple_llm", action="store_true", default=False,
                            help="get responses from multiple llm")
        parser.add_argument("--surrogate_few_shot", action="store_true", default=False,
                            help="get response from surrogate model")
        parser.add_argument("--shot", type=int, default=3,
                            help="prompt shot count")
        parser.add_argument("--eval",action="store_true", default=False,
                            help="run evaluation script")
        parser.add_argument("--model_name",type=str, default=None,
                            help="provide model name")
        
        return parser.parse_args()
    
    args = parse_args()
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)
        
    if args.eval:
        data_path = "data/dataset/custom_allenai_OLMo-2-1124-7B-Instruct_mmlu_results.csv"
        results = metrics.compute_metrics_from_csv(data_path)
        print(results)
    else:    
        if args.surrogate_few_shot:
            ds_file_name = config["data_path"] + f"/custom_{config['model_name'].replace('/', '_')}_{config['dataset_name'].replace('/', '_')}_results.csv"
            
            get_few_shot_surrogate(dataset_path=ds_file_name, shot=args.shot)
        elif args.custom_resp:
            if args.multiple_llm:
                llms = config.get('model_list', [])
                for llm in llms:
                    
                    csv_file_name = config["data_path"] + f"/custom_{llm.replace('/', '_')}_{config['dataset_name'].replace('/', '_')}_results.csv"
                    get_custom_mmlu_response(
                        model_name=llm,
                        config=config,
                        csv_filename=csv_file_name,
                        use_vllm=args.use_vllm,
                    )
            else:
                csv_file_name = f"{config['data_path']}/full/custom_{config['model_name'].replace('/', '_')}_{config['dataset_name'].replace('/', '_')}_results.csv"
                get_custom_mmlu_response(
                    model_name=args.model_name if args.model_name else config['model_name'],
                    config=config,
                    csv_filename=csv_file_name,
                    use_vllm=args.use_vllm,
                )
            metrics = metrics.compute_metrics_from_csv(csv_file_name)
            print(f"Result: {metrics}")
        else:
            csv_file_name = config["data_path"] + f"/{config['model_name'].replace('/', '_')}_{config['dataset_name'].replace('/', '_')}_results.csv"
            recreate_llama_benchmark(
                args.model_name if args.model_name else config['model_name'], 
                config["dataset_name"], 
                config,
                csv_filename=csv_file_name,
                use_vllm=args.use_vllm,
                use_pipeline=args.use_pipeline,
                use_hub=args.use_hub
            )
            metrics = metrics.compute_metrics_from_csv(csv_file_name)
            print(f"Result: {metrics}")