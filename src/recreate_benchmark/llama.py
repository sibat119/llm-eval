import pandas as pd
import csv
from datasets import load_dataset, DownloadMode
from .model_loader import load_model, load_model_pipeline, load_model_vllm, get_response_from_hub
import yaml
from tqdm import tqdm
import argparse

def format_prompt(example, dataset_name):
    # Format: "Question: <question text>\nOptions: A. <opt1> B. <opt2> ...\nAnswer:"
    if dataset_name == "cais/mmlu":
        question = example["question"]
        options = example["choices"]
        option_text = " ".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
        prompt = f"Question: {question}\nOptions: {option_text}\nAnswer:"
    elif dataset_name == "meta-llama/Llama-3.2-3B-evals" or dataset_name == "meta-llama/Llama-3.2-3B-Instruct-evals":
        prompt = example["input_final_prompts"]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return prompt

def compute_accuracy_from_csv(csv_filename):
    """
    Reads the CSV file and computes accuracy by comparing the saved model output to the ground truth.
    """
    df = pd.read_csv(csv_filename)
    # Extract the first element from ground_truth list if it's a list, otherwise use as is
    df["correct"] = df.apply(lambda row: 1 if row["model_output"] == row["ground_truth"] else 0, axis=1)
    accuracy = df["correct"].mean()
    return accuracy

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

    
    
    
    
    


if __name__ == "__main__":
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Run LLM benchmark evaluation")
        parser.add_argument("--use_vllm", action="store_true", default=False, 
                            help="Use VLLM for inference")
        parser.add_argument("--use_pipeline", action="store_true", default=False,
                            help="Use HuggingFace pipeline for inference")
        parser.add_argument("--use_hub", action="store_true", default=False,
                            help="Use HuggingFace hub for inference")
        return parser.parse_args()
    
    args = parse_args()
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)
        
    csv_file_name = config["data_path"] + f"/{config['model_name'].replace('/', '_')}_{config['dataset_name'].replace('/', '_')}_results.csv"
    recreate_llama_benchmark(
        config["model_name"], 
        config["dataset_name"], 
        config,
        csv_filename=csv_file_name,
        use_vllm=args.use_vllm,
        use_pipeline=args.use_pipeline,
        use_hub=args.use_hub
    )
    accuracy = compute_accuracy_from_csv(csv_file_name)
    print(f"Accuracy: {accuracy}")