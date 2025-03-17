from datasets import load_dataset
from src.recreate_benchmark.model_loader import load_model, load_model_pipeline, load_model_vllm, get_response_from_hub
import csv
import argparse
import yaml
from tqdm import tqdm
import pandas as pd
import evaluate

def get_prompt(context, question):
    prompt = """You are a helpful assistant who are good at answering healthcare question.
I will give you, context, which is the sentences split from the context with the format of "index: sentence",
followed by the question, then you need to reply me with two things:
First, find the index of sentences that is the answer-related to the question, namely the evidence sentences.
(You have to find more than one evidence sentence.)
Second, base on the evidence sentences you chose, give me the abstractive answer to the question.
context: {context}
question: {question}
You should reply me with the following format:
Evidence sentences: [index1,index2,index3...]
Answer: <your answer>"""
    prompt = prompt.format(context=context, question=question)
    return prompt

def get_dataset(dataset_name):
    if dataset_name == "mesaqa":
        return load_dataset("riiwang/MESAQA")
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

def get_mesaqa_example(example):
    context = example["context"]
    question = example["question"]
    evidence = example["evidence"]
    answer = example["answer"]
    ground_truth = f"Evidence sentences: {evidence}\nAnswer: {answer}"
    return context, question, ground_truth

def recreate_llama_benchmark(
    model_name: str, 
    dataset_name: str, 
    config: dict, 
    csv_filename: str = None,
    use_vllm: bool = False, 
    use_pipeline: bool = False, 
    use_hub: bool = False):
    
    dataset = get_dataset(dataset_name)
    
    if use_vllm:
        model, tokenizer, sampling_params = load_model_vllm(model_name, config)
    elif use_pipeline:
        model, tokenizer, pipe = load_model_pipeline(model_name, config)
    elif use_hub:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_model(model_name, config)
        
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # CSV header
        writer.writerow(["question", "context", "prompt", "model_output", "ground_truth", "response"])
        
        for example in tqdm(dataset):
            # Extract fields; adjust field names if needed.
            context, question, ground_truth = get_mesaqa_example(example)
            
            prompt = get_prompt(context, question)
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
            
            breakpoint()
            predicted = answer_text[0].upper() if answer_text else ""
            
            writer.writerow([question, context, prompt, predicted, ground_truth, answer_text])
    
    print(f"Saved evaluation results to {csv_filename}")

def compute_accuracy_from_csv(csv_filename):
    """
    Reads the CSV file and computes accuracy by comparing the saved model output to the ground truth.
    """
    df = pd.read_csv(csv_filename)
    df["correct"] = df.apply(lambda row: 1 if row["model_output"] == row["ground_truth"] else 0, axis=1)
    accuracy = df["correct"].mean()
    
    def calculate_f1_recall(pred, truth):
        if not pred or not truth:
            return 0, 0
        pred_tokens = set(pred.lower().split())
        truth_tokens = set(truth.lower().split())
        
        common = len(pred_tokens.intersection(truth_tokens))
        if common == 0:
            return 0, 0
            
        precision = common / len(pred_tokens)
        recall = common / len(truth_tokens)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1, recall
    
    f1_scores = []
    recall_scores = []
    
    for _, row in df.iterrows():
        f1, recall = calculate_f1_recall(row["model_output"], row["ground_truth"])
        f1_scores.append(f1)
        recall_scores.append(recall)
    
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    
    df["f1_score"] = f1_scores
    df["recall_score"] = recall_scores
    return accuracy, avg_f1, avg_recall

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
    accuracy, f1, recall = compute_accuracy_from_csv(csv_file_name)
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")
    print(f"Recall: {recall}")