import pandas as pd
import csv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import torch
from vllm import LLM, SamplingParams
from vllm.transformers_utils.config import get_config
from transformers import pipeline
from tqdm import tqdm
from huggingface_hub import InferenceClient

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

def load_model(model_name, config):
    """
    Load the pre-trained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype='auto',
        trust_remote_code=True,
        # cache_dir=config['model_cache'],
    )
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer

def get_response_from_hub(model_name, prompt):
    client = InferenceClient(
        provider="hf-inference",
        api_key="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"
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    stream = client.chat.completions.create(
        model=model_name, 
        messages=messages, 
        temperature=0.5,
        max_tokens=2048,
        top_p=0.7,
        stream=False
    )

    for chunk in stream:
        print(chunk.choices[0].delta.content, end="")

    return chunk.choices[0].message.content

def load_model_pipeline(model_name, config):
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )


    tokenizer = pipe.tokenizer
    model = pipe.model
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, pipe

def set_tensor_parallel(num_devices, model_name):

    # get number of attention heads for the model
    n_head = get_num_attn_heads(model_name)

    tensor_parallel_size = num_devices
    while n_head%tensor_parallel_size != 0:
        tensor_parallel_size -= 1

    return tensor_parallel_size

def get_num_attn_heads(model_name):
    
    llm_cfg = get_config(model_name, trust_remote_code=True)

    # run through possible names for the number of attention heads in llm_cfg
    # this is necessary because the configs for each LLM are not standardized
    n_head = getattr(llm_cfg, 'num_attention_heads', None)
    n_head = getattr(llm_cfg, 'n_head', n_head)
    n_head = getattr(llm_cfg, 'num_heads', n_head)

    if n_head is None:
        print('n_head not set')
        breakpoint()
        raise ValueError()

    return n_head

def load_model_vllm(model_name, config):
    num_devices = config.get(
        'num_devices',
        torch.cuda.device_count()
    )
    
    # self.is_generation_model = is_generation_model
    tensor_parallel_size = set_tensor_parallel(num_devices, model_name)
    
    sampling_params = SamplingParams(
        top_p=config['top_p'],
        max_tokens=config['num_output_tokens'],
        temperature=config['temperature'],
    )
    model = LLM(
        model_name,
        trust_remote_code=True,
        download_dir=config['model_cache'],
        dtype=config['dtype'],
        # tensor_parallel_size=tensor_parallel_size,
        max_model_len=config['max_length'],
    )
    
    tokenizer = model.get_tokenizer()
    
    return model, tokenizer, sampling_params

def compute_accuracy_from_csv(csv_filename):
    """
    Reads the CSV file and computes accuracy by comparing the saved model output to the ground truth.
    """
    df = pd.read_csv(csv_filename)
    df["correct"] = df.apply(lambda row: 1 if row["model_output"].strip() == row["ground_truth"].strip() else 0, axis=1)
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
    use_vllm: bool = False, 
    use_pipeline: bool = False, 
    use_hub: bool = False):
    if dataset_name == "cais/mmlu":
        dataset = load_dataset(dataset_name, "all", split="test")
        # print(dataset)
    elif dataset_name == "meta-llama/Llama-3.2-3B-evals":
        dataset = load_dataset(dataset_name, "Llama-3.2-3B-evals__mmlu__details", split="latest")
    elif dataset_name == "meta-llama/Llama-3.2-3B-Instruct-evals":
        dataset = load_dataset(dataset_name, "Llama-3.2-3B-Instruct-evals__mmlu__details", split="latest")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    if use_vllm:
        model, tokenizer, sampling_params = load_model_vllm(model_name, config)
    elif use_pipeline:
        model, tokenizer, pipe = load_model_pipeline(model_name, config)
    elif get_response_from_hub:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_model(model_name, config)
        
    csv_filename = config["data_path"] + f"/{model_name.replace('/', '_')}_{dataset_name.replace('/', '_')}_results.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # CSV header
        writer.writerow(["question", "options", "prompt", "model_output", "ground_truth"])
        
        for example in tqdm(dataset):
            # Extract fields; adjust field names if needed.
            question, options, ground_truth = get_mmlu_example(example, dataset_name)
            
            prompt = format_prompt(example, dataset_name)
            
            # Generate output using greedy decoding
            if use_vllm:
                seqs = model.generate(
                    prompt,
                    sampling_params=sampling_params,
                )
                answer_text = [seq.outputs[0].text for seq in seqs]
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
            
            writer.writerow([question, " | ".join(options), prompt, predicted, ground_truth])
    
    print(f"Saved evaluation results to {csv_filename}")

    
    
    
    
    


if __name__ == "__main__":
    with open("data/config/conf.yml", "r") as file:
        config = yaml.safe_load(file)
    recreate_llama_benchmark(
        config["model_name"], 
        config["dataset_name"], 
        config,
        use_pipeline=True
    )
    accuracy = compute_accuracy_from_csv(config["data_path"] + f"/{config['model_name']}_{config['dataset_name']}_results.csv")
    print(f"Accuracy: {accuracy}")