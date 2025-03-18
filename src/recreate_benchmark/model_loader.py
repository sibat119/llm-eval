import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from transformers import pipeline
from huggingface_hub import InferenceClient
from vllm.transformers_utils.config import get_config

def load_model(model_name, config):
    """
    Load the pre-trained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cache_dir = config.get('model_cache', None)
    if cache_dir:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype='auto',
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype='auto',
            trust_remote_code=True,
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
    cache_dir = config.get('model_cache', None)
    if cache_dir is None:
        model = LLM(
            model_name,
            trust_remote_code=True,
            dtype=config['dtype'],
            # tensor_parallel_size=tensor_parallel_size,
            max_model_len=config['max_length'],
        )
    else:
        model = LLM(
            model_name,
            trust_remote_code=True,
            dtype=config['dtype'],
            download_dir=cache_dir,
            max_model_len=config['max_length'],
        )
    
    tokenizer = model.get_tokenizer()
    
    return model, tokenizer, sampling_params