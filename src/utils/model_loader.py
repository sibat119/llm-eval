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
    
    # is_generation_model = is_generation_model
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

def preprocess_model_msg(model_name, usr_msg, sys_msg=None):
        
    if model_name == 'Phind/Phind-CodeLlama-34B-v2':
        msg = f'### User Message\n{usr_msg}\n\n### Assistant\n'
        if sys_msg is not None:
            msg = f'### System Prompt\n{sys_msg}\n\n' + msg
        return msg
    elif 'WizardLM' in model_name:
        msg = f'### Instruction:\n{usr_msg}\n\n### Response:'
        if sys_msg is not None:
            msg = f'{sys_msg.strip()}\n\n' + msg
        else:
            msg = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n' + msg
        return msg
    elif 'codellama' in model_name and 'Instruct' in model_name:
        msg = f'[INST]{usr_msg.strip()}[/INST]'
        if sys_msg is not None:
            msg = f'<<SYS>>{sys_msg.strip()}<</SYS>>' + msg
        return msg
    elif 'Salesforce' in model_name and 'instruct' in model_name:
        msg = f'### Instruction:\n{usr_msg.strip()}\n\n### Response:\n'
        if sys_msg is not None:
            msg = f'{sys_msg.strip()}\n\n' + msg
        else:
            msg = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n' + msg

        return msg
    elif model_name == 'mistralai/Mistral-7B-Instruct-v0.1':
        msg = f'[INST] {usr_msg} [/INST]'
        return msg 
    elif 'lmsys/vicuna' in model_name:
        msg = f'USER: {usr_msg.strip()}\nASSISTANT: '
        if sys_msg is not None:
            msg = sys_msg.strip() + '\n\n' + msg
        return msg
    elif ('Meta-Llama-3-8B-Instruct' in model_name 
            or 'Llama-3.1-8B-Instruct' in model_name
            or 'OLMo-7B-Instruct' in model_name
            or 'open_llama_13b' in model_name
            or 'Qwen/Qwen2.5-7B-Instruct' in model_name
            or 'allenai/OLMo-2-1124-7B-Instruct' in model_name
            ):
        if sys_msg:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ]
        else:
            messages = [
                {"role": "user", "content": usr_msg},
            ]
        return messages
    else:
        return usr_msg

def prepare_batch(model_name, usr_msg, sys_msg=None):

    # convert string input to list
    return_str=False
    if type(usr_msg) == str:
        msg = [preprocess_model_msg(model_name, usr_msg, sys_msg)]
        return_str=True
        return msg, return_str

    if sys_msg is None:
        sys_msg = [None] * len(usr_msg)

    # ensure length of usr_msg and sys_msg match
    if len(usr_msg) != len(sys_msg):
        print('length of usr_msg does not match length of sys_msg')
        raise ValueError()

    msg = [preprocess_model_msg(model_name, prompt, sys)
            for prompt, sys in zip(usr_msg, sys_msg)]
    
    return msg, return_str