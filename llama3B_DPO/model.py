import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from args import Args
args = Args()

# MODEL : Llama-3.2B
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    low_cpu_mem_usage=True, 
    torch_dtype=torch.bfloat16, # llama3b 예시 기준.. easyDPO는 float16
    trust_remote_code=True, 
    device_map= "balanced",
    cache_dir=args.cache_dir).to("cuda:1")

model.config.use_cache = False
model.is_parallelizable = True
model.model_parallel = True
model.config.max_position_embeddings= args.max_prompt_length

print("model's max_position_embeddings :", model.config.max_position_embeddings)


# DDP 모드에서 편향 버퍼를 무시하는 설정 
if args.ignore_bias_buffers:
    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]