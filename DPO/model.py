import torch
from transformers import AutoModelForCausalLM

from args import Args
args = Args()

# SFT MODEL : SmolLM
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    low_cpu_mem_usage=True, # CPU 메모리 사용 최적화? 
    torch_dtype=torch.float16, # 16bit로 로드
    trust_remote_code=True, # 원격코드 신뢰..?  
    device_map="balanced",
    cache_dir=args.cache_dir)

model.config.use_cache = False
model.is_parallelizable = True
model.model_parallel = True
model.config.max_position_embeddings= args.max_prompt_length

print("model's max_position_embeddings :",model.config.max_position_embeddings)


# DDP 모드에서 편향 버퍼를 무시하는 설정 
if args.ignore_bias_buffers:
    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]