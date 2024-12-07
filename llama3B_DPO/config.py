from peft import LoraConfig
from trl.trainer import DPOConfig

from args import Args
args = Args()

peft_config = LoraConfig(
    r = args.lora_r,
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout,
    target_modules = args.lora_target_modules,
    bias = "none",
    task_type = "CAUSAL_LM") # Causal Language Model로 학습

# 학습 인자 설정
training_args = DPOConfig( # model_init_kwargs 없음
    num_train_epochs= args.num_epochs,
    per_device_train_batch_size = args.per_device_train_batch_size,
    per_device_eval_batch_size = args.per_device_eval_batch_size,
    logging_steps = args.logging_steps,
    save_steps = args.save_steps,
    gradient_accumulation_steps = args.gradient_accumulation_steps,
    gradient_checkpointing = args.gradient_checkpointing,
    learning_rate = args.learning_rate,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    output_dir = args.output_dir,
    lr_scheduler_type = args.lr_scheduler_type,
    warmup_steps = args.warmup_steps,
    optim = args.optimizer_type,
    bf16 = True,
    remove_unused_columns = False,
    run_name = "llama3B_dpo_1209",
)