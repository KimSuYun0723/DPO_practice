from dataclasses import dataclass, field
from typing import List

@dataclass
class Args:
    cache_dir: str = field(
        default="C:\_SY\HCNLP\DPO\cache_dir"
    )
    num_epochs: int = field(
        default=3  # 에포크 수
    )
    beta: float = field(
        default=0.1  # DPO에서 사용할 beta값
    )
    datapath: str = field(
        default="nayohan/Neural-DPO-ko"
    )
    model_name_or_path: str = field(
        default="HuggingFaceTB/SmolLM-135M"
    )
    learning_rate: float = field(
        default=1e-4
    )
    lr_scheduler_type: str = field(
        default="linear"  # LR 스케줄러 (linear, cosine, etc)
    )
    warmup_steps: int = field(
        default=25  # 학습 초기 워밍업 스텝
    )
    token_id: str = field(
        default="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"
    )
    weight_decay: float = field(
        default=0.05  # 가중치 감쇠 (Normalization 기법)
    )
    optimizer_type: str = field(
        default="paged_adamw_32bit"  # 최적화 알고리즘
    )
    per_device_train_batch_size: int = field(
        default=4
    )
    per_device_eval_batch_size: int = field(
        default=1
    )
    gradient_accumulation_steps: int = field(
        default=8
    )
    gradient_checkpointing: bool = field(
        default=True  # Gradient checkpointing 메모리 최적화를 위함
    )
    lora_alpha: int = field(
        default=16  # LoRA에서 사용하는 스케일링 파라미터
    )
    lora_dropout: float = field(
        default=0.1
    )
    lora_r: int = field(
        default=16
    )
    max_prompt_length: int = field(
        default=1024  # 싹둑...
    )
    max_length: int = field(
        default=1024
    )
    max_step: int = field(
        default=500
    )
    logging_steps: int = field(
        default=10
    )
    save_steps: int = field(
        default=50
    )
    eval_steps: int = field(
        default=50
    )
    output_dir: str = field(
        default="C:\_SY\HCNLP\DPO\output_dir"
    )
    log_freq: int = field(
        default=1
    )
    sanity_check: bool = field(
        default=False  # 학습 전 샘플 데이터를 이용한 테스트 여부
    )
    report_to: str = field(
        default="wandb"  # 로그 및 메트릭을 저장할 서비스 ('wandb', 'tensorboard' etc)
    )
    ignore_bias_buffers: bool = field(
        default=False  # DDP 학습 시 편향 버퍼 무시 여부
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            'embed_tokens', 'q_proj', 'k_proj', 'v_proj', 
            'gate_proj', 'down_proj', 'up_proj', 'lm_head'
        ]
    )

# 인스턴스 생성
args = Args()