# main.py
from transformers import HfArgumentParser
from huggingface_hub import login

from args import Args
from train import train
from args import Args
from trl import DPOTrainer  # DPOTrainer는 필요한 모듈에서 가져옵니다.


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    parser = HfArgumentParser((Args))
    args = parser.parse_args_into_dataclasses()

    login(token=args.token_id)

    dpo_trainer = DPOTrainer(args)
    train(dpo_trainer, args)