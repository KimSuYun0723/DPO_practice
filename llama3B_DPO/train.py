from trl import DPOTrainer
import os

from prepare_dataset import train_dataset, eval_dataset, tokenizer
from config import peft_config, training_args
from model import model
from args import Args
args = Args()

print("###################################################################################")
print("############################  MODEL was Lodaded in GPU ############################")
print("###################################################################################")

dpo_trainer = DPOTrainer(
    model=model,
    ref_model = None,   # ref 모델을 None으로 놓게 되면 SFT + adapter가 붙은 모델에서 adapter를 떼고, policy에 따른 최적화를 진행하게 됩니다. 두개의 모델을 로드할 필요가 없어 메모리 이득을 꾀할 수 있습니다.
    args = training_args,
    beta = args.beta,
    train_dataset= train_dataset,
    eval_dataset = eval_dataset,
    tokenizer = tokenizer,
    peft_config = peft_config,
    max_prompt_length = args.max_prompt_length,
    max_length = args.max_length,
)

print("###################################################################################")
print("########################  Trainin Process is preparing now  #######################")
print("###################################################################################")

def train(dpo_trainer, args):
    dpo_trainer.train()
    dpo_trainer.save_model(args.output_dir)

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)