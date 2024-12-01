from datasets import Dataset, load_dataset
from typing import Dict
from transformers import AutoTokenizer

from args import Args
args = Args()

# 토크나이저 로드
model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def paired_data_preparation(
    data_dir = "C:\_SY\DPO_practice\llama3B_DPO\data\dataset", # 데이터 경로
    sanity_check: bool = False,
    cache_dir: str = None,
    split_criteria: str = "train",
    num_proc: int=24,
) -> Dataset:
    """
    이 데이터셋은 이후 딕셔너리 형태로 변환되며  다음과 같은 형태로 prompt, chosen, reject로 담기게 됩니다.
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompt의 구조는 다음과 같이 담기게 됩니다(알파카 프롬프트):
      "###질문: " + <prompt> + "\n\n###답변: "
    """

    dataset = load_dataset(data_dir, split=split_criteria ,cache_dir=cache_dir)

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 500)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["###질문:\n" + question + "\n\n###답변:\n" for question in samples["question"]],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

dataset = paired_data_preparation(data_dir= args.datapath)
split_data = dataset.train_test_split(test_size=0.2)
train_dataset = split_data['train'] # 여기 수정 필요
eval_dataset = split_data['test'] # 여기 수정 필요

# 학습/평가용 데이터셋 
train_dataset = train_dataset.filter(
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)

eval_dataset = eval_dataset.filter(
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)