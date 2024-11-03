from datasets import Dataset, load_dataset
from typing import Dict
from transformers import AutoTokenizer

from args import Args
args = Args()

def paired_data_preparation(
    data_dir = "nayohan/Neural-DPO-ko",
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

    # 함수내에 정의되는 함수로, 각 데이터 포맷과 구조에 맞게 매핑해주는 역할을 진행합니다.
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


# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token # 패딩 토큰을 eos로

# 'train'밖에 없는 관계로 trian-test 나눠주기
train_test_split = paired_data_preparation(data_dir= args.datapath).train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# 학습용 데이터셋
train_dataset = train_dataset.filter(
    # 프롬프트+응답 길이 < max_length
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)

# 평가 데이터셋
eval_dataset = eval_dataset.filter(
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)