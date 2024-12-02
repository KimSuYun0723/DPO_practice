from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from typing import Dict
from transformers import AutoTokenizer

from args import Args
args = Args()


# 한국어-영어 데이터셋 column 통일하는 함수
def rename_question_to_prompt(example):
    """
    한국어와 영어 데이터셋이 달라서
    한국어 데이터셋의 column 이름 바꿔주는 함수
    """
    return {
        "prompt": example["question"],  # 'question' -> 'prompt'
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

# DPO format dataset으로 바꿔주는 함수
def paired_data_preparation(
    data_dir : str = None,
    sanity_check: bool = False,
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

    dataset = load_from_disk(data_dir)

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
        disable=True,
    )

dataset_ko = load_dataset("kuotient/orca-math-korean-dpo-pairs", split="train")

# 한국어 데이터셋 열 정리
dataset_ko = dataset_ko.map(rename_question_to_prompt, load_from_cache_file=False)
dataset_ko = dataset_ko.remove_columns(["system", "question"])  # 'system' 열 제거

# 영어 데이터셋
dataset_en = load_dataset("ibivibiv/cleaned_orca_math_dpo_pairs", split="train")

# 중간 확인: 열 구조와 샘플 확인
print("==== 중간 확인: 열 구조와 샘플 확인 ====")
print("- 한국어 데이터셋 열 구조:", dataset_ko.column_names)
print("- 영어 데이터셋 열 구조:", dataset_en.column_names)
print("- 한국어 데이터셋 샘플:", dataset_ko[0])
print("- 영어 데이터셋 샘플:", dataset_en[0])

en_num_5 = int(len(dataset_ko) * 0.5)
en_num_2 = int(len(dataset_ko) * 0.25)
ko_num_5 = int(len(dataset_ko) * 0.5)
ko_num_2 = int(len(dataset_ko) * 0.25)

dataset_en_sampled_5 = dataset_en.shuffle(seed=42).select(range(en_num_5))
dataset_en_sampled_2 = dataset_en.shuffle(seed=42).select(range(en_num_2))
dataset_ko_sampled_5 = dataset_ko.shuffle(seed=42).select(range(ko_num_5))
dataset_ko_sampled_2 = dataset_ko.shuffle(seed=42).select(range(ko_num_2))

#저장
"""ratio_data_path = "C:\_SY\DPO_practice\llama3B_DPO\data\dataset"
dataset_en_sampled_5_path = f"{ratio_data_path}/dataset_en_sampled_5" 
dataset_en_sampled_2_path = f"{ratio_data_path}/dataset_en_sampled_2"
dataset_ko_sampled_5_path = f"{ratio_data_path}/dataset_ko_sampled_5"
dataset_ko_sampled_2_path = f"{ratio_data_path}/dataset_ko_sampled_2"

dataset_en_sampled_5.save_to_disk(dataset_en_sampled_5_path)
dataset_en_sampled_2.save_to_disk(dataset_en_sampled_2_path)
dataset_ko_sampled_5.save_to_disk(dataset_ko_sampled_5_path)
dataset_ko_sampled_2.save_to_disk(dataset_ko_sampled_2_path)"""

# train-test 쪼개기
q = str(input(" 82/55/28?\n:"))
if q == "82":
    en_split_data = dataset_en.train_test_split(test_size=0.2)
    ko_split_data = dataset_ko_sampled_2.train_test_split(test_size=0.2)
elif q == "55":
    en_split_data = dataset_en_sampled_5.train_test_split(test_size=0.2)
    ko_split_data = dataset_ko_sampled_5.train_test_split(test_size=0.2)
elif q == "28":
    en_split_data = dataset_en_sampled_2.train_test_split(test_size=0.2)
    ko_split_data = dataset_ko.train_test_split(test_size=0.2)
else:
    print("Again")

en_train_dataset = en_split_data['train']
ko_train_dataset = ko_split_data['train']

en_eval_dataset = en_split_data['test']
ko_eval_dataset = ko_split_data['test']

concated_train_dataset = concatenate_datasets([en_train_dataset, ko_train_dataset])
concated_eval_dataset = concatenate_datasets([en_eval_dataset, ko_eval_dataset])

print("train dataset 열 구조:", concated_train_dataset.column_names)
print("train dataset 예시:", concated_train_dataset[100])

# 저장할 경로 설정 & 데이터셋 저장
train_save_path = f"C:\_SY\DPO_practice\llama3B_DPO\data\dataset\combined_dataset_{q}/train"
eval_save_path = f"C:\_SY\DPO_practice\llama3B_DPO\data\dataset\combined_dataset_{q}/test"
concated_train_dataset.save_to_disk(train_save_path)
concated_eval_dataset.save_to_disk(eval_save_path)

# DPO format으로 바꾸기
train_dataset = paired_data_preparation(data_dir=train_save_path)
eval_dataset = paired_data_preparation(data_dir=eval_save_path)

train_dataset = train_dataset.filter(
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)

print(train_dataset[:10])

"""# 저장된 데이터셋 로드
train_dataset = load_from_disk(train_save_path)
eval_dataset = load_from_disk(eval_save_path)"""