from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from typing import Dict
import torch

from args import Args
args = Args()


"""# 한국어-영어 데이터셋 column 통일하는 함수
def rename_question_to_prompt(example):
    return {
        "prompt": example["question"],  # 'question' -> 'prompt'
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

# DPO format dataset으로 바꿔주는 함수
def paired_data_preparation(
    data_dir: str = None,
    sanity_check: bool = False,
    num_proc: int = 8,
) -> Dataset:
    dataset = load_from_disk(data_dir)
    original_columns = dataset.column_names

    # Sanity Check
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 500)))

    # GPU 기반 변환 함수
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        # 데이터 배열을 텐서로 변환 (GPU로 이동)
        questions = torch.tensor(samples["question"], dtype=torch.string, device="cuda")
        chosen = torch.tensor(samples["chosen"], dtype=torch.string, device="cuda")
        rejected = torch.tensor(samples["rejected"], dtype=torch.string, device="cuda")

        # 텐서를 다시 문자열로 변환해 반환
        return {
            "prompt": ["###질문:\n" + question + "\n\n###답변:\n" for question in questions.tolist()],
            "chosen": chosen.tolist(),
            "rejected": rejected.tolist(),
        }

    # map 함수 호출
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns
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
ratio_data_path = "/home/nlpgpu8/hdd2/suyun/DPO_practice/llama3B_DPO/data/dataset"
dataset_en_sampled_5_path = f"{ratio_data_path}/dataset_en_sampled_5" 
dataset_en_sampled_2_path = f"{ratio_data_path}/dataset_en_sampled_2"
dataset_ko_sampled_5_path = f"{ratio_data_path}/dataset_ko_sampled_5"
dataset_ko_sampled_2_path = f"{ratio_data_path}/dataset_ko_sampled_2"

dataset_en_sampled_5.save_to_disk(dataset_en_sampled_5_path)
dataset_en_sampled_2.save_to_disk(dataset_en_sampled_2_path)
dataset_ko_sampled_5.save_to_disk(dataset_ko_sampled_5_path)
dataset_ko_sampled_2.save_to_disk(dataset_ko_sampled_2_path)

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
print("train dataset 예시:", concated_train_dataset[100])"""

# DPO format dataset으로 바꿔주는 함수
"""def paired_data_preparation(
    data_dir: str = None,
    sanity_check: bool = False,
    num_proc: int = 1,
) -> Dataset:
    dataset = load_from_disk(data_dir)
    dataset = dataset.to_dict()  # 메모리로 로드
    original_columns = dataset.column_names

    # Sanity Check
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 500)))

    # GPU 기반 변환 함수
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        # 데이터 배열을 텐서로 변환 (GPU로 이동)
        questions = torch.tensor(samples["question"], dtype=torch.string, device="cuda")
        chosen = torch.tensor(samples["chosen"], dtype=torch.string, device="cuda")
        rejected = torch.tensor(samples["rejected"], dtype=torch.string, device="cuda")

        # 텐서를 다시 문자열로 변환해 반환
        return {
            "prompt": ["###질문:\n" + question + "\n\n###답변:\n" for question in questions.tolist()],
            "chosen": chosen.tolist(),
            "rejected": rejected.tolist(),
        }

    # map 함수 호출
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns
    )
q = "82"
# 저장할 경로 설정 & 데이터셋 저장
train_save_path = f"/home/nlpgpu8/hdd2/suyun/DPO_practice/llama3B_DPO/data/dataset/combined_dataset_{q}/train"
eval_save_path = f"/home/nlpgpu8/hdd2/suyun/DPO_practice/llama3B_DPO/data/dataset/combined_dataset_{q}/test"
#concated_train_dataset.save_to_disk(train_save_path)
#concated_eval_dataset.save_to_disk(eval_save_path)

# DPO format으로 바꾸기
train_dataset = paired_data_preparation(data_dir=train_save_path)
eval_dataset = paired_data_preparation(data_dir=eval_save_path)

train_dataset = train_dataset.filter(
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)

print(train_dataset[:5])"""

if __name__ == "__main__":
    import torch
    from datasets import load_from_disk, Dataset
    from typing import Dict

    # DPO format dataset으로 바꿔주는 함수
    def paired_data_preparation(
        data_dir: str = None,
        sanity_check: bool = False,
        num_proc: int = 1,
    ) -> Dataset:
        dataset = load_from_disk(data_dir)
        original_columns = dataset.column_names

        # Sanity Check
        if sanity_check:
            dataset = dataset.select(range(min(len(dataset), 500)))

        # GPU 기반 변환 함수
        def return_prompt_and_responses(samples) -> Dict[str, str]:
            questions = samples["prompt"]
            chosen = samples["chosen"]
            rejected = samples["rejected"]

            # GPU로 옮기지 않고 간단히 처리
            return {
                "prompt": ["###질문:\n" + question + "\n\n###답변:\n" for question in questions],
                "chosen": chosen,
                "rejected": rejected,
            }

        # map 함수 호출
        return dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns
        )

    # 저장할 경로 설정 & 데이터셋 저장
    q = "82"
    train_save_path = f"/home/nlpgpu8/hdd2/suyun/DPO_practice/llama3B_DPO/data/dataset/combined_dataset_{q}/train"
    eval_save_path = f"/home/nlpgpu8/hdd2/suyun/DPO_practice/llama3B_DPO/data/dataset/combined_dataset_{q}/test"

    # DPO format으로 바꾸기
    train_dataset = paired_data_preparation(data_dir=train_save_path, num_proc=8)
    eval_dataset = paired_data_preparation(data_dir=eval_save_path, num_proc=8)

    # 데이터 필터링
    args = {"max_length": 512}  # 예시
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= args["max_length"]
        and len(x["prompt"]) + len(x["rejected"]) <= args["max_length"]
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= args["max_length"]
        and len(x["prompt"]) + len(x["rejected"]) <= args["max_length"]
    )

    # 출력
    print(train_dataset[:5])
