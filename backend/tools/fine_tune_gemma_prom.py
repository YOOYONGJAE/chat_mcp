# 사용자가 제공한 학습 스크립트에 상세한 주석 추가 버전 생성

# 파일 경로 설정
annotated_script_path = "/mnt/data/fine_tune_gemma_annotated.py"

# 원본 코드 줄 단위로 나눔
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ----- [설정 시작] -----
model_name = "google/gemma-2b-it"  # 사용할 사전 학습된 모델 이름
output_dir = "./outputs/gemma2b-it-finetuned-v2"  # 학습된 모델 저장 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 파일의 절대 경로
JSONL_PATH = os.path.join(CURRENT_DIR, "../data/ncube_finetune_data.jsonl")  # 학습 데이터 경로
# -----  [설정 끝]  -----

# ----- [모델 로드 및 준비 시작] -----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 모델을 4비트로 양자화하여 메모리 절약
    bnb_4bit_quant_type="nf4",  # 양자화 방식: NF4 (Normal Float 4-bit)
    bnb_4bit_compute_dtype=torch.bfloat16,  # 계산 시 사용할 데이터 타입
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,  # 사전학습 모델 이름
    quantization_config=bnb_config,  # 위에서 정의한 양자화 설정 적용
    device_map="auto",  # 가능한 GPU 또는 CPU에 자동으로 분배
)

model = prepare_model_for_kbit_training(model)  # 양자화된 모델을 학습 가능하게 준비

lora_config = LoraConfig(
    r=16,  # 랭크 (적은 수의 파라미터로 업데이트 수행)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # LoRA 적용할 모듈
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.05,  # 드롭아웃 비율
    bias="none",  # 바이어스 학습 안 함
    task_type="CAUSAL_LM"  # 언어 모델링 작업임을 명시
)

model = get_peft_model(model, lora_config)  # 모델에 LoRA 설정 적용

# ----- [토크나이저 준비] -----
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)  # 모델용 토크나이저 로드 (종료 토큰 추가)
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰도 eos 토큰으로 설정

# ----- [데이터 로드 및 분할] -----
raw_dataset = load_dataset("json", data_files=JSONL_PATH, split="train")  # JSONL 파일로부터 데이터 로드
dataset_split = raw_dataset.train_test_split(test_size=0.2, seed=42)  # 80% 학습, 20% 검증 데이터로 나눔
train_dataset = dataset_split["train"]  # 학습용 데이터셋
eval_dataset = dataset_split["test"]  # 검증용 데이터셋

# ----- [토크나이징] -----
train_dataset = train_dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)  # 학습 데이터 토크나이징
eval_dataset = eval_dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)  # 검증 데이터 토크나이징

# ----- [SFTTrainer 설정 및 실행] -----
training_args = TrainingArguments(
    output_dir=output_dir,  # 결과 저장 경로
    per_device_train_batch_size=1,  # 디바이스당 배치 크기
    gradient_accumulation_steps=4,  # 여러 step 동안 gradient 누적 후 역전파
    learning_rate=2e-5,  # 학습률
    num_train_epochs=20,  # 전체 학습 epoch 수
    optim="paged_adamw_8bit",  # 8비트 AdamW 옵티마이저 사용
    logging_dir="logs",  # 로그 저장 폴더
    logging_steps=10,  # 로그 출력 주기
    save_strategy="epoch",  # 매 epoch마다 저장
    report_to="tensorboard"  # TensorBoard 로깅
)

trainer = SFTTrainer(
    model=model,  # 학습할 모델
    args=training_args,  # 학습 인자
    train_dataset=train_dataset,  # 학습 데이터셋
    eval_dataset=eval_dataset,  # 검증 데이터셋
)

trainer.train()  # 학습 시작

# ----- [LoRA 어댑터 저장] -----
model.save_pretrained(output_dir)  # 학습된 LoRA 어댑터 저장
print(f"SFTTrainer 학습 완료! 어댑터 저장 위치: {output_dir}")


