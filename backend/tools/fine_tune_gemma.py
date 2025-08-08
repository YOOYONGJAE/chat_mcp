import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer

# ----- [설정 시작] -----
model_name = "google/gemma-2b-it"
# 학습 방식이 근본적으로 바뀌므로, 새로운 결과 폴더를 사용합니다.
output_dir = "./outputs/gemma2b-it-finetuned-정제"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# instruction/output이 분리된 원본 데이터를 사용합니다.
JSONL_PATH = os.path.join(CURRENT_DIR, "../data/ncube_finetune_data_io_정제.jsonl")
# -----  [설정 끝]  -----

# ----- [모델 로드 및 준비 시작] -----
# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # 4비트 양자화 사용
    bnb_4bit_quant_type="nf4", # nf4 양자화 방식 사용
    bnb_4bit_compute_dtype=torch.bfloat16, # 계산 시 bfloat16 사용 (메모리 절약 및 속도 향상
)

# 4비트로 양자화된 Gemma-2B 모델을 불러오고, VRAM 적게 쓰면서 빠르게 작업
model = AutoModelForCausalLM.from_pretrained( # AutoModelForCausalLM는 Causal Language Model을 위한 클래스입니다.
    model_name, # 모델 이름
    quantization_config=bnb_config, # 4비트 양자화 설정
    device_map="auto", # 자동으로 장치 맵핑 (GPU 사용 시 자동 할당)
)

# 토크나이저 로드 및 설정
# AutoTokenizer는 사전 훈련된 토크나이저를 불러오는 클래스입니다.
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
# Gemma는 pad_token이 없으므로, eos_token을 pad_token으로 사용합니다.
tokenizer.pad_token = tokenizer.eos_token

# LoRA 설정
# prepare_model_for_kbit_training는 모델을 LoRA 학습에 적합하게 준비합니다.
# LayerNorm 같은 민감한 레이어는 fp32로 유지
# requires_grad 설정을 조정해 훈련 가능한 파라미터만 남김
# bnb 양자화 모델의 구조에 맞게 LoRA 삽입이 가능하도록 세팅
model = prepare_model_for_kbit_training(model)

# LoRA 설정을 정의합니다.
lora_config = LoraConfig(
    r=16, # LoRA rank (저차원 근사) 클수록 표현력 증가하지만 연산량도 증가함
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # LoRA를 적용할 모듈들
    lora_alpha=32, # LoRA scaling factor
    lora_dropout=0.05, # LoRA dropout 비율
    bias="none", # LoRA bias 설정 (없음)
    task_type="CAUSAL_LM" # 작업 유형 (Causal Language Model)
)

# LoRA 모델을 생성합니다.
# get_peft_model는 LoRA 설정을 모델에 적용합니다.
# get_peft_model(...)은 기존 transformers 모델을 감싸서 LoRA 어댑터가 장착된 새로운 모델로 반환합니다.
model = get_peft_model(model, lora_config)
# -----  [모델 로드 및 준비 끝]  -----

# ----- [데이터 로드 및 전처리 시작 (핵심 수정)] -----
# JSON 형식의 학습 데이터를 datasets 라이브러리를 통해 로딩
# load_dataset는 JSON 파일을 로드하여 데이터셋 객체로 변환합니다.
# split="train"은 전체 데이터셋을 훈련용으로 사용합니다.
# 이 데이터셋은 'instruction'과 'output' 필드를 포함하고 있어, 모델이 질문과 답변을 학습할 수 있도록 구성되어 있습니다
dataset = load_dataset("json", data_files=JSONL_PATH, split="train")


# 이 함수는 데이터셋의 각 샘플을 올바른 학습 형식으로 변환합니다.
# 모델이 '답변' 부분만 학습하도록 'labels'를 마스킹 처리합니다.
def preprocess_function(examples):
    
    # 전체 프롬프트를 Gemma의 공식 프롬프트 템플릿에 맞춰 생성합니다.
    all_input_ids = []
    # labels는 모델이 학습할 정답 부분을 포함합니다.
    all_labels = []
    
    # Gemma의 공식 프롬프트 템플릿
    prompt_template = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
    
    # 각 샘플에 대해 프롬프트를 생성하고 토크나이즈합니다.
    # instruction은 질문, output은 답변입니다.
    # tokenizer는 텍스트를 토큰 ID로 변환합니다.
    for instruction, output in zip(examples["instruction"], examples["output"]):
        # 1. 질문과 답변 템플릿을 합쳐 전체 프롬프트를 만듭니다.
        full_prompt = prompt_template.format(instruction=instruction) + output + tokenizer.eos_token
        
        # 2. 프롬프트를 토크나이징하여 input_ids를 생성합니다.
        # input_ids는 모델이 이해할 수 있는 숫자 형태로 변환된 프롬프트입니다.
        # truncation=True로 설정하여 최대 길이를 초과하는 부분은 잘라냅니다.
        # max_length=1024로 설정하여 최대 토큰 길이를 1024로 제한합니다.
        # padding=False로 설정하여 패딩을 하지 않습니다.
        input_ids = tokenizer(full_prompt, truncation=True, max_length=1024, padding=False).input_ids
        
        # 3. 질문 부분만 토크나이징하여 길이를 계산합니다. (마스킹을 위해)
        # prompt_only는 질문 부분만 포함된 프롬프트입니다.
        # prompt_only_token_len는 질문 부분의 토큰 길이를 계산합니다.
        prompt_only = prompt_template.format(instruction=instruction)
        prompt_only_token_len = len(tokenizer(prompt_only, add_special_tokens=False).input_ids)
        
        # 4. labels를 생성합니다. input_ids를 그대로 복사하되, 질문 부분은 -100으로 마스킹합니다.
        # -100은 모델이 해당 위치의 토큰을 무시하도록 합니다.
        # labels는 모델이 학습할 정답 부분을 포함합니다.
        # input_ids[:prompt_only_token_len] 부분은 -100으로 설정하여 질문 부분은 학습하지 않도록 합니다.
        # 나머지 부분은 그대로 유지하여 답변 부분만 학습하도록 합니다.
        # labels는 모델이 학습할 정답 부분을
        labels = list(input_ids) # 또는 input_ids.copy()
        labels[:prompt_only_token_len] = [-100] * prompt_only_token_len
        
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        
    return {"input_ids": all_input_ids, "labels": all_labels}

# map 함수를 사용하여 전체 데이터셋에 전처리 함수를 적용합니다.
processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
# -----  [데이터 로드 및 전처리 끝]  -----

# ----- [Trainer 설정 및 실행 시작] -----
training_args = TrainingArguments(
    output_dir=output_dir, # 모델 출력 디렉토리
    per_device_train_batch_size=1, # 한 번에 GPU에 올릴 배치 사이즈 
    gradient_accumulation_steps=4, # 한 번에 1개씩 GPU에 올리지만, 4번 누적해서 한 번의 업데이트(loss.backward + optimizer.step)를 하겠다는 뜻
    learning_rate=2e-5, # 학습률
    logging_dir="logs",  # 로그 디렉토리
    logging_steps=5, # 10 스텝마다 로그 기록
    report_to="tensorboard", # TensorBoard에 로그 기록
    save_strategy="epoch", # 에폭마다 모델 저장
    num_train_epochs=10, # 전체 학습 에폭 수
    fp16=True, # 16비트 부동소수점 연산 사용 (메모리 절약 및 속도 향상)
    optim="paged_adamw_8bit" # 8비트 최적화된 AdamW 사용 (메모리 절약 및 속도 향상)
)

# DataCollatorForSeq2Seq는 input_ids와 labels를 받아 동적으로 패딩 처리를 해줍니다.
# Causal LM에도 사용 가능하며, 이 경우 labels의 -100을 올바르게 처리합니다.
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100, # labels의 패딩은 -100으로 처리
    pad_to_multiple_of=8
)

# SFTTrainer 대신 기본 Trainer를 사용합니다.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator, # 직접 만든 데이터 콜레이터를 사용
)

trainer.train()

# LoRA adapter만 저장
model.save_pretrained(output_dir)
print(f"효율적인 학습이 완료되었습니다. 새로운 LoRA 어댑터가 다음 경로에 저장되었습니다: {output_dir}")
