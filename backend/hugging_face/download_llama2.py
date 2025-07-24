# download_llama2.py

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

print("CUDA 사용 가능:", torch.cuda.is_available())
print("사용 중인 장치:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

model_name = "meta-llama/Llama-2-7b-chat-hf"

print("🔄 모델 다운로드 시작 중...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name) CPU 배치

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,     # 더 적은 VRAM 사용 (양자화는 아님)
    device_map="auto"              # GPU에 자동 배치
)

print("✅ 모델 다운로드 완료!")