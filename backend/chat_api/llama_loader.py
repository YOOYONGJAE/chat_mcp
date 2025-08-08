# chatbot/llama_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from pathlib import Path
from django.conf import settings
from peft import PeftModel

# MODEL_NAME = "google/gemma-2b-it"
# MODEL_NAME = "microsoft/phi
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_NAME = "google/gemma-2b-it"
# PEFT_MODEL = os.path.join(CURRENT_DIR, "../tools/outputs/gemma2b-it-finetuned-20250806")
PEFT_MODEL = os.path.join(CURRENT_DIR, "../tools/outputs/gemma2b-it-finetuned-정제")
ADAPTER_PATH = PEFT_MODEL  # 이 라인 추가해줘용!

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print(f"[INFO] {PEFT_MODEL} 모델 사전 로딩 중...")

        # offload_path = os.path.join(settings.BASE_DIR, "offload")
        # os.makedirs(offload_path, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, local_files_only=True, torch_dtype=torch.float16)    
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

        print("----------------")
        print(type(model))
        print(model.base_model.model) 
        print("----------------")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("[INFO] 모델 로딩 완료 (LoRA 적용됨!)")

def get_model_and_tokenizer():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("[WARN] 모델이 로딩되지 않아 load_model()을 자동 호출합니다.")
        load_model()
    return tokenizer, model