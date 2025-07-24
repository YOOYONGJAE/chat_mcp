# chatbot/llama_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "google/gemma-2b-it"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("🧠 LLaMA 모델 사전 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ LLaMA 모델 로딩 완료")

def get_model_and_tokenizer():
    if tokenizer is None or model is None:
        print("[WARN] 모델이 로딩되지 않아 load_model()을 자동 호출합니다.")
        load_model()    
    return tokenizer, model