# chatbot/llama_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "google/gemma-2b-it"
# MODEL_NAME = "microsoft/phi
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print(f"[INFO] {MODEL_NAME} 모델 사전 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] 모델 로딩 완료")

def get_model_and_tokenizer():
    if tokenizer is None or model is None:
        print("[WARN] 모델이 로딩되지 않아 load_model()을 자동 호출합니다.")
        load_model()    
    return tokenizer, model