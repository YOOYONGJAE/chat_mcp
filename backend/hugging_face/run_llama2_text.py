from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# 전체 시작 시간
start_all = time.time()

# 모델 이름
model_name = "meta-llama/Llama-2-7b-chat-hf"

print("🔄 모델 로딩 중...")
start_load = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

end_load = time.time()
print("✅ 모델 로딩 완료 (걸린 시간: {:.2f}초)".format(end_load - start_load))

# 테스트용 질문
question = "안녕하세요!"

# 프롬프트 구성
prompt = f"### 질문:\n{question}\n\n### 답변:\n"

tokenizer.pad_token = tokenizer.eos_token  # 가장 안전한 방식

# 전처리 시작
start_preprocess = time.time()
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
end_preprocess = time.time()

# 생성 시작
print("🧠 텍스트 생성 중...")
start_gen = time.time()
with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        min_new_tokens=20
    )
end_gen = time.time()

# 출력 디코딩
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n💬 모델 응답:", output_text.replace(prompt, "").strip())

# 전체 종료 시간
end_all = time.time()

# 시간 로그 출력
print("\n⏱️ 소요 시간 요약:")
print(f" - 모델 로딩 시간: {end_load - start_load:.2f}초")
print(f" - 전처리 시간: {end_preprocess - start_preprocess:.2f}초")
print(f" - 텍스트 생성 시간: {end_gen - start_gen:.2f}초")
print(f" - 전체 실행 시간: {end_all - start_all:.2f}초")
