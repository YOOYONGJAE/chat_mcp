from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import torch
import time
import re
from transformers import GenerationConfig



# ✅ llama_loader.py 에 있는 모델 로딩 함수 가져오기
from .llama_loader import get_model_and_tokenizer
from tools.query_rag import get_rag_prompt

@api_view(['GET'])
def test(request):
    return Response({'message': 'Hello, world!'})

@api_view(['GET', 'POST'])
def chat_test(request):
    print("[INFO] chat_test 실행")
    question = request.GET.get('question') or request.data.get('question') or ''
    print(f"[INFO] 질문 : {question}")

    # ❗그 외 일반 질문은 기존 RAG + generate 처리
    rag_prompt = get_rag_prompt(question, top_k=4)
    print(f"[INFO] rag_prompt : {rag_prompt}")


    #prompt = f"### 질문:\n{question}\n\n### 답변:\n"
    prompt = rag_prompt

    # print(f"[INFO] prompt : {prompt}")

    # ✅ 전역 모델 가져오기
    tokenizer, model = get_model_and_tokenizer()

    print("[INFO] 모델 로딩 완료")

    # 🔄 전처리
    start_all = time.time()
    start_preprocess = time.time()
    print("[INFO] 전처리 시작")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    end_preprocess = time.time()
    print(f"[INFO] 전처리 완료 : {end_preprocess - start_preprocess}초")

    # 🤖 생성
    start_gen = time.time()

    print("[INFO] GenerationConfig 옵션 설정")

    generation_config = GenerationConfig(
        temperature=0.1,               # [샘플링 온도] 낮을수록 결정적 → 항상 비슷한 답변 생성됨 (0에 가까우면 거의 greedy)
        top_k=50,                       # [상위 k 토큰 제한] 확률이 가장 높은 1개의 토큰만 후보로 사용 (탐색 범위 축소)
        do_sample=True,              # [샘플링 여부] False면 확률이 가장 높은 토큰을 항상 선택 (deterministic)
        max_new_tokens=150,             # [최대 생성 토큰 수] 답변의 최대 길이를 제한
        eos_token_id=tokenizer.eos_token_id
    )

    # 🤖 생성 (with torch.no_grad() 추가)
    start_gen = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )

    end_gen = time.time()
    print(f"[INFO] 생성 완료 : {end_gen - start_gen}초")
    # 📤 결과 디코딩
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # answer = output_text.strip()

    
    # 생성된 텍스트 디코딩
    full_output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # 첫 문단까지만 사용
    answer = full_output.strip().split("\n")[0]
    
    print(f"[INFO] answer : {answer}")    

    end_all = time.time()

    return Response({
        'question': question,
        'answer': answer,
        'timing': {
            'preprocess': round(end_preprocess - start_preprocess, 2),
            'generation': round(end_gen - start_gen, 2),
            'total': round(end_all - start_all, 2)
        }
    })
