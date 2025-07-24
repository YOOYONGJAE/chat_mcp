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
    print("_ chat_test 실행")
    question = request.GET.get('question') or request.data.get('question') or ''
    print(f"_ 질문 : {question}")

    # ✅ "몇 개", "총 몇 개", "프로젝트 수" 같은 표현이면 직접 count
    if re.search(r"(몇\s*개|총\s*\d+\s*개|프로젝트\s*(수|갯수|갯수가))", question):
        print("_ 프로젝트 개수 질문으로 분기")

        # 1. RAG 문서 중 가장 관련 있는 1개 선택
        rag_prompt = get_rag_prompt(question, top_k=1)
        print(f"_ rag_prompt : {rag_prompt}")

        # 2. 텍스트만 추출해서 줄 수 세기
        # 줄 수 세려면 "정보:" ~ "답변:" 사이만 추출
        project_block_match = re.search(r"정보:\s*(.*?)\s*이전에 나온 정보", rag_prompt, re.DOTALL)
        if project_block_match:
            project_block = project_block_match.group(1).strip()
            project_lines = [line for line in project_block.split("\n") if re.match(r"^\d{4}년", line)]
            project_count = len(project_lines)
            answer = f"총 {project_count}개의 프로젝트가 수행되었습니다."
        else:
            answer = "프로젝트 정보를 정확히 찾지 못했습니다."

        return Response({
            'question': question,
            'answer': answer,
            'timing': {'preprocess': 0.0, 'generation': 0.0, 'total': 0.0}
        })

    # ❗그 외 일반 질문은 기존 RAG + generate 처리
    rag_prompt = get_rag_prompt(question, top_k=1)
    print(f"_ rag_prompt : {rag_prompt}")


    #prompt = f"### 질문:\n{question}\n\n### 답변:\n"
    prompt = rag_prompt

    # print(f"_ prompt : {prompt}")

    # ✅ 전역 모델 가져오기
    tokenizer, model = get_model_and_tokenizer()

    print("_ 모델 로딩 완료")

    # 🔄 전처리
    start_all = time.time()
    start_preprocess = time.time()
    print("_ 전처리 시작")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    end_preprocess = time.time()
    print(f"_ 전처리 완료 : {end_preprocess - start_preprocess}초")

    # 🤖 생성
    start_gen = time.time()

    print("_ GenerationConfig 옵션 설정")

    generation_config = GenerationConfig(
    # temperature=0.1,
    # top_k=1,
    do_sample=False,
    # repetition_penalty=1.2,
    max_new_tokens=256,
    # eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
    )

    print("_ 생성 시작")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )

    end_gen = time.time()
    print(f"_ 생성 완료 : {end_gen - start_gen}초")
    # 📤 결과 디코딩
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # answer = output_text.strip()

    
    # 생성된 텍스트 디코딩
    full_output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # 첫 문단까지만 사용
    answer = full_output.strip().split("\n")[0]
    
    print("_answer")
    print(answer)

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
