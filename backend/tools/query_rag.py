# query_rag.py → 함수화: get_rag_prompt(question)

import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import Union


# ===== 설정 =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

JSONL_PATH = os.path.join(CURRENT_DIR, "../data/rag_documents.jsonl")
FAISS_INDEX_PATH = os.path.join(CURRENT_DIR, "../embeddings/faiss_index.index")
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
VECTOR_DIM = 768
TOP_K = 3

# ===== 전역 로드 =====
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    documents = [json.loads(line) for line in f]

def get_rag_prompt(question: str, top_k: int = TOP_K, threshold: float = 140) -> Union[str, None]:
    query_vec = model.encode([question]) # 질문을 벡터 인코딩화 한다.
    D, I = index.search(query_vec, top_k) # RAG 문서에서 질문과 유사한 것을 찾는다.

    # 유사도가 너무 낮으면 RAG 생략
    top_score = D[0][0]
    print(f"🔍 유사도 거리 점수: {top_score}")

    if top_score > threshold:  # 예: 1.0 이상이면 관련 문서 아님
        context = "정보 없음"
        question = "답변할 수 없는 질문입니다. 죄송합니다."

    else : context = "\n".join([documents[i]['text'] for i in I[0]])

    return f"""
질문: {question}
정보: {context}
정보 기반으로만 답변하세요.
정보 외의 내용을 상상하거나 추측하지 마세요.
답변:
"""


# 디버깅용 실행 (직접 실행 시 동작)
if __name__ == "__main__":
    user_input = input("질문을 입력하세요: ")
    prompt = get_rag_prompt(user_input)
    print("\n[RAG 프롬프트 생성 결과]\n")
    print(prompt)
