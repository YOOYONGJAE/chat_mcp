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

def get_rag_prompt(question: str, top_k: int = TOP_K, threshold: float = 1) -> Union[str, None]:
    query_vec = model.encode([question]) # 질문을 벡터 인코딩화 한다.
    D, I = index.search(query_vec, top_k) # RAG 문서에서 질문과 유사한 것을 찾는다.

    # 유사도가 너무 낮으면 RAG 생략
    top_score = D[0][0]
    print(f"[INFO] 유사도 거리 점수: {top_score}")

    if top_score > threshold:  # RAG context is NOT relevant
        # Allow model to use its fine-tuned knowledge
        # context_block = "엔큐브와 관련된 질문이 아니면 '죄송합니다' 라고 대답하세요.\n" # No RAG context provided
        context_block = "" # No RAG context provided
        instruction_suffix = "" # No strict instruction to say "모르겠습니다"
    else : # RAG context IS relevant
        context_data = "\n".join([documents[i]['text'] for i in I[0]])
        context_block = f"다음 정보를 참고하여 질문에 답하세요.\n정보:\n{context_data}\n\n"
        # instruction_suffix = "\n정보에 없는 내용은 '모르겠습니다'라고 답변하세요."
        instruction_suffix = "\n정보에 없는 내용은 절대로 말하지 마세요."

    # Construct the final prompt
    return f"<start_of_turn>user\n{context_block}{question}{instruction_suffix}<end_of_turn>\n<start_of_turn>model\n"
