# embed_documents.py

import json
import faiss
from sentence_transformers import SentenceTransformer
import os

# ===== 설정 =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(CURRENT_DIR, "../data/rag_documents.jsonl")  # JSONL 파일 경로
FAISS_INDEX_PATH = os.path.join(CURRENT_DIR, "../embeddings/faiss_index.index") # 저장할 FAISS 인덱스 경로
# EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"  # 한국어 특화 임베딩 모델 사용
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"  # 한국어 특화 임베딩 모델 사용
VECTOR_DIM = 768  # ko-sroberta-multitask 모델 출력 차원

# ===== 디렉토리 생성 =====
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

# ===== 모델 로드 =====
print("[INFO] 한국어 임베딩 모델 로드 중...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ===== JSONL 파일 로드 =====
print(f"[INFO] JSONL 문서 로딩 중: {JSONL_PATH}")
texts = []
ids = []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        ids.append(obj.get("id", None))

print(f"[INFO] 총 {len(texts)}개 문서 로딩 완료")

# ===== 임베딩 수행 =====
print("[INFO] 문서 임베딩 중...")
embeddings = model.encode(texts, show_progress_bar=True)

# ===== FAISS 인덱스 생성 및 저장 =====
print("[INFO] FAISS 인덱스 생성 중...")
index = faiss.IndexFlatL2(VECTOR_DIM)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)
print(f"[INFO] FAISS 인덱스 저장 완료 → {FAISS_INDEX_PATH}")