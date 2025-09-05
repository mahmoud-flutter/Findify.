# app.py - Findify Assistant

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

MODEL_NAME = "all-MiniLM-L6-v2"

def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            pages.append(t)
    return "\n".join(pages)

def chunk_text(text, chunk_size=500, overlap=50):
    text = text.replace("\r", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]

def build_index(chunks, model):
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def search(query, model, index, chunks, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((float(score), chunks[idx]))
    return results

def main():
    pdf_path = "knowledge.pdf"
    if not os.path.exists(pdf_path):
        print("⚠️ ضع ملف PDF باسم 'knowledge.pdf' في نفس مجلد المشروع.")
        return

    print("📥 تحميل النموذج...")
    model = SentenceTransformer(MODEL_NAME)

    print("📖 قراءة الملف...")
    text = read_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"✅ تم تقسيم الملف إلى {len(chunks)} مقطع.")

    print("⚡ بناء الفهرس...")
    index = build_index(chunks, model)
    print("🚀 جاهز! اكتب سؤال (اكتب exit للخروج):")

    while True:
        q = input("\n❓ سؤالك: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        results = search(q, model, index, chunks, top_k=3)
        print("\n📌 النتائج:")
        for score, chunk in results:
            print(f"\n-- (score={score:.4f})\n{chunk}\n")

if __name__ == "__main__":
    main()
