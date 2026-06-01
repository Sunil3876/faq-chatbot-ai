import json
import os
from fastapi import FastAPI, HTTPException # pyright: ignore[reportMissingImports]
from fastapi.staticfiles import StaticFiles # pyright: ignore[reportMissingImports]
from fastapi.responses import FileResponse # pyright: ignore[reportMissingImports]
from pydantic import BaseModel # pyright: ignore[reportMissingImports]
from langchain_huggingface import HuggingFaceEmbeddings # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import FAISS # pyright: ignore[reportMissingImports]

app = FastAPI(title="Advanced AI FAQ Bot", version="2.0")

# Mounting static files for Frontend
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 1. Load Dataset
with open("faq_data.json", "r") as file:
    faq_data = json.load(file)

# Prepare documents for Vector Store
texts = [f"Question: {item['question']} Answer: {item['answer']}" for item in faq_data]
metadatas = [{"answer": item["answer"], "question": item["question"]} for item in faq_data]

# 2. Initialize AI Embeddings Model (Local & Free)
print("Loading HuggingFace Embeddings Model... Please wait...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create High-Performance FAISS Vector Database
print("Building Vector Indexes...")
vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
print("Vector Store Ready!")

# Pydantic schema for input validation
class ChatQuery(BaseModel):
    message: str

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat_endpoint(query: ChatQuery):
    try:
        user_input = query.message.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Semantic Vector Search with Score
        # k=1 means fetch the top 1 best matching semantic context
        results = vector_store.similarity_search_with_score(user_input, k=1)
        
        if not results:
            return {"response": "I'm sorry, I couldn't find any relevant information."}
        
        doc, score = results[0]
        
        # FAISS score standard threshold (lower score means higher similarity in L2 distance)
        # 0.0 means identical, typically anything below 1.2 is a solid semantic match
        if score > 1.2:
            return {
                "response": "I am not completely sure about that. Could you please rephrase your question or contact our human support team directly?",
                "match_score": float(score),
                "confident": False
            }
        
        return {
            "response": doc.metadata["answer"],
            "matched_question": doc.metadata["question"],
            "match_score": float(score),
            "confident": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn # pyright: ignore[reportMissingImports]
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)