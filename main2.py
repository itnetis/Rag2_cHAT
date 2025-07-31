from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import time
from datetime import datetime
import re
import requests
from typing import List, Dict, Any, Optional
import uuid
from functools import lru_cache

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Constants ===
MODEL_DIR = "final_vectorstore"
IRRELEVANT_QUESTIONS = ["hi", "hello", "hey", "how are you", "date", "time", "what's up"]
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# === Pydantic Models ===
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[Dict[str, Any]]
    processing_time: float

class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]

# === Utility Functions ===
def is_valid_question(q: str) -> bool:
    q = q.strip().lower()
    if len(q) < 4: return False
    if re.fullmatch(r"[a-zA-Z]{1,4}", q): return False
    if not re.search(r"[a-zA-Z]{2,}", q): return False
    return True

def get_page_number(metadata: Dict[str, Any]) -> int:
    return metadata.get("page_number", metadata.get("page", 0) + 1)

def docs_to_jsonable(docs: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "source": os.path.basename(doc.metadata['source']),
            "page_number": get_page_number(doc.metadata),
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
        } for doc in docs
    ]

def call_custom_chat_model(prompt: str) -> str:
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": "hermes-3-llama-3.2-3b",
        "messages": [
            {"role": "system", "content": "You are an expert document assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"⚠️ Failed to call model API: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"⚠️ Invalid response format from model API: {str(e)}"

# === Chat Service ===
class ChatService:
    def __init__(self):
        self.vectorstore = self.load_vectorstore()
        self.rag_chain = self.create_rag_chain()
        self.sessions: Dict[str, List[Dict]] = {}

    @lru_cache
    def load_vectorstore(self):
        from langchain_community.vectorstores import FAISS
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return FAISS.load_local(
            MODEL_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    def create_rag_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.8}
        )

        prompt_template = """You are an expert document assistant.
Use ONLY the context below to answer.
Always cite sources exactly like: [Source: filename (page X)]
If unsure, say: "I couldn't find relevant information."

Context:
{context}

Chat history:
{history}

Current Question:
{question}
"""

        def rag_chain(question: str, history: List[Dict[str, str]] = []):
            docs = retriever.invoke(question)
            context = "\n\n".join([
                f"### {os.path.basename(doc.metadata['source'])} (Page {get_page_number(doc.metadata)})\n{doc.page_content}"
                for doc in docs
            ])

            history_text = "\n".join([
                f"User: {h['query']}\nAssistant: {h['response']}" for h in history[-3:]
            ]) if history else "None"

            prompt = prompt_template.format(context=context, question=question, history=history_text)
            answer = call_custom_chat_model(prompt)
            return answer, docs

        return rag_chain

    def process_question(self, session_id: str, question: str) -> Dict[str, Any]:
        start_time = time.time()
        session = self.sessions.get(session_id, [])

        lower_q = question.lower().strip()
        if lower_q in IRRELEVANT_QUESTIONS:
            response = "🤖 I'm focused on document-related questions. Please ask something relevant."
            docs_list = []
        else:
            try:
                response, docs = self.rag_chain(question, session)
                docs_list = docs_to_jsonable(docs)

                if docs and "Source:" not in response:
                    sources = "\n".join(
                        f"[Source: {doc['source']} (page {doc['page_number']})]"
                        for doc in docs_list
                    )
                    response += f"\n\n{sources}"
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
                docs_list = []

        elapsed = time.time() - start_time
        response += f"\n\n_(Processed in {elapsed:.2f}s)_"

        entry = {
            "query": question,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "sources": docs_list
        }
        session.append(entry)
        self.sessions[session_id] = session

        self.save_chat_log(session_id)

        return {
            "response": response,
            "sources": docs_list,
            "processing_time": elapsed
        }

    def save_chat_log(self, session_id: str):
        if session_id in self.sessions:
            log_file = f"{LOG_DIR}/chat_{session_id}.json"
            with open(log_file, "w") as f:
                json.dump(self.sessions[session_id], f, indent=2)

    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def create_session(self) -> str:
        session_id = f"session_{uuid.uuid4().hex}"
        self.sessions[session_id] = []
        return session_id

# === Instantiate Service ===
chat_service = ChatService()

# === API Routes ===
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat: ChatRequest):
    if not chat.question or not is_valid_question(chat.question):
        session_id = chat.session_id or chat_service.create_session()
        response = "🤖 I'm focused on document-related questions. Please ask something more specific."
        return ChatResponse(
            session_id=session_id,
            response=response + "\n\n_(Processed instantly)_",
            sources=[],
            processing_time=0.01
        )

    if not chat.session_id or chat.session_id not in chat_service.sessions:
        session_id = chat_service.create_session()
    else:
        session_id = chat.session_id

    try:
        result = chat_service.process_question(session_id, chat.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        sources=result["sources"],
        processing_time=result["processing_time"]
    )

@app.post("/session/new")
async def create_new_session():
    session_id = chat_service.create_session()
    return JSONResponse(
        content={"session_id": session_id, "message": "New session created"}
    )

@app.get("/history/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    if session_id not in chat_service.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionHistoryResponse(
        session_id=session_id,
        history=chat_service.get_session_history(session_id)
    )

@app.get("/health")
async def health_check():
    return {"status": "online", "sessions": len(chat_service.sessions)}

# @app.get("/session/all")
# def list_sessions():
#     return {"sessions": list(chat_engine.sessions.keys())}

# Run locally (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
