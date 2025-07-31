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
import logging
from functools import lru_cache

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DocumentAssistant")

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
MODEL_NAME = "hermes-3-llama-3.2-3b"
MODEL_API_URL = "http://localhost:1234/v1/chat/completions"
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
    """Check if question is valid and document-related"""
    q = q.strip().lower()
    if len(q) < 4: 
        return False
    if re.fullmatch(r"\s*", q): 
        return False
    if not re.search(r"\w", q):  # Require at least one word character
        return False
    if re.fullmatch(r"[a-zA-Z]{1,3}", q): 
        return False
    return True

def get_page_number(metadata: Dict[str, Any]) -> int:
    """Extract page number from metadata"""
    return metadata.get("page_number", metadata.get("page", 0) + 1)

def docs_to_jsonable(docs: List[Any]) -> List[Dict[str, Any]]:
    """Convert document objects to serializable format"""
    return [
        {
            "source": os.path.basename(doc.metadata['source']),
            "page_number": get_page_number(doc.metadata),
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "score": doc.metadata.get("score", 0.0)  # Include relevance score
        } for doc in docs
    ]

def call_custom_chat_model(prompt: str) -> str:
    """Call local LLM API with error handling and timeout"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert document assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }

    try:
        response = requests.post(
            MODEL_API_URL, 
            json=payload,
            timeout=30  # 30-second timeout
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        logger.error("Model API request timed out")
        return "⚠️ The model is taking too long to respond. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"Model API connection error: {str(e)}")
        return "⚠️ Unable to connect to the model service."
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid model response: {str(e)}")
        return "⚠️ Received an invalid response from the model."

# === Chat Service ===
class ChatService:
    def __init__(self):
        self.vectorstore = self.load_vectorstore()
        self.rag_chain = self.create_rag_chain()
        self.sessions: Dict[str, List[Dict]] = {}
        self.load_existing_sessions()

    def load_vectorstore(self):
        """Load FAISS vector store with Ollama embeddings"""
        from langchain_community.vectorstores import FAISS
        from langchain_ollama import OllamaEmbeddings

        logger.info("Loading vector store...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return FAISS.load_local(
            MODEL_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    def create_rag_chain(self):
        """Create RAG retrieval and response chain"""
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5, 
                "fetch_k": 25, 
                "lambda_mult": 0.75,
                "score_threshold": 0.4  # Minimum relevance score
            }
        )

        prompt_template = """You are an expert document assistant. Use ONLY the context below to answer.
Always cite sources exactly like: [Source: filename (page X)]. If unsure, say: "I couldn't find relevant information."

Context:
{context}

Chat history (last 3 exchanges):
{history}

Current Question:
{question}
"""

        def rag_chain(question: str, history: List[Dict[str, str]] = []):
            """Execute RAG workflow"""
            docs = retriever.invoke(question)
            context = "\n\n".join([
                f"### {os.path.basename(doc.metadata['source'])} (Page {get_page_number(doc.metadata)})\n{doc.page_content}"
                for doc in docs
            ])

            history_text = "\n".join([
                f"User: {h['query']}\nAssistant: {h['response']}" 
                for h in history[-3:]
            ]) if history else "No history"

            prompt = prompt_template.format(
                context=context, 
                question=question, 
                history=history_text
            )
            
            answer = call_custom_chat_model(prompt)
            return answer, docs

        return rag_chain

    def process_question(self, session_id: str, question: str) -> Dict[str, Any]:
        """Process user question through RAG pipeline"""
        start_time = time.time()
        session = self.sessions.get(session_id, [])
        response = ""
        docs_list = []

        try:
            if not is_valid_question(question):
                response = "🤖 I'm focused on document-related questions. Please ask something more specific."
            elif question.lower().strip() in IRRELEVANT_QUESTIONS:
                response = "🤖 I'm here to help with document questions. What can I find for you?"
            else:
                response, docs = self.rag_chain(question, session)
                docs_list = docs_to_jsonable(docs)


                # Add sources if missing in response
                if docs and "Source:" not in response:
                    sources = "\n".join(
                        f"[Source: {doc['source']} (page {doc['page_number']})]"
                        for doc in docs_list
                    )
                    response += f"\n\n{sources}"
        except Exception as e:
            logger.exception("Processing error")
            response = "⚠️ Please ensure that your inquiry is related to publications. Kindly provide a detailed description of your question for better assistance."
        finally:
            elapsed = time.time() - start_time
            response += f"\n\n_(Processed in {elapsed:.2f}s)_"

        # Update session history
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
        """Persist session history to disk"""
        try:
            if session_id in self.sessions:
                log_file = f"{LOG_DIR}/chat_{session_id}.json"
                with open(log_file, "w") as f:
                    json.dump(self.sessions[session_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving chat log: {str(e)}")

    def load_existing_sessions(self):
        """Load previous sessions from disk on startup"""
        try:
            for file in os.listdir(LOG_DIR):
                if file.startswith("chat_") and file.endswith(".json"):
                    session_id = file[5:-5]
                    with open(f"{LOG_DIR}/{file}", "r") as f:
                        self.sessions[session_id] = json.load(f)
            logger.info(f"Loaded {len(self.sessions)} existing sessions")
        except Exception as e:
            logger.error(f"Error loading sessions: {str(e)}")

    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def create_session(self) -> str:
        session_id = f"session_{uuid.uuid4().hex}"
        self.sessions[session_id] = []
        logger.info(f"Created new session: {session_id}")
        return session_id

# === Instantiate Service ===
chat_service = ChatService()

# === API Routes ===
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat: ChatRequest):
    # Validate question
    if not chat.question or not chat.question.strip():
        session_id = chat.session_id or chat_service.create_session()
        return ChatResponse(
            session_id=session_id,
            response="🤖 Please enter a valid question.",
            sources=[],
            processing_time=0.01
        )

    # Create new session if needed
    if not chat.session_id or chat.session_id not in chat_service.sessions:
        session_id = chat_service.create_session()
    else:
        session_id = chat.session_id

    # Process question
    result = chat_service.process_question(session_id, chat.question)
    
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
    return {
        "status": "online", 
        "sessions": len(chat_service.sessions),
        "model": MODEL_NAME,
        "vectorstore": MODEL_DIR
    }

# Run locally (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
