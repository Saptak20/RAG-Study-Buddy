# 📚 RAG Study Buddy

> An AI-powered study assistant built with Retrieval-Augmented Generation (RAG) — upload your PDFs, lecture notes, and study materials, and get intelligent answers with source citations.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-latest-orange?style=flat-square)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen?style=flat-square&logo=mongodb)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=flat-square&logo=docker)
![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-purple?style=flat-square)

---

## ✨ Features

- **RAG Pipeline** — Retrieves relevant chunks from your documents before generating answers, minimizing hallucinations
- **Citation Tracking** — Every answer includes references to the exact source documents and page numbers used
- **Persistent Chat Memory** — Conversations maintain context across sessions, stored in MongoDB
- **Multi-format Ingestion** — Supports PDFs, lecture notes (`.txt`, `.md`), and quiz sheets
- **JWT Authentication** — Secure, stateless auth with per-user document isolation
- **Fast Vector Search** — FAISS-powered similarity search for sub-second retrieval
- **LLaMA 3 via Groq** — Blazing fast inference using Groq's hardware-accelerated API
- **Dockerized** — Fully containerized for consistent local and cloud environments

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Client    │────▶│              FastAPI Backend                  │
│  (Browser)  │     │                                              │
└─────────────┘     │  ┌─────────────┐    ┌──────────────────┐    │
                    │  │ JWT Auth    │    │  RAG Pipeline    │    │
                    │  │ Middleware  │    │                  │    │
                    │  └─────────────┘    │  1. Embed Query  │    │
                    │                     │  2. FAISS Search │    │
                    │  ┌─────────────┐    │  3. Context Pack │    │
                    │  │  MongoDB    │    │  4. LLaMA3 Gen   │    │
                    │  │  (Chat +    │    │  5. Cite Sources │    │
                    │  │  Metadata)  │    └──────────────────┘    │
                    │  └─────────────┘              │             │
                    │                        ┌──────▼──────┐      │
                    │                        │  Groq API   │      │
                    │                        │  (LLaMA 3)  │      │
                    │                        └─────────────┘      │
                    └──────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | LLaMA 3 (via Groq API) |
| **RAG Framework** | LangChain |
| **Vector Store** | FAISS |
| **Backend** | FastAPI |
| **Database** | MongoDB |
| **Auth** | JWT (JSON Web Tokens) |
| **Containerization** | Docker |
| **Deployment** | Render |
| **Language** | Python 3.11+ |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- MongoDB instance (local or Atlas)
- [Groq API Key](https://console.groq.com/)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-study-buddy.git
cd rag-study-buddy
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Groq
GROQ_API_KEY=your_groq_api_key_here

# MongoDB
MONGODB_URI=mongodb+srv://your_connection_string
MONGODB_DB_NAME=rag_study_buddy

# JWT
JWT_SECRET_KEY=your_super_secret_key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# App
APP_ENV=development
FAISS_INDEX_PATH=./data/faiss_index
```

### 3. Run with Docker (Recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### 4. Run Locally (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --port 8000
```

---

## 📁 Project Structure

```
rag-study-buddy/
├── app/
│   ├── main.py               # FastAPI app entry point
│   ├── api/
│   │   ├── auth.py           # JWT auth routes (register, login)
│   │   ├── chat.py           # Chat endpoint with RAG
│   │   └── documents.py      # File upload & management
│   ├── core/
│   │   ├── config.py         # App settings & env vars
│   │   ├── security.py       # JWT token logic
│   │   └── database.py       # MongoDB connection
│   ├── rag/
│   │   ├── pipeline.py       # Main RAG chain (LangChain)
│   │   ├── embeddings.py     # Embedding model setup
│   │   ├── vector_store.py   # FAISS index management
│   │   ├── retriever.py      # Document retrieval logic
│   │   └── memory.py         # Persistent chat memory
│   └── models/
│       ├── user.py           # User schema (MongoDB)
│       ├── document.py       # Document metadata schema
│       └── chat.py           # Chat history schema
├── data/
│   └── faiss_index/          # Persisted FAISS indexes (per user)
├── tests/
│   ├── test_rag.py
│   ├── test_auth.py
│   └── test_documents.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📡 API Endpoints

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/register` | Register a new user |
| `POST` | `/auth/login` | Login and receive JWT token |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload PDF or text file |
| `GET` | `/documents/` | List all uploaded documents |
| `DELETE` | `/documents/{id}` | Delete a document |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/` | Send a message, get RAG response |
| `GET` | `/chat/history` | Get full conversation history |
| `DELETE` | `/chat/history` | Clear conversation history |

### Example Chat Request

```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the concept of gradient descent from my notes"}'
```

### Example Chat Response

```json
{
  "answer": "Gradient descent is an optimization algorithm used to minimize a loss function by iteratively moving in the direction of steepest descent...",
  "sources": [
    {
      "document": "ML_Lecture_3.pdf",
      "page": 7,
      "chunk": "Gradient descent updates weights by subtracting the gradient..."
    }
  ],
  "conversation_id": "conv_abc123"
}
```

---

## ⚙️ RAG Pipeline Details

```
User Query
    │
    ▼
Embedding (HuggingFace / Groq)
    │
    ▼
FAISS Similarity Search  ──▶  Top-K Relevant Chunks
    │
    ▼
Context Assembly (LangChain)
    │
    ▼
Prompt Construction (Query + Context + Chat History)
    │
    ▼
LLaMA 3 Generation (Groq API)
    │
    ▼
Citation Extraction + Response Formatting
    │
    ▼
Save to MongoDB (Chat History)
    │
    ▼
Return to User
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html
```

---

## 🐳 Docker

```bash
# Build image
docker build -t rag-study-buddy .

# Run container
docker run -p 8000:8000 --env-file .env rag-study-buddy

# Full stack with Docker Compose (app + MongoDB)
docker-compose up -d
```

---

## ☁️ Deployment (Render)

This project is deployed on [Render](https://render.com).

1. Push your code to GitHub
2. Connect repo to Render as a **Web Service**
3. Set all environment variables in the Render dashboard
4. Set the start command:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
5. Deploy — Render auto-detects the `Dockerfile` if present

---

## 🔮 Roadmap

- [ ] Web UI (React frontend)
- [ ] Quiz generation from uploaded notes
- [ ] Multi-user document sharing
- [ ] Voice input support
- [ ] Flashcard export (Anki format)
- [ ] Support for YouTube lecture transcripts

---

## 👨‍💻 Author

**Saptak Mondal**
B.Tech AIML — GLA University, Mathura
[GitHub](https://github.com/Saptak20) · [LinkedIn](https://linkedin.com/in/your-profile)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
