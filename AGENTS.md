# AGENTS.md

## README is aspirational — verify before acting

- The README describes a **target architecture** (FastAPI `app/`, `tests/`, Docker, MongoDB, Groq/LLaMA 3, JWT). Almost none of it exists yet. There is no `requirements.txt`, `.env.example`, `Dockerfile`, `docker-compose.yml`, or CI.
- The real code is a CLI prototype: `rag_study_buddy.py` (entrypoint, interactive chat loop) + `utils_rag.py` (PDF loading, splitting, FAISS helpers).
- README commands (`uvicorn app.main:app`, `pytest tests/`, `docker-compose up`) reference files that don't exist. Don't run them.

## Running

- Run from repo root: `python rag_study_buddy.py`. It fails unless:
  - `data/notes.pdf` exists (directory is not committed — create it).
  - A working OpenAI API key is set — `rag_study_buddy.py` hardcodes a placeholder into `os.environ["OPENAI_API_KEY"]` (not read from `.env`). Don't put a real key there.
- Dependencies are undeclared and unpinned. Required: `langchain`, `openai`, `faiss-cpu`, `pypdf`.

## LangChain quirks

- Imports use **legacy pre-split LangChain paths** (`langchain.embeddings`, `langchain.chat_models`, `langchain.document_loaders`, `langchain.chains`, `langchain.vectorstores`). These fail on modern LangChain — current equivalents are `langchain_openai`, `langchain_community.document_loaders`, `langchain_community.vectorstores`, `langchain_text_splitters`. `RetrievalQA` is deprecated.
- README says Groq/LLaMA 3; the code actually uses OpenAI `gpt-3.5-turbo`.
- Pipeline tuning lives in `utils_rag.py`: `chunk_size=500`, `chunk_overlap=50`, retriever `k=3`.
- The FAISS index is in-memory only — nothing is persisted, despite the README's `FAISS_INDEX_PATH=./data/faiss_index`.

## Workflow

- Single `main` branch; commits go directly to it, no PR flow.
