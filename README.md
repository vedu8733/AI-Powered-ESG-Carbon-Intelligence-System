# ESG & Carbon Intelligence System

AI-powered Environmental, Social, and Governance (ESG) analytics and carbon footprint intelligence platform. Built with free, open-source tools.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.10+, FastAPI |
| AI/LLM | LangChain, Groq API (LLaMA 3) |
| Embeddings | HuggingFace (sentence-transformers) |
| Vector Store | FAISS |
| Frontend | Streamlit |

## Project Structure

```
EDunetProject/
├── app/
│   ├── api/          # API routes and endpoints
│   ├── agents/       # LangChain agents (ESG, carbon)
│   ├── core/         # Config, settings
│   ├── models/       # Pydantic schemas
│   ├── rag/          # RAG pipeline, FAISS, embeddings
│   ├── services/     # Business logic
│   └── utils/        # Helpers
├── main.py           # FastAPI entry point
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)
```

### 4. Run the API server

```bash
# Development
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run Streamlit frontend (when implemented)

```bash
streamlit run app/frontend/streamlit_app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (liveness) |
| GET | `/health/ready` | Readiness probe |
| GET | `/api/v1` | API info |
| GET | `/docs` | Swagger UI (when DEBUG=true) |

## Production Deployment

```bash
# Using Gunicorn + Uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (required for LLM) | - |
| `GROQ_MODEL` | Groq model name | llama3-70b-8192 |
| `HF_EMBEDDING_MODEL` | HuggingFace embedding model | all-MiniLM-L6-v2 |
| `DEBUG` | Enable debug mode | false |

## License

MIT
