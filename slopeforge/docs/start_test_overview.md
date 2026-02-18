# Start and Test Overview

Summary of how to start and test each project based on its README.

| Project | Start / Run | Test / Verify | Sample curl |
| --- | --- | --- | --- |
| codeforge | `pip install -r requirements.txt` then `uvicorn app:app --reload --port 8000` (local), or `docker compose up --build -d` | Curl `/v1/codegen`, `/v1/execute`, `/v1/solve` (documented in README) | `curl -sS -X POST http://localhost:8000/v1/codegen -H 'Content-Type: application/json' -d '{"task":"find average of 1,2,3"}'` |
| vectordb-chroma | Not documented (README is a template) | Not documented | Not documented |
| ScholarRAG-Agentic-Research-Paper-Chatbot | Local: backend `uvicorn app.main:app --reload` + frontend `npm run dev`; Docker: `docker-compose up --build` | Health check and scenario list in README; backend tests: `cd backend && pytest` | `curl -s http://localhost:8000/api/health` |
| LocalRAGStudio-Langchain | `uvicorn app.main:app --reload --app-dir backend --port 8010` | API curl tests and optional `pytest` in README | `curl -s http://localhost:8010/stats` |
| GraphScholar-RAG | Backend: `uvicorn app.main:app --reload --port 8002`; Frontend: `npm run dev` | Manual usage steps (upload PDF, ask questions); tests not listed | Not listed in README |
| SlopeForge-House-Price-Regression-Lab | `python -m slopeforge.cli train ...` and `python -m slopeforge.cli serve --model models/artifacts/model.joblib` | CLI usage + curl `/predict` in README | `curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"OverallQual":7,"GrLivArea":1710}'` |
