# Token Saver

A middleware that compresses prose into structured XML before sending to LLMs. The goal is to reduce token usage without losing meaning, making AI cheaper for end users.

## Status
Pre-v1.0 — project scaffolding

## Stack
- Backend: FastAPI (Python 3.14) on Railway
- Frontend: React on Vercel
- Database: SQLite (v1.0), Postgres (v1.1+)
- LLM: Claude API (Anthropic)

## How it works
User prose → classifier → generic splitter → XML assembler → Claude API → response
Token counts are measured before and after compression to prove savings.
