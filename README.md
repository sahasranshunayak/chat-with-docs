# 📄 Chat With Your Documents

An AI-powered app that lets you upload a **PDF or CSV** and ask questions about it in plain English. Built using **Retrieval-Augmented Generation (RAG)** — the same core technique behind tools like ChatGPT's "upload a file" feature.

**Live demo:** _add your deployed Streamlit Cloud link here once deployed_

## How it works

1. Your document is split into small overlapping text chunks.
2. Each chunk is converted into a vector embedding and stored in a local ChromaDB vector database.
3. When you ask a question, the app retrieves only the most *relevant* chunks (semantic search).
4. Those chunks + your question are sent to an LLM (Llama 3.3 70B via Groq) which answers using only that context — reducing hallucination.

## Tech stack

- **Streamlit** — web UI, no frontend code needed
- **Groq API** — free-tier LLM inference (very fast)
- **ChromaDB** — free, local vector database
- **pypdf / pandas** — file parsing

## Run it locally

1. Clone this repo and open a terminal in the folder.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Get a **free** Groq API key at https://console.groq.com/keys (sign up, no credit card required).
5. Run the app:
   ```bash
   streamlit run app.py
   ```
6. Paste your Groq API key into the sidebar, upload a PDF or CSV, and start asking questions.

## Deploy it for free (so you have a live link to show clients)

1. Push this project to a public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app," pick your repo and `app.py` as the entry point.
4. Deploy. You'll get a public URL like `yourapp.streamlit.app`.
5. **Do not commit your API key.** Users paste their own key into the sidebar at runtime — this is intentional so you can share the live app safely without exposing your key or paying for others' usage.

## Ideas to extend this (great for a v2 commit / case study)

- Support multiple file uploads at once
- Add a "summarize this document" button
- Show which page/row the answer came from (source citations)
- Swap Groq for OpenAI/Anthropic/Gemini to compare answer quality
- Add authentication + persistent storage for a "real product" version

## Why this project exists

This was built as a portfolio project demonstrating practical skills in **LLM integration, vector search / RAG, and Python app deployment** — the exact skill set behind freelance AI-chatbot and AI-integration services in 2026.
