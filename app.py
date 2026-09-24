"""
Chat With Your Documents — an AI app that lets a user upload a PDF or CSV
and ask questions about it in plain English.

Tech stack (all free-tier friendly):
- Streamlit          -> web UI
- Groq API           -> LLM (free, fast, generous limits — openai/gpt-oss-20b)
- ChromaDB           -> local vector database (no cost, runs on your machine)
- pypdf              -> PDF text extraction
- pandas             -> CSV handling

Author: (your name here) — built as a portfolio + freelance demo project.
"""

import os
import uuid
import tempfile

import streamlit as st
import pandas as pd
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Chat With Your Documents", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# CUSTOM STYLING — glassmorphic dark dashboard look, inspired by enterprise
# AI-platform UIs: deep navy background, frosted-glass panels, teal accent
# with a soft gold secondary accent, pill-shaped buttons, small-caps labels.
# ---------------------------------------------------------------------------
ACCENT = "#8b7cf6"          # purple accent (calmer, flat — not the neon gradient from before)
ACCENT_GOLD = "#c9a876"     # soft, slightly muted gold
GLASS = "rgba(255,255,255,0.04)"
GLASS_BORDER = "rgba(255,255,255,0.10)"
TEXT_MUTED = "#8a94a6"

st.markdown(f"""
<style>
    /* Overall page background — deep navy with a very subtle teal glow, like the reference */
    .stApp {{
        background:
            radial-gradient(circle at 30% 10%, rgba(45,212,191,0.06) 0%, transparent 45%),
            #05070a;
        color: #e6e9ef;
    }}

    header[data-testid="stHeader"] {{
        background: #05070a !important;
    }}
    [data-testid="stBottomBlockContainer"], [data-testid="stBottom"] {{
        background: #05070a !important;
        border-top: 1px solid {GLASS_BORDER};
    }}

    /* Sidebar — frosted glass panel */
    section[data-testid="stSidebar"] {{
        background: rgba(10,14,20,0.9);
        backdrop-filter: blur(16px);
        border-right: 1px solid {GLASS_BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: #e6e9ef !important;
    }}

    /* Main title */
    h1 {{
        color: #f2f4f7 !important;
        font-weight: 650 !important;
        letter-spacing: -0.01em;
    }}

    /* Small-caps section labels, teal, like "RI ANALYSIS" in the reference */
    h3, h4, .stMarkdown strong {{
        color: {ACCENT} !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.85em !important;
    }}

    .stApp p, .stApp li, .stApp span {{
        color: #c3c9d4;
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {TEXT_MUTED} !important;
        text-transform: none;
        letter-spacing: normal;
    }}

    /* File uploader — frosted glass card */
    [data-testid="stFileUploaderDropzone"] {{
        background: {GLASS};
        backdrop-filter: blur(12px);
        border: 1px solid {GLASS_BORDER};
        border-radius: 16px;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {ACCENT};
    }}

    /* Buttons — pill-shaped, teal fill */
    .stButton button, [data-testid="stFileUploaderDropzone"] button {{
        background: {ACCENT} !important;
        color: #f2f4f7 !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        padding: 0.5em 1.3em !important;
        box-shadow: none !important;
    }}
    .stButton button:hover {{
        background: #7364e0 !important;
    }}

    /* Chat input — frosted glass pill, like the reference's floating command bar */
    [data-testid="stChatInput"] {{
        background: rgba(15,19,26,0.85) !important;
        backdrop-filter: blur(16px);
        border: 1px solid {GLASS_BORDER};
        border-radius: 999px;
    }}
    [data-testid="stChatInput"] textarea {{
        background: transparent !important;
        color: #e6e9ef !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {ACCENT};
        box-shadow: none;
    }}

    /* Chat messages — glass cards with a thin left accent bar */
    [data-testid="stChatMessage"] {{
        background: {GLASS};
        backdrop-filter: blur(10px);
        border: 1px solid {GLASS_BORDER};
        border-left: 3px solid {ACCENT};
        border-radius: 12px;
        padding: 8px 10px;
    }}

    /* Alerts — glass cards with gold accent for emphasis */
    [data-testid="stAlertContainer"] {{
        background: {GLASS} !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        border: 1px solid {GLASS_BORDER} !important;
        border-left: 3px solid {ACCENT_GOLD} !important;
    }}

    /* Text input (API key box) */
    input[type="password"], input[type="text"] {{
        background: rgba(15,19,26,0.8) !important;
        color: #e6e9ef !important;
        border-radius: 999px !important;
        border: 1px solid {GLASS_BORDER} !important;
    }}
</style>
""", unsafe_allow_html=True)

GROQ_MODEL = "openai/gpt-oss-20b"   # free-tier Groq model, good quality and very fast
CHUNK_SIZE = 1000        # characters per chunk when splitting documents
CHUNK_OVERLAP = 200      # overlap between chunks so context isn't cut mid-sentence
TOP_K = 6                # how many relevant chunks to retrieve per question (was 4 — too few for longer docs)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def extract_text_from_csv(file) -> str:
    df = pd.read_csv(file)
    # Turn the dataframe into readable text chunks (one "row description" per line)
    # This lets the LLM reason about rows in natural language.
    lines = [", ".join(f"{col}: {row[col]}" for col in df.columns) for _, row in df.iterrows()]
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """
    The underlying Chroma client is safe to share across users (it's just a
    connection), but each user session must get its OWN collection so that
    different visitors' documents never mix together. That mixing was the bug
    causing "I don't have enough information" answers during testing.
    """
    return chromadb.Client()


def get_chroma_collection():
    """
    Creates a brand-new, isolated collection for THIS browser session only.
    Not cached with @st.cache_resource on purpose — that was the bug.
    """
    client = get_chroma_client()
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(name=collection_name, embedding_function=embed_fn)
    return collection


def index_document(collection, text: str, source_name: str):
    chunks = chunk_text(text)
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def retrieve_context(collection, question: str, top_k: int = TOP_K):
    results = collection.query(query_texts=[question], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    return docs


def ask_llm(client: Groq, question: str, context_chunks: list) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions using the provided "
        "document context. Use your best judgment to answer using whatever "
        "relevant information IS present, even if it's not a complete answer. "
        "Only say you don't have enough information if the context is truly "
        "unrelated to the question — never fabricate facts that aren't in the context."
    )
    user_prompt = f"Context from the document:\n{context}\n\nQuestion: {question}\n\nAnswer clearly and concisely."

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("📄 Chat With Your Documents")
st.caption("Upload a PDF or CSV, then ask questions about it in plain English — powered by AI.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
    )
    st.markdown("[Get a free Groq API key →](https://console.groq.com/keys)")
    st.markdown("---")
    st.markdown(
        "**How this works:**\n"
        "1. Your file is split into small chunks\n"
        "2. Chunks are converted into vectors and stored locally\n"
        "3. Your question retrieves only the *relevant* chunks\n"
        "4. Those chunks + your question are sent to the AI model\n"
        "\nThis is called **RAG** (Retrieval-Augmented Generation) — "
        "it's how most modern AI document tools work."
    )

if "collection" not in st.session_state:
    st.session_state.collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

if uploaded_file is not None and uploaded_file.name not in st.session_state.indexed_files:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar first.")
    else:
        with st.spinner(f"Reading and indexing {uploaded_file.name}..."):
            if uploaded_file.name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = extract_text_from_csv(uploaded_file)

            if st.session_state.collection is None:
                st.session_state.collection = get_chroma_collection()

            num_chunks = index_document(st.session_state.collection, text, uploaded_file.name)
            st.session_state.indexed_files.append(uploaded_file.name)
            st.success(f"Indexed {uploaded_file.name} into {num_chunks} chunks. Ask away!")

if st.session_state.indexed_files:
    st.info(f"📚 Loaded documents: {', '.join(st.session_state.indexed_files)}")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your document..."):
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar first.")
    elif st.session_state.collection is None:
        st.warning("Please upload a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context_chunks = retrieve_context(st.session_state.collection, question)
                client = get_groq_client(api_key)
                answer = ask_llm(client, question, context_chunks)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
