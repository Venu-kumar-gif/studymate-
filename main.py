"""
StudyMate: PDF-based Conversational Q&A (End-to-End)
---------------------------------------------------
A complete Streamlit app that lets users upload one or more PDF files,
extracts and chunks text with PyMuPDF, embeds chunks with Sentence-Transformers,
indexes them in FAISS, retrieves relevant chunks for a user question, and
produces an answer using either:
  • IBM watsonx (Mixtral-8x7B-Instruct) — if credentials are provided, OR
  • A local Transformers pipeline fallback (FLAN-T5) — runs on CPU for demo.

This file is self-contained. Run it with:
  streamlit run studymate_streamlit_app.py

Requirements (install once):
  pip install streamlit pymupdf sentence-transformers faiss-cpu transformers torch numpy pandas nltk tqdm
Optional (for IBM watsonx):
  pip install ibm-watsonx-ai

Environment variables for IBM watsonx (if you plan to use it):
  export WATSONX_API_KEY="your_api_key"
  export WATSONX_URL="https://us-south.ml.cloud.ibm.com"   # or your region endpoint
  export WATSONX_PROJECT_ID="your_project_id"              # if required by your account setup

Notes:
- The IBM SDK usage here is minimal and guarded; if credentials are missing, the app will
  automatically fallback to a local text2text model (FLAN-T5 base) for answer generation.
- For hackathon speed, chunking is kept simple (word windows with overlap). You can tweak
  CHUNK_SIZE and CHUNK_OVERLAP in the sidebar.
- The FAISS index is built in-memory and stored in st.session_state for quick iteration.
- On large PDFs, first time embedding will take time; embeddings are cached in-session.
"""

import os
import io
import gc
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from tqdm import tqdm

# PDF processing
import fitz  # PyMuPDF

# Embeddings & Retrieval
from sentence_transformers import SentenceTransformer
import faiss

# LLMs (fallback)
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Optional: IBM watsonx
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Model
    WATSONX_AVAILABLE = True
except Exception:
    WATSONX_AVAILABLE = False

# ---------------------------
# Configuration & Utilities
# ---------------------------

st.set_page_config(page_title="StudyMate – PDF Q&A", page_icon="📚", layout="wide")

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOCAL_T2T_MODEL = "google/flan-t5-base"  # CPU-friendly fallback

@dataclass
class Chunk:
    doc_id: str         # user-visible name (filename)
    page: int           # 1-based page index
    start_word: int     # window start (word index in page)
    end_word: int       # window end (word index in page)
    text: str           # chunk text


def read_pdf_text(file_bytes: bytes, doc_id: str) -> List[Tuple[int, str]]:
    """Extract per-page plain text using PyMuPDF. Returns list of (page_num_1based, text)."""
    pages: List[Tuple[int, str]] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            txt = page.get_text("text") or ""
            pages.append((i + 1, txt))
    return pages


def clean_text(t: str) -> str:
    # Minimal cleaning; you can expand (remove headers/footers, normalize spaces, etc.)
    t = t.replace("\u00a0", " ")
    t = "\n".join([line.strip() for line in t.splitlines() if line.strip()])
    return t


def chunk_page_words(doc_id: str, page_num: int, text: str, win: int, overlap: int) -> List[Chunk]:
    words = text.split()
    chunks: List[Chunk] = []
    if not words:
        return chunks
    step = max(1, win - overlap)
    for start in range(0, len(words), step):
        end = min(len(words), start + win)
        if end - start < max(20, win // 5):  # avoid too-short tails
            break
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(doc_id=doc_id, page=page_num, start_word=start, end_word=end, text=chunk_text))
    return chunks


def build_chunks_from_pdfs(files: List[Tuple[str, bytes]], win: int, overlap: int) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for fname, fbytes in files:
        pages = read_pdf_text(fbytes, fname)
        for page_num, raw in pages:
            cleaned = clean_text(raw)
            page_chunks = chunk_page_words(fname, page_num, cleaned, win, overlap)
            all_chunks.extend(page_chunks)
    return all_chunks


def embed_texts(encoder: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    vectors = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


def build_faiss_index(embeds: np.ndarray) -> faiss.IndexFlatIP:
    # Using Inner Product (cosine if vectors are normalized)
    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)
    return index


def top_k_search(index: faiss.IndexFlatIP, qvec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(qvec, k)
    return D, I


# ---------------------------
# LLM Backends
# ---------------------------

class AnswerGenerator:
    def __init__(self):
        self.backend = None
        self.device = 0 if torch.cuda.is_available() else -1
        # Try watsonx first if available & creds present
        self._init_watsonx_if_possible()
        if self.backend is None:
            self._init_local_flan()

    def _init_watsonx_if_possible(self):
        if not WATSONX_AVAILABLE:
            return
        api_key = os.getenv("WATSONX_API_KEY")
        url = os.getenv("WATSONX_URL")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        if not api_key or not url:
            return
        try:
            credentials = Credentials(api_key=api_key, url=url)
            # Mixtral instruct id can vary per account; common alias below:
            model_id = "mistralai/mixtral-8x7b-instruct-v0.1"
            params = {
                "decoding_method": "greedy",
                "max_new_tokens": 512,
                "temperature": 0.2,
                "repetition_penalty": 1.05,
            }
            self.wxs_model = Model(model_id=model_id, params=params, credentials=credentials, project_id=project_id)
            self.backend = "watsonx"
        except Exception as e:
            # Fallback will be initialized
            self.backend = None

    def _init_local_flan(self):
        # Lightweight CPU-capable model for hackathon demo
        try:
            self.local_pipe = pipeline(
                "text2text-generation",
                model=DEFAULT_LOCAL_T2T_MODEL,
                device=self.device
            )
            self.backend = "local"
        except Exception:
            self.backend = None

    def generate(self, question: str, contexts: List[Chunk]) -> str:
        """Combine retrieved chunks and ask the model to answer grounded in them."""
        context_blocks = []
        for i, ch in enumerate(contexts, start=1):
            tag = f"[Source {i}: {ch.doc_id} • p.{ch.page}]"
            context_blocks.append(f"{tag}\n{ch.text}")
        context = "\n\n".join(context_blocks)

        prompt = (
            "You are StudyMate, an academic assistant. Answer the question using ONLY the provided sources.\n"
            "If the answer is not in the sources, say you don't have enough information from the PDFs.\n"
            "Provide a concise explanation and include source tags like [Source 1], [Source 2].\n\n"
            f"Question: {question}\n\nSources:\n{context}\n\nAnswer:"
        )

        if self.backend == "watsonx":
            try:
                response = self.wxs_model.generate_text(prompt=prompt)
                return response.strip()
            except Exception as e:
                # If watsonx fails at runtime, fall back locally
                pass

        if self.backend == "local" and hasattr(self, "local_pipe"):
            out = self.local_pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
            return out.strip()

        # Final fallback: simple extractive answer (join top contexts)
        return (
            "I couldn't access a generation model. Here are the most relevant excerpts from your PDFs:\n\n"
            + "\n\n".join([f"[Source {i+1}] {c.text}" for i, c in enumerate(contexts)])
        )


# ---------------------------
# Streamlit App
# ---------------------------

st.title("📚 StudyMate – PDF Q&A Assistant")
st.caption("Upload PDFs → Ask questions → Get grounded answers with sources.")

with st.sidebar:
    st.header("Settings")
    CHUNK_SIZE = st.number_input("Chunk size (words)", min_value=100, max_value=1000, value=350, step=50)
    CHUNK_OVERLAP = st.number_input("Chunk overlap (words)", min_value=0, max_value=500, value=80, step=10)
    TOP_K = st.slider("Top-K chunks to retrieve", 1, 10, 5)
    EMBED_MODEL = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL, help="HuggingFace Sentence-Transformers model id")
    st.divider()
    st.subheader("LLM Backend")
    use_watson = st.toggle("Prefer IBM watsonx if available", value=True)
    st.write("If disabled, the app will use a local FLAN-T5 model.")

# Session state containers
if "chunks" not in st.session_state:
    st.session_state.chunks: List[Chunk] = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "encoder_name" not in st.session_state:
    st.session_state.encoder_name = None
if "encoder" not in st.session_state:
    st.session_state.encoder = None
if "answer_gen" not in st.session_state:
    st.session_state.answer_gen = None

# File uploader
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

colA, colB = st.columns([1,1])
with colA:
    build_btn = st.button("🔧 Build Knowledge Base", type="primary")
with colB:
    clear_btn = st.button("🧹 Clear Session")

if clear_btn:
    st.session_state.chunks = []
    st.session_state.embeddings = None
    st.session_state.faiss_index = None
    st.session_state.encoder = None
    st.session_state.encoder_name = None
    st.session_state.answer_gen = None
    gc.collect()
    st.success("Session cleared.")

# Build index pipeline
if build_btn:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        # Read files into memory (filename, bytes)
        files: List[Tuple[str, bytes]] = []
        for f in uploaded_files:
            files.append((f.name, f.read()))

        with st.status("Extracting & chunking PDFs…", expanded=True) as status:
            st.write("Reading and cleaning text…")
            chunks = build_chunks_from_pdfs(files, win=int(CHUNK_SIZE), overlap=int(CHUNK_OVERLAP))
            st.write(f"Created {len(chunks)} chunks.")

            st.write("Loading embedding model…")
            if st.session_state.encoder is None or st.session_state.encoder_name != EMBED_MODEL:
                st.session_state.encoder = SentenceTransformer(EMBED_MODEL)
                st.session_state.encoder_name = EMBED_MODEL
            encoder = st.session_state.encoder

            st.write("Embedding chunks… (this can take a while the first time)")
            texts = [c.text for c in chunks]
            embeds = embed_texts(encoder, texts)

            st.write("Building FAISS index…")
            index = build_faiss_index(embeds)

            st.session_state.chunks = chunks
            st.session_state.embeddings = embeds
            st.session_state.faiss_index = index
            status.update(label="Knowledge base ready!", state="complete")
            st.success("✅ Knowledge base built. You can start asking questions below.")

st.divider()

# Chat/Q&A Section
st.subheader("Ask a question from your PDFs")
question = st.text_input("Your question")
ask_btn = st.button("🔎 Retrieve & Answer")

if ask_btn:
    if not question.strip():
        st.warning("Please type a question.")
    elif st.session_state.faiss_index is None:
        st.warning("Please upload PDFs and click ‘Build Knowledge Base’ first.")
    else:
        # Encode question and retrieve top-k
        encoder = st.session_state.encoder
        qvec = encoder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = top_k_search(st.session_state.faiss_index, qvec, k=TOP_K)
        indices = I[0].tolist()
        scores = D[0].tolist()
        retrieved = [st.session_state.chunks[i] for i in indices]

        # Initialize answer generator if needed
        if st.session_state.answer_gen is None:
            ag = AnswerGenerator()
            if not use_watson and ag.backend == "watsonx":
                # Force local if user disabled watsonx toggle
                ag.backend = None
                ag._init_local_flan()
            st.session_state.answer_gen = ag
        ag = st.session_state.answer_gen

        with st.spinner("Generating answer…"):
            answer = ag.generate(question, retrieved)

        # Display results
        st.markdown("### ✅ Answer")
        st.write(answer)

        st.markdown("### 📎 Sources")
        src_df = pd.DataFrame([
            {
                "Source #": i+1,
                "File": c.doc_id,
                "Page": c.page,
                "Snippet": (c.text[:300] + ("…" if len(c.text) > 300 else "")),
                "Score (cosine)": round(scores[i], 4),
            }
            for i, c in enumerate(retrieved)
        ])
        st.dataframe(src_df, use_container_width=True, hide_index=True)

        with st.expander("Show raw retrieved chunks"):
            for i, c in enumerate(retrieved, start=1):
                st.markdown(f"**Source {i} — {c.doc_id} (p.{c.page})**")
                st.write(c.text)
                st.markdown("---")

st.divider()

st.markdown(
    """
**Tips**
- Use smaller *Chunk size* (e.g., 250) for fine-grained answers, larger (e.g., 500+) for summaries.
- Increase *Top-K* if the question spans multiple places in the PDFs.
- If IBM watsonx creds are added as environment variables, the app will automatically prefer it.
    """
)
