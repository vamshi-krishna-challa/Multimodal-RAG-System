"""Streamlit application for a local multimodal RAG system."""
from __future__ import annotations

import streamlit as st

from src.document_processing import (
    extract_text_from_audio,
    extract_text_from_image,
    extract_text_from_pdf,
    extract_text_from_text_file,
)
from src.rag_pipeline import MultimodalRAGPipeline

st.set_page_config(page_title="Multimodal RAG System", layout="wide")

st.title("Multimodal RAG System")
st.write(
    "Upload PDFs, images, audio files, or text files. The app extracts text, builds a local vector index, "
    "and answers questions using a local Ollama LLM."
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = MultimodalRAGPipeline()
if "text_blocks" not in st.session_state:
    st.session_state.text_blocks = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

uploaded_files = st.file_uploader(
    "Upload files",
    type=["pdf", "png", "jpg", "jpeg", "wav", "mp3", "ogg", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Extract Text"):
    st.session_state.text_blocks = []
    for file in uploaded_files:
        suffix = file.name.lower().split(".")[-1]
        try:
            if suffix == "pdf":
                text = extract_text_from_pdf(file)
            elif suffix in {"png", "jpg", "jpeg"}:
                text = extract_text_from_image(file)
            elif suffix in {"wav", "mp3", "ogg"}:
                text = extract_text_from_audio(file, suffix=f".{suffix}")
            else:
                text = extract_text_from_text_file(file)
            st.session_state.text_blocks.append((file.name, text))
        except Exception as exc:
            st.error(f"Could not process {file.name}: {exc}")

    st.success(f"Extracted text from {len(st.session_state.text_blocks)} file(s).")

if st.session_state.text_blocks:
    with st.expander("Preview extracted text", expanded=False):
        for source, text in st.session_state.text_blocks:
            st.subheader(source)
            st.text_area("Extracted text", text[:5000], height=160, key=f"preview_{source}")

    if st.button("Build RAG Index"):
        try:
            chunk_count = st.session_state.pipeline.build_index(st.session_state.text_blocks)
            st.session_state.index_ready = True
            st.success(f"Vector index created with {chunk_count} chunks.")
        except Exception as exc:
            st.error(f"Index creation failed: {exc}")

st.divider()

question = st.text_input("Ask a question about the uploaded files")
if question and st.session_state.index_ready:
    try:
        result = st.session_state.pipeline.answer(question)
        st.subheader("Answer")
        st.write(result.answer)
        with st.expander("Retrieved context"):
            st.write(result.context)
    except Exception as exc:
        st.error(f"Question answering failed: {exc}")
elif question and not st.session_state.index_ready:
    st.warning("Please build the RAG index before asking questions.")
