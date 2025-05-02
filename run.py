import streamlit as st
import os
from pathlib import Path

from core.chunker import ChapterAwareChunker
from core.document_processor import DocumentProcessor
from core.embedding_generator import EmbeddingGenerator
from core.minio_client import MinioUploader
from core.nvidia_llm import NVIDIAChunkAnalyzer

# Constants
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="LexiChat Uploader", layout="wide")
st.title("📘 LexiChat - Load Legal Document")

uploaded_file = st.file_uploader("Upload a legal PDF (e.g., Indian Penal Code)", type=["pdf"])

if uploaded_file is not None:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Saved to: {file_path}")

    # Upload to MinIO
    uploader = MinioUploader()
    object_path = uploader.upload_file(file_path)
    st.success(f"Uploaded to MinIO at: {object_path}")

    # Download it again locally to process with fitz (simulate production flow)
    downloaded_path = uploader.download_file(object_path.split("/", 1)[1], UPLOAD_DIR / f"minio_{uploaded_file.name}")

    # Process with PyMuPDF
    processor = DocumentProcessor(downloaded_path)
    full_text = processor.extract_text()
    st.success("Text extracted from MinIO-stored PDF ✅")
    chunker = ChapterAwareChunker()
    chapter_chunks = chunker.chunk(full_text)

    st.subheader("📘 Chapter-Aware Chunks")
    st.text(f"{len(chapter_chunks)} total chunks ✅")

    analyzer = NVIDIAChunkAnalyzer()
    results = []

    progress_bar = st.progress(0)
    for i, chunk in enumerate(chapter_chunks[:15]):
        metadata = {
            "chunk_number": i+1,
            "chunk_size": len(chunk),
            "file_name": uploaded_file.name,
            "file_extension": Path(uploaded_file.name).suffix
        }
        result = analyzer.analyze_chunk(chunk)
        results.append(result)
        embedding_generator = EmbeddingGenerator()  # Initialize the class
        embeddings = embedding_generator.generate_embeddings(result.get("rewritten_chunk"))

        progress_bar.progress((i + 1) / len(chapter_chunks))

    st.success("All chunks processed via NVIDIA LLM ✅")
    #

    if results:
        st.subheader("✅ Example LLM Response")
        st.json(results[0])

    st.text(chapter_chunks[0][:500])

    st.text(full_text[:1000])
