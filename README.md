# Multimodal RAG System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

Multimodal RAG System is a local Retrieval-Augmented Generation application that supports document-based question answering across multiple input types, including PDFs, images, audio files, plain text, and Markdown files.

The system extracts text from each modality, converts the extracted content into embeddings, stores the embeddings in ChromaDB, retrieves relevant context, and generates grounded responses using a local Ollama LLM.

## Features

- PDF text extraction
- OCR-based image understanding using Tesseract
- Audio transcription using Whisper
- Text and Markdown ingestion
- Text chunking for retrieval
- Embedding generation using Ollama embeddings
- ChromaDB vector storage
- Local LLM response generation with Ollama
- Streamlit web interface
- Retrieved-context transparency for debugging and review

## Architecture

```text
PDF / Image / Audio / Text
          |
          v
Text Extraction Layer
          |
          v
Chunking and Preprocessing
          |
          v
Embedding Model
          |
          v
ChromaDB Vector Store
          |
          v
Retriever
          |
          v
Local LLM with Ollama
          |
          v
Grounded Answer + Retrieved Context
```

## Technology Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Ollama
- Whisper
- PyTesseract
- PyPDF
- Pillow

## Installation

Clone the repository:

```bash
git clone https://github.com/vamshi-krishna-challa/Multimodal-RAG-System.git
cd Multimodal-RAG-System
```

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama serve
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Example Workflow

1. Upload a PDF, image, audio file, or text document.
2. Extract text from the uploaded files.
3. Build the RAG index.
4. Ask a question about the uploaded content.
5. Review the grounded answer and retrieved context.

## Project Structure

```text
Multimodal-RAG-System/
|
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── .env.example
|
├── src/
│   ├── config.py
│   ├── document_processing.py
│   ├── rag_pipeline.py
│   └── __init__.py
|
├── assets/
├── docs/
└── sample_documents/
```

## Use Cases

- Research paper question answering
- Audio lecture summarization
- Image text extraction and analysis
- Enterprise document search
- Industrial maintenance documentation analysis
- Multimodal knowledge assistant

## Future Improvements

- Add hybrid search
- Add citations for each answer
- Add Docker support
- Add user authentication
- Add evaluation metrics for retrieval quality
- Add support for image captioning models
- Add support for multi-turn chat memory
- Add LangGraph workflow orchestration

## Author

**Vamshi Krishna Challa**  
Ph.D. Student, Computational Analysis and Modeling  
Louisiana Tech University

Research interests: Generative AI, RAG, Agentic AI, Industrial AI, Time-Series Forecasting, and Deep Learning.
