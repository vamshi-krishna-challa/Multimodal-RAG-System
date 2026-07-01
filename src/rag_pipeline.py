"""RAG pipeline using LangChain, Ollama, and ChromaDB."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CONFIG


@dataclass
class RAGResult:
    answer: str
    context: str
    source_count: int


class MultimodalRAGPipeline:
    """Builds a vector index from extracted multimodal text and answers questions."""

    def __init__(self) -> None:
        self.llm = ChatOllama(
            model=CONFIG.ollama_model,
            base_url=CONFIG.ollama_base_url,
            temperature=0.1,
        )
        self.embeddings = OllamaEmbeddings(
            model=CONFIG.embedding_model,
            base_url=CONFIG.ollama_base_url,
        )
        self.vectorstore = None
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful multimodal RAG assistant. Answer only using the provided context. "
                    "If the answer is not in the context, say that the provided documents do not contain enough information.",
                ),
                ("system", "Context:\n{context}"),
                ("human", "Question: {question}"),
            ]
        )

    def build_index(self, text_blocks: List[Tuple[str, str]]) -> int:
        documents = [Document(page_content=text, metadata={"source": source}) for source, text in text_blocks if text.strip()]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.chunk_size,
            chunk_overlap=CONFIG.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CONFIG.chroma_db_path,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": CONFIG.top_k})
        return len(chunks)

    def answer(self, question: str) -> RAGResult:
        if self.retriever is None:
            raise RuntimeError("Build the vector index before asking questions.")
        retrieved_docs = self.retriever.invoke(question)
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in retrieved_docs
        )
        messages = self.prompt.format_messages(context=context, question=question)
        response = self.llm.invoke(messages)
        return RAGResult(answer=response.content, context=context, source_count=len(retrieved_docs))
