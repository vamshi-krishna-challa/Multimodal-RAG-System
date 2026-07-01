"""Utilities for text extraction from PDFs, images, audio, and text files."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from PIL import Image
import pytesseract
import whisper
from pypdf import PdfReader


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_image(file) -> str:
    image = Image.open(file)
    return pytesseract.image_to_string(image).strip()


def extract_text_from_audio(file, suffix: str = ".wav", model_name: str = "base") -> str:
    model = whisper.load_model(model_name)
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name
    result = model.transcribe(temp_path)
    return result.get("text", "").strip()


def extract_text_from_text_file(file) -> str:
    raw = file.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore").strip()
    return str(raw).strip()


def save_text_sample(text: str, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(text, encoding="utf-8")
