import fitz  # PyMuPDF
from pathlib import Path

class DocumentProcessor:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        doc = fitz.open(str(self.pdf_path))
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text.strip()