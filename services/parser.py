"""简历解析服务 —— 支持 PDF / DOCX。"""
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document


class ResumeParser:
    """解析简历文件，提取文本内容和元信息。"""

    SUPPORTED = {".pdf", ".docx"}

    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_and_parse(self, filename: str, content: bytes) -> dict:
        """保存文件并解析，返回 {'resume_id', 'filename', 'text', 'page_count'}。"""
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持 {self.SUPPORTED}")

        resume_id = uuid.uuid4().hex[:12]
        save_path = self.upload_dir / f"{resume_id}{ext}"
        save_path.write_bytes(content)

        text, page_count = self._parse(save_path, ext)
        return {
            "resume_id": resume_id,
            "filename": filename,
            "text": text,
            "page_count": page_count,
        }

    def _parse(self, path: Path, ext: str) -> tuple[str, int]:
        if ext == ".pdf":
            return self._parse_pdf(path)
        return self._parse_docx(path)

    @staticmethod
    def _parse_pdf(path: Path) -> tuple[str, int]:
        doc = fitz.open(path)
        pages = [page.get_text() for page in doc]
        text = "\n---\n".join(pages)
        return text, len(pages)

    @staticmethod
    def _parse_docx(path: Path) -> tuple[str, int]:
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text, 1
