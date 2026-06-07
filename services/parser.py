"""简历解析服务 —— 支持 PDF / DOCX + LLM 验证。"""
import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document

# 常见中文简历段落标题
SECTION_HEADERS = [
    "教育背景", "教育经历",
    "工作经历", "工作经验", "工作背景",
    "项目经历", "项目经验",
    "专业技能", "技术栈", "技能清单", "专业技能",
    "自我评价", "个人简介", "个人总结", "自我评估",
    "证书", "荣誉", "获奖", "证书与荣誉",
    "语言能力", "语言",
    "实习经历",
    "科研经历", "研究成果",
    "社会活动", "校园活动", "社团经历",
]


class ResumeParser:
    """解析简历文件，提取文本内容和元信息。"""

    SUPPORTED = {".pdf", ".docx"}

    def __init__(self, upload_dir: str, llm=None):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm  # 用于解析后验证的 LLM

    def save_and_parse(self, filename: str, content: bytes) -> dict:
        """保存文件并解析，返回 {'resume_id', 'filename', 'text', 'sections', 'page_count', 'quality_warnings'}。"""
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持 {self.SUPPORTED}")

        resume_id = uuid.uuid4().hex[:12]
        save_path = self.upload_dir / f"{resume_id}{ext}"
        save_path.write_bytes(content)

        text, page_count = self._parse(save_path, ext)
        sections = self._detect_sections(text)
        warnings = self._validate(text) if self.llm else []

        return {
            "resume_id": resume_id,
            "filename": filename,
            "text": text,
            "sections": sections,
            "page_count": page_count,
            "quality_warnings": warnings,
        }

    @staticmethod
    def _detect_sections(text: str) -> list[dict]:
        """检测简历中的段落边界，返回 [{"section": "工作经验", "text": "..."}]。"""
        lines = text.split("\n")
        sections = []
        current_section = "其他"
        current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append(line)
                continue

            # 判断该行是否是段落标题
            header = None
            for h in SECTION_HEADERS:
                # 去除常见序号前缀后匹配
                clean = stripped.lstrip("一二三四五六七八九十零0123456789、.）)　 ")
                if clean and (clean == h or clean.startswith(h + " ") or clean.startswith(h + "：")):
                    header = stripped
                    break

            if header:
                if current_lines:
                    sections.append({"section": current_section, "text": "\n".join(current_lines).strip()})
                current_section = header
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({"section": current_section, "text": "\n".join(current_lines).strip()})

        return [s for s in sections if s["text"].strip()]

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

    def _validate(self, text: str) -> list[str]:
        """用 LLM 验证解析质量。"""
        if not text.strip():
            return ["解析结果为空"]
        if len(text) < 50:
            return ["解析文本过短，可能未正确提取内容"]

        prompt = (
            f"你是一个简历解析质量检查 Agent。检查下面这段从 PDF/DOCX 提取的文本，"
            f"判断是否有以下问题：\n"
            f"1. 存在大量乱码或不可读字符\n"
            f"2. 内容明显截断、不完整\n"
            f"3. 表格数据丢失导致信息断裂\n"
            f"4. 提取的文本结构异常（如姓名、联系方式等关键信息缺失）\n\n"
            f"文本内容：\n{text[:2000]}\n\n"
            f"如果一切正常，回复：OK\n"
            f"如果有问题，按行列出问题描述，每行一个。"
        )
        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}])
            result = result.strip()
            if result.upper().startswith("OK"):
                return []
            return [line.strip() for line in result.split("\n") if line.strip()]
        except Exception:
            return []
