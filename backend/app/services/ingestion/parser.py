import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from pypdf import PdfReader

class ParsedDocument:
    def __init__(
        self,
        title: str,
        content: str,
        file_type: str,
        content_hash: str,
        byte_size: int,
        tags: List[str],
        links: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.title = title
        self.content = content
        self.file_type = file_type
        self.content_hash = content_hash
        self.byte_size = byte_size
        self.tags = tags
        self.links = links
        self.metadata = metadata or {}

class DocumentParser:
    """Parses various document formats into standardized ParsedDocument instances."""

    @staticmethod
    def compute_sha256(raw_bytes: bytes) -> str:
        return hashlib.sha256(raw_bytes).hexdigest()

    @classmethod
    def parse_bytes(cls, filename: str, raw_bytes: bytes) -> ParsedDocument:
        ext = Path(filename).suffix.lower()
        content_hash = cls.compute_sha256(raw_bytes)
        byte_size = len(raw_bytes)

        if ext in [".md", ".markdown"]:
            return cls._parse_markdown(filename, raw_bytes.decode("utf-8", errors="replace"), content_hash, byte_size)
        elif ext == ".pdf":
            return cls._parse_pdf(filename, raw_bytes, content_hash, byte_size)
        elif ext in [".py", ".go", ".java", ".c", ".cpp", ".js", ".ts", ".sh", ".rs"]:
            return cls._parse_code(filename, raw_bytes.decode("utf-8", errors="replace"), ext, content_hash, byte_size)
        else:
            # Fallback to plain text
            text = raw_bytes.decode("utf-8", errors="replace")
            title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
            tags = cls._extract_tags(text)
            links = cls._extract_links(text)
            return ParsedDocument(
                title=title,
                content=text,
                file_type="text",
                content_hash=content_hash,
                byte_size=byte_size,
                tags=tags,
                links=links
            )

    @classmethod
    def _parse_markdown(cls, filename: str, text: str, content_hash: str, byte_size: int) -> ParsedDocument:
        # Extract title from frontmatter or first # H1
        title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        metadata: Dict[str, Any] = {}

        # Check for YAML frontmatter
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
        content_body = text
        if fm_match:
            frontmatter_raw = fm_match.group(1)
            content_body = fm_match.group(2)
            for line in frontmatter_raw.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip().lower()
                    val = val.strip().strip('"').strip("'")
                    metadata[key] = val
            if "title" in metadata:
                title = metadata["title"]

        # If title not found in frontmatter, find first markdown H1
        if not metadata.get("title"):
            h1_match = re.search(r"^#\s+(.+)$", content_body, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1).strip()

        # Extract tags (#tag or frontmatter tags)
        tags = cls._extract_tags(content_body)
        if "tags" in metadata:
            raw_fm_tags = metadata["tags"]
            if isinstance(raw_fm_tags, str):
                tags.extend([t.strip().lstrip("#") for t in raw_fm_tags.replace("[", "").replace("]", "").split(",") if t.strip()])

        # Extract links (e.g. [[ObsidianLink]] or [Markdown](link))
        links = cls._extract_links(content_body)

        return ParsedDocument(
            title=title,
            content=content_body.strip(),
            file_type="markdown",
            content_hash=content_hash,
            byte_size=byte_size,
            tags=list(set(tags)),
            links=list(set(links)),
            metadata=metadata
        )

    @classmethod
    def _parse_pdf(cls, filename: str, raw_bytes: bytes, content_hash: str, byte_size: int) -> ParsedDocument:
        import io
        reader = PdfReader(io.BytesIO(raw_bytes))
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(f"--- [Page {i+1}] ---\n{page_text}")

        full_content = "\n\n".join(extracted_pages)
        title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title

        tags = cls._extract_tags(full_content)
        links = cls._extract_links(full_content)

        return ParsedDocument(
            title=title,
            content=full_content.strip(),
            file_type="pdf",
            content_hash=content_hash,
            byte_size=byte_size,
            tags=list(set(tags)),
            links=list(set(links)),
            metadata={"num_pages": len(reader.pages)}
        )

    @classmethod
    def _parse_code(cls, filename: str, code: str, ext: str, content_hash: str, byte_size: int) -> ParsedDocument:
        title = filename
        tags = [ext.lstrip("."), "code"]
        links = cls._extract_links(code)
        return ParsedDocument(
            title=title,
            content=code,
            file_type="code",
            content_hash=content_hash,
            byte_size=byte_size,
            tags=tags,
            links=links,
            metadata={"language": ext.lstrip(".")}
        )

    @staticmethod
    def _extract_tags(text: str) -> List[str]:
        # Match hashtags like #distributed-systems, #raft, #python
        # Avoid matching headers like # Header
        raw_tags = re.findall(r"(?:^|\s)#([a-zA-Z0-9_\-]+)", text)
        # Filter out purely numeric or markdown header artifacts
        tags = [t.lower() for t in raw_tags if not t.isdigit() and len(t) > 1]
        return list(set(tags))

    @staticmethod
    def _extract_links(text: str) -> List[str]:
        # Match [[WikiLink]] style or markdown [Anchor](target.md)
        wiki_links = re.findall(r"\[\[(.*?)\]\]", text)
        cleaned_wiki = [w.split("|")[0].strip() for w in wiki_links]
        
        md_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
        target_links = [target.strip() for _, target in md_links if not target.startswith("http")]

        return list(set(cleaned_wiki + target_links))
