import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class ChildChunkDTO:
    chunk_index: int
    content: str
    contextual_content: str
    token_count: int

@dataclass
class ParentChunkDTO:
    chunk_index: int
    header_path: str
    content: str
    token_count: int
    children: List[ChildChunkDTO]

class HierarchicalChunker:
    """
    Splits documents into semantic Parent Sections and granular Child Chunks.
    Child chunks are enriched with contextual document/section breadcrumbs
    for precise vector similarity while preserving broad context for LLM generation.
    """

    def __init__(
        self,
        parent_max_chars: int = 1500,
        child_max_chars: int = 350,
        child_overlap: int = 60
    ):
        self.parent_max_chars = parent_max_chars
        self.child_max_chars = child_max_chars
        self.child_overlap = child_overlap

    def chunk_document(self, title: str, text: str) -> List[ParentChunkDTO]:
        if not text.strip():
            return []

        # Step 1: Split into parent sections by markdown headers or large paragraph blocks
        parent_sections = self._split_into_parent_sections(text)
        
        result_parents: List[ParentChunkDTO] = []
        for p_idx, (header_path, p_content) in enumerate(parent_sections):
            p_content = p_content.strip()
            if not p_content:
                continue

            p_token_count = self._approx_token_count(p_content)

            # Step 2: Subdivide parent section into child chunks
            children = self._split_parent_into_children(
                doc_title=title,
                header_path=header_path,
                parent_content=p_content
            )

            result_parents.append(
                ParentChunkDTO(
                    chunk_index=p_idx,
                    header_path=header_path,
                    content=p_content,
                    token_count=p_token_count,
                    children=children
                )
            )

        return result_parents

    def _split_into_parent_sections(self, text: str) -> List[Tuple[str, str]]:
        """Splits markdown/text by headers (#, ##, ###) while maintaining section hierarchy."""
        lines = text.splitlines(keepends=True)
        sections: List[Tuple[str, str]] = []
        
        current_header = "Introduction"
        current_buffer: List[str] = []

        header_regex = re.compile(r"^(#{1,4})\s+(.+)$")

        for line in lines:
            h_match = header_regex.match(line.strip())
            if h_match:
                # Flush previous section if it has content
                if current_buffer:
                    sections.append((current_header, "".join(current_buffer)))
                    current_buffer = []
                current_header = h_match.group(2).strip()
            current_buffer.append(line)

        if current_buffer:
            sections.append((current_header, "".join(current_buffer)))

        # If sections are too large, split by double newlines into parent blocks
        refined_sections: List[Tuple[str, str]] = []
        for header, content in sections:
            if len(content) <= self.parent_max_chars:
                refined_sections.append((header, content))
            else:
                paragraphs = content.split("\n\n")
                sub_buf: List[str] = []
                sub_len = 0
                sub_idx = 1
                for p in paragraphs:
                    if sub_len + len(p) > self.parent_max_chars and sub_buf:
                        refined_sections.append((f"{header} (Part {sub_idx})", "\n\n".join(sub_buf)))
                        sub_buf = []
                        sub_len = 0
                        sub_idx += 1
                    sub_buf.append(p)
                    sub_len += len(p) + 2
                if sub_buf:
                    refined_sections.append((f"{header} (Part {sub_idx})" if sub_idx > 1 else header, "\n\n".join(sub_buf)))

        return refined_sections

    def _split_parent_into_children(
        self,
        doc_title: str,
        header_path: str,
        parent_content: str
    ) -> List[ChildChunkDTO]:
        """
        Splits a parent section into overlapping child windows.
        Prepends contextual metadata to prevent vector loss of context.
        """
        words = parent_content.split()
        if not words:
            return []

        children: List[ChildChunkDTO] = []
        step = max(1, (self.child_max_chars - self.child_overlap) // 6)  # approx ~6 chars per word
        window_size = max(1, self.child_max_chars // 6)

        c_idx = 0
        i = 0
        while i < len(words):
            chunk_words = words[i : i + window_size]
            child_text = " ".join(chunk_words).strip()

            if child_text:
                # Contextual enrichment (Anthropic Contextual Retrieval pattern)
                contextual_header = f"[Document: {doc_title} | Section: {header_path}]\n"
                contextual_content = contextual_header + child_text

                children.append(
                    ChildChunkDTO(
                        chunk_index=c_idx,
                        content=child_text,
                        contextual_content=contextual_content,
                        token_count=self._approx_token_count(child_text)
                    )
                )
                c_idx += 1

            i += step
            if i >= len(words):
                break

        return children

    @staticmethod
    def _approx_token_count(text: str) -> int:
        return max(1, len(text) // 4)
