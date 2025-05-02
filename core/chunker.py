import re
from typing import List
from tqdm import tqdm
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter

class RegexSectionChunker:
    def __init__(self, min_length: int = 100):
        self.section_pattern = re.compile(r"(SECTION\s+\d+[A-Z]?\s*[\.-]?)", re.IGNORECASE)
        self.min_length = min_length

    def chunk(self, text: str) -> List[str]:
        sections = self.section_pattern.split(text)
        chunks = []

        # Combine section headers and bodies
        for i in range(1, len(sections), 2):
            title = sections[i].strip()
            content = sections[i + 1].strip() if (i + 1) < len(sections) else ""
            full = f"{title}\n{content}".strip()
            if len(full) >= self.min_length:
                chunks.append(full)

        return chunks

class RecursiveChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

class ChapterAwareChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Initialize the LangChain splitter with optimal separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],  # Ordered by priority
            length_function=len,
            is_separator_regex=False
        )

    def chunk(self, text: str) -> List[str]:
        """Split text into chunks while preserving chapter boundaries."""
        print("\nChunking Process:")
        print("1. Identifying chapters...")
        
        # First split by chapters
        chapter_pattern = r"CHAPTER\s+[IVXLCDM]+"
        chapter_splits = re.split(f"({chapter_pattern})", text)
        
        chunks = []
        total_chapters = len(chapter_splits) // 2 + (1 if len(chapter_splits) % 2 else 0)
        print(f"✓ Found {total_chapters} chapters")
        
        print("\n2. Processing chapters into chunks...")
        # Create a single progress bar for all processing
        with tqdm(total=total_chapters, desc="Processing chapters", file=sys.stdout) as pbar:
            for i in range(0, len(chapter_splits), 2):
                if i + 1 < len(chapter_splits):
                    # This is a chapter header
                    current_chapter = chapter_splits[i + 1].strip()
                    chapter_text = chapter_splits[i + 2] if i + 2 < len(chapter_splits) else ""
                else:
                    # This is the text after the last chapter header
                    current_chapter = "Unknown"
                    chapter_text = chapter_splits[i]
                
                if not chapter_text.strip():
                    pbar.update(1)
                    continue
                
                # Use LangChain's splitter to process the chapter
                chapter_chunks = self.splitter.split_text(chapter_text)
                chunks.extend(chapter_chunks)
                
                # Update progress after processing each chapter
                pbar.update(1)
                print(f"  ✓ Chapter {current_chapter}: {len(chapter_chunks)} chunks")
        
        print(f"\n✓ Created {len(chunks)} total chunks")
        print(f"✓ Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.0f} characters")
        
        return chunks
