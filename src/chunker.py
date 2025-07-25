import os
import re
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

class Chunker:
    """
    A class to process and semantically chunk a book into smaller parts.

    Attributes:
        raw_book (str): The raw text of the book.
        breakpoint_threshold_type (str): The threshold type for chunking (e.g. "percentile").
        breakpoint_threshold_amount (float): The numerical threshold used in chunking.
        min_chunk_size (int): Minimum character length for a chunk.
    """

    def __init__(
        self,
        book: str,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 50.0,
        min_chunk_size: int = 200,
    ):
        """
        Initialize the Chunker with book text and chunking parameters.
        
        Args:
            book (str): The full raw text of the book.
            breakpoint_threshold_type (str): Method used to determine chunk breakpoints.
            breakpoint_threshold_amount (float): Amount used in threshold calculation.
            min_chunk_size (int): Minimum length of each chunk in characters.
        """
        self.raw_book = book
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size

    def clean_book_file(self) -> str:
        """
        Clean up malformed or extraneous characters from the raw book text.

        Returns:
            str: The cleaned book text.
        """
        book = self.raw_book.replace('¡¦', "'").replace('¡¥', "'").replace('¡X', '-') \
               .replace('¡K', '...').replace('¡¨', '"').replace('¡©', '"')
        book = re.sub(r'\n{2,}', '\n\n', book)  # Collapse multiple newlines
        book = re.sub(r'[ \t]+', ' ', book)     # Collapse repeated spaces/tabs
        book = re.sub(r'^Appendix.*$', '', book, flags=re.MULTILINE)  # Remove appendices
        book = re.sub(r'\[[^\]]*\]', '', book)  # Remove content in brackets
        return book.strip()

    def split_by_chapters(self, book_text: str) -> List[Dict[str, str]]:
        """
        Split the book text into chapters using regex pattern matching.

        Args:
            book_text (str): The cleaned text of the book.

        Returns:
            List[Dict[str, str]]: A list of chapter dictionaries with 'chapter' and 'content' keys.
        """
        pattern = re.compile(r'(?=^Chapter\s+\w+)', re.MULTILINE | re.IGNORECASE)
        sections = pattern.split(book_text)
        if sections and sections[0].strip() == '':
            sections = sections[1:]

        chapters = []
        for section in sections:
            lines = section.strip().splitlines()
            title = lines[0].strip() if lines else "Unknown"
            content = "\n".join(lines[1:]).strip()
            chapters.append({
                'chapter': title,
                'content': content
            })
        return chapters

    def semantic_chunk(self, chapters: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Perform semantic chunking on chapter content using OpenAI embeddings.

        Args:
            chapters (List[Dict[str, str]]): List of chapter dictionaries.

        Returns:
            List[Dict[str, str]]: List of chunk dictionaries with 'chapter' and 'content' keys.
        """
        text_splitter = SemanticChunker(
            OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY')),
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            min_chunk_size=self.min_chunk_size,
        )

        chunks = []
        for chapter in chapters:
            docs = text_splitter.create_documents([chapter['content']])
            for doc in docs:
                chunks.append({
                    'chapter': chapter['chapter'],
                    'content': doc.page_content
                })

        print('**************** Chunk data ****************')
        print(f"\nNumber of chunks: {len(chunks)}")
        average_chunk_size = sum(len(chunk['content']) for chunk in chunks) / len(chunks)
        print(f"\nAverage chunk size: {average_chunk_size}")

        return chunks

    def run(self) -> List[Dict[str, str]]:
        """
        Execute the full chunking pipeline:
        1. Clean raw book text.
        2. Split into chapters.
        3. Perform semantic chunking.
        4. Save to 'assets/chunks.json'.

        Returns:
            List[Dict[str, str]]: Final list of chunks.
        """
        book_text = self.clean_book_file()
        chapters = self.split_by_chapters(book_text)
        chunks = self.semantic_chunk(chapters)

        os.makedirs("assets", exist_ok=True)
        with open('assets/chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return chunks
