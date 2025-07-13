import os
import re
import json
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

class Chunker: 
    def __init__(
        self,
        book,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=50.0,
        min_chunk_size=200,
    ):
        self.raw_book = book
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size

    def clean_book_file(self):
        # remove common malformatted characters
        book = self.raw_book.replace('¡¦', "'").replace('¡¥', "'").replace('¡X', '-') \
               .replace('¡K', '...').replace('¡¨', '"').replace('¡©', '"')
        book = re.sub(r'\n{2,}', '\n\n', book)  # collapse big blocks of empty lines
        book = re.sub(r'[ \t]+', ' ', book)    # collapse repeated spaces/tabs
        book = re.sub(r'^Appendix.*$', '', book, flags=re.MULTILINE) # remove appendices
        book = re.sub(r'\[[^\]]*\]', '', book)
        book = book.strip() 
        return book

    def split_by_chapters(self, book_text):
        pattern = re.compile(r'(?=^Chapter\s+\w+)', re.MULTILINE | re.IGNORECASE)
        
        sections = pattern.split(book_text)
        if sections[0].strip() == '':
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

    def semantic_chunk(self, chapters):
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

        average_chunk_size = sum([len(chunk) for chunk in chunks]) / len(chunks)
        print(f"\nAverage chunk size: {average_chunk_size}")

        return chunks
    
    def run(self):
        book_text = self.clean_book_file()
        chapters = self.split_by_chapters(book_text)
        chunks = self.semantic_chunk(chapters)

         # write chunks to assets/chunks.json
        with open('assets/chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return chunks
