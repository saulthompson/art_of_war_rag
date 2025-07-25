import os
import json
from typing import List, Dict, Any

from src.embeddings_generator import Generator
from src.chunker import Chunker

# --- Database setup helper class ---

class DB_setup_helper:
    """
    Helper class to create and populate the database table for storing book chunks and embeddings.
    """

    def __init__(self) -> None:
        """
        Initialize the insert query for the table.
        """
        self.insert_chunk_query: str = """
        INSERT INTO art_of_war_book_english (chunk, chapter, embedding)
        VALUES (%s, %s, %s)
        """

    def create_table(self) -> None:
        """
        Create the required table and vector extension in the PostgreSQL database if they don't exist.
        """
        try:
            self.conn = db_pool.getconn()
            self.cur = self.conn.cursor()

            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS art_of_war_book_english (
                    id SERIAL PRIMARY KEY,
                    chapter TEXT NOT NULL,
                    chunk TEXT NOT NULL,
                    embedding vector(1536)
                );
            """)
            self.conn.commit()
            print("Database setup complete!")
        except Exception as e:
            print("Error during setup:", e)
        finally:
            self.cur.close()
            self.conn.close()

    def insert_chunks_to_db(self, generator: Generator, batch_size: int = 100) -> None:
        """
        Insert generated chunks and their embeddings into the database in batches.

        Args:
            generator (Generator): A Generator instance used to produce embeddings.
            batch_size (int): Number of records to insert per batch.
        """
        try:
            self.conn = db_pool.getconn()
            self.cur = self.conn.cursor()
            batch: List[tuple] = []

            for chunk, embedding in generator.generate_chunk_embeddings():
                batch.append((chunk['content'], chunk['chapter'], embedding))

                if len(batch) > batch_size:
                    self.cur.executemany(self.insert_chunk_query, batch)
                    self.conn.commit()
                    print(f"Inserted batch of {len(batch)}")
                    batch = []

            if batch:
                self.cur.executemany(self.insert_chunk_query, batch)
                self.conn.commit()
                print(f"Inserted batch of {len(batch)}")

        except Exception as e:
            print("error while storing chunks in db:", e)
        finally:
            self.cur.close()
            self.conn.close()


# --- Load or generate chunks ---

def load_or_create_chunks(filepath: str, raw_text_path: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSON file if it exists, otherwise generate them from raw text.

    Args:
        filepath (str): Path to the JSON file for precomputed chunks.
        raw_text_path (str): Path to the raw book text file.

    Returns:
        List[Dict[str, Any]]: A list of chunk dictionaries.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            print(f"✅ Loaded {len(chunks)} chunks from JSON file.")
            return chunks
    else:
        with open(raw_text_path, encoding='iso-8859-1') as f:
            raw_book = f.read()
        chunker = Chunker(raw_book)
        chunks = chunker.run()
        print('✅ Created new chunks.')
        return chunks


# --- Main execution ---

if __name__ == "__main__":
    db_setup_helper = DB_setup_helper()
    db_setup_helper.create_table()

    chunks = load_or_create_chunks(
        filepath='assets/chunks.json',
        raw_text_path='assets/art_of_war_for_rag.txt'
    )

    gen = Generator(chunks)
    db_setup_helper.insert_chunks_to_db(gen, batch_size=100)
