import os
import json
from src.embeddings_generator import Generator
from src.chunker import Chunker

db_setup_helper = DB_setup_helper()
db_setup_helper.create_table()

# load chunks from file-system if they already exist, otherwise generate new chunks
if os.path.exists('assets/chunks.json'):
  with open('assets/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks from JSON file.")
else:
  with open('assets/art_of_war_for_rag.txt', encoding='iso-8859-1') as f:
    raw_book = f.read()
  chunker = Chunker(raw_book)
  chunks = chunker.run()
  print('created new chunks')

class DB_setup_helper:
    def __init__(self):
        self.insert_chunk_query = """
        INSERT INTO art_of_war_book_english (chunk, chapter, embedding)
        VALUES (%s, %s, %s)
        """
    
    def create_table(self):
        try:
            self.conn = db_pool.getconn()
            self.cur = self.conn.cursor()

            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.cur.execute(
                """
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
        

    def insert_chunks_to_db(self, generator, batch_size=100):
        try:
            self.conn = db_pool.getconn()
            self.cur = self.conn.cursor()
            batch = []

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



gen = Generator(chunks)
db_setup_helper.insert_chunks_to_db(gen, batch_size=100)

