from src.db_pool import db_pool

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

