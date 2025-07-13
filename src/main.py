import os
import json
from setup_db import DB_setup_helper
from embeddings_generator import Generator
from chunker import Chunker

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


gen = Generator(chunks)
db_setup_helper.insert_chunks_to_db(gen, batch_size=100)
