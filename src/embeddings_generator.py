import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self, chunks, embedding_model='text-embedding-3-small'):
      self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
      self.chunks = chunks
      self.embedding_model = embedding_model

    def generate_single_embedding(self, text):
      try:
        response = self.openai_client.embeddings.create(
          model=self.embedding_model,
          input=text
        )

        return response.data[0].embedding

      except Exception as e:
        print('Error generating embedding:', e)

    def generate_chunk_embeddings(self):
      """yield embeddings for each chunk"""
      try:
          for chunk in self.chunks:
              embedding = self.generate_single_embedding(chunk)
              yield (chunk, embedding) 

      except Exception as e:
          print("Error generating embeddings:", e)
