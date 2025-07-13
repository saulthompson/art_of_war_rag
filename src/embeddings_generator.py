import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self, chunks):
      self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
      self.chunks = chunks

    def generate_embeddings(self):
      """yield embeddings for each chunk"""
      try:
          for chunk in self.chunks:
              # Create embedding with OpenAI
              response = self.openai_client.embeddings.create(
                  model="text-embedding-3-small",
                  input=chunk['content']
              )
              embedding = response.data[0].embedding
              yield (chunk, embedding) 

      except Exception as e:
          print("Error generating embeddings:", e)
