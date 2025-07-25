import os
from typing import Optional, List, Generator, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Generator:
    """
    A class to generate embeddings for text chunks using OpenAI's embedding models.
    
    Attributes:
        chunks (Optional[List[str]]): A list of text chunks to embed.
        embedding_model (str): The name of the OpenAI embedding model to use.
        openai_client (OpenAI): OpenAI API client instance.
    """

    def __init__(self, chunks: Optional[List[str]] = None, embedding_model: str = 'text-embedding-3-small'):
        """
        Initialize the Generator with optional text chunks and an embedding model.
        
        Args:
            chunks (Optional[List[str]]): A list of strings to be embedded.
            embedding_model (str): The OpenAI model to use for embeddings.
        """
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chunks = chunks
        self.embedding_model = embedding_model

    def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding for a single text input.
        
        Args:
            text (str): The input text to embed.
        
        Returns:
            Optional[List[float]]: The embedding vector, or None if an error occurred.
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print('Error generating embedding:', e)
            return None

    def generate_chunk_embeddings(self) -> Generator[Tuple[str, Optional[List[float]]], None, None]:
        """
        Generate embeddings for all chunks stored in the instance.

        Yields:
            Tuple[str, Optional[List[float]]]: Each chunk and its corresponding embedding.
        """
        try:
            for chunk in self.chunks:
                embedding = self.generate_single_embedding(chunk)
                yield (chunk, embedding)
        except Exception as e:
            print("Error generating embeddings:", e)
