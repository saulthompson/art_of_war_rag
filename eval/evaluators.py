from openai import OpenAI
import numpy as np
from embeddings_generator import Generator
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()
generator = Generator()

def semantic_similarity(a, b):
    """Compute cosine similarity between embeddings of a and b."""
    try:
        emb_a = client.embeddings.create(model=generator.embedding_model, input=a).data[0].embedding
        emb_b = client.embeddings.create(model=generator.embedding_model, input=b).data[0].embedding
        return np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    except Exception as e:
        print("Error computing similarity:", e)
        return None

def exact_match(a, b):
    """Naive baseline: is gold answer substring in answer?"""
    return int(b in a) if b else None
