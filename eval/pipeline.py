import os
import subprocess
import yaml
from dotenv import load_dotenv
from langsmith import wrappers
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

def load_target():
    # Setup retriever
    conn_str = (
        f"postgresql+psycopg://{os.environ['PGVECTOR_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['PGVECTOR_HOST']}:{os.environ['PGVECTOR_PORT']}/{os.environ['PGVECTOR_DB']}"
    )

    vectorstore = PGVector(
        embeddings=OpenAIEmbeddings(model='text-embedding-3-small'),
        collection_name="art_of_war_book_english",
        connection=conn_str,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # Setup LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    def rag_bot(question: str) -> dict:
        docs = retriever.invoke(question)
        docs_string = "".join(doc.page_content for doc in docs)

        instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.
        Use the following source documents to answer the user's questions.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.

        Documents:
        {docs_string}"""
        
        ai_msg = llm.invoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ],
        )
        return {"answer": ai_msg.content, "documents": docs}

    def target(inputs):
        return rag_bot(inputs["question"])

    return target

def load_batch_metadata():
    with open("eval/batch_config.yaml") as f:
        batch_config = yaml.safe_load(f)
    commit_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    batch_config["git_sha"] = commit_sha
    return batch_config
