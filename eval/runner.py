import os
import json
import yaml
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client, wrappers, traceable
from langchain_postgres import PGVector
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.retriever import Retriever
from src.query import QueryMachine
from src.embeddings_generator import Generator
from openai import OpenAI
from eval.evaluators import correctness, groundedness, relevance, retrieval_relevance


load_dotenv()

generator = Generator()
query_machine = QueryMachine()

langsmith_client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
openai_client = wrappers.wrap_openai(OpenAI())

DATASET_NAME = "art_of_war"

existing = list(langsmith_client.list_datasets(dataset_name=DATASET_NAME))
print('existing:', dir(existing))

if existing:
    dataset = existing[0]
    print(f" Using existing dataset: {dataset.name}")
else:
    dataset = langsmith_client.create_dataset(
        dataset_name=DATASET_NAME
    )
    print(f" Created new dataset: {dataset.id}")


with open('eval/data/queries.jsonl') as f:
    queries_and_reference_answers = [json.loads(line) for line in f]
    langsmith_client.create_examples(dataset_id=dataset.id, examples=queries_and_reference_answers)

conn_str = (
    f"postgresql+psycopg://{os.environ['PGVECTOR_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['PGVECTOR_HOST']}:{os.environ['PGVECTOR_PORT']}/{os.environ['PGVECTOR_DB']}"
)

print(conn_str)

vectorstore = PGVector(
    embeddings=OpenAIEmbeddings(model='text-embedding-3-small'),
    collection_name="art_of_war_book_english",
    connection=conn_str,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


@traceable()
def rag_bot(question: str) -> dict:
    # langchain Retriever will be automatically traced
    docs = retriever.invoke(question)
    
    print('docs: ', docs)
    docs_string = "".join(doc.page_content for doc in docs)
    instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.       Use the following source documents to answer the user's questions.       If you don't know the answer, just say that you don't know.       Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}"""
    # langchain ChatModel will be automatically traced
    ai_msg = llm.invoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    )

    return {"answer": ai_msg.content, "documents": docs}

def target(inputs):
    return rag_bot(inputs["question"])


with open("eval/batch_config.yaml") as f:
    batch_config = yaml.safe_load(f)

commit_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
batch_config["git_sha"] = commit_sha

experiment_results = langsmith_client.evaluate(
    target,
    data=DATASET_NAME,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix=f"rag-{batch_config['version']}",
    metadata=batch_config
)

with open('eval/data/results.jsonl', 'w') as f:
    for result in experiment_results:
        json.dump(result, f)
        f.wirte('\n')


