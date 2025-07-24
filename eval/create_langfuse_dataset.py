import json
from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

langfuse.create_dataset(
    name="art_of_war",
    metadata={
        "author": "Saul",
        "date": "2025-07-23",
        "type": "benchmark",
        "version": "1.1"
    }
)

with open('eval/data/queries.jsonl', 'r') as f:
  data = [json.loads(line) for line in f]

for query_pair in data:
  langfuse.create_dataset_item(
    dataset_name="art_of_war",
    input = query_pair["inputs"]["question"],
    expected_output = query_pair["outputs"]["answer"]
  )



# langfuse.create_dataset_item(
#     dataset_name="<dataset_name>",
#     # any python object or value, optional
#     input={
#         "text": "hello world"
#     },
#     # any python object or value, optional
#     expected_output={
#         "text": "hello world"
#     },
#     # metadata, optional
#     metadata={
#         "model": "llama3",
#     }
# )