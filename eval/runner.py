import os
import json
from langsmith import Client
from eval.evaluators import correctness, groundedness, relevance, retrieval_relevance
from eval.pipeline import load_target, load_batch_metadata

DATASET_NAME = "art_of_war"

def main():
    langsmith_client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
    target = load_target()
    batch_metadata = load_batch_metadata()

    existing = langsmith_client.list_datasets(dataset_name=DATASET_NAME)
    if existing:
        dataset = list(existing)[0]
        print(f"✅ Using existing dataset: {dataset.name}")
    else:
        dataset = langsmith_client.create_dataset(dataset_name=DATASET_NAME)
        print(f"📦 Created new dataset: {dataset.id}")

        with open('eval/data/queries.jsonl') as f:
            queries = [json.loads(line) for line in f]
            langsmith_client.create_examples(dataset_id=dataset.id, examples=queries)

    experiment_results = langsmith_client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix=f"rag-{batch_metadata['version']}",
        metadata=batch_metadata
    )

    output_file = 'eval/data/results.jsonl'
    with open(output_file, 'w') as f:
        for result in experiment_results:
            json.dump(result, f)
            f.write('\n')

    print(f"✅ Evaluation complete. Results saved to: {output_file}")

if __name__ == "__main__":
    main()
