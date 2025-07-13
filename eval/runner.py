import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client
from retriever import Retriever
from generator import QueryMachine
from evaluators import semantic_similarity, exact_match

load_dotenv()

client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

retriever = Retriever()
generator = QueryMachine()

def run_eval():
    with open('eval/data/queries.jsonl') as f:
        queries = [json.loads(line) for line in f]

    results = []

    for q in queries:
        question = q['question']
        gold = q.get('gold_answer')

        run = client.create_run(
            name="rag_eval",
            inputs={"question": question},
            run_type="chain"
        )

        try:
            retrieved_chunks = retriever.find_similar(question, limit=6)
            run.log_output({"retrieved_chunks": retrieved_chunks})

            answer = query_machine.get_answer(question, retrieved_chunks)
            run.log_output({"answer": answer})

            # Apply evaluators
            sim_score = semantic_similarity(answer, gold) if gold else None
            em_score = exact_match(answer, gold) if gold else None

            run.log_metric("semantic_similarity", sim_score)
            run.log_metric("exact_match", em_score)

            results.append({
                "question": question,
                "gold_answer": gold,
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "semantic_similarity": sim_score,
                "exact_match": em_score
            })

            run.end()

        except Exception as e:
            run.end(error=str(e))
            print(f"Error: {e}")

    # Save local copy
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M')
    os.makedirs("eval/results", exist_ok=True)
    with open(f'eval/results/run_{run_id}.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"✅ Saved {len(results)} results to eval/results/run_{run_id}.jsonl")

if __name__ == "__main__":
    run_eval()
