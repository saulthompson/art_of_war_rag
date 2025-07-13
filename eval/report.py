import json
import pandas as pd
import glob

def load_latest_results():
    files = sorted(glob.glob('eval/results/*.jsonl'))
    if not files:
        print("No results found.")
        return None
    latest = files[-1]
    print(f"Loading {latest}")
    with open(latest) as f:
        return pd.DataFrame([json.loads(line) for line in f])

df = load_latest_results()
if df is not None:
    print(df[['question', 'semantic_similarity', 'exact_match']])
    print("\nAverage similarity:", df['semantic_similarity'].mean())
    print("Exact match rate:", df['exact_match'].mean())

    import matplotlib.pyplot as plt
    df['semantic_similarity'].hist(bins=10)
    plt.title("Semantic similarity distribution")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.show()
