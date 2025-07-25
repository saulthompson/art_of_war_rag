# Art of War RAG Bot

This project is an experimental Retrieval-Augmented Generation (RAG) application that transforms Chinese author Hua Shan's full-length book commentary on Sun-Tzu's *The Art of War* into an interactive semantic interface. By combining dense vector embeddings with symbolic reasoning via a graph database, it delivers more context-aware and entity-sensitive answers than traditional RAG systems.

---

## 1. 🚀 Getting Started

### 📁 Clone the Repository

```bash
git clone https://github.com/yourusername/art-of-war-rag.git
cd art-of-war-rag
```

### 🐍 Set Up the Python Environment

We recommend using [Poetry](https://python-poetry.org/) for dependency and environment management.

```bash
poetry install
poetry shell
```

If you prefer using `venv`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Set Up the Vector Database (PostgreSQL + pgvector)

Use the included scripts in the repo to:

- Initialize the database and schema
- Load the chunked CSV file (available on request)
- Generate and store vector embeddings

No manual SQL setup is necessary.

### Set Up the Graph Database (Neo4j)

Neo4j is used to store and query named entities and their relationships.

1. Install and launch Neo4j (Desktop or Docker)
2. Use the provided ingestion scripts to:
   - Load entities and relationships into Neo4j
   - Link entities to text chunks

> Datasets (`entities.csv`, `chunks.csv`, etc.) available on request.

### Launch the App with Uvicorn

Once your vector and graph databases are set up, start the app:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

The Gradio interface will be available at:

```
http://localhost:7860
```

---

## 2. Project Concept

Traditional RAG systems work well with semantically similar text retrieval, but they struggle with **named entities**—people, events, and other proper nouns—especially when those entities carry nuanced or niche domain-specific significance.

This project attempts to solve that problem by turning *The Art of War* into an interactive book, blending:

- **Dense vector embeddings** for capturing semantic similarity, and
- **Symbolic knowledge graphs** for structured, relational understanding.

---

## 3. The Hybrid Approach: Graph + Vector RAG

### The Challenge with Proper Nouns

Semantic embeddings encode general meaning, but are not well-suited for precise references to:

- Historical figures (e.g., "Sun Tzu")
- Little-known geographic place names (e.g., "the River Fei")
- Relational questions (e.g., "What did Sun Tzu say about generals?")

Named entities often have similar embeddings, leading to irrelevant results. For example, semantic retrieval struggles to differentiate between Tao Kan (a military leader), and the River Tao.

### The Solution

We combine two retrieval strategies:

1. **Vector Search** via pgvector:

   - Uses cosine similarity to find semantically similar text chunks.
   - Great for general or abstract questions or when entities are not important.

2. **Graph Search** via Neo4j:

   - Uses Cypher queries to locate specific nodes (entities) and their connected chunks.
   - Ideal for proper nouns, related entities, or multi-entity queries.

By combining both types of retrieval, we generate more accurate, grounded responses. We used the Spacy NLP library to extract over 2000 distinct named entities from the source text.

---

## 4. 🧪 Evaluation and Debugging with Langfuse

We use [Langfuse](https://www.langfuse.com/) to trace, monitor, and evaluate the app in real time.

### Built-in Evaluators

Langfuse offers a `contextRelevance` evaluator that scores how well a retrieved context matches a user query.

### The NER Limitation

Just like the model itself, the built-in evaluator struggled with proper nouns and failed to assess entity-based context relevance reliably.

### Custom Evaluator

To address this, we created a custom evaluator inspired by `contextRelevance`, but with an enhanced prompt that explicitly highlights named entities and demands attention to them when scoring relevance.

---

## 5. Advanced Graph Features

### Multi-Entity Handling

When a user query contains multiple named entities:

- **If entities co-occur in the same chunk**: that chunk is prioritized.
- **If entities are in different contexts**: a Cypher query fetches related data through intermediate relationships.

### Category Mapping for Generic Queries

Some user queries don’t mention specific entities but reference abstract categories:

- E.g., "What dynasties are mentioned?", "Who were the main people?", "Which events are described?"

These keywords are mapped to graph labels like `DATE`, `PERSON`, or `EVENT`, enabling the system to return a representative sample of related context. These labels were generated automatically during Spacy's NLP process.

---

## 6. Future Directions

### Graph Expansion

- Add more entity types and relationships (e.g., causal, temporal)
- Link to external graphs like Wikidata for richer context

### Query Understanding

- Enrich or rephrase user prompts with context hints
- Improve NER accuracy using domain-specific training

### Context Reranking

- Combine vector and graph results into a ranked list
- Implement re-ranking using LLM scoring or hybrid heuristics

### Pipeline Reusability

- Abstract the ingestion and retrieval pipeline into a reusable framework
- Apply it to any text corpus or book for general-purpose semantic reading interfaces

---

## 📬 Questions or Requests?

- Want the chunked dataset or graph data?
- Have feedback on improving the hybrid pipeline?

**Reach out or open an issue!**

