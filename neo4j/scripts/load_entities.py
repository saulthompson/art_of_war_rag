import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
ENTITIES = ["PERSON", "ORG", "GPE", "NOUN", "DATE", "WORK_OF_ART", "NORP", "EVENT", "LOC", "LAW", "FAC", "LANGUAGE"]

df = pd.read_csv("assets/entities.csv")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.getenv('NEO4J_PASSWORD')))

def load_entities(tx, row):
    label = row["label"]
    if label not in ENTITIES:
        raise ValueError(f"unexpected entity label: {label}")

    query = f"""
        MERGE (c:Chunk {{id: $chunk_id}})
        MERGE (e:Entity:{label} {{name: $entity_text, type: $entity_type}})
        MERGE (e)-[:MENTIONED_IN]->(c)
    """

    tx.run(
        query,
        chunk_id=row["chunk_id"],
        entity_text=row["entity_text"],
        entity_type=label
    )

with driver.session() as session:
    for _, row in df.iterrows():
        print('row:', row)
        session.execute_write(load_entities, row)
