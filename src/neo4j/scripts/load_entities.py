import os
import pandas as pd
from typing import Dict, Any
from neo4j import GraphDatabase, Transaction, Driver
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the list of valid entity labels
ENTITIES = [
    "PERSON", "ORG", "GPE", "NOUN", "DATE", "WORK_OF_ART",
    "NORP", "EVENT", "LOC", "LAW", "FAC", "LANGUAGE"
]

# Load CSV file containing entity data
df: pd.DataFrame = pd.read_csv("assets/entities.csv")

# Initialize the Neo4j driver
driver: Driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", os.getenv('NEO4J_PASSWORD'))
)

def load_entities(tx: Transaction, row: Dict[str, Any]) -> None:
    """
    Write a row of entity data into the Neo4j database using MERGE statements.
    
    Ensures the entity is associated with a chunk and labeled correctly.

    Args:
        tx (Transaction): The Neo4j transaction object.
        row (Dict[str, Any]): A row from the CSV containing entity data.

    Raises:
        ValueError: If the entity label is not in the expected list.
    """
    label: str = row["label"]
    if label not in ENTITIES:
        raise ValueError(f"unexpected entity label: {label}")

    query: str = f"""
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

# Apply the entity creation for each row in the dataframe
with driver.session() as session:
    for _, row in df.iterrows():
