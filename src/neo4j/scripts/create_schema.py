import os
from typing import List
from neo4j import GraphDatabase, Driver, Session
from dotenv import load_dotenv

load_dotenv()


def get_driver() -> Driver:
    """
    Initialize and return a Neo4j database driver using credentials from environment variables.
    
    Returns:
        Driver: A Neo4j driver instance connected to the database.
    """
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))


def get_schema_queries() -> List[str]:
    """
    Return a list of Cypher schema constraint queries to ensure uniqueness of various node types.
    
    Returns:
        List[str]: A list of Cypher CREATE CONSTRAINT queries.
    """
    return [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:CHUNK) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:PERSON) REQUIRE p.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (o:ORG) REQUIRE o.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:GPE) REQUIRE g.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:NOUN) REQUIRE n.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d: DATE) REQUIRE d.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (w: WORK_OF_ART) REQUIRE w.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (no: NORP) REQUIRE no.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e: EVENT) REQUIRE e.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (l: LOC) REQUIRE l.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (la: LAW) REQUIRE la.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (en: ENTITY) REQUIRE en.text IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f: FAC) REQUIRE f.text IS UNIQUE"
    ]


def apply_schema_constraints(driver: Driver, queries: List[str]) -> None:
    """
    Apply each Cypher schema constraint query to the Neo4j database.
    
    Args:
        driver (Driver): The Neo4j driver instance.
        queries (List[str]): A list of Cypher schema constraint queries.
    """
    with driver.session() as session:
        for query in queries:
            session.run(query)
            print(f"✅ Ran: {query}")


if __name__ == "__main__":
    driver = get_driver()
    schema_queries = get_schema_queries()
    apply_schema_constraints(driver, schema_queries)
