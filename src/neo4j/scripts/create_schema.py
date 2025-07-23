import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.getenv('NEO4J_PASSWORD')))

schema_queries = [
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

with driver.session() as session:
    for query in schema_queries:
        session.run(query)
        print(f"✅ Ran: {query}")
