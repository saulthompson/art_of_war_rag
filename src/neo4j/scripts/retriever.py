import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from src.spacy_helper import SpacyHelper
from typing import List

load_dotenv()

class GraphModel:
  def __init__(self):
    self.spacy_helper = SpacyHelper()
    self.queries = []
    self.driver = GraphDatabase.driver(
      os.getenv('NEO4J_URI'),
      auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )

  def build_query(self, entities):
    match_clauses = []
    where_clauses = []
    entity_vars = []

    for i, entity in enumerate(entities):
        var = f"e{i}"
        entity_vars.append(var)
        match_clauses.append(f"MATCH ({var}:{entity["label"]} {{name: $name{i}}})-[:MENTIONED_IN]->(c:CHUNK)")

    query = " \n".join(match_clauses) + " \nRETURN DISTINCT c.content AS content"
    params = {f"name{i}": e["text"] for i, e in enumerate(entities)}
    
    self.queries.append((query, params))


  def execute_newest_query(self) -> List[str]:
    [query, params] = self.queries[-1]

    print('query here', query)
    try:
      with self.driver.session() as session:
        result = list(session.run(query, params))
        chunks = [record["content"] for record in result]
        return chunks
    except Exception as e:
      print("Neo4j query failed:", e)
      return []
  
  def run(self, user_message):
      named_entities = self.spacy_helper.parse_user_query_for_entities(user_message)
      self.build_query(named_entities)
      return self.execute_newest_query()
