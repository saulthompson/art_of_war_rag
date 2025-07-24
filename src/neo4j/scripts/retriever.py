import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from src.spacy_helper import SpacyHelper

load_dotenv()

class GraphModel:
  def __init__(self, max_chunks=25):
    self.spacy_helper = SpacyHelper()
    self.queries = []
    self.MAX_CHUNKS = max_chunks
    self.driver = GraphDatabase.driver(
      os.getenv('NEO4J_URI'),
      auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )

  def build_query(self, entities):
    match_clauses = []
    where_clauses = []
    entity_vars = []

    for i, entity in enumerate(entities):
        print('query builder entities:', entities)
        var = f"e{i}"
        entity_vars.append(var)
        match_clauses.append(f"MATCH ({var}:{entity["label"]} {{name: $name{i}}})-[:MENTIONED_IN]->(c:CHUNK)")

    query = " \n".join(match_clauses) + " \nRETURN DISTINCT c.content AS content"
    params = {}
    for i, e in enumerate(entities):
        params[f"name{i}"] = e["text"]
        params[f"label{i}"] = e["label"]
        
    self.queries.append((query, params))


  def execute_newest_query(self):
    [query, params] = self.queries[-1]

    try:
      with self.driver.session() as session:
        result = list(session.run(query, params))
        chunks = [record["content"] for record in result]

        # if multiple entities are not mentioned in any one chunk, get chunks for each entity separately
        if len(chunks) == 0 and len(params.keys()) >= 3:
          print('no combined chunks')
          for key, value in list(params.items()):
            print('key:', key, "value:", value)
            if key.startswith("name"):
              label_key = key.replace("name", "label")
              label_value = params.get(label_key)

              if label_value is None:
                  print(f"Could not find label for {key}")
                  continue

              self.build_query([{"text": value, "label": label_value}])
              print('queries:', self.queries[-1])
              chunks += self.execute_newest_query()

        return chunks
    except Exception as e:
      print("Neo4j query failed:", e)
      return []
  
  def run(self, user_message):
      named_entities = self.spacy_helper.parse_user_query_for_entities(user_message)
      self.build_query(named_entities)
      chunks = self.execute_newest_query()
      if len(chunks) > self.MAX_CHUNKS:
        chunks = chunks[:self.MAX_CHUNKS]
      
      return chunks
