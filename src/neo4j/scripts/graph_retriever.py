import os
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from src.spacy_helper import get_spacy_helper

load_dotenv()

class GraphModel:
    """
    GraphModel interacts with a Neo4j graph to retrieve document chunks 
    related to named entities and generic entity categories found in user queries.
    """

    def __init__(self, max_chunks: int = 25):
        """
        Initializes the GraphModel with a spaCy helper and Neo4j driver.

        Args:
            max_chunks (int): Maximum number of text chunks to return.
        """
        self.spacy_helper = get_spacy_helper()
        self.queries: List[Tuple[str, Dict[str, Any]]] = []
        self.MAX_CHUNKS = max_chunks
        self.driver: Driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )

    def build_query(self, entities: List[Dict[str, str]]) -> None:
        """
        Builds and stores a Cypher query for specific named entities.

        Args:
            entities (List[Dict[str, str]]): List of entities with 'text' and 'label' keys.
        """
        if not entities:
            return

        match_clauses = []
        params: Dict[str, str] = {}

        for i, entity in enumerate(entities):
            var = f"e{i}"
            label = entity["label"]
            name = entity["text"]
            match_clauses.append(
                f"MATCH ({var}:{label} {{name: $name{i}}})-[:MENTIONED_IN]->(c:CHUNK)"
            )
            params[f"name{i}"] = name
            params[f"label{i}"] = label

        query = "\n".join(match_clauses) + "\nRETURN DISTINCT c.content AS content"
        self.queries.append((query, params))

    def build_generic_query(self, generics: List[str]) -> int:
        """
        Builds and stores generic entity queries (e.g., for 'event', 'place').

        Args:
            generics (List[str]): List of generic entity types extracted from the user query.

        Returns:
            int: Number of generic queries added.
        """
        label_map = {
            "people": "PERSON",
            "person": "PERSON",
            "who": "PERSON",
            "historical figure": "PERSON",
            "event": "EVENT",
            "events": "EVENT",
            "battle": "EVENT",
            "dynasty": "DATE",
            "when": "DATE",
            "period": "DATE",
            "place": "LOC",
            "location": "LOC"
        }

        queries_added = 0

        for i, word in enumerate(generics):
            label = label_map.get(word.lower())
            if not label:
                continue

            query = f"""
                MATCH (e:{label})-[:MENTIONED_IN]->(c:CHUNK)
                WITH e, c
                ORDER BY e, rand() 
                WITH e, collect(c)[0] AS one_chunk 
                RETURN one_chunk.content AS content
                LIMIT 10
            """
            self.queries.append((query, {}))
            queries_added += 1

        return queries_added

    def execute_newest_query(self) -> List[str]:
        """
        Executes the most recent Cypher query in the queue.

        Returns:
            List[str]: List of chunk contents.
        """
        if not self.queries:
            return []

        query, params = self.queries[-1]

        try:
            with self.driver.session() as session:
                result = list(session.run(query, params))
                chunks = [record["content"] for record in result]

                # Fallback if no combined chunk was found
                if not chunks and len(params.keys()) >= 3:
                    print('no combined chunks')
                    for key, value in list(params.items()):
                        if key.startswith("name"):
                            label_key = key.replace("name", "label")
                            label_value = params.get(label_key)

                            if label_value is None:
                                print(f"Could not find label for {key}")
                                continue

                            self.build_query([{"text": value, "label": label_value}])
                            chunks += self.execute_newest_query()

                return chunks

        except Exception as e:
            print("Neo4j query failed:", e)
            return []

    def run(self, user_message: str) -> List[str]:
        """
        Main entry point. Parses user message, builds and runs appropriate queries.

        Args:
            user_message (str): The user's natural language input.

        Returns:
            List[str]: Text chunks retrieved from the graph.
        """
        named_entities, generics = self.spacy_helper.parse_user_query_for_entities(user_message)

        self.build_query(named_entities)
        chunks = self.execute_newest_query()

        if generics:
            generic_query_count = self.build_generic_query(generics)
            for _ in range(generic_query_count):
                chunks += self.execute_newest_query()

        if len(chunks) > self.MAX_CHUNKS:
            chunks = chunks[:self.MAX_CHUNKS]

        return chunks
