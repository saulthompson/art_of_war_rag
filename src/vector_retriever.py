import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from typing import Generator, List, Optional, Dict, Any, Tuple
from src.db_pool import db_pool


class Retriever:
    """
    Provides methods for retrieving vector-based similarity search results
    from PostgreSQL tables using pgvector.
    """

    @contextmanager
    def get_cursor(self) -> Generator[psycopg2.extras.RealDictCursor, None, None]:
        """
        Context manager that yields a dictionary-based PostgreSQL cursor.

        Yields:
            Generator[RealDictCursor, None, None]: A database cursor for executing queries.
        """
        conn = db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
        finally:
            conn.close()

    def find_similar(self, embedding: List[float], limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Find the most similar text chunks in 'art_of_war_book_english' based on vector similarity.

        Args:
            embedding (List[float]): The query embedding vector.
            limit (int): Maximum number of results to return.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of matching rows with similarity scores.
        """
        try:
            query = """
                SELECT 
                    id,
                    chunk,
                    chapter,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM art_of_war_book_english
                ORDER BY similarity DESC
                LIMIT %s;
            """
            with self.get_cursor() as cur:
                cur.execute(query, (embedding, limit))
                return cur.fetchall()
        except Exception as e:
            print('Error while retrieving similar chunks:', e)
            return None

    def find_similar_above_threshold(
        self, 
        embedding: List[float], 
        threshold: float = 0.5, 
        limit: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find activities in 'travel_activity' table with similarity above a given threshold.

        Args:
            embedding (List[float]): The query embedding vector.
            threshold (float): Minimum similarity score to include.
            limit (int): Maximum number of results to return.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of activities meeting the threshold.
        """
        try:
            query = """
                SELECT 
                    id,
                    activity,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM travel_activity
                WHERE (1 - (embedding <=> %s::vector)) >= %s
                ORDER BY similarity DESC
                LIMIT %s;
            """
            with self.get_cursor() as cur:
                cur.execute(query, (embedding, embedding, threshold, limit))
                return cur.fetchall()
        except Exception as e:
            print(f'Error while retrieving chunks above threshold {threshold}:', e)
            return None

    def find_most_average(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find the top `limit` activities whose embeddings are closest to the average vector (centroid).

        Args:
            limit (int): Number of most average activities to return.

        Returns:
            List[Dict[str, Any]]: A list of the most average activities.
        """
        query = """
            WITH centroid AS (
                SELECT avg(embedding) AS center_vector FROM travel_activity
            )
            SELECT 
                id,
                activity,
                embedding <=> (SELECT center_vector FROM centroid) AS distance_to_center
            FROM travel_activity
            ORDER BY distance_to_center ASC
            LIMIT %s;
        """
        with self.get_cursor() as cur:
            cur.execute(query, (limit,))
            return cur.fetchall()

    def find_outliers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find the top `limit` activities whose embeddings are furthest from the average (semantic outliers).

        Args:
            limit (int): Number of most distant outliers to return.

        Returns:
            List[Dict[str, Any]]: A list of the most semantically distant activities.
        """
        query = """
            WITH centroid AS (
                SELECT avg(embedding) AS center_vector FROM travel_activity
            )
            SELECT 
                id,
                activity,
                embedding <=> (SELECT center_vector FROM centroid) AS distance_from_center
            FROM travel_activity
            ORDER BY distance_from_center DESC
            LIMIT %s;
        """
        with self.get_cursor() as cur:
            cur.execute(query, (limit,))
            return cur.fetchall()
