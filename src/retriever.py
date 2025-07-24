import psycopg2
import psycopg2.extras
from src.db_pool import db_pool
from contextlib import contextmanager

class Retriever:
    @contextmanager
    def get_cursor(self):
        conn = db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
        finally:
            conn.close()

    def find_similar(self, embedding, limit=5):
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
            print('Error while retrieving similar chunks', e)

    def find_similar_above_threshold(self, embedding, threshold = 0.5, limit = 5):
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
            print('Error while retrieving chunks above threshold ', threshold, e)

    def find_most_average(self, limit=5):
        """
        Find the activity whose embedding is closest to the average embedding (centroid).
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


    def find_outliers(self, limit=5):
        """
        Find the top `limit` activities whose embeddings are furthest from the average embedding (centroid).
        These are semantic outliers.
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
