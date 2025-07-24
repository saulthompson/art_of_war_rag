from psycopg2 import pool
import os
from dotenv import load_dotenv
load_dotenv()

db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=22,
    host="localhost",
    port=5432,
    user="postgres",
    database="art_of_war",
    password=os.getenv('DB_PASSWORD')
)