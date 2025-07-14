import os
import uuid
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DB_URL)


def insert_job_title(title: str, embedding: list[float]):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO job_titles (uuid, title, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (title) DO NOTHING;
        """,
        (str(uuid.uuid4()), title, embedding),
    )
    conn.commit()
    cur.close()
    conn.close()


def query_similar_titles(query_embedding: list[float], top_k: int = 5):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT uuid, title, score
        FROM (
            SELECT uuid, title, embedding <=> %s::vector AS score
            FROM job_titles
        ) AS scored
        ORDER BY score ASC
        LIMIT %s;
        """,
        (query_embedding, top_k),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [{"uuid": row[0], "title": row[1], "score": 1 - row[2]} for row in results]
