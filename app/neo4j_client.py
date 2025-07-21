from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


class Neo4jClient:
    def run_query(self, query: str, params: dict = None):
        with driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def close(self):
        driver.close()


neo4j_client = Neo4jClient()
