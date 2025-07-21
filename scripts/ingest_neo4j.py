from app.neo4j_client import driver
from app.job_title_match import get_connection


def fetch_job_titles():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT uuid, title, type, onet_code FROM job_titles;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def ingest_into_neo4j():
    rows = fetch_job_titles()
    print(f"Fetched {len(rows)} job titles from Postgres")

    with driver.session() as session:
        print("Creating JobTitle nodes in Neo4j...")
        for uuid, title, type_, onet_code in rows:
            session.run(
                """
                MERGE (jt:JobTitle {uuid: $uuid})
                SET jt.title = $title, jt.type = $type, jt.onet_code = $onet_code
                """,
                {
                    "uuid": str(uuid),
                    "title": title,
                    "type": type_,
                    "onet_code": onet_code,
                },
            )

        print("Creating VARIANT_OF relationships...")
        session.run(
            """
            MATCH (alt:JobTitle {type: 'alt_title'}), (onet:JobTitle {type: 'onet_role'})
            WHERE alt.onet_code = onet.onet_code
            MERGE (alt)-[:VARIANT_OF]->(onet)
            """
        )

    print("Neo4j ingestion complete!")


if __name__ == "__main__":
    ingest_into_neo4j()
