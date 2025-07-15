from alembic import op

def create_pgvector_extension():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

def drop_pgvector_extension():
    op.execute("DROP EXTENSION IF EXISTS vector")

def create_pgtrgm_extension():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

def drop_pgtrgm_extension():
    op.execute("DROP EXTENSION IF EXISTS pg_trgm;")

# HNSW index
def create_hnsw_index(table_name: str, column_name: str, index_name: str = None, m: int = 16, ef_construction: int = 64):
    if not index_name:
        index_name = f"{table_name}_{column_name}_hnsw_idx"
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {table_name}
        USING hnsw ({column_name} vector_cosine_ops)
        WITH (m = {m}, ef_construction = {ef_construction})
    """)

def drop_hnsw_index(index_name: str):
    op.execute(f"DROP INDEX IF EXISTS {index_name}")
