CREATE INDEX IF NOT EXISTS job_titles_embedding_hnsw_idx
ON job_titles
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 64
);