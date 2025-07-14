CREATE TABLE IF NOT EXISTS job_titles (
    id SERIAL PRIMARY KEY,
    uuid UUID NOT NULL,
    title TEXT NOT NULL UNIQUE,
    embedding vector(384)
);