-- Enable pgvector and create table

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE job_titles (
    id SERIAL PRIMARY KEY,
    uuid UUID NOT NULL,
    title TEXT NOT NULL UNIQUE,
    embedding vector(384)  -- Adjust if your model has a different dim
);