from fastapi import FastAPI
from app.api import router

app = FastAPI(
    title="Normalization Service APIs",
    description="A collection or normalization APIs designed to standardize / normalize / match raw text entities with entites in the appropriate normalized dataset.",
    version="1.0.0",
)

app.include_router(router)
