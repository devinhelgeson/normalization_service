from fastapi import FastAPI
from app.api import router

app = FastAPI(
    title="City Suggestion API",
    description="Provides autocomplete suggestions for large cities in the US and Canada.",
    version="1.0.0",
)

app.include_router(router)
