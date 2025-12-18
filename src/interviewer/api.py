"""
FastAPI server for interview response semantic search.
Uses polars-fastembed retrieve() for search.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from polars_fastembed import register_model

MODEL_ID = "snowflake/snowflake-arctic-embed-xs"
BASE_DIR = Path(__file__).parents[2]
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Global state
df: pl.DataFrame


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df
    print(f"Registering model: {MODEL_ID}")
    register_model(
        MODEL_ID, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    parquet_path = OUTPUT_DIR / "interview_embeddings.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No embeddings found at {parquet_path}. Run 'interviewer-embed' first."
        )

    print(f"Loading embeddings from {parquet_path}")
    df = pl.read_parquet(parquet_path)
    print(f"Loaded {len(df)} responses")
    yield


app = FastAPI(
    title="Interviewer Explorer",
    description="Explore Claude interview responses with semantic search",
    lifespan=lifespan,
)


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), k: int = Query(30, ge=1, le=100)):
    """
    Semantic search over interview responses.
    Returns top k results with similarity scores.
    """
    result = df.fastembed.retrieve(
        query=f"{QUERY_PREFIX}{q}",
        model_name=MODEL_ID,
        embedding_column="embedding",
        k=k,
    )
    return [
        {
            "id": row["transcript_id"],
            "split": row["split"],
            "turn": row["turn"],
            "response": row["response"],
            "score": round(row["similarity"], 4),
            "x": round(row["x"], 4),
            "y": round(row["y"], 4),
        }
        for row in result.iter_rows(named=True)
    ]


@app.get("/api/responses")
def get_all_responses():
    """
    Return all responses with UMAP coordinates for the 2D scatter plot.
    No embeddings sent to client.
    """
    return [
        {
            "id": row["transcript_id"],
            "split": row["split"],
            "turn": row["turn"],
            "preview": row["response"][:100] + "..."
            if len(row["response"]) > 100
            else row["response"],
            "x": round(row["x"], 4),
            "y": round(row["y"], 4),
        }
        for row in df.iter_rows(named=True)
    ]


@app.get("/api/transcript/{transcript_id}")
def get_transcript(transcript_id: str):
    """
    Get all responses from a specific transcript.
    """
    responses = df.filter(pl.col("transcript_id") == transcript_id).sort("turn")
    return [
        {
            "turn": row["turn"],
            "response": row["response"],
            "x": round(row["x"], 4),
            "y": round(row["y"], 4),
        }
        for row in responses.iter_rows(named=True)
    ]


@app.get("/api/stats")
def get_stats():
    """Return dataset statistics."""
    return {
        "total_responses": len(df),
        "splits": df.group_by("split").len().sort("split").to_dicts(),
        "transcripts": df["transcript_id"].n_unique(),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def run():
    uvicorn.run(
        "interviewer.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
