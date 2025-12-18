"""
Embed interview responses using polars-fastembed.
Run once to generate the parquet file.
"""

import re
from pathlib import Path

import numpy as np
import polars as pl
from polars_fastembed import register_model
from umap import UMAP

MODEL_ID = "snowflake/snowflake-arctic-embed-xs"
HF_DATASET = "Anthropic/AnthropicInterviewer"
BASE_DIR = Path(__file__).parents[2]
OUTPUT_DIR = BASE_DIR / "output"


def extract_responses(text: str) -> list[dict]:
    """Extract individual user responses from a transcript."""
    responses = []

    # Split by "User:" to get user turns
    parts = re.split(r"\bUser:\s*", text)

    for i, part in enumerate(parts[1:], 1):  # Skip first part (before first User:)
        # Get text up to next "AI:" or "Assistant:"
        user_text = re.split(r"\b(?:AI|Assistant):\s*", part)[0].strip()

        if user_text and len(user_text) > 20:  # Filter out very short responses
            responses.append(
                {
                    "response": user_text,
                    "turn": i,
                }
            )

    return responses


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading interview dataset...")

    # Load all splits
    all_dfs = []
    for split in ["workforce", "creatives", "scientists"]:
        try:
            df = pl.read_parquet(
                f"hf://datasets/{HF_DATASET}@~parquet/AnthropicInterviewer/{split}/*.parquet"
            )
            df = df.with_columns(pl.lit(split).alias("split"))
            all_dfs.append(df)
            print(f"  Loaded {len(df)} transcripts from {split}")
        except Exception as e:
            print(f"  Warning: Could not load {split}: {e}")

    df = pl.concat(all_dfs)
    print(f"Total: {len(df)} transcripts")

    # Extract individual responses from transcripts
    print("\nExtracting user responses from transcripts...")

    all_responses = []
    for row in df.iter_rows(named=True):
        responses = extract_responses(row["text"])
        for resp in responses:
            all_responses.append(
                {
                    "transcript_id": row["transcript_id"],
                    "split": row["split"],
                    "turn": resp["turn"],
                    "response": resp["response"],
                }
            )

    df_responses = pl.DataFrame(all_responses)
    print(f"Extracted {len(df_responses)} user responses")

    print(f"\nRegistering model: {MODEL_ID}")
    register_model(
        MODEL_ID, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # Create text for embedding - just use the response
    df_responses = df_responses.with_columns(
        pl.col("response")
        .str.slice(0, 512)
        .alias("text_to_embed")  # Truncate long responses
    )

    print("\nEmbedding responses...")
    df_emb = df_responses.fastembed.embed(
        columns="text_to_embed",
        model_name=MODEL_ID,
        output_column="embedding",
    )

    print("\nRunning UMAP for 2D visualization...")
    embeddings = np.array(df_emb["embedding"].to_list(), dtype=np.float32)
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords_2d = umap.fit_transform(embeddings)

    df_final = df_emb.with_columns(
        pl.Series("x", coords_2d[:, 0].tolist()),
        pl.Series("y", coords_2d[:, 1].tolist()),
    ).drop("text_to_embed")

    parquet_path = OUTPUT_DIR / "interview_embeddings.parquet"
    df_final.write_parquet(parquet_path)
    print(f"\nSaved: {parquet_path}")
    print(f"Size: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Print some stats
    print(f"\nSplit breakdown:")
    print(df_final.group_by("split").len().sort("split"))


if __name__ == "__main__":
    run()
