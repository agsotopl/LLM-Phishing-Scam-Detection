"""Embeds the merged DIFrauD text column with OpenAI text-embedding-3-small"""

import os

import numpy as np
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "text-embedding-3-small"
MAX_TOKENS = 8191
BATCH_SIZE = 100
INPUT_PATH = "data/processed/difraud_merged.parquet"
EMBEDDINGS_PATH = "data/processed/embeddings.npy"

client = OpenAI()
encoding = tiktoken.encoding_for_model(MODEL)


def truncate(text, max_tokens=MAX_TOKENS):
    """Truncates text to max_tokens using the model's tokenizer."""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def embed_batch(texts):
    """Calls the embeddings API for one batch and returns a list of vectors."""
    response = client.embeddings.create(model=MODEL, input=texts)
    return [d.embedding for d in response.data]


def main():
    df = pd.read_parquet(INPUT_PATH)
    texts = [truncate(t) for t in df["text"]]

    if os.path.exists(EMBEDDINGS_PATH):
        print(f"{EMBEDDINGS_PATH} already exists, skipping re-embedding")
        return

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        all_embeddings.extend(embed_batch(batch))
        print(f"embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    embeddings = np.array(all_embeddings)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"saved {EMBEDDINGS_PATH} with shape {embeddings.shape}")


if __name__ == "__main__":
    main()