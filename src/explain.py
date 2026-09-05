"""Generates a short plain-language explanation for every test-set message, using Claude Haiku."""

import json
import os
import time

import numpy as np
import pandas as pd
from anthropic import Anthropic, APIError
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/processed/difraud_resplit.parquet"
METRICS_PATH = "outputs/metrics.json"
CACHE_PATH = "outputs/explanations.jsonl"
MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3

client = Anthropic()

SYSTEM_PROMPT = (
    "You review short messages (emails or texts) that have been scored by a "
    "phishing/scam detection model. Given the message text and the model's "
    "prediction, write a short paragraph (2-4 sentences) explaining what in "
    "the message content supports or contradicts that prediction. Point to "
    "specific red flags (urgency, requests for credentials or payment, "
    "suspicious links, impersonation) if the message was flagged, or "
    "specific reasons it reads as legitimate if it wasn't. Do not repeat "
    "the raw score. Be concrete and reference the actual message content."
)


def load_cache(path=CACHE_PATH):
    """Loads already-generated explanations keyed by row index."""
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            cache[row["index"]] = row
    return cache


def append_cache(row, path=CACHE_PATH):
    """Appends one explanation record to the cache file."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def generate_explanation(text, predicted_label, true_label):
    """Calls Claude to explain one prediction, with basic retry on API errors."""
    predicted = "phishing/scam" if predicted_label == 1 else "legitimate"
    user_prompt = (
        f"Message:\n{text}\n\n"
        f"Model prediction: {predicted}\n\n"
        "Explain what supports or contradicts this prediction."
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except APIError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    df = pd.read_parquet(DATA_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    print(f"loaded metrics for reference (overall f1={metrics['overall']['f1']:.4f})")

    import joblib

    embeddings = np.load("data/processed/embeddings.npy")
    test_mask = (df["split"] == "test").values

    model = joblib.load("outputs/model.joblib")
    predictions = model.predict(embeddings[test_mask])

    cache = load_cache()
    print(f"{len(cache)} explanations already cached")

    for i, row in test_df.iterrows():
        if i in cache:
            continue
        explanation = generate_explanation(row["text"], predictions[i], row["label"])
        record = {
            "index": i,
            "domain": row["domain"],
            "true_label": int(row["label"]),
            "predicted_label": int(predictions[i]),
            "explanation": explanation,
        }
        append_cache(record)
        if (i + 1) % 50 == 0:
            print(f"explained {i + 1}/{len(test_df)}")

    print("done")


if __name__ == "__main__":
    main()