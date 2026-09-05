"""Flags test-set rows with a near-duplicate (very high cosine similarity)"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/processed/difraud_resplit.parquet"
EMBEDDINGS_PATH = "data/processed/embeddings.npy"
SIMILARITY_THRESHOLD = 0.98


def main():
    df = pd.read_parquet(DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values

    train_emb = embeddings[train_mask]
    test_emb = embeddings[test_mask]
    test_df = df.loc[test_mask].reset_index(drop=True)

    sims = cosine_similarity(test_emb, train_emb)
    max_sims = sims.max(axis=1)

    flagged = test_df[max_sims >= SIMILARITY_THRESHOLD]
    print(f"{len(flagged)} of {len(test_df)} test rows have a train "
          f"near-duplicate at similarity >= {SIMILARITY_THRESHOLD}")
    print(flagged["domain"].value_counts())

    flagged.to_csv("outputs/near_duplicate_test_rows.csv", index=False)
    print("saved outputs/near_duplicate_test_rows.csv for review")


if __name__ == "__main__":
    main()