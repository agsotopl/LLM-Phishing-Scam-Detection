"""Clusters near-duplicate texts across the full dataset and reassigns train/test/validation splits so no near-duplicate cluster spans a split boundary."""

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

DATA_PATH = "data/processed/difraud_merged.parquet"
EMBEDDINGS_PATH = "data/processed/embeddings.npy"
OUTPUT_PATH = "data/processed/difraud_resplit.parquet"
SIMILARITY_THRESHOLD = 0.98
SPLIT_PROPORTIONS = {"train": 0.80, "test": 0.10, "validation": 0.10}
RANDOM_SEED = 42


def build_clusters(embeddings, threshold=SIMILARITY_THRESHOLD):
    """Builds a sparse near-duplicate graph and returns a cluster id per row."""
    radius = 1 - threshold
    nn = NearestNeighbors(radius=radius, metric="cosine")
    nn.fit(embeddings)
    graph = nn.radius_neighbors_graph(embeddings, mode="connectivity")
    n_clusters, labels = connected_components(graph, directed=False)
    return labels, n_clusters


def assign_splits(df, cluster_ids, proportions=SPLIT_PROPORTIONS, seed=RANDOM_SEED):
    """Greedily assigns whole clusters to splits, stratified by each
    cluster's majority label so class balance stays consistent across
    train/test/validation."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["cluster"] = cluster_ids

    cluster_stats = df.groupby("cluster")["label"].agg(["size", "mean"])
    cluster_stats["majority_label"] = (cluster_stats["mean"] >= 0.5).astype(int)

    cluster_to_split = {}
    current = {split: 0 for split in proportions}

    for majority_label in [0, 1]:
        group = cluster_stats[cluster_stats["majority_label"] == majority_label]
        group_total = group["size"].sum()
        group_targets = {s: prop * group_total for s, prop in proportions.items()}
        group_current = {s: 0 for s in proportions}

        cluster_order = list(group.index)
        rng.shuffle(cluster_order)
        cluster_order.sort(key=lambda c: group.loc[c, "size"], reverse=True)

        for cluster in cluster_order:
            size = group.loc[cluster, "size"]
            deficits = {s: group_targets[s] - group_current[s] for s in proportions}
            split = max(deficits, key=deficits.get)
            cluster_to_split[cluster] = split
            group_current[split] += size
            current[split] += size

    df["split"] = df["cluster"].map(cluster_to_split)
    return df.drop(columns="cluster"), current


def main():
    df = pd.read_parquet(DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    cluster_ids, n_clusters = build_clusters(embeddings)
    print(f"found {n_clusters} clusters across {len(df)} rows")

    df_resplit, counts = assign_splits(df, cluster_ids)
    print("new split sizes:", counts)
    print(df_resplit["split"].value_counts())
    print("fraud rate by split:")
    print(df_resplit.groupby("split")["label"].mean())

    df_resplit.to_parquet(OUTPUT_PATH, index=False)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()