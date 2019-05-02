import embeddings.embedding_base as eb
import networkx as nx
import numpy as np
import math


def compute_graph_weights(graph: nx.Graph, src_v: np.array, tar_v: np.array, ns_baseline: float) -> np.array:
    results = []
    for u, v in zip(src_v, tar_v):
        if not graph.has_edge(u, v):
            results.append(0)
            continue

        w = graph.edges[src_v, tar_v].get('weight', 1)
        results.append(math.exp(w) / (math.exp(w) + ns_baseline))

    return np.array(results)


def compute_embedding_weights(embedding: eb.EmbeddingBase, src_v: np.array, tar_v: np.array, ns_baseline: float) -> np.array:
    weights = embedding.estimate_weights(src_v, tar_v)
    baseline = np.exp(weights) + ns_baseline
    return np.exp(weights) / baseline


def get_negative_sample(graph: nx.Graph, excluded_vertices: np.array, sample_size: int) -> np.array:
    result = []
    while len(result) < sample_size:
        src_v, tar_v = np.random.choice(graph.nodes, size=(2,))

        # Excluded edges
        if src_v in excluded_vertices or tar_v in excluded_vertices:
            continue

        # Non-Observed edges
        if not graph.has_edge(src_v, tar_v):
            continue

        result.append((src_v, tar_v))

    return np.array(result)


def compute_ns_graph(graph: nx.Graph, negative_sample: np.array) -> float:
    sum = 0.0
    for (u, v) in negative_sample:
        w = graph.edges[u, v].get('weight', 1)
        sum += math.exp(w)

    return sum


def compute_ns_embedding(embedding: eb.EmbeddingBase, negative_sample: np.array) -> float:
    src_v = []
    tar_v = []

    for u, v in negative_sample:
        src_v.append(u)
        tar_v.append(v)

    src_v = np.array(src_v)
    tar_v = np.array(tar_v)

    weights = embedding.estimate_weights(src_v, tar_v).diagonal()
    return np.sum(np.exp(weights))
