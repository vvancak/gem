import reconstruction.negsam_reconstr as ens
import embeddings.embedding_base as eb
import networkx as nx
import typing as t


def mse_observed(graph: nx.Graph, embedding: eb.EmbeddingBase, max_k: int, ns_size=10) -> t.Dict:
    num_correct = 0
    err = 0.0

    for src_v in graph.nodes:
        emb_edges, graph_edges = ens.get_top_edges_ns(graph, embedding, src_v, max_k, ns_size)
        for (ew, (u, v)), (gw, (_, _)) in zip(emb_edges, graph_edges):
            if graph.has_edge(u, v):
                err += (ew - gw) ** 2
                num_correct += 1

    mseo = err / num_correct if num_correct > 0 else 0
    return {
        "MSE_OBS": round(mseo, 5)
    }


def mse_new_obs_at_k(graph: nx.Graph, learn_graph: nx.Graph, embedding: eb.EmbeddingBase, max_k: int, batch_size: int, ns_size: int = 10) -> t.Dict:
    num_correct = 0
    err = 0.0

    errors = []
    emb_edges, graph_edges = ens.get_top_new_edges_ns(graph, learn_graph, embedding, max_k, ns_size, batch_size)
    for (ew, (u, v)), (gw, (_, _)) in zip(emb_edges, graph_edges):
        if graph.has_edge(u, v):
            err += (ew - gw) ** 2
            num_correct += 1

        mseo = err / num_correct if num_correct > 0 else 0
        errors.append(round(mseo, 5))

    return {
        "MSE_OBS": errors
    }
