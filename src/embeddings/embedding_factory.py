import embeddings.methods.laplacian_eigenmaps as le
import embeddings.methods.deep_walk as dw
import embeddings.methods.node2vec as n2v
import embeddings.methods.sdne as sdn
import embeddings.methods.line as lne
import embeddings.methods.random as rnd
import embeddings.embedding_base as eb
import networkx as nx
import typing as t

EM_CONFIG = "../config/em_config.json"


class Embeddings:
    @staticmethod
    def laplacian_eigenmaps(graph, d, seed, **kwargs):
        return le.LaplacianEigenmaps(graph, d, **kwargs)

    @staticmethod
    def deep_walk(graph, d, seed, **kwargs):
        return dw.DeepWalk(graph, d, seed, **kwargs)

    @staticmethod
    def node2vec(graph, d, seed, **kwargs):
        return n2v.Node2Vec(graph, d, seed, **kwargs)

    @staticmethod
    def LINE(graph, d, seed, **kwargs):
        return lne.LINE(graph, d, seed, **kwargs)

    @staticmethod
    def SDNE(graph, d, seed, **kwargs):
        return sdn.SDNE(graph, d, seed, **kwargs)

    @staticmethod
    def random(graph, d, seed, **kwargs):
        return rnd.Random(graph, d, seed, **kwargs)


class EmbeddingFactory:
    def __init__(self, config: t.Dict) -> None:
        self.config = config

    def get_embedding(self, graph: nx.Graph, method_name: str, dimension: int, seed: int) -> eb.EmbeddingBase:
        kwargs = self.config[method_name]
        method = getattr(Embeddings, method_name)

        return method(graph, dimension, seed, **kwargs)
