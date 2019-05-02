import evaluations.evaluation_factory as evf
import embeddings.embedding_factory as emf
import embeddings.embedding_base as emb
import normalization.init_norms as inn
import graphs.graph_factory as gf
import graphs.edge_hiding as eh
import graphs.node_mapper as mp
import run.print_headers as ph
import networkx as nx
import typing as t
import numpy as np


class RunStages:
    def __init__(self, seed, dataset, embed_method, eval_method):
        self._seed = seed
        self._dataset = dataset
        self._embed_method = embed_method
        self._eval_method = eval_method

        print(f"{ph.INFO} Fixing np random seed {seed}")
        np.random.seed(seed)

    def load_graphs(self, config, norm_method: str = None, hide_perc: int = None) -> (nx.Graph, nx.Graph, mp.NodeMapper):
        # Factory & Load
        print(f"{ph.INFO} Loading - {self._dataset}")
        graph, mapper = gf.GraphFactory(config).get_graph(self._dataset)

        # Check graph loaded
        if graph is None:
            print(f"{ph.ERROR} Failed to load Graph")
            exit(1)

        print(f"{ph.OK} Done")

        # Normalize
        if norm_method is not None:
            print(f"{ph.INFO} Normalizing with {norm_method}")
            normalizer = getattr(inn, norm_method)
            graph = normalizer(graph)
            print(f"{ph.OK} Done")

        # Hide edges
        learn_graph = graph
        if hide_perc is not None:
            print(f"{ph.INFO} Hiding {hide_perc}% edges")
            learn_graph = eh.hide_edges(learn_graph, hide_perc)
            print(f"{ph.OK} Done")

        self.set_graphs(graph, learn_graph, mapper)

        return graph, learn_graph, mapper

    def set_graphs(self, eval_graph: nx.Graph, learn_graph: nx.Graph, mapper: mp.NodeMapper) -> None:
        self._eval_graph = eval_graph
        self._learn_graph = learn_graph
        self._mapper = mapper

    def learn_embedding(self, config: t.Dict, dimension: int) -> (emb.EmbeddingBase, float):
        print(f"{ph.INFO} Learning - {self._embed_method}")

        # Check config exists
        if not self._embed_method in config.keys():
            print(f"{ph.ERROR} Embedding Method config not found!")
            exit(1)

        # Factory
        ef = emf.EmbeddingFactory(config)
        embedding = ef.get_embedding(self._learn_graph, self._embed_method, dimension, self._seed)

        # Check embedding retrieved correctly
        if embedding is None:
            print(f"{ph.ERROR} Failed to initialize Embedding!")
            exit(1)

        # Learn
        time = embedding.learn()
        print(f"{ph.OK} Learned in {time:.3f} seconds")

        self._embedding = embedding
        return embedding, time

    def load_embedding(self, config: t.Dict, file: str, dimension: int) -> emb.EmbeddingBase:
        print(f"{ph.INFO} Loading - {self._embed_method}")

        # Check config exists
        if not self._embed_method in config.keys():
            print(f"{ph.ERROR} Embedding Method config not found!")
            exit(1)

        # Factory
        ef = emf.EmbeddingFactory(config)
        embedding = ef.get_embedding(self._learn_graph, self._embed_method, dimension, self._seed)

        # Check embedding retrieved correctly
        if embedding is None:
            print(f"{ph.ERROR} Failed to initialize Embedding!")
            exit(1)

        # Load
        embedding.load_embedding(self._mapper, file)
        print(f"{ph.OK} Done")

        self._embedding = embedding
        return embedding

    def store_embedding(self, path: str) -> None:
        print(f"{ph.INFO} Storing embedding in {path}")
        self._embedding.save_embedding(self._mapper, path)
        print(f"{ph.OK} Done")

    def evaluate_embedding(self, config: t.Dict, out_path: str = None) -> t.Dict:
        print(f"{ph.INFO} Evaluating - {self._eval_method}")

        ef = evf.EvaluationFactory(config)
        evaluation = ef.get_evaluation(self._eval_graph, self._learn_graph, self._eval_method)

        result = evaluation.run(self._embedding, (out_path is not None), out_path)
        print(f"{ph.OK} Done")

        self._result = result
        return result
