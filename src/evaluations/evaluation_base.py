from abc import ABC, abstractmethod
import typing as t
import embeddings.embedding_base as emb
import networkx as nx
import json


class EvaluationBase(ABC):
    def __init__(self, graph: nx.Graph, learning_graph: nx.Graph) -> None:
        self._graph = graph
        self._learning_graph = learning_graph
        self._max_weight = max(graph.edges(data=True), key=lambda e: e[2].get("weight", 1))
        self._max_weight = self._max_weight[2].get('weight', 1)
        super().__init__()

    def _store(self, result: t.Dict, file_path: str) -> None:
        with open(f"{file_path}.txt", "a+") as fp:
            json.dump(result, fp)

    @abstractmethod
    def _show(self, result: t.Dict) -> None:
        pass

    @abstractmethod
    def _run(self, embedding: emb.EmbeddingBase) -> t.Dict:
        pass

    def run(self, embedding: emb.EmbeddingBase, store: bool = False, file_path: str = None) -> t.Dict:
        result = self._run(embedding)

        if store:
            self._store(result, file_path)
        self._show(result)

        return result
