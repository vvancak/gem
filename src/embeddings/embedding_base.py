from abc import ABC, abstractmethod
import graphs.node_mapper as mp
import networkx as nx
import typing as t
import numpy as np


class EmbeddingBase(ABC):
    def __init__(self, graph: nx.Graph, d: int):
        super().__init__()
        self._d = d
        self._graph = graph
        self._embedding = None

    # region === ABSTRACT ===
    @abstractmethod
    def learn(self) -> float:
        pass

    @abstractmethod
    def _estimate_weights(self, src_v: np.array, tar_v: np.array) -> np.ndarray:
        pass

    # endregion

    # region === PUBLIC ===
    def __getitem__(self, pos: t.Union[int, np.array]) -> np.array:
        if self._embedding is None:
            print("[WARNING]: Embeddings not learned yet, learning...")
            self.learn()

        return self._embedding[pos]

    def estimate_weights(self, src_v: np.array, tar_v: np.array = None) -> np.ndarray:
        if tar_v is None:
            tar_v = src_v
        return self._estimate_weights(src_v, tar_v)

    def save_embedding(self, mapper: mp.NodeMapper, output_file: str) -> None:
        with open(f"{output_file}", "w+") as outfile:
            # Header
            outfile.write(f"node_id")
            for i in range(len(self._embedding[0])):
                outfile.write(f",emb_{i}")
            outfile.write("\n")

            # Embeddings
            for vertex, id in zip(self._embedding, mapper.node_ids_sorted()):
                outfile.write(f"{id}")
                for yi in vertex:
                    outfile.write(f",{yi:.3f}")
                outfile.write('\n')

    def load_embedding(self, mapper: mp.NodeMapper, input_file: str) -> None:
        # Load embeddings into dictionary {mapped_id:embedding}
        embedding = {}
        with open(f"{input_file}", "r") as infile:
            for line in infile:
                tokens = line.split(",")

                # Mapped id
                node_id = tokens[0]
                node_id = mapper[node_id]

                # Embedding
                embed_vector = np.array(tokens[1:]).astype(float)
                embedding[node_id] = embed_vector

        # Sort Embeddings by mapped_id and put into np.array
        em_ordered = [embedding[d] for d in sorted(embedding.keys())]
        self._embedding = np.array(em_ordered)
    # endregion
