import evaluations.evaluation_base as evb
import embeddings.embedding_base as emb
import sklearn.manifold as skm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import typing as t


class Visualization(evb.EvaluationBase):
    def __init__(self, graph: nx.Graph, learning_graph: nx.Graph, labels=False) -> None:
        super().__init__(graph, learning_graph)
        self._labels = labels

    # region === OVERRIDES ===
    def _store(self, result: t.Dict, file_path: str) -> None:
        plt.savefig(f"{file_path}.png")

    def _show(self, result: t.Dict):
        plt.show()

    def _run(self, embedding: emb.EmbeddingBase) -> t.Dict:
        emb_ndarr = embedding.get_ndarray

        num_nodes, d = np.shape(emb_ndarr)
        if d > 2:
            emb_ndarr = skm.TSNE(n_components=2).fit_transform(emb_ndarr)
        self._vis_2d(emb_ndarr, num_nodes)
        return {}

    # endregion

    # region === PRIVATE ===
    def _vis_2d(self, emb_ndarr: np.ndarray, num_nodes: int) -> None:
        labelled_nodes = nx.get_node_attributes(self._graph, 'label')

        for v in range(num_nodes):
            emb = emb_ndarr[v]

            if v in labelled_nodes.keys():
                plt.scatter(emb[0], emb[1], marker='D', c='m', zorder=1)

                if self._labels:
                    plt.text(emb[0], emb[1], s=str(labelled_nodes[v]), fontdict={'color': 'red'})

            else:
                plt.scatter(emb[0], emb[1], marker='o', c="y", zorder=1)

        for e in self._graph.edges(data=True):
            emb_from = emb_ndarr[e[0]]
            emb_to = emb_ndarr[e[1]]
            line = list(zip(emb_from, emb_to))

            weight = 1 - (e[2].get("weight", 1) / self._max_weight)
            plt.plot(line[0], line[1], c=str(weight), zorder=0)
    # endregion
