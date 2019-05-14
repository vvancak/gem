import reconstruction.helpers.reconstr_network as rne
import reconstruction.direct_reconstr as dr
import embeddings.embedding_base as eb
import run.print_headers as ph
import networkx as nx
import numpy as np

# TODO: Replace this; experimental stuff
LR = 0.01
IT = 10
SEED = 42


# TODO: Actually, refactor whole this crap if works
class ModelReconstruction():
    def __init__(self, learn_graph: nx.Graph, embedding: eb.EmbeddingBase, batch_size: int, seed: int = SEED):
        self._batch_size = batch_size
        self._embedding = embedding
        self._learn_graph = learn_graph
        self._network = rne.ReconstrNetwork(seed=seed)
        self._network.construct(len(self._embedding[0]), 0.001)

    def learn(self):
        print(f"{ph.INFO} Learning Edge Weight Reconstruction model...")

        for i in range(IT):
            loss = []
            x1, x2, ws = [], [], []

            for u, v, w in self._learn_graph.edges.data("weight", default=1):
                x1.append(u)
                x2.append(v)
                ws.append(w)

                # TODO: Replace this; inefficient, batch size designed for ...
                # TODO: ... processing batch_size * batch_size matrix
                if len(x1) == self._batch_size:
                    l = self._train_batch(x1, x2, ws)
                    loss.append(l)
                    x1, x2, ws = [], [], []

            # Last batch
            l = self._train_batch(x1, x2, ws)
            loss.append(l)

            loss = np.array(loss)
            print(f"{ph.INFO} Traning - Ep {i} | Loss: {np.mean(loss):.5f}")

    def _train_batch(self, x1, x2, w):
        if len(x1) == 0:
            return

        # Duplicate in opposite direction
        x1, x2, w = np.array(x1), np.array(x2), np.array(w)
        x1a = np.concatenate((x1, x2), axis=0)
        x2a = np.concatenate((x2, x1), axis=0)
        w = np.concatenate((w, w), axis=0)

        return self._network.train(self._embedding[x1a], self._embedding[x2a], w)

    def get_best_new_edges(self, graph: nx.Graph, max_k: int):
        edges = dr.get_best_new_edges(graph, self._learn_graph, self._embedding, max_k, self._batch_size)

        final_edges = []
        x1, x2 = [], []
        for (w, (u, v)) in edges:
            x1.append(u)
            x2.append(v)

            if len(x1) == self._batch_size:
                x1, x2 = np.array(x1), np.array(x2)
                weights = self._network.predict(self._embedding[x1], self._embedding[x2])
                for i, w in enumerate(weights):
                    final_edges.append((w, (x1[i], x2[i])))
                x1, x2 = [], []

        # Last Batch
        x1, x2 = np.array(x1, dtype=int), np.array(x2, dtype=int)
        weights = self._network.predict(self._embedding[x1], self._embedding[x2])
        for i, w in enumerate(weights):
            final_edges.append((w, (x1[i], x2[i])))

        return np.array(final_edges)
