import evaluations.metrics.mean_sqr_err as mse
import evaluations.metrics.prec_at_k as pak
import evaluations.evaluation_base as evb
import embeddings.embedding_base as emb
import networkx as nx
import typing as t


class LinkPrediction(evb.EvaluationBase):
    def __init__(self, graph: nx.Graph, learning_graph: nx.Graph, max_k: int) -> None:
        super().__init__(graph, learning_graph)
        self._max_k = max_k

    def _show(self, result: t.Dict) -> None:
        print(result)

    def _run(self, embedding: emb.EmbeddingBase) -> t.Dict:
        prec_at_k = pak.precision_at_k(self._graph, self._learning_graph, embedding, self._max_k)
        mse_o = mse.mse_new_obs_at_k(self._graph, self._learning_graph, embedding, self._max_k)

        # Report
        return {**prec_at_k, **mse_o}
