import evaluations.metrics.mean_avg_prec as meavp
import evaluations.metrics.mean_sqr_err as mse
import evaluations.evaluation_base as evb
import embeddings.embedding_base as emb
import networkx as nx
import typing as t


class GraphReconstruction(evb.EvaluationBase):
    def __init__(self, graph: nx.Graph, learning_graph: nx.Graph, max_k: int) -> None:
        super().__init__(graph, learning_graph)
        self._max_k = max_k

    def _show(self, result: t.Dict) -> None:
        print(result)

    def _run(self, embedding: emb) -> t.Dict:
        # Mean squared error of observed edges
        mse_observed = mse.mse_observed_old(self._graph, embedding, self._max_k)

        # Mean Average Precision
        mean_avg_prec = meavp.mean_avg_prec(self._graph, embedding, self._max_k)

        # Report
        return {**mean_avg_prec, **mse_observed}
