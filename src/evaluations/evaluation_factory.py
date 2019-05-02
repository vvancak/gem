import evaluations.methods.graph_reconstruction as grr
import evaluations.methods.link_prediction as lpr
import evaluations.methods.visualization as vis
import evaluations.evaluation_base as eb
import networkx as nx
import typing as t
import json

EV_CONFIG = "../config/ev_config.json"


class Evaluations:
    @staticmethod
    def visualization(graph, learning_graph, **kwargs) -> vis.Visualization:
        return vis.Visualization(graph, learning_graph, **kwargs)

    @staticmethod
    def graph_reconstruction(graph, learning_graph, **kwargs) -> grr.GraphReconstruction:
        return grr.GraphReconstruction(graph, learning_graph, **kwargs)

    @staticmethod
    def link_prediction(graph, learning_graph, **kwargs) -> lpr.LinkPrediction:
        return lpr.LinkPrediction(graph, learning_graph, **kwargs)


class EvaluationFactory:
    def __init__(self, config: t.Dict) -> None:
        self._config = config

    def get_evaluation(self, graph: nx.Graph, learning_graph: nx.Graph, method_name: str) -> eb.EvaluationBase:
        kwargs = self._config[method_name]
        method = getattr(Evaluations, method_name)

        return method(graph, learning_graph, **kwargs)
