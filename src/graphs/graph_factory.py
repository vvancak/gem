import graphs.graph_parsers as gp
import graphs.node_mapper as nmp
import run.print_headers as ph
import networkx as nx
import typing as t

CONF_PATH = "path"
CONF_PARSER = "parser"
CONF_PARAMS = "parameters"


class GraphFactory:
    def __init__(self, config: t.Dict) -> None:
        self._config = config

    def get_graph(self, dataset: str) -> (nx.Graph, nmp.NodeMapper):
        if dataset not in self._config.keys():
            print(f"{ph.ERROR} No entry for dataset {dataset} !")
            return None, None

        # Get dataset configuration
        ds_entry = self._config[dataset]

        # Which parser to use (method from graph_parsers)
        parser = ds_entry[CONF_PARSER]
        parser = getattr(gp, parser)

        # Parser parameters
        ds_path = ds_entry.get(CONF_PATH, f"../ds/{dataset}")
        parameters = ds_entry.get(CONF_PARAMS, {})

        # Get graph
        graph, mapper = parser(ds_path, **parameters)
        return graph, mapper
