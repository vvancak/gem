import graphs.node_mapper as mp
import networkx as nx
import numpy as np


def edgelist(path: str, directed=False, weighted=False, separator=" ", comment="%", labels=np.array([])) -> (nx.Graph, mp.NodeMapper):
    graph = nx.DiGraph() if directed else nx.Graph()
    mapper = mp.NodeMapper()

    with open(path, "r") as file:
        for line in file:
            # Skip Comments
            line = line.strip()
            if (line[0] == comment):
                continue

            # Split into tokens
            tokens = line.split(separator)
            src_v, tar_v = tokens[0:2]

            # Add weighted/unweighted edge into the graph + apply edge mapping
            if weighted:
                graph.add_edge(mapper[src_v], mapper[tar_v], weight=float(tokens[2]))
            else:
                graph.add_edge(mapper[src_v], mapper[tar_v])

    # Mark important nodes
    for k, v in mapper.mapping.items():
        if isinstance(labels, str) or k in labels:
            graph.add_nodes_from([v], label=k)

    return graph, mapper
