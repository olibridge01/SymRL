import math
from networkx.generators.geometric import euclidean

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    _is_scipy_available = False
else:
    _is_scipy_available = True

import networkx as nx
from networkx.utils import nodes_or_number, py_random_state


@py_random_state(5)
@nodes_or_number(0)
def kaiser_hilgetag_graph(n, beta=0.4, alpha=0.1, domain=(0, 0, 1, 1),
                 metric=None, seed=None):
    """
    Generate a Kaiser-Hilgetag graph.
    
    Args:
    - n (int or iterable): number of nodes or iterable of nodes
    - beta (float): KH parameter
    - alpha (float): KH parameter
    - domain (tuple): bounding box of the graph positions
    - metric (callable): a distance metric
    - seed (int, random_state, or None): seed for random number generator
    """
    n_name, nodes = n
    n_nodes = len(nodes)
    G = nx.Graph()

    (xmin, ymin, xmax, ymax) = domain
    xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
    G.add_node(0, pos=(xcenter, ycenter))

    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = euclidean

    i = 1
    while True:
        pos_i = seed.uniform(xmin, xmax), seed.uniform(ymin, ymax)
        cands = list(range(0, i))

        pos = nx.get_node_attributes(G, 'pos')
        # `pair` is the pair of nodes to decide whether to join.

        def should_join_with(cand):
            dist = metric(pos_i, pos[cand])
            s = seed.random()
            v = beta * math.exp(-dist * alpha)
            return s < v

        nodes_to_connect = filter(should_join_with, cands)
        edges_to_add = [(i, j) for j in nodes_to_connect]

        if len(edges_to_add) > 0:
            G.add_node(i, pos=pos_i)
            G.add_edges_from(edges_to_add)
            i += 1

        if i == n_nodes:
            break

    return G