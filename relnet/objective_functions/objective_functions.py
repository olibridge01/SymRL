import numpy as np
import networkx as nx
from objective_functions_ext import *
from relnet.state.graph_state import get_graph_hash, GeometricRelnetGraph, GeometricRoutingGraph, GeometricCoveringGraph


def extract_kwargs(s2v_graph, kwargs):
    num_mc_sims = 20
    random_seed = 42
    if 'mc_sims_multiplier' in kwargs:
        num_mc_sims = int(s2v_graph.num_nodes * kwargs['mc_sims_multiplier'])
    if 'random_seed' in kwargs:
        random_seed = kwargs['random_seed']
    return num_mc_sims, random_seed


class LargestComponentSizeTargeted(object):
    """Robustness objective function."""
    name = "lcs_targeted"
    upper_limit = 0.5

    @staticmethod
    def compute(s2v_graph, **kwargs):
        num_mc_sims, random_seed = extract_kwargs(s2v_graph, kwargs)
        N, M, edges = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs
        graph_hash = get_graph_hash(s2v_graph)
        lcs = size_largest_component_targeted(N, M, edges, num_mc_sims, graph_hash, random_seed)
        return lcs

class GlobalEfficiency(object):
    """Global efficiency objective function."""
    name = "global_eff"
    upper_limit = 1

    @staticmethod
    def compute(s2v_graph, **kwargs):
        if type(s2v_graph) != GeometricRelnetGraph:
            raise ValueError("cannot compute efficiency for non-geometric graphs!")

        N, M, edges, edge_lengths = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs, s2v_graph.get_edge_lengths_as_arr()
        pairwise_dists = s2v_graph.get_all_pairwise_distances()
        eff = global_efficiency(N, M, edges, edge_lengths, pairwise_dists)

        if eff == -1.:
            eff = 0.
        return eff
    
class NegativeTourLength(object):
    """Negative tour length objective function (for vanilla TSP)."""
    name = "negative_tour_length"

    @staticmethod
    def compute(s2v_graph, **kwargs):
        if type(s2v_graph) != GeometricRoutingGraph:
            raise ValueError("Require routing graph in order to use tour length objective!")

        N, M, edges, edge_lengths = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs, list(s2v_graph.compute_edge_lengths().values())
        pairwise_dists = s2v_graph.get_all_pairwise_distances()
        tour_length = -np.sum(edge_lengths) / (N * np.amax(pairwise_dists))
        return tour_length
    
class NegativeCoveringSetSize(object):
    """Negative covering set size objective function (covering problems)."""
    name = "covering_set_size"

    @staticmethod
    def compute(s2v_graph, **kwargs):
        if type(s2v_graph) != GeometricCoveringGraph:
            raise ValueError("Require covering graph in order to use covering set size objective!")

        N, M, edges = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs

        if len(edges) > 0:
            return 0
            # raise ValueError("Objective function only evaluated on fully covered graphs!")
        
        # Normalise objective function by num_nodes to be in the range [0, 1]
        covering_set_size = -len(s2v_graph.covering_nodes) / N
        return covering_set_size
    
class SimpleObjective(object):
    """Simple objective function; sum of edge lengths."""
    name = "simple_objective"

    @staticmethod
    def compute(s2v_graph, **kwargs):
        if type(s2v_graph) != GeometricRelnetGraph:
            raise ValueError("Require GeometricRelnet graph in order to use simple objective!")

        edge_lengths = list(s2v_graph.compute_edge_lengths().values())
        # return np.sum(edge_lengths)
        return len(edge_lengths)
    
class GlobalEntropy(object):
    """Graph entropy objective."""
    name = "shannon"
    upper_limit = 100.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        g = s2v_graph.to_networkx()
        deg = nx.degree_histogram(g)
        N = g.number_of_nodes()
        s = 0
        for d in deg:
            if d != 0:
                prob = d / N
                s -= prob * np.log2(prob)
        return s