import math
import json
from abc import ABC
from pathlib import Path

import networkx as nx
import numpy as np

from relnet.state import networkx_extension as nx_ext
from relnet.state.graph_state import GeometricRelnetGraph, GeometricRoutingGraph, GeometricCoveringGraph
from relnet.state.network_generators import NetworkGenerator
from relnet.utils.config_utils import local_seed
from relnet.data.data_preprocessor import DataPreprocessor
from relnet.data.euroroad_preprocessor import EuroroadDataPreprocessor
from relnet.eval.file_paths import FilePaths


class GeometricNetworkGenerator(NetworkGenerator, ABC):
    def post_generate_instance(self, instance):
        pos_x = nx.get_node_attributes(instance, 'pos_x')
        pos_y = nx.get_node_attributes(instance, 'pos_y')
        pos = {}
        for node, x in pos_x.items():
            pos[node] = (x, pos_y[node])
        node_positions = np.array([p[1] for p in sorted(pos.items(), key=lambda x: x[0])], dtype=np.float32)

        state = GeometricRelnetGraph(instance, node_positions)
        return state

    def rem_edges_if_needed(self, gen_params, nx_graph, random_seed):
        if 'rem_edges_prop' in gen_params:
            edges = list(nx_graph.edges())
            n_to_rem = math.floor(gen_params['rem_edges_prop'] * len(edges))

            with local_seed(random_seed):
                edges_to_rem_idx = np.random.choice(np.arange(len(edges)), n_to_rem, replace=False)
            nx_graph.remove_edges_from([edges[idx] for idx in edges_to_rem_idx])

    def rem_nodes_if_needed(self, gen_params, nx_graph, random_seed):
        if 'rem_nodes_prop' in gen_params:
            nodes = list(nx_graph.nodes())
            n_to_rem = math.floor(gen_params['rem_nodes_prop'] * len(nodes))
            with local_seed(random_seed):
                nodes_to_rem = np.random.choice(nodes, n_to_rem, replace=False)

            edges_to_rem = []
            for node in nodes_to_rem:
                node_edges = list(nx_graph.edges(node))
                edges_to_rem.extend(node_edges)
            nx_graph.remove_edges_from(edges_to_rem)

    def set_position_attributes(self, nx_graph):
        pos = nx.get_node_attributes(nx_graph, 'pos')
        for node, (x, y) in pos.items():
            nx_graph.node[node]['pos_x'] = x
            nx_graph.node[node]['pos_y'] = y
            del nx_graph.node[node]['pos']


class KHNetworkGenerator(GeometricNetworkGenerator):
    name = 'kaiser_hilgetag'
    conn_radius_modifiers = {'range': 1.5,
                            'max_current': 1}

    def generate_instance(self, gen_params, random_seed):
        n = gen_params['n']
        alpha, beta = gen_params['alpha_kh'], gen_params['beta_kh']
        nx_graph = nx_ext.kaiser_hilgetag_graph(n, alpha=alpha, beta=beta, seed=random_seed)
        self.set_position_attributes(nx_graph)
        self.rem_nodes_if_needed(gen_params, nx_graph, random_seed)
        return nx_graph
    
class RealWorldNetworkGenerator(GeometricNetworkGenerator):
    name = 'euroroad'

    def __init__(self, store_graphs=False, graph_storage_root=None, logs_file=None, original_dataset_dir=None):
        super().__init__(store_graphs=store_graphs, graph_storage_root=graph_storage_root, logs_file=logs_file)

        if original_dataset_dir is None:
            raise ValueError(f"{original_dataset_dir} cannot be None")
        self.original_dataset_dir = original_dataset_dir

        graph_metadata_file = original_dataset_dir / self.name / DataPreprocessor.DATASET_METADATA_FILE_NAME
        with open(graph_metadata_file.resolve(), "r") as fh:
            graph_metadata = json.load(fh)
            self.num_graphs, self.graph_names = graph_metadata['num_graphs'], graph_metadata['graph_names']

    def generate_instance(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)
        filepath = self.original_dataset_dir / self.name / f"{graph_name}.graphml"
        nx_graph = self.read_graphml_with_ordered_int_labels(filepath)
        return nx_graph

    def get_num_graphs(self):
        return self.num_graphs

    def get_graph_name(self, random_seed):
        graph_idx = random_seed % self.num_graphs
        graph_name = self.graph_names[graph_idx]
        return graph_name

    def get_data_filename(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)
        filename = f"{random_seed}-{graph_name}.graphml"
        return filename

    def get_drawing_filename(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)
        filename = f"{random_seed}-{graph_name}.png"
        return filename
    
class WaxmanNetworkGenerator(GeometricNetworkGenerator):
    name = 'waxman'
    conn_radius_modifiers = {'range': 1.5,
                            'max_current': 1}

    def generate_instance(self, gen_params, random_seed):
        n = gen_params['n']
        alpha, beta = gen_params['alpha_waxman'], gen_params['beta_waxman']
        nx_graph = nx.waxman_graph(n, alpha=alpha, beta=beta, seed=random_seed)
        self.set_position_attributes(nx_graph)
        self.rem_nodes_if_needed(gen_params, nx_graph, random_seed)
        return nx_graph
    
class GridNetworkGenerator(GeometricNetworkGenerator):
    name = 'grid'
    conn_radius_modifiers = {'range': 1.5,
                            'max_current': 1}

    def generate_instance(self, gen_params, random_seed):
        n = gen_params['n']
        # Generate geometric grid graph with positions
        nx_graph = nx.grid_2d_graph(n, n)
        pos = {}
        for i, node in enumerate(nx_graph.nodes()):
            pos[i] = node

        pos[12] = (pos[12][0] + 0.1, pos[12][1] + 0.1)
        # Label nodes with single index rather than tuple
        nx.relabel_nodes(nx_graph, {node: i for i, node in enumerate(nx_graph.nodes())}, copy=False)
        nx.set_node_attributes(nx_graph, pos, 'pos')

        self.set_position_attributes(nx_graph)
        self.rem_nodes_if_needed(gen_params, nx_graph, random_seed)
        return nx_graph
    
class KHNetworkGeneratorMVC(GeometricNetworkGenerator):
    name = 'kaiser_hilgetag_mvc'
    conn_radius_modifiers = {'range': 1.5,
                            'max_current': 1}

    def generate_instance(self, gen_params, random_seed):
        n = gen_params['n']
        alpha, beta = gen_params['alpha_kh'], gen_params['beta_kh']
        nx_graph = nx_ext.kaiser_hilgetag_graph(n, alpha=alpha, beta=beta, seed=random_seed)
        self.set_position_attributes(nx_graph)
        self.rem_nodes_if_needed(gen_params, nx_graph, random_seed)
        return nx_graph
    
    def post_generate_instance(self, instance):
        pos_x = nx.get_node_attributes(instance, 'pos_x')
        pos_y = nx.get_node_attributes(instance, 'pos_y')
        pos = {}
        for node, x in pos_x.items():
            pos[node] = (x, pos_y[node])
        node_positions = np.array([p[1] for p in sorted(pos.items(), key=lambda x: x[0])], dtype=np.float32)

        state = GeometricCoveringGraph(instance, node_positions)
        return state
    
class RandomNetworkGenerator(GeometricNetworkGenerator):
    name = 'random_geometric'
    conn_radius_modifiers = {'range': 1.5,
                            'max_current': 1}
    
    def generate_instance(self, gen_params, random_seed):
        n = gen_params['n']
        radius = 0.2
        
        nx_graph = nx.random_geometric_graph(n, radius, seed=random_seed)
        self.set_position_attributes(nx_graph)
        return nx_graph
    
    def post_generate_instance(self, instance):
        pos_x = nx.get_node_attributes(instance, 'pos_x')
        pos_y = nx.get_node_attributes(instance, 'pos_y')
        pos = {}
        for node, x in pos_x.items():
            pos[node] = (x, pos_y[node])
        node_positions = np.array([p[1] for p in sorted(pos.items(), key=lambda x: x[0])], dtype=np.float32)

        state = GeometricRelnetGraph(instance, node_positions)
        return state
    
class NetworkFromFile(GeometricNetworkGenerator):
    name = 'network_from_file'

    def generate_instance(self, positions):
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(len(positions)))
        for i, pos in enumerate(positions):
            nx_graph.node[i]['pos_x'] = pos[0]
            nx_graph.node[i]['pos_y'] = pos[1]
        return nx_graph
    
    def post_generate_instance(self, instance):
        pos_x = nx.get_node_attributes(instance, 'pos_x')
        pos_y = nx.get_node_attributes(instance, 'pos_y')
        pos = {}
        for node, x in pos_x.items():
            pos[node] = (x, pos_y[node])
        node_positions = np.array([p[1] for p in sorted(pos.items(), key=lambda x: x[0])], dtype=np.float32)

        state = GeometricRoutingGraph(instance, node_positions, starting_node=0, current_node=0)
        return state
    

if __name__ == '__main__':
    def get_file_paths(exp_id):
        """Get file paths for experiment."""
        parent_dir = '/experiment_data'
        file_paths = FilePaths(parent_dir, exp_id, setup_directories=True)
        return file_paths
    
    # gen = WaxmanNetworkGenerator()
    # gen = KHNetworkGenerator()
    gen = RandomNetworkGenerator()
    # gen = GridNetworkGenerator() 
    # fp = get_file_paths('test')

    # gen = RealWorldNetworkGenerator(original_dataset_dir=fp.processed_graphs_dir)
    gen_params = {'n': 20, 'alpha_kh': 10, 'beta_kh': 0.001, 'alpha_waxman': 0.25, 'beta_waxman': 0.5}

    # Generate a random geometric graph
    nx_graph = gen.generate_instance(gen_params, random_seed=0)

    pos = {node: (nx_graph.nodes[node]['pos_x'], nx_graph.nodes[node]['pos_y']) for node in nx_graph.nodes}

    # Print number of nodes and edges
    print(nx_graph.number_of_nodes())
    print(nx_graph.number_of_edges())

    # Plot the graph
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    # pos = nx.get_node_attributes(nx_graph, 'pos')
    nx.draw(nx_graph, pos, with_labels=True, node_size=100, font_size=5)

    # Save nx_graph to graphml file
    # nx.write_graphml(nx_graph, 'kh_fig.graphml')

    # Print number of graph edges
    print(nx_graph.number_of_edges())

    plt.savefig('random_geometric_graph.png', dpi=600)
    plt.show()