import json
import math
from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx

from relnet.state.graph_state import RelnetGraph
from relnet.utils.config_utils import get_logger_instance

num_graphs_synth = 25

class NetworkGenerator(ABC):
    """Base network generator class."""
    enforce_connected = True

    def __init__(self, store_graphs=False, graph_storage_root=None, logs_file=None, **kwargs):
        super().__init__()
        self.store_graphs = store_graphs
        if self.store_graphs:
            self.graph_storage_root = graph_storage_root
            self.graph_storage_dir = graph_storage_root / self.name
            self.graph_storage_dir.mkdir(parents=True, exist_ok=True)

        if logs_file is not None:
            self.logger_instance = get_logger_instance(logs_file)
        else:
            self.logger_instance = None

        self.num_graphs, self.graph_names = num_graphs_synth, [f"synth_{i}" for i in range(num_graphs_synth)]

    def generate(self, gen_params, random_seed):
        if self.store_graphs:
            filename = self.get_data_filename(gen_params, random_seed)
            filepath = self.graph_storage_dir / filename

            should_create = True
            if filepath.exists():
                try:
                    instance = self.read_graphml_with_ordered_int_labels(filepath)
                    state = self.post_generate_instance(instance)
                    should_create = False
                except Exception:
                    should_create = True

            if should_create:
                instance = self.generate_instance(gen_params, random_seed)
                state = self.post_generate_instance(instance)
                nx.readwrite.write_graphml(instance, filepath.resolve())

                drawing_filename = self.get_drawing_filename(gen_params, random_seed)
                drawing_path = self.graph_storage_dir / drawing_filename
                state.draw_to_file(drawing_path)
        else:
            instance = self.generate_instance(gen_params, random_seed)
            state = self.post_generate_instance(instance)

        return state

    def get_num_graphs(self):
        return self.num_graphs

    def get_graph_name(self, random_seed):
        graph_idx = random_seed % self.num_graphs
        graph_name = self.graph_names[graph_idx]
        return graph_name

    @staticmethod
    def read_graphml_with_ordered_int_labels(filepath):
        instance = nx.readwrite.read_graphml(filepath.resolve())
        num_nodes = len(instance.nodes)
        relabel_map = {str(i): i for i in range(num_nodes)}
        nx.relabel_nodes(instance, relabel_map, copy=False)

        G = nx.Graph()
        G.add_nodes_from(sorted(instance.nodes(data=True)))
        G.add_edges_from(instance.edges(data=True))

        return G

    def generate_many(self, gen_params, random_seeds):
        return [self.generate(gen_params, random_seed) for random_seed in random_seeds]

    @abstractmethod
    def generate_instance(self, gen_params, random_seed):
        pass

    @abstractmethod
    def post_generate_instance(self, instance):
        pass


    def get_data_filename(self, gen_params, random_seed):
        n = gen_params['n']
        filename = f"{n}-{random_seed}.graphml"
        return filename

    def get_drawing_filename(self, gen_params, random_seed):
        n = gen_params['n']
        filename = f"{n}-{random_seed}.png"
        return filename

    @staticmethod
    def compute_number_edges(n, edge_percentage):
        total_possible_edges = (n * (n - 1)) / 2
        return int(math.ceil((total_possible_edges * edge_percentage / 100)))

    @staticmethod
    def construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs):
        train_seeds = list(range(0, num_train_graphs))
        validation_seeds = list(range(num_train_graphs, num_train_graphs + num_validation_graphs))
        offset = num_train_graphs + num_validation_graphs
        test_seeds = list(range(offset, offset + num_test_graphs))
        return train_seeds, validation_seeds, test_seeds


    @staticmethod
    def get_common_generator_params():
        gp = {}
        gp['m_ba'] = 2
        gp['m_percentage_er'] = 20
        gp['m_ws'] = 2
        gp['p_ws'] = 0.1
        gp['d_reg'] = 2

        gp['radius_rgg'] = 0.25

        gp['alpha_kh'] = 10
        gp['beta_kh'] = 0.001
        gp['alpha_waxman'] = 0.25
        gp['beta_waxman'] = 0.5
        return gp
    
    @staticmethod
    def get_custom_generator_params(n):
        """Get generator params for a custom number of nodes."""
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = n
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

    @staticmethod
    def get_default_generator_params_tiny():
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = 10
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

    @staticmethod
    def get_default_generator_params():
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = 25
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

    @staticmethod
    def get_default_generator_params_med():
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = 50
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

    @staticmethod
    def get_default_generator_params_large():
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = 75
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

    @staticmethod
    def get_default_generator_params_xlarge():
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = 100
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

    @staticmethod
    def get_default_generator_params_xxlarge():
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = 125
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

    @staticmethod
    def get_default_generator_params_xxxlarge():
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = 150
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

    @staticmethod
    def get_default_generator_params_4xlarge():
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = 175
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

    @staticmethod
    def get_default_generator_params_5xlarge():
        gp = NetworkGenerator.get_common_generator_params()
        gp['n'] = 200
        gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
        return gp

class OrdinaryGraphGenerator(NetworkGenerator, ABC):
    def post_generate_instance(self, instance):
        state = RelnetGraph(instance)
        state.populate_banned_actions()
        return state


class GNMNetworkGenerator(OrdinaryGraphGenerator):
    name = 'random_network'
    num_tries = 10000

    def generate_instance_new(self, gen_params, random_seed):
        # This is a refactored, better version; but incompatible with NeurIPS submitted results.
        number_vertices = gen_params['n']
        number_edges = gen_params['m']

        if not self.enforce_connected:
            random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices, number_edges, seed=random_seed)
            return random_graph
        else:
            for try_num in range(1, self.num_tries + 1):
                random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices, number_edges,
                                                                            seed=((random_seed + 1) * try_num))
                if nx.is_connected(random_graph):
                    return random_graph
                else:
                    continue
            raise ValueError("Maximum number of tries exceeded, giving up...")

    # This was used to generate NeurIPS draft results.
    def generate_instance(self, gen_params, random_seed):
        number_vertices = gen_params['n']
        number_edges = gen_params['m']

        if not self.enforce_connected:
            random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices, number_edges, seed=random_seed)
            return random_graph
        else:
            for try_num in range(0, self.num_tries):
                random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices, number_edges,
                                                                            seed=(random_seed + (try_num * 1000)))
                if nx.is_connected(random_graph):
                    return random_graph
                else:
                    continue
            raise ValueError("Maximum number of tries exceeded, giving up...")

class BANetworkGenerator(OrdinaryGraphGenerator):
    name = 'barabasi_albert'

    def generate_instance(self, gen_params, random_seed):
        n, m = gen_params['n'], gen_params['m_ba']
        ba_graph = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=random_seed)
        return ba_graph


class WattsStrogatzNetworkGenerator(OrdinaryGraphGenerator):
    name = 'watts_strogatz'

    def generate_instance(self, gen_params, random_seed):
        n, m, p = gen_params['n'], gen_params['m_ws'], gen_params['p_ws']
        ws_graph = nx.generators.random_graphs.connected_watts_strogatz_graph(n, m, p, seed=random_seed)
        return ws_graph


class RegularNetworkGenerator(OrdinaryGraphGenerator):
    name = 'regular'
    num_tries = 10000

    def generate_instance(self, gen_params, random_seed):
        d, n = gen_params['d_reg'], gen_params['n']
        for try_num in range(0, self.num_tries):
            seed_used = random_seed + (try_num * 1000)
            reg_graph = nx.generators.random_graphs.random_regular_graph(d, n, seed=seed_used)
            if nx.is_connected(reg_graph):
                return reg_graph
            else:
                continue
        raise ValueError("Maximum number of tries exceeded, giving up...")


class StarNetworkGenerator(OrdinaryGraphGenerator):
    name = 'star'

    def generate_instance(self, gen_params, random_seed):
        n = gen_params['n']
        star_graph = nx.generators.classic.star_graph(n - 1)
        return star_graph

