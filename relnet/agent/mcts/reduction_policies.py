import math
from abc import ABC, abstractmethod
from copy import copy, deepcopy

import networkx as nx
import numpy as np

from relnet.state.graph_state import GeometricRelnetGraph


class ReductionPolicy(ABC):
    def __init__(self, ar_fun, ar_modifier, **kwargs):
        self.ar_fun = ar_fun
        self.ar_modifier = ar_modifier

    def get_max_allowed_acts(self, g):
        N = g.num_nodes
        if self.ar_fun == 'off':
            return N
        if self.ar_fun == 'sqrt':
            max_acts = math.ceil(math.sqrt(self.ar_modifier * N))
        elif self.ar_fun == 'percentage':
            max_acts = math.ceil((self.ar_modifier * N) / 100)
        else:
            raise ValueError(f"unknown action reduction method {self.ar_fun}")

        return max_acts

    def get_state_max_acts(self, start_state, valid_acts):
        num_acts = len(valid_acts)
        max_acts = self.get_max_allowed_acts(start_state)
        subset_n = min(max_acts, num_acts)
        return subset_n

    def reduce_valid_actions(self, start_state, valid_acts, i):
        subset_n = self.get_state_max_acts(start_state, valid_acts)

        valid_acts_list = list(valid_acts)
        subset = self.apply_reduction_policy(start_state, valid_acts_list, subset_n, i)
        # print(f"Reducing to subset {subset}")
        return subset

    def extract_policy_data(self, env, g_list, initial_obj_values):
        pass

    @abstractmethod
    def apply_reduction_policy(self, start_state, valid_acts, n, i):
        pass

    @staticmethod
    def get_reduction_policy_instance(policy_name, ar_fun, ar_modifier, **kwargs):
        if policy_name == DummyReductionPolicy.policy_name:
            return DummyReductionPolicy('off', -1, **kwargs)
        elif policy_name == RandomReductionPolicy.policy_name:
            return RandomReductionPolicy(ar_fun, ar_modifier, **kwargs)
        elif policy_name == DegreeReductionPolicy.policy_name:
            return DegreeReductionPolicy(ar_fun, ar_modifier, **kwargs)
        elif policy_name == InvDegreeReductionPolicy.policy_name:
            return InvDegreeReductionPolicy(ar_fun, ar_modifier, **kwargs)
        elif policy_name == LBHBReductionPolicy.policy_name:
            return LBHBReductionPolicy(ar_fun, ar_modifier, **kwargs)
        elif policy_name == NumConnsReductionPolicy.policy_name:
            return NumConnsReductionPolicy(ar_fun, ar_modifier, **kwargs)
        elif policy_name == BestEdgeReductionPolicy.policy_name:
            return BestEdgeReductionPolicy(ar_fun, ar_modifier, **kwargs)
        elif policy_name == BestEdgeCSReductionPolicy.policy_name:
            return BestEdgeCSReductionPolicy(ar_fun, ar_modifier, **kwargs)
        elif policy_name == AvgEdgeReductionPolicy.policy_name:
            return AvgEdgeReductionPolicy(ar_fun, ar_modifier, **kwargs)
        elif policy_name == AvgEdgeCSReductionPolicy.policy_name:
            return AvgEdgeCSReductionPolicy(ar_fun, ar_modifier, **kwargs)
        else:
            raise ValueError(f"unknown reduction policy {policy_name}")

    def partition_n_lowest(self, valid_acts, n, partition_value):
        partition_idxes = list(np.argpartition(partition_value, n)[:n])
        chosen_acts = {valid_acts[idx] for idx in partition_idxes}
        return chosen_acts


class DummyReductionPolicy(ReductionPolicy):
    policy_name = 'dummy'

    def apply_reduction_policy(self, start_state, valid_acts, n, i):
        return valid_acts


class RandomReductionPolicy(ReductionPolicy):
    policy_name = 'random'

    def __init__(self, ar_fun, ar_modifier, **kwargs):
        super().__init__(ar_fun, ar_modifier, **kwargs)

    def apply_reduction_policy(self, start_state, valid_acts, n, i):
        rand_acts = set(np.random.choice(tuple(valid_acts), n, replace=False))
        return rand_acts


class DegreeReductionPolicy(ReductionPolicy):
    policy_name = 'degree'

    def __init__(self, ar_fun, ar_modifier, **kwargs):
        super().__init__(ar_fun, ar_modifier, **kwargs)

    def apply_reduction_policy(self, start_state, valid_acts, n, i):
        degrees = np.array([start_state.node_degrees[fa] for fa in valid_acts])
        degrees_rev = np.max(degrees) - degrees

        chosen_acts = self.partition_n_lowest(valid_acts, n, degrees_rev)
        return chosen_acts


class InvDegreeReductionPolicy(ReductionPolicy):
    policy_name = 'invdegree'

    def __init__(self, ar_fun, ar_modifier, **kwargs):
        super().__init__(ar_fun, ar_modifier, **kwargs)

    def apply_reduction_policy(self, start_state, valid_acts, n, i):
        degrees = np.array([start_state.node_degrees[fa] for fa in valid_acts])

        chosen_acts = self.partition_n_lowest(valid_acts, n, degrees)
        return chosen_acts

class LBHBReductionPolicy(ReductionPolicy):
    policy_name = 'lbhb'

    def __init__(self, ar_fun, ar_modifier, **kwargs):
        super().__init__(ar_fun, ar_modifier, **kwargs)

    def apply_reduction_policy(self, start_state, valid_acts, n, i):
        n_high = math.ceil(n / 2)
        n_low = math.floor(n / 2)

        G = start_state.to_networkx()
        nx.set_edge_attributes(G, start_state.edge_lengths, "weight")
        betweeness = nx.algorithms.centrality.betweenness_centrality(G, weight="weight")
        betweeness_arr = np.array([betweeness[fa] for fa in valid_acts])
        min_betweeness_arr = 1 - betweeness_arr

        low_b_nodes = self.partition_n_lowest(valid_acts, n_low, betweeness_arr)
        high_b_nodes = self.partition_n_lowest(valid_acts, n_high, min_betweeness_arr)

        assert(len(low_b_nodes.intersection(high_b_nodes)) == 0)

        chosen_acts = low_b_nodes.union(high_b_nodes)
        return chosen_acts

class NumConnsReductionPolicy(ReductionPolicy):
    policy_name = 'numconns'

    def __init__(self, ar_fun, ar_modifier, **kwargs):
        super().__init__(ar_fun, ar_modifier, **kwargs)

    def apply_reduction_policy(self, start_state, valid_acts, n, i):
        num_conns = np.array(
            [len(self.initial_allowed_cons[i][fa]) if fa in self.initial_allowed_cons[i] else 0 for fa in
             valid_acts])
        num_conns_rev = np.max(num_conns) - num_conns

        chosen_acts = self.partition_n_lowest(valid_acts, n, num_conns_rev)

        # print(f"chosen acts had values {[len(self.initial_allowed_cons[i][a]) for a in chosen_acts]}")
        return chosen_acts

    def extract_policy_data(self, env, g_list, initial_obj_values):
        env_ref = copy(env)
        env_ref.logger_instance = None
        env_ref.setup([g.copy() for g in g_list],
                      [g_initial_obj_value.copy() for g_initial_obj_value in initial_obj_values],
                      training=False)

        initial_allowed_conns = []

        for i in range(len(g_list)):
            g_conns = deepcopy(env_ref.g_list[i].allowed_connections)
            initial_allowed_conns.append(g_conns)

        self.initial_allowed_cons = initial_allowed_conns
        env_ref.tear_down()

class EdgeReductionPolicy(ReductionPolicy, ABC):
    def extract_policy_data(self, env, g_list, initial_obj_values):

        env_ref = copy(env)
        env_ref.logger_instance = None
        env_ref.setup([g.copy() for g in g_list],
                      [g_initial_obj_value.copy() for g_initial_obj_value in initial_obj_values],
                      training=False)

        initial_best_node_vals = []

        for i in range(len(g_list)):
            g = env_ref.g_list[i]
            init_g_F = env_ref.objective_function_values[0, i]
            non_edges = list(env_ref.get_graph_non_edges_for_idx(i))

            node_vals = {act: [] for act in g.node_labels}

            for first, second in non_edges:
                g_copy = g.copy()
                next_g, _ = g_copy.add_edge(first, second)
                next_g_F = env_ref.get_objective_function_value(next_g)

                if not self.is_cost_sensitive:
                    edge_val = next_g_F - init_g_F
                else:
                    cost_spent = GeometricRelnetGraph.get_cost_spent(g, next_g)
                    edge_val = (next_g_F - init_g_F) / cost_spent

                node_vals[first].append(edge_val)
                node_vals[second].append(edge_val)


            agg_fun = np.max if self.policy_name.startswith("best") else np.mean
            node_best_vals = {act: agg_fun(node_vals[act]) for act in node_vals.keys() if len(node_vals[act]) > 0}
            initial_best_node_vals.append(node_best_vals)

        self.initial_best_node_vals = initial_best_node_vals
        env_ref.tear_down()

    def apply_reduction_policy(self, start_state, valid_acts, n, i):
        nv = self.initial_best_node_vals[i]
        node_vals = np.array([nv[fa] for fa in valid_acts])
        node_vals_rev = np.max(node_vals) - node_vals

        chosen_acts = self.partition_n_lowest(valid_acts, n, node_vals_rev)
        # print(f"chosen acts had values {[nv[a] for a in chosen_acts]}")
        return chosen_acts


class BestEdgeReductionPolicy(EdgeReductionPolicy):
    policy_name = 'best_edge'
    is_cost_sensitive = False

    def __init__(self, ar_fun, ar_modifier, **kwargs):
        super().__init__(ar_fun, ar_modifier, **kwargs)


class BestEdgeCSReductionPolicy(EdgeReductionPolicy):
    policy_name = 'best_edge_cs'
    is_cost_sensitive = True

    def __init__(self, ar_fun, ar_modifier, **kwargs):
        super().__init__(ar_fun, ar_modifier, **kwargs)


class AvgEdgeReductionPolicy(EdgeReductionPolicy):
    policy_name = 'avg_edge'
    is_cost_sensitive = False

    def __init__(self, ar_fun, ar_modifier, **kwargs):
        super().__init__(ar_fun, ar_modifier, **kwargs)


class AvgEdgeCSReductionPolicy(EdgeReductionPolicy):
    policy_name = 'avg_edge_cs'
    is_cost_sensitive = True

    def __init__(self, ar_fun, ar_modifier, **kwargs):
        super().__init__(ar_fun, ar_modifier, **kwargs)