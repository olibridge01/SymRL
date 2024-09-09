from copy import copy, deepcopy
from random import Random

import numpy as np
import scipy as sp

from relnet.agent.baseline.baseline_agent import MinCostAgent
from relnet.env.graph_edge_env import GraphEdgeEnv


class SimulationPolicy(object):
    def __init__(self, random_state, **kwargs):
        self.local_random = Random()
        self.local_random.setstate(random_state)

    def get_random_state(self):
        return self.local_random.getstate()

    def reset(self, tracked_edges):
        pass

class RandomSimulationPolicy(SimulationPolicy):
    def __init__(self,  start_state, start_budget, random_state, **kwargs):
        super().__init__(random_state)

    def choose_action(self, state, rem_budget, total_depth, possible_actions):
        available_acts = tuple(possible_actions)
        # print(f'Available actions: {available_acts}')

        # # Get distances to all possible actions
        # dists = [1 / state.compute_edge_length(act, state.current_node) for act in available_acts]
        # print(f'Dists: {dists}')
        # # Choose action weighted by distance
        # probs = sp.special.softmax(dists)
        # print(f'Probs: {probs}')
        # chosen_action = np.random.choice(available_acts, p=probs)
        # print(f'Chosen action: {chosen_action}')
        # print('-' * 50)

        chosen_action = self.local_random.choice(available_acts)
        return chosen_action

class MinCostSimulationPolicyFast(SimulationPolicy):
    DEFAULT_POLICY_BIAS = 1
    BASE_PROB_MODIFIER = 0.05

    def __init__(self, start_state, start_budget, random_state, **kwargs):
        super().__init__(random_state)
        self.bias = kwargs.pop('bias', self.DEFAULT_POLICY_BIAS)
        self.pairwise_dists = kwargs.pop('pairwise_dists', None)

        dists_matrix = self.pairwise_dists
        non_edges_list = self.find_all_allowed_edges(start_state, start_budget)
        self.non_edges = non_edges_list
        self.non_edges_range = np.arange(len(self.non_edges))

        non_edges_idxes = np.array( [list(ne) for ne in self.non_edges] )
        row_idxes = non_edges_idxes[:, 0].ravel()
        col_idxes = non_edges_idxes[:, 1].ravel()

        self.ne_dists = dists_matrix[row_idxes, col_idxes]
        scores = np.array(self.ne_dists, copy=True)

        max_dists_idxes = np.argwhere(scores == np.amax(scores))
        inv_dists = np.max(scores) - scores

        default_prob = self.BASE_PROB_MODIFIER * (1 / len(self.non_edges))
        inv_dists[max_dists_idxes] = default_prob

        biased_scores = np.power(inv_dists, self.bias)

        self.precomputed_scores_orig = biased_scores
        self.ne_to_pos_index = {ne: i for i, ne in enumerate(self.non_edges)}
        self.next_action = None

        self.build_other_nodes_edges_idx()


    def build_other_nodes_edges_idx(self):
        node_set = set()
        for ne in self.non_edges:
            node_set.add(ne[0])
            node_set.add(ne[1])

        self.other_nodes_edges_idx = {}
        for first_node in node_set:
            self.other_nodes_edges_idx[first_node] = [self.ne_to_pos_index[ne] for ne in self.non_edges if ne[0] != first_node]

    def reset(self, tracked_edges):
        self.next_action = None # Added to fix a potential bug
        self.precomputed_scores = np.array(self.precomputed_scores_orig, copy=True)
        for tracked_edge in tracked_edges:
            self.mark_edge_as_used(tracked_edge)

    def find_all_allowed_edges(self, start_state, start_budget):
        if start_state.first_node is None:
            state_ref = start_state
        else:
            state_cp = deepcopy(start_state)
            state_cp.first_node = None
            state_cp.populate_banned_actions(start_budget)
            state_ref = state_cp

        non_edges_orig = GraphEdgeEnv.get_graph_non_edges(state_ref, start_budget)
        non_edges_set = deepcopy(non_edges_orig)
        # for ne in non_edges_orig:
        #     edge_rev = ne[1], ne[0]
        #     if edge_rev not in non_edges_set:
        #         non_edges_set.add(edge_rev)
        non_edges_list = list(non_edges_set)

        return non_edges_list

    def get_indices_as_mask(self, start_state, non_edges):
        N = start_state.num_nodes
        mask_dims = (N, N)
        mask_array = np.ones(mask_dims, dtype=bool)
        flat_index_array = np.ravel_multi_index(non_edges, mask_array.shape)
        np.ravel(mask_array)[flat_index_array] = False
        return mask_array

    def remove_over_budget(self, rem_budget):
        self.precomputed_scores[self.ne_dists >= rem_budget] = 0.

    def choose_action(self, state, rem_budget, total_depth, possible_actions):
        if total_depth % 2 == 0:
            self.remove_over_budget(rem_budget)
            chosen_edge = self.sample_edge_by_dist()
            first_node = chosen_edge[0]
            second_node = chosen_edge[1]
            self.next_action = second_node

            self.mark_edge_as_used(chosen_edge)

            return first_node
        else:
            if self.next_action is None:
                first_node = state.first_node

                if first_node is None:
                    raise ValueError(f"first_node shouldn't be None!")

                self.remove_over_budget(rem_budget)
                chosen_edge = self.sample_edge_by_dist(first_node=first_node)
                self.mark_edge_as_used(chosen_edge)
                return chosen_edge[1]
            else:
                next_act = copy(self.next_action)
                self.next_action = None
                return next_act

    def mark_edge_as_used(self, chosen_edge):
        self.precomputed_scores[self.ne_to_pos_index[chosen_edge]] = 0.
        chosen_edge_rev = chosen_edge[1], chosen_edge[0]
        if chosen_edge_rev in self.ne_to_pos_index:
            self.precomputed_scores[self.ne_to_pos_index[chosen_edge_rev]] = 0.

    def sample_edge_by_dist(self, first_node=None):
        if first_node is None:
            scores = np.array(self.precomputed_scores, copy=False)
        else:
            scores = np.array(self.precomputed_scores, copy=True)
            scores[self.other_nodes_edges_idx[first_node]] = 0.

        probs = scores / np.sum(scores)
        chosen_edge_idx = np.random.choice(self.non_edges_range, p=probs)
        chosen_edge = self.non_edges[chosen_edge_idx]
        return chosen_edge



