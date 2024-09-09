from copy import deepcopy

import numpy as np

from relnet.state.graph_state import RelnetGraph, GeometricRelnetGraph
from relnet.state.network_generators import NetworkGenerator


class GraphEdgeEnv(object):
    def __init__(self, objective_function, objective_function_kwargs,
                 edge_budget_percentage,
                 conn_radius_modifier=1,
                 restriction_mechanism='max_current'):
        """Spatial Graph Construction (SGC) environment class."""
        self.objective_function = objective_function
        self.original_objective_function_kwargs = objective_function_kwargs
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)

        self.edge_budget_percentage = edge_budget_percentage
        self.conn_radius_modifier = conn_radius_modifier
        self.restriction_mechanism = restriction_mechanism

        self.num_mdp_substeps = 2

        self.reward_eps = 1e-4
        self.reward_scale_multiplier = 1

        self.g_list = None


    def setup(self, g_list, initial_objective_function_values, training=False):
        """
        Set up the environment with a list of graphs and initial objective function values.
        
        Args:
        - g_list (list): list of graphs to be used in the environment.
        - initial_objective_function_values (np.ndarray): initial objective function values for each graph.
        - training (bool): whether the environment is being used for training or not.
        """
        self.g_list = g_list
        self.n_steps = 0

        self.edge_budgets = self.compute_edge_budgets(self.g_list, self.edge_budget_percentage)
        self.used_edge_budgets = np.zeros(len(g_list), dtype=np.float)
        self.exhausted_budgets = np.zeros(len(g_list), dtype=np.bool)

        for i in range(len(self.g_list)):
            g = g_list[i]
            g.first_node = None
            g.compute_restrictions(self.restriction_mechanism, self.conn_radius_modifier)
            g.populate_banned_actions(self.edge_budgets[i])

        self.training = training

        self.objective_function_values = np.zeros((2, len(self.g_list)), dtype=np.float)
        self.objective_function_values[0, :] = initial_objective_function_values
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)
        self.rewards = np.zeros(len(g_list), dtype=np.float)

        if self.training:
            self.objective_function_values[0, :] = np.multiply(self.objective_function_values[0, :], self.reward_scale_multiplier)

    def tear_down(self):
        """Tear down the environment."""
        if self.g_list is not None:
            self.g_list = None
            self.edge_budgets = None
            self.used_edge_budgets = None
            self.exhausted_budgets = None

            self.training = None

            self.objective_function_values = None
            # self.objective_function_kwargs = None
            self.rewards = None
    
    def pass_logger_instance(self, logger):
        """Pass a logger instance to the environment."""
        self.logger_instance = logger

    # Retrieving objective function values
    def get_final_values(self):
        """Get the final objective function values for each graph."""
        return self.objective_function_values[-1, :]

    def get_initial_values(self):
        """Get the initial objective function values for each graph."""
        return self.objective_function_values[0, :]

    def get_objective_function_value(self, graph):
        """Get the objective function value for a given graph."""
        obj_function_value = self.objective_function.compute(graph, **self.objective_function_kwargs)
        return obj_function_value

    def get_objective_function_values(self, graphs):
        """Get the objective function values for a list of graphs."""
        return np.array([self.get_objective_function_value(g) for g in graphs])

    def get_graph_non_edges_for_idx(self, i):
        """Get the non-edges for a given graph index."""
        g = self.g_list[i]
        budget = self.get_remaining_budget(i)
        return self.get_graph_non_edges(g, budget)

    @staticmethod
    def get_graph_non_edges(g, budget):
        banned_first_nodes = g.banned_actions
        valid_acts = GraphEdgeEnv.get_valid_actions(g, banned_first_nodes)
        non_edges = set()
        for first in valid_acts:
            banned_second_nodes = g.get_invalid_edge_ends(first, budget)
            valid_second_nodes = GraphEdgeEnv.get_valid_actions(g, banned_second_nodes)

            for second in valid_second_nodes:
                non_edges.add((first, second))
        return non_edges

    @staticmethod
    def get_graph_non_edges_first_picked(g, budget):
        first = g.first_node
        non_edges = set()
        banned_second_nodes = g.get_invalid_edge_ends(first, budget)
        valid_second_nodes = GraphEdgeEnv.get_valid_actions(g, banned_second_nodes)

        for second in valid_second_nodes:
            non_edges.add((first, second))
        return non_edges

    def get_remaining_budget(self, i):
        return self.edge_budgets[i] - self.used_edge_budgets[i]

    @staticmethod
    def compute_edge_budgets(g_list, edge_budget_percentage):
        """Compute the initial edge budget for each graph in the list."""
        edge_budgets = np.zeros(len(g_list), dtype=np.float)

        # Budget is a percentage of the total edge length/number of edges in the graph
        for i in range(len(g_list)):
            g = g_list[i]
            if type(g) == RelnetGraph:
                n = g.num_nodes
                edge_budgets[i] = NetworkGenerator.compute_number_edges(n, edge_budget_percentage)
            elif type(g) == GeometricRelnetGraph:
                total_edge_length = sum(g.edge_lengths.values())
                budget = (total_edge_length * edge_budget_percentage) / 100

                if budget == 0:
                    budget = 1

                edge_budgets[i] = budget
            else:
                raise ValueError(f"Unrecognized graph class {type(g)}!")

        return edge_budgets


    @staticmethod
    def get_valid_actions(g, banned_actions):
        """Retrieve the valid actions for a given graph."""
        all_nodes_set = g.all_nodes_set
        valid_first_nodes = all_nodes_set - banned_actions
        return valid_first_nodes

    @staticmethod
    def apply_action(g, action, remaining_budget, copy_state=False):
        """Apply an action to a graph."""
        if g.first_node is None:
            if copy_state:
                g_ref = g.copy()
            else:
                g_ref = g
            g_ref.first_node = action
            g_ref.populate_banned_actions(remaining_budget)
            return g_ref, remaining_budget
        else:
            new_g, edge_cost = g.add_edge(g.first_node, action)
            new_g.first_node = None

            updated_budget = remaining_budget - edge_cost
            new_g.populate_banned_actions(updated_budget)
            return new_g, updated_budget

    @staticmethod
    def apply_action_in_place(g, action, remaining_budget):
        """Apply an action to a graph in place."""
        if g.first_node is None:
            g.first_node = action
            g.populate_banned_actions(remaining_budget)
            return remaining_budget
        else:
            edge_cost = g.add_edge_dynamically(g.first_node, action)
            g.first_node = None

            updated_budget = remaining_budget - edge_cost
            g.populate_banned_actions(updated_budget)
            return updated_budget

    def step(self, actions):
        """Step through the environment."""
        for i in range(len(self.g_list)):
            if not self.exhausted_budgets[i]:
                if actions[i] == -1:
                    if self.logger_instance is not None:
                        self.logger_instance.warn("budget not exhausted but trying to apply dummy action!")
                        self.logger_instance.error(f"the remaining budget: {self.get_remaining_budget(i)}")
                        g = self.g_list[i]

                        self.logger_instance.error(f"first_node selection: {g.first_node}")

                        if type(g) == GeometricRelnetGraph:
                            self.logger_instance.error("graph is geometric.")
                            self.logger_instance.error(f"state allowed connections: {g.allowed_connections}")
                            self.logger_instance.error(f"state shortest allowed connection: {g.shortest_allowed_connection}")
                        else:
                            self.logger_instance.error("graph is not geometric.")

                remaining_budget = self.get_remaining_budget(i)
                self.g_list[i], updated_budget = self.apply_action(self.g_list[i], actions[i], remaining_budget)
                self.used_edge_budgets[i] += (remaining_budget - updated_budget)

                if self.n_steps % 2 == 1:
                    if self.g_list[i].banned_actions == self.g_list[i].all_nodes_set:
                        self.mark_exhausted(i)

        self.n_steps += 1

    def mark_exhausted(self, i):
        """Mark the edge budget for a given graph as exhausted."""
        self.exhausted_budgets[i] = True
        objective_function_value = self.get_objective_function_value(self.g_list[i])
        if self.training:
            objective_function_value = objective_function_value * self.reward_scale_multiplier
        self.objective_function_values[-1, i] = objective_function_value
        reward = objective_function_value - self.objective_function_values[0, i]
        if abs(reward) < self.reward_eps:
            reward = 0.
        self.rewards[i] = reward

    def exploratory_actions(self, agent_exploration_policy):
        act_list_t0, act_list_t1 = [], []
        for i in range(len(self.g_list)):
            first_node, second_node = agent_exploration_policy(i)

            act_list_t0.append(first_node)
            act_list_t1.append(second_node)

        return act_list_t0, act_list_t1

    def get_max_graph_size(self):
        max_graph_size = np.max([g.num_nodes for g in self.g_list])
        return max_graph_size

    def is_terminal(self):
        return np.all(self.exhausted_budgets)

    def get_state_ref(self):
        cp_first = [g.first_node for g in self.g_list]
        b_list = [g.banned_actions for g in self.g_list]
        return zip(self.g_list, cp_first, b_list)

    def clone_state(self, indices=None):
        if indices is None:
            cp_first = [g.first_node for g in self.g_list][:]
            b_list = [g.banned_actions for g in self.g_list][:]
            return list(zip(deepcopy(self.g_list), cp_first, b_list))
        else:
            cp_g_list = []
            cp_first = []
            b_list = []

            for i in indices:
                cp_g_list.append(deepcopy(self.g_list[i]))
                cp_first.append(deepcopy(self.g_list[i].first_node))
                b_list.append(deepcopy(self.g_list[i].banned_actions))

            return list(zip(cp_g_list, cp_first, b_list))
        
    def get_num_mdp_substeps(self):
        return 1
    
    @staticmethod
    def from_env_instance(env_instance):
        new_instance = GraphEdgeEnv(deepcopy(env_instance.objective_function),
                                   deepcopy(env_instance.objective_function_kwargs),
                                   deepcopy(env_instance.edge_budget_percentage),
                                   deepcopy(env_instance.conn_radius_modifier),
                                   deepcopy(env_instance.restriction_mechanism)
                                   )
        return new_instance
    
    def get_num_node_feats(self):
        return 5

    def get_num_edge_feats(self):
        return 0