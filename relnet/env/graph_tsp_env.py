from copy import deepcopy

import numpy as np

from relnet.state.graph_state import RelnetGraph, GeometricRelnetGraph
from relnet.state.network_generators import NetworkGenerator


class GraphRoutingEnv(object):
    """Graph environment for routing problems (currently just vanilla TSP)."""
    def __init__(self, objective_function, objective_function_kwargs):
        self.objective_function = objective_function
        self.original_objective_function_kwargs = objective_function_kwargs
        self.objective_function_kwargs = deepcopy(objective_function_kwargs)

        self.reward_eps = 1e-4
        self.reward_scale_multiplier = 1

        self.g_list = None

    def setup(self, g_list, initial_objective_function_values, training=False):
        """Set up the environment."""
        self.g_list = g_list
        self.n_steps = 0
        self.terminal = np.zeros(len(g_list), dtype=np.bool)

        for i in range(len(self.g_list)):
            g = g_list[i]
            g.first_node = None # Need to change this (this is syntax for SGC)
            # Don't compute restrictions?
            g.populate_banned_actions() # Need to change graph state functions for TSP
        
        self.training = training

        self.objective_function_values = np.zeros((2, len(self.g_list)), dtype=np.float)
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)
        self.rewards = np.zeros(len(g_list), dtype=np.float)

    def tear_down(self):
        """Tear down the environment."""
        if self.g_list is not None:
            self.g_list = None
            self.terminal = None

            self.training = None

            self.objective_function_values = None
            self.objective_function_kwargs = None
            self.rewards = None

    def pass_logger_instance(self, logger):
        """Pass the logger instance to the environment."""
        self.logger_instance = logger

    def get_final_values(self):
        """Get the final objective function values."""
        return self.objective_function_values[-1, :]

    def get_initial_values(self):
        """Get the initial objective function values."""
        return self.objective_function_values[0, :]
    
    def get_objective_function_value(self, s2v_graph):
        """Compute the objective function for a given graph."""
        obj_function_value = self.objective_function.compute(s2v_graph, **self.objective_function_kwargs)
        return obj_function_value
    
    def get_objective_function_values(self, s2v_graphs):
        """Compute the objective function for a list of graphs."""
        return np.array([self.get_objective_function_value(g) for g in s2v_graphs])
    
    @staticmethod
    def get_valid_actions(g, banned_actions):
        """Get valid actions (i.e. unvisited nodes) for given graph."""
        all_nodes_set = g.all_nodes_set
        valid_first_nodes = all_nodes_set - banned_actions

        if len(valid_first_nodes) == 0 and not g.edge_exists(g.current_node, g.starting_node):
            valid_first_nodes.add(g.starting_node)

        return valid_first_nodes
    
    @staticmethod
    def apply_action(g, action):
        """Apply selected action to a given graph."""
        # Add new edge to route
        new_g, edge_cost = g.add_edge(g.current_node, action)
        new_g.current_node = action # Add current_node attribute to graph state

        new_g.populate_banned_actions() # Need to change graph state functions for TSP
        return new_g, edge_cost
    
    @staticmethod
    def apply_action_in_place(g, action):
        """Apply selected action to a given graph in place."""
        # Add new edge to route
        edge_cost = g.add_edge_dynamically(g.current_node, action)
        g.current_node = action

        g.populate_banned_actions() # Need to change graph state functions for TSP
        return edge_cost
    
    def step(self, actions):
        """Take a step in the environment."""
        for i in range(len(self.g_list)):

            # SGC has some code here about using dummy action when budgets not exhausted. Extend to TSP?

            self.g_list[i], edge_cost = self.apply_action(self.g_list[i], actions[i])

            if self.g_list[i].banned_actions == self.g_list[i].all_nodes_set:
                current_node = self.g_list[i].current_node
                starting_node = self.g_list[i].starting_node
                final_edge_exists = self.g_list[i].edge_exists(current_node, starting_node)

                # Only mark exhausted if tour has been completed
                if final_edge_exists:
                    self.mark_exhausted(i)
       
        self.n_steps += 1
    
    def mark_exhausted(self, i):
        """Mark the graph as exhausted (tour completed)."""
        self.terminal[i] = True
        objective_function_value = self.get_objective_function_value(self.g_list[i])
        if self.training:
            objective_function_value = self.reward_scale_multiplier * objective_function_value
        self.objective_function_values[-1, i] = objective_function_value

        self.rewards[i] = objective_function_value
    
    def exploratory_actions(self, agent_exploration_policy):
        """Get exploratory actions for each graph in the list."""
        act_list_t0, act_list_t1 = [], []
        for i in range(len(self.g_list)):
            first_node, second_node = agent_exploration_policy(i)

            act_list_t0.append(first_node)
            act_list_t1.append(second_node)

        return act_list_t0, act_list_t1
    
    def get_max_graph_size(self):
        """Get the size of largest graph in the list."""
        max_graph_size = np.max([g.num_nodes for g in self.g_list])
        return max_graph_size

    def is_terminal(self):
        """Check if all graphs are terminal (tours all complete)."""
        return np.all(self.terminal)
    
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