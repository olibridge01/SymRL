import gc
import warnings
from math import sqrt, log
from pathlib import Path

import matplotlib.pyplot as plt
import network2tikz
import scipy as sp
from billiard.pool import Pool
from networkx.drawing.nx_agraph import graphviz_layout

from relnet.agent.base_agent import Agent
from relnet.agent.mcts.mcts_tree_node import MCTSTreeNode, TranspositionTreeNode
from relnet.agent.mcts.reduction_policies import *
from relnet.agent.mcts.simulation_policies import RandomSimulationPolicy, MinCostSimulationPolicyFast
from relnet.eval.eval_utils import *
from relnet.state.graph_state import get_graph_hash, get_node_hash, budget_eps


class MonteCarloTreeSearchAgent(Agent):
    is_deterministic = False
    is_trainable = False

    def __init__(self, environment):
        super().__init__(environment)
        self.draw_trees = False
        self.root_nodes = None

        self.final_action_strategies = {'max_child': self.pick_max_child,
                                        'robust_child': self.pick_robust_child,
                                        }

    def init_root_information(self, t, training):
        """From timestep t, initialize root nodes starting from current graph states."""
        self.root_nodes = []
        self.fa_subsets = []
        self.node_expansion_budgets = []

        if self.sim_policy.startswith('min_cost'):
            self.pairwise_dists = []


        for i in range(len(self.environment.g_list)):
            start_state = self.environment.g_list[i]
            remaining_budget = self.environment.get_remaining_budget(i)

            fa_subset = self.get_first_actions_subset(start_state, i, training)
            self.node_hashmap = {} # Clear node hashmap
            root_node, _ = self.initialize_tree_node(None, start_state, None, remaining_budget, fa_subset, training, with_depth=t)
            exp_budget = int(start_state.num_nodes * self.expansion_budget_modifier)

            self.root_nodes.append(root_node)
            self.fa_subsets.append(fa_subset)
            self.node_expansion_budgets.append(exp_budget)
            if self.sim_policy == 'min_cost':
                self.pairwise_dists.append(sp.spatial.distance.squareform(start_state.get_all_pairwise_distances()))


    def make_actions(self, t, **kwargs):
        raise ValueError("method not supported -- should call eval using overriden method")
        force_init = kwargs['force_init'] if 'force_init' in kwargs else False
        training = kwargs['training'] if 'training' in kwargs else False
        #self.run_search_for_g_list(t, force_init=force_init, training=training)
        self.run_search_for_g_list(t, force_init=True, training=training)
        return self.pick_children()

    def eval(self, g_list,
             initial_obj_values=None,
             validation=False,
             make_action_kwargs=None):
        """Evaluate the agent on a list of graphs and return the average reward."""

        parallel_tasks = []
        env_ref = copy(self.environment)
        env_ref.logger_instance = None
        env_ref.tear_down()

        trajectories = []
        for i, g in enumerate(g_list):
            starting_graph = g.copy()
            starting_graph_initial_obj_value = env_ref.get_objective_function_value(starting_graph)
            
            opts_copy = copy(self.options)
            opts_copy['log_tf_summaries'] = False
            opts_copy['random_seed'] = (i + 1) * (self.random_seed + 1)
            hyps_copy = copy(self.hyperparams)

            if not self.parallel_eval:
                trajectory = self.get_trajectory_for_graph(self.__class__,
                                                           env_ref,
                                                           0,
                                                           None,
                                                           hyps_copy,
                                                           opts_copy,
                                                           starting_graph,
                                                           starting_graph_initial_obj_value,
                                                           )
                trajectories.append(trajectory)

            parallel_tasks.append((self.__class__,
                                   env_ref,
                                   0,
                                   None,
                                   hyps_copy,
                                   opts_copy,
                                   starting_graph,
                                   starting_graph_initial_obj_value,
                                   ))

        if self.parallel_eval:
            for local_trajectory in self.eval_pool.starmap(self.get_trajectory_for_graph, parallel_tasks):
                trajectories.append(local_trajectory)

        env_ref.tear_down()
        # rewards = [t[2] for t in trajectories]
        print(f'Best trajectory: {[t[1] for t in trajectories]}')
        # return float(np.mean(rewards))
        return trajectories

    def run_search_for_g_list(self, t, force_init=True, training=False):
        """Run Monte Carlo Tree Search for a list of graphs at timestep t."""
        if t == 0:
            self.moves_so_far = []
            self.best_trajectories_found = []
            self.best_Rs = []

            self.C_ps = []
            self.starting_budgets = []
            self.rollout_limits = []

            self.reduction_policy.extract_policy_data(self.environment, 
                                                      self.environment.g_list, 
                                                      self.environment.get_objective_function_values(self.environment.g_list))
            self.reduced_actions = []

            for i in range(len(self.environment.g_list)):
                g = self.environment.g_list[i]

                self.moves_so_far.append([])
                self.best_trajectories_found.append([])
                self.best_Rs.append(float("-inf"))

                starting_budget = self.environment.get_remaining_budget(i)
                self.starting_budgets.append(starting_budget)

                if self.hyperparams['rollout_depth'] == -1:
                    self.rollout_limits.append(starting_budget)
                else:
                    rollout_limit = math.ceil(starting_budget * self.hyperparams['rollout_depth'])
                    self.rollout_limits.append(rollout_limit)
                self.C_ps.append(self.hyperparams['C_p'])

                valid_acts = self.environment.get_valid_actions(g, g.banned_actions)
                self.reduced_actions.append(self.reduction_policy.reduce_valid_actions(g, valid_acts, i))

            if self.hyperparams['adjust_C_p']:
                self.init_root_information(t, training)
                for i in range(len(self.root_nodes)):
                    self.execute_search_step(i, t, training)

        if t == 0 or force_init:
            self.init_root_information(t, training)
        for i in range(len(self.root_nodes)):
            self.execute_search_step(i, t, training)

    def execute_search_step(self, i, t, training):
        """Perform search step for a single graph at timestep t."""
        # print(f"Executing step {t}...")
        self.num_node_updates = 0
        self.tree_depth = []
        root_node = self.root_nodes[i]
        # print(f'Root node tracked_edges before: {root_node.state.tracked_edges}')
        root_node.state.start_edge_tracking()
        # print(f'Root node tracked_edges after: {root_node.state.tracked_edges}')

        fa_subset = self.fa_subsets[i]
        node_expansion_budget = self.node_expansion_budgets[i]

        starting_budget = self.starting_budgets[i]
        rollout_limit = self.rollout_limits[i]

        ## init sim policy here, actually...
        if self.sim_policy == 'random':
            sim_policy_class = RandomSimulationPolicy
            sim_policy_kwargs = {}

        elif self.sim_policy == 'min_cost':
            sim_policy_class = MinCostSimulationPolicyFast
            sim_policy_kwargs = {"bias": self.hyperparams['sim_policy_bias'],
                                 "pairwise_dists": self.pairwise_dists[i]
                                 }

        else:
            raise ValueError(f"sim policy {self.sim_policy} not recognised!")

        # _, rem_budget = MonteCarloTreeSearchAgent.find_budget_at_leaf(root_node, rollout_limit, root_node)
        self.sim_policy_inst = sim_policy_class(root_node.state, starting_budget, self.local_random.getstate(), **sim_policy_kwargs)

        hit_terminal_depth = False
        self.expansion_N = 0

        while True:
            # follow tree policy to reach a certain node
            tree_nodes, tree_actions = self.follow_tree_policy(root_node, i, fa_subset, training)
            self.tree_depth.append(len(tree_nodes))
            # For debugging - printing tree state and values
            # print('-' * 50)
            # for j, node in enumerate(tree_nodes):
            #     if j == 0:
            #         print(f'Root node: {node}')
            #         print(f'First node: {node.state.first_node}')
            #     else:
            #         print(f'Tree node: {node}')
            #         print(f"First node: {node.state.first_node}")
            #         print(f'Action: {tree_actions[j - 1]}')
            #     # print(f"Node Q: {node.Q}")
            #     # print(f"Node N: {node.N}")
            # print(f'tree_actions: {tree_actions}')
            # print('-' * 50)

            # if root_node.N == 3500:
            #     print(node.state.convert_edges())
            #     print('')
            #     for item in node.state.allowed_connections.items():
            #         print(item)


            if len(tree_actions) == 0:
                hit_terminal_depth = True

            v_l = tree_nodes[-1]
            simulation_results = self.execute_simulation_policy(v_l, root_node, i, fa_subset, starting_budget, rollout_limit)
            self.obj_fun_eval_count += 1

            # NEED A FUNCTION TO GET ALL TREE NODES TO BACKUP
            if self.transposition:
                # valid_nodes, tree_nodes_to_backup = self.get_all_nodes_to_backup(v_l, root_node)
                tree_nodes_to_backup = [tree_nodes[::-1]]
                # print(f'Backing up {len(tree_nodes_to_backup)} nodes. Specific path has {len(tree_nodes)} nodes.')
            else:
                tree_nodes_to_backup = tree_nodes

            # print(f'Backing up {len(valid_nodes)} distinct nodes. Backing up {len(tree_nodes_to_backup)} paths. Specific path has {len(tree_nodes)} nodes.')
            self.backup_values(tree_nodes_to_backup, tree_actions, simulation_results)
            self.update_best_trajectories(i, t, tree_actions, simulation_results)

            # Debugging
            # print(f'Current root node N: {root_node.N}')
            self.expansion_N += 1


            if self.expansion_N >= node_expansion_budget:
                root_Q = root_node.Q
                if self.hyperparams['adjust_C_p']:
                    self.C_ps[i] = self.hyperparams['C_p'] * root_Q
                break
        # print(f"picked action {action} using strategy {self.final_action_strategy}.")
        # print(f"prior to executing, we had {root_node.state.moves_budget} moves left.")
        if self.draw_trees:
            if hit_terminal_depth:
                self.draw_search_tree_with_values(i, root_node, t, max_depth=1, drawing_type=self.drawing_type)
            else:
                self.draw_search_tree_with_values(i, root_node, t, drawing_type=self.drawing_type)

    def get_all_nodes_to_backup(self, leaf_node, root_node):
        """
        Perform depth-first search from leaf node to root node, collecting all nodes to backup.
        """
        def dfs(node, path, all_paths, current_path_nodes):
            path.append(node)
            current_path_nodes.add(node)
            if node == root_node:
                all_paths.append(path.copy())
                for n in path:
                    valid_path_nodes.add(n)
            elif node.depth - root_node.depth >= 1:
                for parent in node.parent_nodes:
                    dfs(parent, path, all_paths, current_path_nodes)
            path.pop()
            current_path_nodes.remove(node)
        
        all_paths = []
        valid_path_nodes = set()
        dfs(leaf_node, [], all_paths, set())
        return list(valid_path_nodes), all_paths

    def pick_children(self):
        """Pick the children of the root nodes based on accumulated Q-value statistics."""
        actions = []
        for i in range(len(self.root_nodes)):
            root_node = self.root_nodes[i]
            
            # Debgging - print root_node's children
            # print(f"Root node's children: {root_node.children}")

            if len(root_node.children) > 0:
                action, selected_child = self.final_action_strategy(root_node)
                # print(f'Selected child: {selected_child}')
                # print(f"Selected child's children: {selected_child.children}")
                self.root_nodes[i] = selected_child
                self.root_nodes[i].parent_node = None
                del root_node
                gc.collect()
            else:
                # raise ValueError("Hello!")
                self.environment.mark_exhausted(i)
                action = -1

            self.moves_so_far[i].append(action)
            actions.append(action)
        
        self.tree_depths.append(np.mean(self.tree_depth))
        self.updates.append(self.num_node_updates)
        # print(f'Picked actions: {actions}. Num node updates: {self.num_node_updates}')
        # print(f'Average tree depth: {np.mean(self.tree_depth)}')
        return actions

    def pick_robust_child(self, root_node):
        """Pick the child node with highest N and Q values."""
        # print(f"there are {len(root_node.children)} possible actions")
        # print(f"there have been {root_node.state.number_actions_taken} actions so far.")
        # print(f"move budget is {root_node.state.moves_budget}.")
        # print(f'Root node children: {root_node.children}')
        return sorted(root_node.children.items(), key=lambda x: (x[1].N, x[1].Q), reverse=True)[0]


    def pick_max_child(self, root_node):
        return sorted(root_node.children.items(), key=lambda x: (x[1].Q, x[1].N), reverse=True)[0]

    def follow_tree_policy(self, node, i, fa_subset, training):
        """Follow the tree policy to reach a leaf node. If a node is not fully expanded, expand it."""
        traversed_nodes = []
        actions_taken = []

        # print(node.state.first_node)

        if node.num_valid_actions == 0:
            traversed_nodes.append(node)
            return traversed_nodes, actions_taken

        while True:
            traversed_nodes.append(node)
            state = node.state

            if len(node.children) < node.num_valid_actions:
                if hasattr(self, 'step'):
                    global_step = self.step
                else:
                    global_step = 1

                chosen_action = node.choose_action(int(self.random_seed * self.expansion_N * global_step))
                next_state, updated_budget = self.environment.apply_action(state, chosen_action, node.remaining_budget, copy_state=True)

                next_node, seen = self.initialize_tree_node(node, next_state, chosen_action, updated_budget, fa_subset, training)

                node.children[chosen_action] = next_node
                actions_taken.append(chosen_action)

                if not seen:
                    traversed_nodes.append(next_node)

                node = next_node

                if not seen:
                    break
                # break
            else:
                if node.num_valid_actions == 0:
                    break
                else:

                    highest_ucb, ucb_action, ucb_node = self.pick_best_child(node, i, self.C_ps[i])
                    # if ucb_node is None:
                    #     print(f"caught the None node.")
                    #     print(f"highest_ucb is {highest_ucb}")
                    #     print(f"ucb_action is {ucb_action}")
                    #     remaining_budget = self.environment.get_remaining_budget(0)
                    #     print(f"rem budget is {remaining_budget}")
                    #     print(f"num valid acts is {node.num_valid_actions}")
                    #     print(f"value of obj function is {self.environment.get_objective_function_value(node.state)}")
                    #     self.pick_best_child(node, self.C_p, print_vals=True)

                    # Go to the best child node.
                    node = ucb_node
                    actions_taken.append(ucb_action)
                    continue
                    
        # print(actions_taken)
        # print(f"Traversed {len(traversed_nodes)} nodes.")
        return traversed_nodes, actions_taken

    def initialize_tree_node(self, parent_node, node_state, chosen_action, updated_budget, fa_subset, training, with_depth=-1):
        """Initialize a new tree node with the given parameters."""
        seen = False

        if self.transposition:
            graph_hash = get_graph_hash(node_state, size=64, include_first=True)
            if graph_hash in self.node_hashmap:
                # print('Node already in hashmap! Linking new parent...')

                next_node = self.node_hashmap[graph_hash] # Get next node from hash map
                next_node.parent_nodes.append(parent_node) # Add parent node to list of child's parents
                seen = True
                self.num_transpositions += 1

                return next_node, seen

        banned_actions = node_state.banned_actions
        next_node_actions = self.environment.get_valid_actions(node_state, banned_actions)

        if node_state.first_node is None:
            next_node_actions = next_node_actions.intersection(fa_subset)

        next_node_actions = list(next_node_actions)
        depth = parent_node.depth + 1 if with_depth == -1 else with_depth

        predictor_vals = self.get_predictor_values(node_state, next_node_actions)

        if self.transposition:
            next_node = TranspositionTreeNode2(node_state, parent_node, chosen_action, next_node_actions, 
                                               remaining_budget=updated_budget, depth=depth)
            
            # Add new node to hashmap
            self.node_hashmap[graph_hash] = next_node
            self.num_tree_nodes += 1

        else:
            next_node = MCTSTreeNode(node_state, parent_node, chosen_action, next_node_actions, 
                                     remaining_budget=updated_budget, depth=depth)
            self.num_tree_nodes += 1

        next_node.assign_predictor_values(predictor_vals)
        return next_node, seen

    def get_first_actions_subset(self, start_state, i, training):
        return self.reduced_actions[i]

    def get_predictor_values(self, state, actions):
        n = len(actions)
        if n == 0:
            return []
        uniform_probs = np.full(n, 1, dtype=np.float32)
        return uniform_probs


    def pick_best_child(self, node, i, c, print_vals=False):
        """Pick the child node with the highest UCB1 value."""
        highest_value = float("-inf")
        best_node = None
        best_action = None

        # Get the UCB1 value for each child node.
        child_values = {action: self.node_selection_strategy(node, i, action, child_node)
                        for action, child_node in node.children.items()}


        if print_vals:
            print(f"child values were {child_values}")

        # Select the best action (child)
        for action, value in child_values.items():
            if value > highest_value:
                highest_value = value
                best_node = node.children[action]
                best_action = action
        return highest_value, best_action, best_node

    def node_selection_strategy(self, parent_node, i, action, child_node):
        """Given a child node, return the value of the node based on the selection strategy."""
        # Based on UCB1 for default UCT.
        
        if self.transposition:
            Q, parent_N, child_N = child_node.Q, parent_node.N, child_node.action_counts[parent_node]
        else:
            Q, parent_N, child_N = child_node.Q, parent_node.N, child_node.N

        predictor_value = parent_node.get_predictor_value(action)

        node_value = self.get_ucb1_term(Q, parent_N, child_N, self.C_ps[i], predictor_value)

        if math.isnan(node_value):
            node_value = 0.0

        return node_value

    def get_ucb1_term(self, Q, parent_N, child_N, C_p, model_prior):
        """Get the UCB1 value for a child node."""
        ci_term = sqrt((2 * log(parent_N)) / child_N)
        ucb1_value = Q + C_p * model_prior * ci_term
        return ucb1_value

    def execute_simulation_policy(self, node, root_node, i, fa_subset, starting_budget, rollout_limit):
        """Execute the simulation policy."""
        obj_fun_computation = self.environment.objective_function.compute
        obj_fun_kwargs = self.environment.objective_function_kwargs
        if node.num_valid_actions == 0:
            return [([], self.get_final_node_val(node.state, obj_fun_computation, obj_fun_kwargs), None, 0)]

        valid_actions_finder = self.environment.get_valid_actions

        action_applier = self.environment.apply_action_in_place
        simulation_results = []


        for sim_number in range(self.num_simulations):
            # print(f"yeah, doing a sim!")
            # print(f'Tracked edges: {node.state.tracked_edges}')
            self.sim_policy_inst.reset(node.state.tracked_edges)
            out_of_tree_acts, R, post_random_state, sim_number = self.sim_policy_episode(self.sim_policy_inst,
                                                                                                node,
                                                                                                root_node,
                                                                                                fa_subset,
                                                                                                starting_budget,
                                                                                                rollout_limit,
                                                                                                sim_number,
                                                                                                valid_actions_finder,
                                                                                                action_applier,
                                                                                                obj_fun_computation,
                                                                                                obj_fun_kwargs,
                                                                                                self.local_random.getstate(),
                                                                                                )
            simulation_results.append((out_of_tree_acts, R, post_random_state, sim_number))
            self.local_random.setstate(post_random_state)
        return simulation_results

    @staticmethod
    def sim_policy_episode(sim_policy,
                                  node,
                                  root_node,
                                  fa_subset,
                                  starting_budget,
                                  rollout_limit,
                                  sim_number,
                                  valid_actions_finder,
                                  action_applier,
                                  obj_fun_computation,
                                  obj_fun_kwargs,
                                  random_state,
                                  ):

        """
        when do we want to stop?
            A: when total used budget from _current root_ exceeds the rollout limit
            alternatively, since env needs remaining budget:
                - subtract from root remaining budget the budget of the current node (that's how much was used up
                from root to here)
                - then, subtract from rollout limit this value: that's your remaining budget
                - take the minimum between this value and rem budget at starting node; this will be hit when search
                is nearly completed
        """

        initial_depth, rem_budget = MonteCarloTreeSearchAgent.find_budget_at_leaf(root_node, rollout_limit, node)

        # orig_budget = rem_budget
        # print(f"used so far: {used_so_far}, rem at node: {node.remaining_budget}")
        # print(f"computed me budget as {rem_budget}")

        state = node.state.copy()
        out_of_tree_actions = []

        state.init_dynamic_edges()
        current_rollout_depth = 0


        while True:
            possible_actions = valid_actions_finder(state, state.banned_actions)
            total_depth = initial_depth + current_rollout_depth
            if total_depth % 2 == 0:
                possible_actions = possible_actions.intersection(fa_subset)

            if rem_budget <= budget_eps or len(possible_actions) == 0:
                break
                # raise ValueError("Shouldn't run simulation from node when no actions are available!")

            chosen_action = sim_policy.choose_action(state, rem_budget, total_depth, possible_actions)
            out_of_tree_actions.append(chosen_action)

            rem_budget = action_applier(state, chosen_action, rem_budget)
            # print(f"yay, adding an action from simulation policy! budget updated from {orig_budget} to {rem_budget}")

            current_rollout_depth += 1

        final_state = state.apply_dynamic_edges()
        node_val = MonteCarloTreeSearchAgent.get_final_node_val(final_state, obj_fun_computation,
                                                                obj_fun_kwargs)
        post_random_state = sim_policy.get_random_state()
        return out_of_tree_actions, node_val, post_random_state, sim_number

    @staticmethod
    def find_budget_at_leaf(root_node, rollout_limit, node):
        initial_depth = node.depth
        used_so_far = root_node.remaining_budget - node.remaining_budget
        rem_budget = rollout_limit - used_so_far
        rem_budget = min(node.remaining_budget, rem_budget)
        rem_budget = max(0, rem_budget)
        return initial_depth, rem_budget

    @staticmethod
    def get_final_node_val(final_state, obj_fun_computation, obj_fun_kwargs):
        """Get final node value by the value of the objective function."""
        final_value = obj_fun_computation(final_state, **obj_fun_kwargs)
        node_val = final_value
        return node_val


    def update_best_trajectories(self, i, t, tree_actions, simulation_results):
        for out_of_tree_acts, R, _, sim_num in simulation_results:
            if R > self.best_Rs[i]:
                self.best_Rs[i] = R

                best_traj = deepcopy(self.moves_so_far[i])
                best_traj.extend(tree_actions)
                best_traj.extend(out_of_tree_acts)

                self.best_trajectories_found[i] = best_traj

                # print(f'New best trajectory found with reward {R - self.environment.get_initial_values()[i]}! {best_traj}')


    def backup_values(self, tree_nodes, tree_actions, simulation_results):
        """Backup the values of the nodes in the tree after a simulation."""
        # print([x[1] for x in simulation_results])
        if self.transposition:
            # if len(tree_nodes) > 2:
            #     print(f'Backing up {len(tree_nodes)} paths.')
            # if len(tree_nodes) > 50:
            #     print(tree_nodes)

            # next_nodes = set()
            # for path in tree_nodes:
            #     next_nodes.add(path[-2])
            # if len(next_nodes) > 2:
            #     print(f'Backing up {len(next_nodes)} distinct nodes, {len(tree_nodes)} paths.')
            for path in tree_nodes:
                for j, tree_node in enumerate(path):
                    parent_node = path[j+1] if j < len(path) - 1 else None
                    for _, R, _, _ in simulation_results:
                        tree_node.update_estimates(R)
                        tree_node.action_counts[parent_node] = tree_node.action_counts.get(parent_node, 0) + 1
                        self.num_node_updates += 1
        else:
            for tree_node in tree_nodes:
                for _, R, _, _ in simulation_results:
                    tree_node.update_estimates(R)
                    self.num_node_updates += 1
            

    @staticmethod
    def get_trajectory_for_graph(agent_class, environment, step, step_dep_params, hyperparams, options, starting_graph,
                                 starting_graph_initial_obj_values):

        agent = agent_class(environment)
        agent.setup(options, hyperparams)
        if step_dep_params is not None:
            agent.restore_step_dependent_params(step, step_dep_params)
        agent.environment.setup([starting_graph],
                                [starting_graph_initial_obj_values],
                                training=True)
    
        trajectory = agent.run_trajectory_collection()

        # Plot final state
        final_state = agent.environment.g_list[0]
        # final_state, _ = final_state.add_edge(final_state.current_node, final_state.starting_node)
        final_state.draw_to_file('test.png') 

        agent.environment.tear_down()
        agent.finalize()
        del agent
        return trajectory

    def run_trajectory_collection(self):
        acts = []
        self.tree_depths = []
        self.updates = []
        self.num_tree_nodes = 0
        self.num_transpositions = 0

        t = 0
        while not self.environment.is_terminal():
            # if self.log_progress:
            #     self.logger.info(f"executing inner search step {t}")
            #     self.logger.info(f"{get_memory_usage_str()}")
            self.obj_fun_eval_count = 0
            self.log_timings_if_required(t, "before", 1, self.obj_fun_eval_count)
            self.run_search_for_g_list(t, force_init=self.force_init, training=True)
            list_at = self.pick_children()
            self.log_timings_if_required(t, "after", 1, self.obj_fun_eval_count)

            acts.append(list_at[0])
            self.environment.step(list_at)
            t += 1

        subset = self.reduced_actions[0]

        if self.btm:
            best_acts = self.best_trajectories_found[0]
            best_F = self.best_Rs[0] - self.environment.get_initial_values()[0]
        else:
            best_acts = self.moves_so_far[0]
            best_F = self.environment.rewards[0]

        if math.isnan(best_F):
            best_F = 0.

        return subset, best_acts, best_F, np.mean(self.tree_depths), np.mean(self.updates), self.num_tree_nodes, self.num_transpositions

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        if "draw_trees" in options:
            self.draw_trees = options['draw_trees']
            if self.draw_trees:
                self.tree_illustration_path = Path(options['tree_illustration_path'])
                self.tree_illustration_path.mkdir(parents=True, exist_ok=True)
                self.drawing_type = options['drawing_type']
        else:
            self.draw_trees = False

        if 'num_simulations' in options:
            self.num_simulations = options['num_simulations']
        else:
            self.num_simulations = 1

        if 'eval_pool_size' in options:
            self.parallel_eval = True
            self.eval_pool = Pool(processes=self.options['eval_pool_size'])
        else:
            self.parallel_eval = False

        self.ar_fun = self.hyperparams['ar_fun']
        self.ar_modifier = self.hyperparams['ar_modifier']
        self.reduction_policy = ReductionPolicy.get_reduction_policy_instance(self.hyperparams['reduction_policy'],
                                                                              self.ar_fun,
                                                                              self.ar_modifier)

        self.C_p = hyperparams['C_p']
        self.expansion_budget_modifier = hyperparams['expansion_budget_modifier']
        self.btm = hyperparams['btm']

        if 'sim_policy' in hyperparams:
            self.sim_policy = hyperparams['sim_policy']
        else:
            self.sim_policy = 'random'

        if 'rollout_depth' in hyperparams:
            self.rollout_depth = hyperparams['rollout_depth']
        else:
            self.rollout_depth = -1

        if 'final_action_strategy' in hyperparams:
            self.final_action_strategy = self.final_action_strategies[hyperparams['final_action_strategy']]
        else:
            self.final_action_strategy = self.pick_robust_child

        if 'transposition' in hyperparams:
            self.transposition = hyperparams['transposition']

            if self.transposition:
                self.node_hashmap = {}
        else:
            self.transposition = False

        if 'force_init' in hyperparams:
            self.force_init = hyperparams['force_init']
        else:
            self.force_init = False


    def get_default_hyperparameters(self):
        default_hyperparams = {
            'C_p': 0.1,
            'adjust_C_p': False,
            'btm': False,
            'final_action_strategy': 'robust_child',
            'expansion_budget_modifier': 500,
            'sim_policy': 'random',
            'sim_policy_bias': 5,
            'ar_fun': 'sqrt',
            'ar_modifier': 1,
            'rollout_depth': 1,
            'transposition': False,
            'force_init': True,
            'reduction_policy': 'dummy'
        }
        return default_hyperparams

    def draw_search_tree_with_values(self, graph_index, root_node, t, max_depth=1, max_breadth=20, drawing_type="mpl", write_dot=False):
        with warnings.catch_warnings():

            G = nx.DiGraph()
            root_node_label = self.construct_node_label(root_node, 1.0, drawing_type)
            G.add_node(root_node_label, value=root_node.Q)
            self.add_mcts_edges_recursively(G, root_node, root_node_label, drawing_type, depth=0, max_depth=max_depth,
                                            max_breadth=max_breadth)


            if drawing_type == "mpl":
                pos = graphviz_layout(G, prog='dot')
                dot_file = self.tree_illustration_path / f"mcts_tree_g{graph_index}_{self.random_seed}_{t}.dot"
                dot_filename = str(dot_file)
                png_filename = str(self.tree_illustration_path / f"mcts_tree_g{graph_index}_{self.random_seed}_{t}.png")

                plt.figure(figsize=(20, 10))
                plt.title('Monte Carlo Search Tree')

                node_data = list(G.nodes())
                node_colours = [float(x.split("\n")[0].split(":")[1]) for x in node_data]

                labels = {}
                for u, v, data in G.edges(data=True):
                    labels[(u, v)] = data['action']

                nx.draw_networkx(G, pos, with_labels=True, arrows=True, node_size=1500, font_size=8, alpha=0.95, vmin=0,
                                 vmax=1,
                                 cmap="Reds", node_color=node_colours)
                nx.draw_networkx_edges(G, pos, edge_color='g', width=5)



                nx.draw_networkx_edge_labels(G, pos, font_size=8, edge_labels=labels)
                plt.axis('off')
                plt.savefig(png_filename, bbox_inches="tight")
                plt.close()

                # subprocess.Popen(["dot", "-Tpng", dot_file, "-o", png_file])

                if write_dot:
                    A = nx.nx_agraph.to_agraph(G)
                    A.write(dot_filename)

            elif drawing_type == "tikz":
                pos = graphviz_layout(G, prog='dot', args='-Gnodesep=7.5 -Granksep=75')

                tikz_filename = str(self.tree_illustration_path / f"tikz_mcts_tree_g{graph_index}_{self.random_seed}_{t}.tex")
                my_style = {}

                lightblue = (171, 215, 230)
                my_style["canvas"] = (12, 2.5)
                my_style["edge_color"] = ['darkgray' for edge in G.edges]
                my_style['node_color'] = [lightblue for n in G.nodes]
                my_style["node_label"] = [n for n in G.nodes()]
                my_style["node_label_size"] = [0 for n in G.nodes()]

                my_style["node_size"] = [0.28 for n in G.nodes()]
                my_style["edge_width"] = [0.85 for e in G.edges()]
                my_style["edge_style"] = ["-" for e in G.edges()]

                my_style["edge_label_size"] = [0 for e in G.edges()]
                network2tikz.plot(G, tikz_filename, **my_style, layout=pos, standalone=False)



    def add_mcts_edges_recursively(self, G, parent_node, parent_node_label, drawing_type, depth, max_depth, max_breadth):
        if max_depth == 0 or depth == max_depth:
            return

        children_items = parent_node.children.items()
        visited_children = [entry for entry in children_items]
        children = sorted(visited_children, key=lambda x: x[1].Q, reverse=True)[0:max_breadth]

        for action, child_node in children:
            predictor_value = parent_node.get_predictor_value(action)
            child_node_label = self.construct_node_label(child_node, predictor_value, drawing_type)
            G.add_node(child_node_label, value=child_node.Q)
            G.add_edge(parent_node_label, child_node_label, action=f"{action}")
            self.add_mcts_edges_recursively(G, child_node, child_node_label, drawing_type, depth + 1, max_depth, max_breadth)

    def construct_node_label(self, node, predictor_value, drawing_type):
        state_id = get_graph_hash(node.state, size=64, include_first=True)
        if drawing_type == "mpl":
            node_label = f"Q: {node.Q: .3f}\n" \
                        f"N: {node.N}\n" \
                        f"B: {node.remaining_budget: .3f}\n" \
                        f"P: {predictor_value:.3f}\n" \
                        f"A: {node.parent_action}\n" \
                        f"id: {state_id}"
            return node_label
        else:
            return state_id

    def finalize(self):
        if self.parallel_eval:
            if self.eval_pool is not None:
                self.eval_pool.close()
                del self.eval_pool

        self.root_nodes = None
        self.fa_subsets = None
        self.node_expansion_budgets = None



class StandardMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'off'
        hyps_copy['ar_modifier'] = -1
        hyps_copy['reduction_policy'] = 'dummy'
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class BTMMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_btm'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'off'
        hyps_copy['ar_modifier'] = -1
        hyps_copy['reduction_policy'] = DummyReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = True
        super().setup(options, hyps_copy)

class MinCostMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_mincost'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'off'
        hyps_copy['ar_modifier'] = -1
        hyps_copy['reduction_policy'] = DummyReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'min_cost'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR80RandMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_rand_80'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 80
        hyps_copy['reduction_policy'] = RandomReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR60RandMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_rand_60'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 60
        hyps_copy['reduction_policy'] = RandomReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR40RandMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_rand_40'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['reduction_policy'] = RandomReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR40DegreeMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_deg_40'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['reduction_policy'] = DegreeReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR40InvDegreeMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_invdeg_40'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['reduction_policy'] = InvDegreeReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR40LBHBMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_lbhb_40'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['reduction_policy'] = LBHBReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR40NumConnsMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_numconns_40'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['reduction_policy'] = NumConnsReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR40BestEdgeMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_be_40'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['reduction_policy'] = BestEdgeReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR40BestEdgeCSMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_becs_40'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['reduction_policy'] = BestEdgeCSReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR40AvgEdgeMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_ae_40'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['reduction_policy'] = AvgEdgeReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class AR40AvgEdgeCSMCTSAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uct_aecs_40'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['reduction_policy'] = AvgEdgeCSReductionPolicy.policy_name
        hyps_copy['sim_policy'] = 'random'
        hyps_copy['btm'] = False
        super().setup(options, hyps_copy)


class SGUCTAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'sg_uct'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['ar_fun'] = 'percentage'
        hyps_copy['ar_modifier'] = 40
        hyps_copy['sim_policy'] = 'min_cost'
        hyps_copy['sim_policy_bias'] = 25
        hyps_copy['btm'] = True
        hyps_copy['reduction_policy'] = AvgEdgeCSReductionPolicy.policy_name
        super().setup(options, hyps_copy)