import sys
import argparse
import time as time
from copy import deepcopy
from billiard.pool import Pool
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append('/relnet')
from relnet.agent.baseline.baseline_agent import GreedyAgent, CostSensitiveGreedyAgent
from relnet.agent.mcts.mcts_agent import AR40RandMCTSAgent, SGUCTAgent, StandardMCTSAgent, BTMMCTSAgent, MinCostMCTSAgent
from relnet.env.graph_edge_env import GraphEdgeEnv
from relnet.eval.file_paths import FilePaths
from relnet.objective_functions.objective_functions import LargestComponentSizeTargeted, GlobalEfficiency, SimpleObjective, GlobalEntropy
from relnet.state.network_generators import NetworkGenerator
from relnet.state.geometric_network_generators import KHNetworkGenerator, WaxmanNetworkGenerator, GridNetworkGenerator, RandomNetworkGenerator, RealWorldNetworkGenerator

def get_options(file_paths, seed):
    """Get options for MCTS agent."""
    mcts_opts = {}
    mcts_opts['random_seed'] = 42 * seed
    mcts_opts['draw_trees'] = False
    mcts_opts['log_progress'] = True
    mcts_opts['log_filename'] = file_paths.construct_log_filepath()
    mcts_opts['parallel_eval'] = False
    mcts_opts['num_simulations'] = 1
    mcts_opts['tree_illustration_path'] = file_paths.figures_dir
    mcts_opts['drawing_type'] = 'mpl'
    # mcts_opts['eval_pool_size'] = 8
    return mcts_opts

def get_file_paths(exp_id):
    """Get file paths for experiment."""
    parent_dir = '/experiment_data'
    file_paths = FilePaths(parent_dir, exp_id, setup_directories=True)
    return file_paths

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run MCTS experiment.')
    parser.add_argument('--exp_id', type=str, default='psn_development', help='Experiment ID')
    parser.add_argument('-a', '--agent', type=str, default='StandardMCTSAgent', help='Agent to use')
    parser.add_argument('-o', '--obj_fun', type=str, default='GlobalEfficiency', help='Objective function to use')
    parser.add_argument('-n', '--num_graphs', type=int, default=5, help='Number of graphs to generate')
    parser.add_argument('-v', '--graph_nodes', type=int, default=20, help='Number of nodes in graph')
    parser.add_argument('-c', '--exploration_param', type=float, default=0.05, help='UCT exploration parameter')
    parser.add_argument('-b', '--edge_budget', type=int, default=10, help='Edge budget')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-e', '--expansion_budget', type=int, default=20, help='Expansion budget (multiplier of number of nodes)')
    parser.add_argument('-p', '--parallel_seeds', type=int, default=10, help='Number of parallel seeds to evaluate per graph')
    parser.add_argument('-t', '--transposition', action='store_true', help='Use transposition-enabled agent')
    args = parser.parse_args()

    assert args.agent in ['StandardMCTSAgent', 'SGUCTAgent', 'BTMMCTSAgent', 'MinCostMCTSAgent'], 'Agent must be one of: StandardMCTSAgent, SGUCTAgent, BTMMCTSAgent'
    assert args.obj_fun in ['GlobalEfficiency', 'LargestComponentSizeTargeted', 'SimpleObjective', 'GlobalEntropy'], 'Objective function not recognised'

    return args

def main():
    # Parse arguments
    args = parse_args()

    # Default values for edge budget, seed and objective function kwargs
    seed = args.seed
    edge_percentage = args.edge_budget
    obj_fun_kwargs = {"random_seed": 42, "mc_sims_multiplier": 0.25}

    # Set up agent, objective function, experiment ID and number of nodes
    agent_class = globals()[args.agent]
    obj_fun = globals()[args.obj_fun]()
    exp_id = args.exp_id
    n = args.graph_nodes

    gen_params = NetworkGenerator.get_custom_generator_params(n)
    storage_root = Path('/experiment_data/stored_graphs')
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}
    fp = get_file_paths(exp_id)
    gen = KHNetworkGenerator(**kwargs)
    # gen = RealWorldNetworkGenerator(original_dataset_dir=fp.processed_graphs_dir, **kwargs)
    # gen = WaxmanNetworkGenerator(**kwargs)
    # gen = GridNetworkGenerator(**kwargs)
    # gen = RandomNetworkGenerator(**kwargs)

    n_graphs = args.num_graphs
    graph_seeds = NetworkGenerator.construct_network_seeds(0, 0, n_graphs)
    _, _, test_graph_seeds = graph_seeds

    # Only test graphs needed (for now)
    test_graphs = gen.generate_many(gen_params, test_graph_seeds)
    options = get_options(fp, seed)

    if args.parallel_seeds > 1:
        options['eval_pool_size'] = args.parallel_seeds

    g_start = time.time()
     
    # Set up environment
    env = GraphEdgeEnv(obj_fun, obj_fun_kwargs, edge_percentage, restriction_mechanism='none')
    agent = agent_class(env)

    # Hyperparams and options
    hyperparams = agent.get_default_hyperparameters()
    hyperparams['transposition'] = args.transposition
    hyperparams['C_p'] = args.exploration_param
    hyperparams['expansion_budget_modifier'] = args.expansion_budget

    opts_copy = deepcopy(options)
    opts_copy['model_identifier_prefix'] = f"{agent_class.algorithm_name}_default_{n}_{hyperparams['sim_policy']}"
    agent.setup(opts_copy, hyperparams)

    # Run multiple parallel evaluations of the agent for the single test graph
    traj = agent.eval(test_graphs)
    res = [t[2] for t in traj]
    depth = [t[3] for t in traj]
    update = [t[4] for t in traj]
    num_nodes = [t[5] for t in traj]
    trans = [t[6] for t in traj]

    g_end = time.time()
    runtime = g_end - g_start    
    print(res)

    # Save results to csv in experiment directory
    results_df = pd.DataFrame()
    results_df['reward'] = res
    results_df['mean'] = [np.mean(res)] * len(res)
    results_df['std'] = [np.std(res)] * len(res)
    results_df['depth'] = depth
    results_df['update'] = update
    results_df['num_nodes'] = num_nodes
    results_df['num_trans'] = trans
    results_df['time'] = [runtime] * len(res)

    obj_fun_savenames = {
        'GlobalEfficiency': 'ge',
        'LargestComponentSizeTargeted': 'lcst',
        'SimpleObjective': 'so',
        'GlobalEntropy': 'shan'
    }

    if args.transposition:
        save_name = f"{agent_class.algorithm_name}_n{n}_{obj_fun_savenames[args.obj_fun]}_e{args.expansion_budget}_b{args.edge_budget}_c{args.exploration_param}_t.csv"
    else:
        save_name = f"{agent_class.algorithm_name}_n{n}_{obj_fun_savenames[args.obj_fun]}_e{args.expansion_budget}_b{args.edge_budget}_c{args.exploration_param}.csv"
    results_df.to_csv(fp.experiment_dir / save_name)

if __name__ == '__main__':
    main()