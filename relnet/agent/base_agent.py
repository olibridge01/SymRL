import random
import time
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import numpy as np

from relnet.eval.eval_utils import eval_on_dataset, get_values_for_g_list
from relnet.eval.file_paths import FilePaths
from relnet.state.graph_state import GeometricRelnetGraph
from relnet.utils.config_utils import get_logger_instance


class Agent(ABC):
    def __init__(self, environment):
        self.environment = environment
        self.obj_fun_eval_count = 0


    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        pass

    def eval(self, g_list,
             initial_obj_values=None,
             validation=False,
             make_action_kwargs=None):

        eval_nets = [deepcopy(g) for g in g_list]
        initial_obj_values, final_obj_values = get_values_for_g_list(self, eval_nets, initial_obj_values, validation, make_action_kwargs)
        return eval_on_dataset(initial_obj_values, final_obj_values)

    @abstractmethod
    def make_actions(self, t, **kwargs):
        pass

    def setup(self, options, hyperparams):
        self.options = options
        if 'log_timings' in options:
            self.log_timings = options['log_timings']
        else:
            self.log_timings = False
            self.timings_out = None

        if self.log_timings:
            self.setup_timings_file()


        if 'log_filename' in options:
            self.log_filename = options['log_filename']
        if 'log_progress' in options:
            self.log_progress = options['log_progress']
        else:
            self.log_progress = False
        if self.log_progress:
            self.logger = get_logger_instance(self.log_filename)
            self.environment.pass_logger_instance(self.logger)
        else:
            self.logger = None

        if 'random_seed' in options:
            self.set_random_seeds(options['random_seed'])
        else:
            self.set_random_seeds(42)
        self.hyperparams = hyperparams

    def get_default_hyperparameters(self):
        return {}

    def setup_timings_file(self):
        self.timings_path = self.options['timings_path']
        timings_filename = self.timings_path / FilePaths.construct_timings_file_name(self.options['model_identifier_prefix'])
        timings_file = Path(timings_filename)
        if timings_file.exists():
            timings_file.unlink()
        self.timings_out = open(timings_filename, 'a')

    def log_timings_if_required(self, t, entry_tag, num_graphs, obj_fun_eval_count):
        if self.timings_out is not None:
            ms_since_epoch = time.time() * 1000
            self.timings_out.write('%d,%s,%d,%.6f,%d\n' % (t, entry_tag, num_graphs, ms_since_epoch, obj_fun_eval_count))
            try:
                self.timings_out.flush()
            except BaseException:
                if self.logger is not None:
                    self.logger.warn("caught an exception when trying to flush timings data.")
                    self.logger.warn(traceback.format_exc())

    def finalize(self):
        if self.log_timings:
            if self.timings_out is not None and not self.timings_out.closed:
                self.timings_out.close()

    def pick_random_actions(self, i):
        g = self.environment.g_list[i]
        banned_first_nodes = g.banned_actions

        first_valid_acts = self.environment.get_valid_actions(g, banned_first_nodes)
        if len(first_valid_acts) == 0:
            return -1, -1

        first_node = self.local_random.choice(tuple(first_valid_acts))
        rem_budget = self.environment.get_remaining_budget(i)
        banned_second_nodes = g.get_invalid_edge_ends(first_node, rem_budget)
        second_valid_acts = self.environment.get_valid_actions(g, banned_second_nodes)

        if second_valid_acts is None or len(second_valid_acts) == 0:
            if self.logger is not None:
                self.logger.error(f"caught an illegal state: allowed first actions disagree with second")
                self.logger.error(f"first node valid acts: {first_valid_acts}")
                self.logger.error(f"second node valid acts: {second_valid_acts}")
                self.logger.error(f"the remaining budget: {rem_budget}")

                self.logger.error(f"first_node selection: {g.first_node}")


                if type(g) == GeometricRelnetGraph:
                    self.logger.error("graph is geometric.")
                    self.logger.error(f"state allowed connections: {g.allowed_connections}")
                    self.logger.error(f"state shortest allowed connection: {g.shortest_allowed_connection}")
                else:
                    self.logger.error("graph is not geometric.")

            return -1, -1
        else:
            second_node = self.local_random.choice(tuple(second_valid_acts))
            return first_node, second_node

    def set_random_seeds(self, random_seed):
        self.random_seed = random_seed
        self.local_random = random.Random()
        self.local_random.seed(self.random_seed)
        np.random.seed(self.random_seed)
