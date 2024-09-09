import os
import time
from datetime import datetime
from pathlib import Path

import psutil

from relnet.evaluation.file_paths import FilePaths
from relnet.state.geometric_network_generators import GeometricNetworkGenerator
from relnet.state.network_generators import NetworkGenerator, RealWorldNetworkGenerator

date_format = "%Y-%m-%d-%H-%M-%S"


def find_latest_file(files_dir):
    filenames = [f.name for f in list(files_dir.glob("*.pickle"))]
    latest_file_date = datetime.utcfromtimestamp(0)
    latest_file = None
    for f in filenames:
        datestring = "".join(f.split(".")[:-1])
        file_date = datetime.strptime(datestring, date_format)
        if file_date > latest_file_date:
            latest_file = f
            latest_file_date = file_date
    if latest_file is not None:
        return files_dir / latest_file
    else:
        return None

def get_memory_usage_str():
    mb_used = psutil.Process(os.getpid()).memory_info().vms / 1024 ** 2
    return f"Process memory usage: {mb_used} MBs."

def is_time_expired(time_started, time_allowed):
    return get_current_time_millis() - time_started > time_allowed

def get_current_time_millis():
    return int(time.time() * 1000)

def retrieve_generator_class(generator_class_name):
    subclasses = list(get_subclasses(NetworkGenerator))
    subclasses.extend(GeometricNetworkGenerator.__subclasses__())
    subclass = [c for c in subclasses if hasattr(c, "name") and generator_class_name == c.name][0]
    return subclass

def create_generator_instance(generator_class, file_paths):
    processed_graph_dir = file_paths.processed_graphs_dir if isinstance(file_paths, FilePaths) else Path(file_paths['processed_graphs_dir'])
    graph_storage_dir = file_paths.graph_storage_dir if isinstance(file_paths, FilePaths) else Path(file_paths['graph_storage_dir'])
    log_filename = file_paths.construct_log_filepath() if isinstance(file_paths, FilePaths) else \
        Path(file_paths['logs_dir']) / FilePaths.construct_log_filename()

    if check_is_real_world(generator_class):
        original_dataset_dir = processed_graph_dir
        gen_kwargs = {'store_graphs': True, 'graph_storage_root': graph_storage_dir, 'logs_file': log_filename, 'original_dataset_dir': original_dataset_dir}
    else:
        gen_kwargs = {'store_graphs': True, 'graph_storage_root': graph_storage_dir, 'logs_file': log_filename}
    if type(generator_class) == str:
        generator_class = retrieve_generator_class(generator_class)
    gen_instance = generator_class(**gen_kwargs)
    return gen_instance

def check_is_real_world(generator_class):
    if type(generator_class) == str:
        # given as a name from retrieved experiment conditions
        subclasses = [str(c.name) for c in RealWorldNetworkGenerator.__subclasses__() if hasattr(c, "name")]
        if generator_class in subclasses:
            return True
    else:
        return issubclass(generator_class, RealWorldNetworkGenerator)

def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass

def get_graph_ids_to_iterate(train_individually, generator_class, file_paths):
    if train_individually:
        graph_ids = []
        network_generator_instance = create_generator_instance(generator_class, file_paths)
        num_graphs = network_generator_instance.get_num_graphs()
        for g_num in range(num_graphs):
            graph_id = network_generator_instance.get_graph_name(g_num)
            graph_ids.append(graph_id)
        return graph_ids
    else:
        return [None]