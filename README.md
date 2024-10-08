# Symmetries in Model-Based Graph Reinforcement Learning

This repository contains the code for my MSc thesis on symmetries in model-based graph reinforcement learning. Specifically, the code implements a transposition-enabled MCTS algorithm for graph-based combinatorial optimisation (CO) environments, and value function approximators parametrised by Message Passing Neural Networks (MPNNs).

## Prerequisites

The repository prerequisites are managed through a Docker container. Details on how to set up the container are provided below.

## Configuration

Create a file `relnet.env` at the root of the project (see `example.env`) and adjust the paths within: this is where some data generated by the container will be stored.

Add the following lines to `.bashrc`, replacing `/Users/john/git/relnet` with the path where the repository is cloned. 

```bash
export RN_SOURCE_DIR='/Users/john/git/relnet'
set -a
. $RN_SOURCE_DIR/relnet_example.env
set +a

export PATH=$PATH:$RN_SOURCE_DIR/scripts
```

Make the scripts executable (e.g. `chmod u+x scripts/*`) the first time after cloning the repository.

## Managing the container
To build the container:
```bash
update_container.sh
```
To start it:
```bash
manage_container.sh up
```
To stop it:
```bash
manage_container.sh stop
```

## Setting up graph data

### Synthetic data
Synthetic data will be automatically generated when the experiments are ran and stored to `$RN_EXPERIMENT_DIR/stored_graphs`.

## Running experiments
The file `relnet/experiment_launchers/run_experiment.py` is a script that can be used to run the MCTS and transposition-enabled MCTS algorithms on synthetic data. The script takes the following arguments to run experiments on synthetic (KH) graphs:

```bash
manage_container.sh up
docker exec -it symrl-psn /bin/bash -c  \
"source activate symrl-cenv && python relnet/experiment_launchers/run_experiment.py \
    --exp_id experiment_id \
    -a agent \
    -o objective_function \
    -n num_graphs \
    -v graph_size \
    -c exploration_param \
    -b edge_budget \
    -s seed \
    -e expansion_budget \
    -p parallel_cpus \
    -t (for transpositions)"
```


