[![Build Status](https://travis-ci.com/RealVNF/deep-rl-network-service-coordination.svg?token=NVCqJRthxojczFXPsprz&branch=master)](https://travis-ci.com/RealVNF/deep-rl-network-service-coordination)

# Self-Driving Network and Service Coordination Using Deep Reinforcement Learning

<p align="center">
  <img src="docs/realvnf_logo.png" height="150" hspace="30"/>
	<img src="docs/upb.png" width="200" hspace="30"/>
	<img src="docs/huawei_horizontal.png" width="250" hspace="30"/>
</p>

## Setup

You need to have [Python 3.6](https://www.python.org/downloads/release/)+ and [venv](https://docs.python.org/3/library/venv.html) module installed.

### Create a venv

On your local machine:

```bash
# create venv once
python3.6 -m venv ./venv
# activate the venv (always)
source venv/bin/activate
# update setuptools
pip install -U setuptools
```

### Install dependencies

```bash
pip install -r requirements.txt
```

This also installs the required [coord-sim](https://github.com/RealVNF/coord-sim/tree/cnsm2020) simulator and [common-utils](https://github.com/RealVNF/common-utils/tree/cnsm2020) package.

## Use the RL agent

All options:

```bash
$ rlsp -h
Usage: rlsp [OPTIONS] AGENT_CONFIG NETWORK SERVICE SIM_CONFIG STEPS

  rlsp cli for learning and testing

Options:
  --seed INTEGER               Specify the random seed for the environment and
                               the learning agent.
  -t, --test TEXT              Name of the training run whose weights should
                               be used for testing.
  -w, --weights TEXT           Continue training with the specified weights
                               (similar to testing)
  -a, --append-test            Append a test run of the previously trained
                               agent.
  -v, --verbose                Set console logger level to debug. (Default is
                               INFO)
  -b, --best                   Test the best of the trained agents so far.
  -e, --test-episodes INTEGER  Set the number of testing episodes
  -ss, --sim-seed INTEGER      Set the simulator seed
  -gs, --gen-scenario PATH     Diff. sim config file for additional scenario
                               test
  -h, --help                   Show this message and exit.
```

### Training and testing

Example for short training then testing:

```bash
rlsp res/config/agent/sample_agent.yaml res/networks/sample_network.graphml res/service_functions/abc.yaml res/config/simulator/sample_config.yaml 1000 --append-test
```

Results are stored under `results/` according to the input arguments and the current time stamp.
There, you'll find copies of the used inputs, the trained weights, logs, and all result files of any test runs that you performed with these weights.

### Testing

To run another test run with the trained weights, specify the `<timestamp_seed>` of the training run. For testing, it is recommended to use 200 steps as it is the duration of one episode and use `-e` to specify the number of testing episodes.
For example:

```bash
rlsp res/config/agent/sample_agent.yaml res/networks/sample_network.graphml res/service_functions/abc.yaml res/config/simulator/sample_config.yaml 200 -t <timestamp_seed> -e 1
```

### Testing with a different simulator configuration (Generalization)

To train an agent and test it on multiple scenarios (simulator configurations), use the `-gs` to specify a different simulator config file to test in combination with `--append-test`.

Example for testing with generalization:

```bash
rlsp res/config/agent/sample_agent.yaml res/networks/sample_network.graphml res/service_functions/abc.yaml res/config/simulator/sample_config.yaml 1000 --append-test -gs res/config/simulator/sample_config.yaml
```

## Learning Curves using Tensorboard

To view the learning curve of all agents, i.e., the episode reward over time, use `tensorboard`:

```bash
tensorboard --logdir==./graph
```

You can also filter to only show curves of a specific agent config, network (and service and config) by setting the `--logdir` correspondingly:

```bash
tensorboard --logdir==./graph/<agent_config>/<network>/<service>/<simulator_config>
```

## Visualizing/Analyzing Results

To get a better understanding of what the agent is doing, there is an Juypter notebook `example_eval.ipynb`.
It's just an example; you won't be able to run it without all the results (which are too large for the repo).

To create a similar notebook for evaluation:

```bash
# first time installation
pip install -r eval_requirements.txt
# run jupyter server
jupyter lab
```

## Training and testing on multiple scenarios

There is script provided in the `scripts` folder that utilizes the [GNU Parallel](https://www.gnu.org/software/parallel/) utility to run multiple agents at the same time to speed up the training process.

```bash
./scripts/run_parallel.sh
```

To run long training sessions in remote environments without risking to stop the sessions due to possible connectivity issues, it is recommended to run the experiments with the `screen` linux tool.

- For that, start a new `screen` with `screen -S rl-parallel`.
- Configure the agent and the network, service, config files in the `scripts` directory to match the scenarios that you want to run. Here, the lines of network, service, config are matched by lines (not all permutations), eg, 1. network is matched with 1. service and 1. config. Then all seeds are used for all scenarios.
- Inside the screen, with the venv activated, run `./scripts/run_parallel.sh` from the project root.

## Acknowledgement

This project has received funding from German Federal Ministry of Education and Research ([BMBF](https://www.bmbf.de/)) through Software Campus grant 01IS17046 ([RealVNF](https://realvnf.github.io/)).

<p align="center">
	<img src="docs/software_campus.png" width="200"/>
	<img src="docs/BMBF_sponsored_by.jpg" width="250"/>
</p>
