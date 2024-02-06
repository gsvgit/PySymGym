# PySymGym
Python infrastructure to train paths selectors for symbolic execution engines.

We treat paths selection as a game where current state of symbolic execution process, represented as a interprocedural control flow graph equipped with information about execution details, is a map of the world (game map). States of symbolic machine are chips that player able to move. Each step, having current game map, player (AI agent) selects state to move and sends it to game server. Server moves selected state and return updated map to the player. Depending on scoring function, player can be aimed to achieve 100% coverage in minimal number of moves, or achieve 100% coverage with minimal number of tests generated, or something else.

Thus we introduced the following components.
- Game server
- Game maps
- AI agent (player)
- Training gym

As far as we use Json-based format to transfer data between server and agent, together with Json-based game maps description, our gym can be used to train networks using different symbolic execution engines.


## Quick Start

This repository contains submodules, so use the following command to get sources locally. 
```sh
git clone https://github.com/gsvgit/PySymGym.git
git submodule update --init --recursive
```

Build .net game server (V#)
```sh
cd GameServers/VSharp
dotnet build -c Release
```

Create & activate virtual environment:
```bash
python3 -m pip install virtualenv
python3 -m virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
```

### GPU installation:

To use GPU, the correct `torch` and `torch_geometric` version should be installed depending on your host device. You may first need to `pip uninstall` these packages, provided by requirements.
Then follow installation instructions provided on [torch](https://pytorch.org/get-started/locally/) and [torch_geometric](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html#installation-from-wheels) websites.

## Repo structure

- **AIAgent** contains Python agent and related infrastructure to train network, prepare data, etc.
- **GameServers** contains (as submodules) different symbolic execution engines extended to communicate with the agent, generate data for raining, etc.  
- **maps** contains target projects that used as inputs for symbolic execution engines, as data for training.

## Usage

Typical steps.

- Build game server
- Build game maps
- Create Json with maps description 
- Generate initial data
- Convert initial data to dataset for training
- Run training process

Currently we use V# as a primary game server. You can see example of typical workflow in [our automation](.github/workflows/build_and_run.yaml).

### Dataset description

Json

### Agent-server protocol description

Json

## Linting tools

Install [black](https://github.com/psf/black) code formatter by running following command in repo root to check your code style before committing:
```sh
pip install pre-commit && pre-commit install
```
