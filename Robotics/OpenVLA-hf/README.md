# OpenVLA
This folder is a project about exploring OpenVLA. It mainly covers setting up a LIBERO environment and calling Huggingface to make OpenVLA inferences in simulation.

## Usage
1. `conda activate pickagent`
1. `cd Robotics/OpenVLA/entrypoints`
1. `python libero_openvla.py` or one of the other entrypoint files

I've personally set up my own pickagent environment with a bunch of steps.

To create your own pickagent conda environment with all the right libraries etc, see my fork here for what I did: https://github.com/tjadams/PickAgent

## Folder structure
 - entrypoints: contains the python files to run as a user
 - shared: contains shared python files that are used across other files