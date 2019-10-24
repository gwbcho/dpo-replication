# Distributional Policy Optimization: Reproducing the Generative Actor Critic Algorithm


## Objectives

The paper describes a method to represent arbitrary distribution functions over a continuous action space allowing for the construction of an optimal stationary stochastic policy. Without making assumptions about the underlying distribution, their generative scheme, called Generative Actor Critic, can overcome some limitations of more traditional policy gradient methods.

The purpose of this project is to reproduce Figure 4 which plots the training curves of 6 different methods on 6 DeepMind MuJoCo control suites. Reproducing this figure requires that we reproducing the entire paper as 3 variations of GAC need to be implemented and tested, while 3 previous methods have code available online.

As the paper has stated, GAC is computationally more expensive than current PG methods. Therefore, we anticipate we need the most computationally powerful Google Cloud instance or equivalence, e.g. c2-standard-60 with 60 vCPUs.


## Project Layout

The four main components of the Generative Actor Critic (GAC) Algorithm are the Actor Architecture, tne Value Network, the Critic Network, and the Delayed Actor which tracks the explorer Actor using a Polyak averaging scheme. These major componentes can be found under the policies/gac directory while the tests and other policies to reproduce from the original paper (https://arxiv.org/pdf/1905.09855.pdf) can be found under policies/ddpg.


## Run Commands

To run this project use the command

    python3 -m main

from the main directory.


## Requirements

The current requirements for this project are:
- tensorflow
- docker
- numpy
- mujoco - see explanation here: https://github.com/openai/mujoco-py (still need to add this to requirements)
- gym
- tqdm - for tracking experiment time left
- visdom - for visualization of the learning process


## Docker commands

The main docker commands to be concerned with for this project are as follows.

    docker-compose up --build

Which will construct the docker container and report any logs from said container to the standard output.

    docker-compose up -d

Which will run the docker container in the background.

    docker ps

Will return a list of all running docker containers.

    docker exec -it <image name> bash

And all related commands will allow the user to effectively ssh into the docker container. More useful commands (especially commands to handle cleanup of containers, volumes, and your environment) can be found here: https://docs.docker.com/engine/reference/commandline/docker/.


## Project Tests

To run the tests in the tests file (currently there are only functional tests) use the following command from the home directory.

    python3 -m tests.unit.unit_test_suites


## Current Bugs

There are no current bugs.


## Contributors

Gregory Cho, Jiuyang Bai, Linlin Liu, Liu Yang, Xingchi Yan
