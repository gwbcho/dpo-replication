# Distributional Policy Optimization: Reproducing the Generative Actor Critic Algorithm

## Project Layout

The four main components of the Generative Actor Critic (GAC) Algorithm are the Actor Architecture, tne Value Network, the Critic Network, and the Delayed Actor which tracks the explorer Actor using a Polyak averaging scheme. These major componentes can be found under the gac directory while the tests to reproduce from the original paper (https://arxiv.org/pdf/1905.09855.pdf) can be found under tests.


## Requirements

The current requirements for this project are tensorflow, docker, and numpy.


## Docker commands

The main docker commands to be concerned with for this project are as follows.

    docker-compose up --build

Which will construct the docker container and report any logs from said container to the standard output.

    docker-compose up -d

Which will run the docker container in the background.

    docker ps

Will return a list of all running docker containers.

    docker exec

And all related commands will allow the user to effectively ssh into the docker container. More useful commands (especially commands to handle cleanup of containers, volumes, and your environment) can be found here: https://docs.docker.com/engine/reference/commandline/docker/.


## Current Bugs

There are no current bugs.


## Contributors

Gregory Cho, Haoze Zhang, Jiuyang Bai, Linlin Liu, Liu Yang, Xingchi Yan
