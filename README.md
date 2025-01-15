# PyPSA-China：An Open Optimisation model of the Chinese Energy System

This is the PIK implementation of the PyPSA-China power model, first published by Hailiang Liu et al for their study of [hydro-power in china](https://doi.org/10.1016/j.apenergy.2019.02.009) and extended by Xiaowei Zhou et al for their  ["Multi-energy system horizon planning: Early decarbonisation in China avoids stranded assets"](doi.org/10.1049/ein2.12011) paper. It is adapted from the Zhou version by the [PIK RD3-ETL team](https://www.pik-potsdam.de/en/institute/labs/energy-transition/energy-transition-lab), with the aim of coupling it to the [REMIND](https://www.pik-potsdam.de/en/institute/departments/transformation-pathways/models/remind) integrated assessment model.

PyPSA-China should be understood as a modelling worklow, using snakemake as workflow manager, around the [PyPSA python power system analysis](https://pypsa.org/) package. The workflow collects data, builds the power system network and plots the results. It is akin to its more mature sister project, [PyPSA-EUR](https://github.com/PyPSA/pypsa-eur), from which it is derived.

Unlike PyPSA-EUR, which simplifies high resolution data into a user-defined network size, the PyPSA-China network is currently fixed to one node per province. This is in large part due to data availability issues.

The PyPSA can perform a number of different study types (investment decision, operational decisions, simulate AC power flows). Currently only capacity expansion problems are explicitly implemented in PyPSA-China.

The PyPSA-CHINA-PIK is currently under development. Please contact us if you


# Installation

## Set-up on the PIK cluster
Gurobi license activation from the compute nodes requries internet access. The workaround is an ssh tunnel to the login nodes, which can be set-up on the compute nodes with
```
# interactive session on the compute nodes
srun --qos=priority --pty bash
# key pair gen (here ed25518 but can be rsa)
ssh-keygen -t ed25519 -f ~/.ssh/id_rsa.cluster_internal_exchange -C "$USER@cluster_internal_exchange"
# leave the compute nodes
exit
```
You will then need to add the contents of the public key `~/.ssh/id_rsa.cluster_internal_exchange.pub` to your authorised `~/.ssh/authorized_keys`

In addition you should have your .profile setup as per https://gitlab.pik-potsdam.de/rse/rsewiki/-/wikis/Cluster-Access
and add `module load anaconda/2024.10` (or latest) to it

## General installation
- Create the conda environment in workflow/envs/ (maybe snakemake does it automatically for you provided the profile has use-conda) `conda env create --file path_to_env` (name is opt.). You can use either the pinned (exact) or the loose env (will install later package versions too).
- If you experience issues switch to the pinned environment #TODO: generate
- NB! you may need to modify atlite for things to work. Instructions to follow.


## Getting the data
- some of the data is downloaded by the snakemake workflow (e.g. cutouts). Just make sure te relevant config options are set to true if it is your first run
- the shapely files can be generated with the build_province_shapes script
- the [zeonodo bundle](https://zenodo.org/records/13987282) from the pypsa-China v3 comes with the data but in the old format, you will have to manually restructure it (or we can write a script)
- you can also copy the data from the tmp folder

# Usage
- If you are not running on the PIK hpc, you will need make a new profile for your machine under `config/<myprofile>/config.yaml`
- The workflow can be launched with `snakemake --profile config/pik_hpc_profile`

# Changelog
- restructure project to match snakemake8 guidelines & update to snakemake8
- move hardcoded to centralised store constants.py (file paths still partially hardcoded)
- start adding typing
- add scripts to pull data
- add derived_data/ folder and change target of data cleaning/prep steps for clarity

# TODOs:
- see issues
- add pinned env
- make a PR on atlite to add the squeeze array, which may be needed
- integrate various into settings
