# PyPSA-China：An Open Optimisation model of the Chinese Energy System

An exploratory version for the REMIND-PyPSA PANDA coupling based on the version published by Xiaowei Zhou et al for their  "Multi-energy system horizon planning: Early decarbonisation in China avoids stranded assets" (doi.org/10.1049/ein2.12011)

Adapted by Ivan Ramirez

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
- Create the conda environment in workflow/envs/ (maybe snakemake does it automatically for you provided the profile has use-conda)
- If you experience issues switch to the pinned environment #TODO: generate
- NB! you may need to modify atlite for things to work. Instructions to follow.     


## Getting the data
- some of the data is downloaded by the snakemake workflow (e.g. cutouts). Just make sure te relevant config options are set to true if it is your first run
- the shapely files can be generated with the build_province_shapes script
- the [zeonodo bundle](https://zenodo.org/records/13987282) from the pypsa-China v3 comes with the data but in the old format, you will have to manually restructure it (or we can write a script)

# Usage
- If you are not running on the PIK hpc, you will need make a new profile for your machine under `config/<myprofile>/config.yaml` and edit the `run_snakemake.sh` script to point to your profile
- The workflow can be launched with `sbatch un_snakemake.sh`
# Changelog
- restructure project to match snakemake8 guidelines & update to snakemake8
- move hardcoded to centralised store constants.py (file paths still partially hardcoded)
- start adding typing
- add scripts to pull data
- add derived_data/ folder and change target of data cleaning/prep steps for clarity

# TODOs:
- separate functionalities
- add sources to all data
- fix readme
- add pinned env
- make a PR on atlite to add the squeeze
- integrate various into settings
