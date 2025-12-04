# PyPSA-China：An Open-Source Optimisation model of the Chinese Energy System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://pik-piam.github.io/PyPSA-China-PIK/)
[![GitHub release](https://img.shields.io/github/v/release/pik-piam/PyPSA-China-PIK)](https://github.com/pik-piam/PyPSA-China-PIK/releases)

PyPSA-China (PIK) is a open-source model of the Chinese energy system covering electricity and heat. It co-optimizes dispatch and investments under user-set constraints, such as limits to environmental impacts, to minimize costs. The model works at provincial resolution and can simulate a full year at hourly resolution.

## PIK version
This is the PIK implementation of the PyPSA-China power model, first published by Hailiang Liu et al for their study of [hydro-power in china](https://doi.org/10.1016/j.apenergy.2019.02.009) and extended by Xiaowei Zhou et al for their  ["Multi-energy system horizon planning: Early decarbonisation in China avoids stranded assets"](doi.org/10.1049/ein2.12011) paper. It is adapted from the Zhou version by the [PIK RD3-ETL team](https://www.pik-potsdam.de/en/institute/labs/energy-transition/energy-transition-lab), with the aim of coupling it to the [REMIND](https://www.pik-potsdam.de/en/institute/departments/transformation-pathways/models/remind) integrated assessment model. A reference guide is available as part of the [documentation](https://pik-piam.github.io/PyPSA-China-PIK/).

## Overview
PyPSA-China should be understood as a modelling worklow, using snakemake as workflow manager, around the [PyPSA python power system analysis](https://pypsa.org/) package. The workflow collects data, builds the power system network and plots the results. It is akin to its more mature sister project, [PyPSA-EUR](https://github.com/PyPSA/pypsa-eur), from which it is derived.

Unlike PyPSA-EUR, which simplifies high resolution electricity grid data to a user-defined network size, the PyPSA-China network is currently fixed to one node per province (with a 340 node version in the works). This is in large part due to data availability issues.

The PyPSA can perform a number of different study types (investment decision, operational decisions, simulate AC power flows). Currently only capacity expansion problems are explicitly implemented in PyPSA-China.

The PyPSA-CHINA-PIK is currently under development. Please contact us if you intend to use it for publications.

## Quick Links

- 📖 [Documentation](https://pik-piam.github.io/PyPSA-China-PIK/)
- 📝 [Changelog](CHANGELOG.md)
- 🚀 [Releases](https://github.com/pik-piam/PyPSA-China-PIK/releases)
- 🤝 [Contributing Guide](CONTRIBUTING.md)
- 📋 [Release Guide](docs/release-guide.md) (for maintainers)

# License
The code is released under the [MIT license](https://github.com/irr-github/PyPSA-China-PIK/blob/main/LICENSES/MIT.txt), however some of the data used is more restrictive. 

# Documentation
The documentation can be found at https://pik-piam.github.io/PyPSA-China-PIK/

# Getting started

## Installation

An installation guide is provided at https://pik-piam.github.io/PyPSA-China-PIK/ 

## Getting the data
You will need to enable data retrieval in the config
```yaml
enable:
  build_cutout: false # if you want to build your own (requires ERA5 api access)
  retrieve_cutout: true # if you want to download the pre-computed one from zenodo
  retrieve_raster: true # get raster data
```
Some of the files are very large - expect a slow process!

- You can also download the data manually and  copy it over to the correct folder. The source and target destinations are the input/output of the `fetch_` rules in `workflow/rules/fetch_data.smk`
- **PIK HPC users only** you can also copy the data from other users

## Usage

Detailed instructions in the documentation. 
### local execution
- local execution can be started (once the environment is activated) with `snakemake`
- to customize the options, create `my_config.yaml` and launch `snakemake --configfile `my_config.yaml`. Configuration options are summarised in the documentation.
### Remote execution
This is relevant for slurm HPCs and other remotes with a submit job command
- The workflow can be launched with `snakemake --profile config/compute_profile`
- [PIK HPC users only] use `snakemake --profile config/pik_hpc_profile`
- If you are not running on the PIK hpc, you will need make a new profile for your machine under `config/<compute_profile>/config.yaml`




