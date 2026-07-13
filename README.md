# PyPSA-China：An Open-Source Optimisation model of the Chinese Energy System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://pypsa.github.io/PyPSA-China/)
[![GitHub release](https://img.shields.io/github/v/release/pypsa/PyPSA-China)](https://github.com/pypsa/PyPSA-China/releases)

PyPSA-China is a open-source model of the Chinese energy system covering electricity and heat. It co-optimizes dispatch and investments under user-set constraints, such as limits to environmental impacts, to minimize costs. The model works at provincial resolution and can simulate a full year at hourly resolution.

The PyPSA-China power model, was first developed by H Liu et al for their study of [hydro-power in china](https://doi.org/10.1016/j.apenergy.2019.02.009) and then extended and released by X Zhou et al for their  ["Multi-energy system horizon planning: Early decarbonisation in China avoids stranded assets"](doi.org/10.1049/ein2.12011) paper. The current version has been heavily reworked and is currently maintained by the [Potsdam Institute for Climate Impact Studies ETL team](https://www.pik-potsdam.de/en/institute/labs/energy-transition/energy-transition-lab).

## Overview
PyPSA-China should be understood as a modelling worklow, using snakemake as workflow manager, around the [PyPSA python power system analysis](https://pypsa.org/) package. The workflow collects data, builds the power system network and plots the results. It is akin to its more mature sibbling, [PyPSA-EUR](https://github.com/PyPSA/pypsa-eur), from which it is derived.

Unlike PyPSA-EUR, which simplifies high resolution electricity grid data to a user-defined network size, the PyPSA-China network is currently fixed to one node per province (a 341 node version in the works and will be released this year)

The PyPSA can perform a number of different study types (investment decision, operational decisions, simulate AC power flows). Currently only capacity expansion problems are explicitly implemented in PyPSA-China.

PyPSA-China natively supports coupling with the [REMIND](https://www.pik-potsdam.de/en/institute/departments/transformation-pathways/models/remind) integrated assessment model.

## Quick Links

- 📖 [Documentation](https://pypsa.github.io/PyPSA-China/)
- 📝 [Changelog](CHANGELOG.md)
- 🚀 [Releases](https://github.com/pypsa/PyPSA-China/releases)
- 🤝 [Contributing Guide](CONTRIBUTING.md)
- 📋 [Release Guide](docs/release-guide.md) (for maintainers)

# License
The code is released under the [MIT license](https://github.com/pypsa/PyPSA-China/blob/main/LICENSES/MIT.txt), however some of the data used is more restrictive.

# Documentation
The documentation can be found at https://pypsa.github.io/PyPSA-China/

# Getting started

## Installation

An installation guide is provided at https://pypsa.github.io/PyPSA-China/

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
