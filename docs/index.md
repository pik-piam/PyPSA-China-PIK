# Welcome to the PyPSA-China-PIK documentation!

This is the documentation for the Potsdam Institute for Climate Impact Studies' (PIK) **Python Power System Analysis for China** (`PyPSA-China-PIK`) model, maintained by the [Energy Transition Lab](https://www.pik-potsdam.de/en/institute/labs/energy-transition/energy-transition-lab) at PIK. `PyPSA-China-PIK` is an open model to simulate the future of energy in China at provincial level. Currently, electricity and heat are covered as energy sectors. 
The model can be partially coupled to the [REMIND](https://www.pik-potsdam.de/en/institute/departments/transformation-pathways/models/remind) Integrated Assesment Model to obtain multi-sectoral demand pathways. In this mode, battery electric vehicles can also be modelled.

## What is PyPSA-China-PIK?

`PyPSA-China-PIK` is a highly configurable, open-source and open data power system model that co-optimises investments (capacity expansion) and dispatch of future energy systems in China. The model covers several end-use sectors, including electricity, battery electric vehicles and heat. It comes with default data for costs, technical parameters, loads and existing infrastructure that can easily be replaced. The execution options are controlled by configuration files, meaning the code can be run with limited or even no coding expertise. A highly modular structure also makes it easy to extend the model.

`PyPSA-China-PIK` is built around the [PyPSA](https://pypsa.org/) "Python for Power System Analysis" (pronounced "pipes-ah") energy system modelling toolbox. PyPSA provides python objects that represent mixed AC and DC electricity networks, generators with optional unit commitment or variable generation, storage units and transformers as well as efficient bindings to open-source and commercial solvers.

Currently, hourly (and lower) time resolutions and provincial level demand and transmission are supported. It is possible to improve the spatial resolution as for other `PyPSA` workflows (EUR, USA, earth) if open data is available.  


## Capabilities

| Capability                | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Sectors                   | Models both electricity and heat demand and supply                          |
| Open Data & Code          | Fully open-source model and datasets                                        |
| Provincial Resolution     | Simulates energy systems at the provincial level across China               |
| Renewables & Hydro        | Models renewables and hydroelectricity availability based on historic data using [Atlite](https://atlite.readthedocs.io/en/latest/). Renewables can be split by capacity factor grades                        |
| Storage                   | Long and short-term duration storage with hydrogen, pumped hydro and batteries.                                                                        |
| Coupling with REMIND      | Can be partially coupled to the REMIND IAM for multi-sectoral analysis      |
| Cost Optimization         | Minimizes system costs using a solver of your choice                             |
| Flexible Workflow         | Managed by Snakemake for reproducible and automated analysis                |
| Customizable         | Modular structure can easily be customized                |
| Post-processing Tools     | Detailed results analysis and visualization                                 |


The model has been validated against short term energy trends. **Todo: add figure.**

## Learning

The model comes with an [installation guide](installation/quick_start), [model overview](model) [basic tutorials](tutorials/running/) and references for the [code](reference/SUMMARY/) and [configuration options](configuration). `PyPSA-China-PIK` is best understood as workflow, managed by the low-code [snakemake tool](https://snakemake.readthedocs.io/en/stable/). It is possible to run the workflow with minimal knowledge of `snakemake` and we have listed a few useful [tricks](tutorials/snakemake_tricks/) but we recommend going over the snakemake documentation.  

The workflow consists of gathering and preparing relevant data, formulating the problem as a PyPSA network object, minimising the system costs using a solver and post-processing the data. The `atlite` package is used to compute renewable generator availability and potentials. You may want to look into the [PyPSA documentation](https://pypsa.readthedocs.io/en/stable/) and the [atlite documentation](https://atlite.readthedocs.io/en/latest/). 

![PyPSA-China Workflow](./assets/img/pypsa-china-workflow.png)

## Authors and Credits

This version has is maintained by the [PIK RD3-ETL team](https://www.pik-potsdam.de/en/institute/labs/energy-transition/energy-transition-lab). It is not yet published, please contact us in case you are interested in using the model.

The model is based on the `PyPSA-EUR` work by the Tom Brown Group, originally adapted to China by Hailiang Liu et al for their study of [hydro-power in china](https://doi.org/10.1016/j.apenergy.2019.02.009) and extended by Xiaowei Zhou et al for their  ["Multi-energy system horizon planning: Early decarbonisation in China avoids stranded assets"](https://doi.org/10.1049/ein2.12011) paper. It has received significant upgrades.


[![GitHub stars](https://img.shields.io/github/stars/pik-piam/PyPSA-China-PIK?style=social)](https://github.com/pik-piam/PyPSA-China-PIK/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/pik-piam/PyPSA-China-PIK?style=social)](https://github.com/pik-piam/PyPSA-China-PIK/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://pik-piam.github.io/PyPSA-China-PIK/)