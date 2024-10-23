# PyPSA-China：An Open Optimisation model of the Chinese Energy System

An exploratory version for the REMIND-PyPSA PANDA coupling based on the version published by Xiaowei Zhou et al for their  "Multi-energy system horizon planning: Early decarbonisation in China avoids stranded assets" (doi.org/10.1049/ein2.12011)

Adapted by Ivan Ramirez

## Installation

### Getting the data
- some of the data is downloaded by the snakemake workflow (e.g. cutouts). Just make sure te relevant config options are set to true if it is your first run
- the shapely files can be generated with the build_province_shapes script