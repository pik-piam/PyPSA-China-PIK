# PyPSA-China：An Open Optimisation model of the Chinese Energy System

An exploratory version for the REMIND-PyPSA PANDA coupling based on the version published by Xiaowei Zhou et al for their  "Multi-energy system horizon planning: Early decarbonisation in China avoids stranded assets" (doi.org/10.1049/ein2.12011)

Adapted by Ivan Ramirez

## Changelog
- restructure project to match snakemake8 guidelines & update to snakemake8
- move hardcoded to centralised store constants.py (file paths still partially hardcoded)
- start adding typing
- add scripts to pull data
- add derived_data/ folder and change target of data cleaning/prep steps for clarity

## TODOs:
- separate functionalities
- add sources to all data
- integrate various into settings

## Installation

### Getting the data
- some of the data is downloaded by the snakemake workflow (e.g. cutouts). Just make sure te relevant config options are set to true if it is your first run
- the shapely files can be generated with the build_province_shapes script
- the zeonodo link now comes with the data but in the old format
