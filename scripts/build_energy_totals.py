import logging
from _helpers import configure_logging

from functions import pro_names
import pandas as pd

logger = logging.getLogger(__name__)

idx = pd.IndexSlice

nodes = pd.Index(pro_names)


def build_energy_totals():
    """ list the provinces' annual space heating, hot water and electricity consumption """

    with pd.HDFStore(snakemake.input.infile1, mode='r') as store:
        population_count = store['population']

    with pd.HDFStore(snakemake.input.infile2, mode='r') as store:
        population_gridcell_map = store['population_gridcell_map']

    # using DH data 2016 assuming relatively good insulation
    # MWh/hour/C/m2  source: urbanization yearbooks
    unit_space_heating = 2.4769322112272924e-06 

    # In 2010, the city of Helsingborg, Sweden used 28% of its total heating 4270TJ for hot water.
    # and it has a population of 100,000 
    # source: Svend DH book and wiki
    # MWh/capita/year = 4270 * 1e12 / 3.6e9 * 0.28 / 1e5
    #unit_hot_water = 3.321111111111111

    # In 2008 China 228.4 Twh for urban residential DHW
    # MWh/capita/year = 228.4 * 1e6 / 62403/1e4 = 0.366008
    # MWh/capita/year = 228.4 * 1e6 / 141013/1e4 = 0.161971
    unit_hot_water = 0.366008

    # m2/capital source: urbanization yearbooks
    if snakemake.wildcards.planning_horizons == 2020:
        floor_space_per_capita = 18.57 # 1226627/66067
    elif snakemake.wildcards.planning_horizons == 2025:
        floor_space_per_capita = 18.57
    elif snakemake.wildcards.planning_horizons == 2030:
        floor_space_per_capita = 18.57
    elif snakemake.wildcards.planning_horizons == 2035:
        floor_space_per_capita = 18.57
    elif snakemake.wildcards.planning_horizons == 2040:
        floor_space_per_capita = 18.57
    elif snakemake.wildcards.planning_horizons == 2045:
        floor_space_per_capita = 18.57
    elif snakemake.wildcards.planning_horizons == 2050:
        floor_space_per_capita = 18.57
    elif snakemake.wildcards.planning_horizons == 2055:
        floor_space_per_capita = 18.57
    else:
        floor_space_per_capita = 18.57

    #2020 27.28; 2025 32.75; 2030 36.98; 2035 40.11; 2040 42.34; 2045 43.89; 2050 44.96; 2055 45.68; 2060 46.18

    # MWh per hdh
    space_heating_per_hdd = unit_space_heating * floor_space_per_capita * population_count

    # MWh per day
    hot_water_per_day = unit_hot_water * population_count / 365.

    with pd.HDFStore(snakemake.output.outfile1, mode='w', complevel=4) as store:
        store['space_heating_per_hdd'] = space_heating_per_hdd

    with pd.HDFStore(snakemake.output.outfile1, mode='a', complevel=4) as store:
        store['hot_water_per_day'] = hot_water_per_day

    return space_heating_per_hdd, hot_water_per_day


if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_energy_totals',
                                   planning_horizons=2020)
    configure_logging(snakemake)

    space_heating_per_hdd, hot_water_per_day = build_energy_totals()
