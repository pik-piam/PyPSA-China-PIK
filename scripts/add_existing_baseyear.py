# coding: utf-8

import logging
logger = logging.getLogger(__name__)

import pandas as pd
idx = pd.IndexSlice

import numpy as np
import xarray as xr

import pypsa
import yaml

from add_electricity import load_costs
from _helpers import override_component_attrs, define_spatial
from functions import pro_names, offwind_nodes

from types import SimpleNamespace
spatial = SimpleNamespace()

def add_build_year_to_new_assets(n, baseyear):
    """
    Parameters
    ----------
    n : pypsa.Network
    baseyear : int
        year in which optimized assets are built
    """

    # Give assets with lifetimes and no build year the build year baseyear
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        assets = c.df.index[(c.df.lifetime!=np.inf) & (c.df.build_year==0) & (c.df[attr + "_nom_extendable"]==True)]
        c.df.loc[assets, "build_year"] = baseyear

        # add -baseyear to name
        rename = pd.Series(c.df.index, c.df.index)
        rename[assets] += "-" + str(baseyear)
        c.df.rename(index=rename, inplace=True)

        # rename time-dependent
        selection = (
            n.component_attrs[c.name].type.str.contains("series")
            & n.component_attrs[c.name].status.str.contains("Input")
        )
        for attr in n.component_attrs[c.name].index[selection]:
            c.pnl[attr].rename(columns=rename, inplace=True)


def add_existing_capacities(df_agg):

    carrier = {
        "coal": "coal",
        "CHP": "coal",
        "OCGT": "gas",
        "solar": "solar",
        "onwind": "onwind",
        "offwind": "offwind",
        "coal boiler": "coal",
        "gas boiler": "gas",
        "resistive heater": "AC",
        "solar thermal": "solar thermal"}

    for tech in ['coal','CHP', 'OCGT','solar', 'onwind', 'offwind','coal boiler','resistive heater','solar thermal']:

        df = pd.read_csv(snakemake.input[f"existing_{tech}"], index_col=0).fillna(0.)
        df.columns = df.columns.astype(int)
        df = df.sort_index()

        for year in df.columns:
            for node in df.index:
                name = f"{node}-{tech}-{year}"
                capacity = df.loc[node, year]
                if capacity > 0.:
                    df_agg.at[name, "Fueltype"] = carrier[tech]
                    df_agg.at[name, "Tech"] = tech
                    df_agg.at[name, "Capacity"] = capacity
                    df_agg.at[name, "DateIn"] = year
                    df_agg.at[name, "cluster_bus"] = node


def add_power_capacities_installed_before_baseyear(n, grouping_years, costs, baseyear, config):
    """
    Parameters
    ----------
    n : pypsa.Network
    grouping_years :
        intervals to group existing capacities
    costs :
        to read lifetime to estimate YearDecomissioning
    baseyear : int
    """
    print("adding power capacities installed before baseyear")

    df_agg = pd.DataFrame()

    # include renewables in df_agg
    add_existing_capacities(df_agg)

    df_agg["grouping_year"] = np.take(
        grouping_years,
        np.digitize(df_agg.DateIn, grouping_years, right=True)
    )

    df = df_agg.pivot_table(
        index=["grouping_year", 'Tech'],
        columns='cluster_bus',
        values='Capacity',
        aggfunc='sum'
    )

    df.fillna(0)

    for grouping_year, generator in df.index:

        # capacity is the capacity in MW at each node for this
        capacity = df.loc[grouping_year, generator]
        capacity = capacity[~capacity.isna()]
        capacity = capacity[capacity > config['existing_capacities']['threshold_capacity']]

        if generator in ['solar', 'onwind', 'offwind']:
            p_max_pu = n.generators_t.p_max_pu[capacity.index + " " + generator]
            n.madd("Generator",
                   capacity.index,
                   suffix=' ' + generator + "-" + str(grouping_year),
                   bus=capacity.index,
                   carrier=generator,
                   p_nom=capacity,
                   p_nom_min=capacity,
                   p_nom_extendable=False,
                   marginal_cost=costs.at[generator, 'marginal_cost'],
                   capital_cost=costs.at[generator, 'capital_cost'],
                   efficiency=costs.at[generator, 'efficiency'],
                   p_max_pu=p_max_pu.rename(columns=n.generators.bus),
                   build_year=grouping_year,
                   lifetime=costs.at[generator, 'lifetime']
                   )

        if generator == "coal":
            n.madd("Generator",
                   capacity.index,
                   suffix=' ' + generator + "-" + str(grouping_year),
                   bus=capacity.index,
                   carrier=generator,
                   p_nom=capacity,
                   p_nom_min=capacity,
                   p_nom_extendable=False,
                   marginal_cost=costs.at[generator, 'efficiency'] * costs.at[generator, 'marginal_cost'],
                   capital_cost=costs.at[generator, 'efficiency'] * costs.at[generator, 'capital_cost'],
                   efficiency=costs.at[generator, 'efficiency'],
                   build_year=grouping_year,
                   lifetime=costs.at[generator, 'lifetime']
                   )

        if generator == "solar thermal":
            decentral_percentage = pd.read_csv('data/existing_infrastructure/decentral solar thermal percentrage.csv',index_col=0)
            decentral_percentage = decentral_percentage.squeeze()
            for cat in [" central "]:
                p_max_pu = n.generators_t.p_max_pu[capacity.index + cat + generator]
                p_max_pu.columns = capacity.index
                n.madd("Generator",
                       capacity.index,
                       suffix=cat + generator + "-" + str(grouping_year),
                       bus=capacity.index + cat + "heat",
                       carrier=generator,
                       p_nom=capacity*(1-decentral_percentage ),
                       p_nom_min=capacity*(1-decentral_percentage ),
                       p_nom_extendable=False,
                       marginal_cost=costs.at["central "+ generator, 'marginal_cost'],
                       capital_cost=costs.at["central " + generator, 'capital_cost'],
                       p_max_pu=p_max_pu,
                       build_year=grouping_year,
                       lifetime=costs.at["central " + generator, 'lifetime']
                       )
            for cat in [" decentral "]:
                p_max_pu = n.generators_t.p_max_pu[capacity.index + cat + generator]
                p_max_pu.columns = capacity.index
                n.madd("Generator",
                       capacity.index,
                       suffix=cat + generator + "-" + str(grouping_year),
                       bus=capacity.index + cat + "heat",
                       carrier=generator,
                       p_nom=capacity*decentral_percentage,
                       p_nom_min=capacity*decentral_percentage,
                       p_nom_extendable=False,
                       marginal_cost=costs.at["decentral " + generator, 'marginal_cost'],
                       capital_cost=costs.at["decentral " + generator, 'capital_cost'],
                       p_max_pu=p_max_pu,
                       build_year=grouping_year,
                       lifetime=costs.at["decentral " + generator, 'lifetime']
                       )

        if generator == "CHP":
            bus0 = capacity.index + " coal"
            n.madd("Link",
                   capacity.index,
                   suffix=" " + generator + "-" + str(grouping_year),
                   bus0=bus0,
                   bus1=capacity.index,
                   bus2=capacity.index + " central heat",
                   carrier=generator,
                   marginal_cost=costs.at['central coal CHP', 'efficiency'] * costs.at['central coal CHP', 'VOM'],  # NB: VOM is per MWel
                   capital_cost=costs.at['central coal CHP', 'efficiency'] * costs.at['central coal CHP', 'capital_cost'],  # NB: fixed cost is per MWel,
                   p_nom=capacity,
                   p_nom_min=capacity,
                   p_nom_extendable=False,
                   efficiency=config['chp_parameters']['eff_el'],
                   efficiency2=config['chp_parameters']['eff_th'],
                   build_year=grouping_year,
                   lifetime=costs.at["central coal CHP", 'lifetime']
                   )

        if generator == "OCGT":
            bus0 = capacity.index + " gas"
            n.madd("Link",
                   capacity.index,
                   suffix=" " + generator + "-" + str(grouping_year),
                   bus0=bus0,
                   bus1=capacity.index,
                   marginal_cost=costs.at[generator, 'efficiency'] * costs.at[generator, 'VOM'],  # NB: VOM is per MWel
                   capital_cost=costs.at[generator,'efficiency'] * costs.at[generator, 'capital_cost'],
                   # NB: fixed cost is per MWel
                   p_nom=capacity,
                   p_nom_min=capacity,
                   p_nom_extendable=False,
                   efficiency=costs.at[generator, 'efficiency'],
                   build_year=grouping_year,
                   lifetime=costs.at[generator,'lifetime']
                   )

        if generator == "coal boiler":
            bus0 = capacity.index + " coal"
            decentral_percentage = pd.read_csv('data/existing_infrastructure/decentral coal boiler percentrage.csv',index_col=0)
            decentral_percentage = decentral_percentage.squeeze()
            for cat in [" central "]:
                n.madd("Link",
                       capacity.index,
                       suffix="" + cat + generator + "-" + str(grouping_year),
                       bus0=bus0,
                       bus1=capacity.index + cat + "heat",
                       marginal_cost=costs.at[cat.lstrip() + generator, 'VOM'],
                       capital_cost=costs.at[cat.lstrip() + generator, 'efficiency'] * costs.at[
                           cat.lstrip() + generator, 'capital_cost'],
                       p_nom=capacity*(1-decentral_percentage ),
                       p_nom_min=capacity*(1-decentral_percentage ),
                       p_nom_extendable=False,
                       efficiency=costs.at[cat.lstrip() + generator, 'efficiency'],
                       build_year=grouping_year,
                       lifetime=costs.at[cat.lstrip() + generator, 'lifetime']
                       )
            for cat in [" decentral "]:
                n.madd("Link",
                       capacity.index,
                       suffix="" + cat + generator + "-" + str(grouping_year),
                       bus0=bus0,
                       bus1=capacity.index + cat + "heat",
                       marginal_cost=costs.at[cat.lstrip() + generator, 'VOM'],
                       capital_cost=costs.at[cat.lstrip() + generator, 'efficiency'] * costs.at[
                           cat.lstrip() + generator, 'capital_cost'],
                       p_nom=capacity*decentral_percentage,
                       p_nom_min=capacity*decentral_percentage,
                       p_nom_extendable=False,
                       efficiency=costs.at[cat.lstrip() + generator, 'efficiency'],
                       build_year=grouping_year,
                       lifetime=costs.at[cat.lstrip() + generator, 'lifetime']
                       )

        if generator == "resistive heater":
            for cat in [" decentral "]:
                n.madd("Link",
                       capacity.index,
                       suffix=' ' + cat + generator + "-" + str(grouping_year),
                       bus0=capacity.index,
                       bus1=capacity.index + cat + "heat",
                       carrier=generator,
                       p_nom=capacity,
                       p_nom_min=capacity,
                       p_nom_extendable=False,
                       capital_cost=costs.at[cat.lstrip() + generator, 'efficiency'] * costs.at[cat.lstrip() + generator, 'capital_cost'],
                       efficiency=costs.at[cat.lstrip() + generator, 'efficiency'],
                       build_year=grouping_year,
                       lifetime=costs.at[cat.lstrip() + generator, 'lifetime']
                       )


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'add_existing_baseyear',
            co2_reduction='1.0',
            opts='ll',
            planning_horizons=2020
        )

    logging.basicConfig(level=snakemake.config['logging']['level'])

    options = snakemake.config["sector"]
    # sector_opts = '168H-T-H-B-I-solar+p3-dist1'
    # opts = sector_opts.split('-')

    baseyear = snakemake.config['scenario']["planning_horizons"][0]

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    # define spatial resolution of carriers
    spatial = define_spatial(n.buses[n.buses.carrier=="AC"].index, options)
    # add_build_year_to_new_assets(n, baseyear)

    Nyears = n.snapshot_weightings.generators.sum() / 8760.

    config = snakemake.config
    tech_costs = snakemake.input.tech_costs
    cost_year = snakemake.wildcards.planning_horizons
    costs = load_costs(tech_costs,config['costs'],config['electricity'],cost_year, Nyears)

    grouping_years = config['existing_capacities']['grouping_years']
    add_power_capacities_installed_before_baseyear(n, grouping_years, costs, baseyear, config)

    ## update renewable potentials

    # for tech in ['onwind', 'offwind', 'solar']:
    #     if tech == 'offwind':
    #         for node in offwind_nodes:
    #             n.generators.p_nom_max.loc[(n.generators.bus == node) & (n.generators.carrier == tech)] -= \
    #             n.generators[(n.generators.bus == node) & (n.generators.carrier == tech)].p_nom.sum()
    #     else:
    #         for node in pro_names:
    #             n.generators.p_nom_max.loc[(n.generators.bus == node) & (n.generators.carrier == tech)] -= \
    #             n.generators[(n.generators.bus == node) & (n.generators.carrier == tech)].p_nom.sum()

    n.export_to_netcdf(snakemake.output[0])
