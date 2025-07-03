# coding: utf-8

"""
Functions for myopic pathway network generation snakemake rules

Add assets from previous planning horizon network solution to
network to solve for the current planning horizon.

Usage:
- use via a snakemake rule
- debug: use as standalone with mock_snakemake (reads snakefile)
"""
import logging
import pandas as pd
import pypsa

import numpy as np
import xarray as xr

from add_existing_baseyear import add_build_year_to_new_assets
from _helpers import mock_snakemake, configure_logging
from constants import PROV_NAMES, OFFSHORE_WIND_NODES

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


def add_brownfield(n: pypsa.Network, n_p: pypsa.Network, year: int):
    """Add paid for assets as p_nom to the current network

    Args:
        n (pypsa.Network): next network to prep & optimize in the planning horizon
        n_p (pypsa.Network): previous optimized network
        year (int): the planning year
    """

    logger.info("Adding brownfield")
    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    # dc_i = n.links[n.links.carrier=="DC"].index
    # n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]
    # update links
    n.links.loc[(n.links.length > 0) & (n.links.lifetime == np.inf), "p_nom"] = n_p.links.loc[
        (n_p.links.carrier == "AC") & (n_p.links.build_year == 0), "p_nom_opt"
    ]
    n.links.loc[(n.links.length > 0) & (n.links.lifetime == np.inf), "p_nom_min"] = n_p.links.loc[
        (n_p.links.carrier == "AC") & (n_p.links.build_year == 0), "p_nom_opt"
    ]

    if year == 2025:
        add_build_year_to_new_assets(n_p, 2020)

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):

        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.mremove(c.name, c.df.index[c.df.lifetime == np.inf])
        # remove assets whose build_year + lifetime < year
        n_p.mremove(c.name, c.df.index[c.df.build_year + c.df.lifetime < year])
        # remove assets if their optimized nominal capacity is lower than a threshold
        # since CHP heat Link is proportional to CHP electric Link, ensure threshold is compatible
        chp_heat = c.df.index[(c.df[attr + "_nom_extendable"] & c.df.index.str.contains("CHP"))]

        threshold = snakemake.config["existing_capacities"]["threshold_capacity"]

        if not chp_heat.empty:
            threshold_chp_heat = (
                threshold * c.df.loc[chp_heat].efficiency2 / c.df.loc[chp_heat].efficiency
            )
            n_p.mremove(
                c.name,
                chp_heat[c.df.loc[chp_heat, attr + "_nom_opt"] < threshold_chp_heat],
            )

        n_p.mremove(
            c.name,
            c.df.index[
                c.df[attr + "_nom_extendable"]
                & ~c.df.index.isin(chp_heat)
                & (c.df[attr + "_nom_opt"] < threshold)
            ],
        )

        # copy over assets but fix their capacity
        c.df[attr + "_nom"] = c.df[attr + "_nom_opt"]
        c.df[attr + "_nom_extendable"] = False
        c.df[attr + "_nom_max"] = np.inf

        n.import_components_from_dataframe(c.df, c.name)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains("series") & n.component_attrs[
            c.name
        ].status.str.contains("Input")

        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr].set_index(n.snapshots), c.name, tattr)

    for tech in ["onwind", "offwind", "solar"]:
        ds_tech = xr.open_dataset(snakemake.input["profile_" + tech])
        p_nom_max_initial = ds_tech["p_nom_max"].to_pandas()

        if tech == "offwind":
            for node in OFFSHORE_WIND_NODES:
                n.generators.loc[
                    (n.generators.bus == node)
                    & (n.generators.carrier == tech)
                    & (n.generators.build_year == year),
                    "p_nom_max",
                ] = (
                    p_nom_max_initial[node]
                    - n_p.generators[
                        (n_p.generators.bus == node) & (n_p.generators.carrier == tech)
                    ].p_nom_opt.sum()
                )
        else:
            for node in PROV_NAMES:
                n.generators.loc[
                    (n.generators.bus == node)
                    & (n.generators.carrier == tech)
                    & (n.generators.build_year == year),
                    "p_nom_max",
                ] = (
                    p_nom_max_initial[node]
                    - n_p.generators[
                        (n_p.generators.bus == node) & (n_p.generators.carrier == tech)
                    ].p_nom_opt.sum()
                )

    n.generators.loc[(n.generators.p_nom_max < 0), "p_nom_max"] = 0

    # retrofit coal power plant with carbon capture
    n.generators.loc[n.generators.carrier == "coal power plant", "p_nom_extendable"] = True
    n.generators.loc[
        n.generators.index.str.contains("retrofit") & ~n.generators.index.str.contains(str(year)),
        "p_nom_extendable",
    ] = False


if __name__ == "__main__":

    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_brownfield",
            topology="current+FCG",
            co2_pathway="exp175default",
            heating_demand="positive",
            planning_horizons=2025,
        )

    configure_logging(snakemake, logger=logger)

    year = int(snakemake.wildcards.planning_horizons)

    n = pypsa.Network(snakemake.input.network)

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p)

    add_brownfield(n, n_p, year)

    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    n.export_to_netcdf(snakemake.output.network_name, compression=compression)

    logger.info(
        f"Brownfield extension successfully added for {snakemake.wildcards.planning_horizons}"
    )
