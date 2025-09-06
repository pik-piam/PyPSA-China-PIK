"""Function suite and script to define the network to be solved. Network components are added here.
Additional constraints require the linopy model and are added in the solve_network script.

These functions are currently only for the overnight mode. Myopic pathway mode contains near
        duplicates which need to merged in the future. Idem for solve_network.py
"""

# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# for non-pathway network
# TODO WHY DO WE USE VRESUTILS ANNUITY IN ONE PLACE AND OUR OWN CALC ELSEWHERE?

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import ConfigManager, configure_logging, mock_snakemake
from _pypsa_helpers import (
    assign_locations,
    make_periodic_snapshots,
    shift_profile_to_planning_year,
)
from add_electricity import load_costs, sanitize_carriers
from build_biomass_potential import estimate_co2_intensity_xing
from constants import (
    CRS,
    FOM_LINES,
    INFLOW_DATA_YR,
    NUCLEAR_EXTENDABLE,
    PROV_NAMES,
)
from functions import haversine
from readers_geospatial import read_province_shapes
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# TODO add a heat bus that can absorb heat for free in non-coupled mode
#   (e.g. Hydrogen electrolysis, sabatier)
# TODO add heat disipator?


def add_biomass_chp(
    network: pypsa.Network,
    costs: pd.DataFrame,
    nodes: pd.Index,
    biomass_potential: pd.DataFrame,
    prov_centroids: gpd.GeoDataFrame,
    add_beccs: bool = True,
):
    """Add biomass to the network. Biomass is here a new build (and not a retrofit)
    and is not co-fired with coal. An optional CC can be added to biomass

    NOTE THAT THE CC IS NOT CONSTRAINED TO THE BIOMASS?

    Args:
        network (pypsa.Network): the pypsa network
        costs (pd.DataFrame): the costs dataframe
        nodes (pd.Index): the nodes
        biomass_potential (pd.DataFrame): the biomass potential
        prov_centroids (gpd.GeoDataFrame): the x,y locations of the nodes
        add_beccs (bool, optional): whether to add BECCS. Defaults to True.
    """

    suffix = " biomass"
    biomass_potential.index = biomass_potential.index.map(
        lambda x: x + suffix if not x.endswith(suffix) else x
    )

    network.add(
        "Bus",
        nodes,
        suffix=suffix,
        x=prov_centroids.x,
        y=prov_centroids.y,
        carrier="biomass",
    )
    logger.info("Adding biomass buses")
    logger.info(f"{nodes + suffix}")
    logger.info("potentials")
    # aggricultural residue biomass
    # NOTE THIS CURRENTLY DOESN'T INCLUDE TRANSPORT between nodes
    # NOTE additional emissions from treatment/remedials are missing
    network.add(
        "Store",
        nodes + suffix,
        bus=nodes + suffix,
        e_nom_extendable=False,
        e_nom=biomass_potential,
        e_initial=biomass_potential,
        carrier="biomass",
    )
    biomass_co2_intsty = estimate_co2_intensity_xing()
    network.add(
        "Link",
        nodes + " central biomass CHP",
        bus0=nodes + " biomass",
        bus1=nodes,
        bus2=nodes + " central heat",
        bus3=nodes + " CO2",
        p_nom_extendable=True,
        carrier="biomass",
        efficiency=costs.at["biomass CHP", "efficiency"],
        efficiency2=costs.at["biomass CHP", "efficiency-heat"],
        efficiency3=biomass_co2_intsty,
        capital_cost=costs.at["biomass CHP", "efficiency"]
        * costs.at["biomass CHP", "capital_cost"],
        marginal_cost=costs.at["biomass CHP", "efficiency"]
        * costs.at["biomass CHP", "marginal_cost"]
        + costs.at["solid biomass", "fuel"],
        lifetime=costs.at["biomass CHP", "lifetime"],
    )
    if add_beccs:
        network.add(
            "Link",
            nodes + " central biomass CHP capture",
            bus0=nodes + " CO2",
            bus1=nodes + " CO2 capture",
            bus2=nodes,
            p_nom_extendable=True,
            carrier="CO2 capture",
            efficiency=costs.at["biomass CHP capture", "capture_rate"],
            efficiency2=-1
            * costs.at["biomass CHP capture", "capture_rate"]
            * costs.at["biomass CHP capture", "electricity-input"],
            capital_cost=costs.at["biomass CHP capture", "capture_rate"]
            * costs.at["biomass CHP capture", "capital_cost"],
            lifetime=costs.at["biomass CHP capture", "lifetime"],
        )

    network.add(
        "Link",
        nodes + " decentral biomass boiler",
        bus0=nodes + " biomass",
        bus1=nodes + " decentral heat",
        p_nom_extendable=True,
        carrier="biomass",
        efficiency=costs.at["biomass boiler", "efficiency"],
        capital_cost=costs.at["biomass boiler", "efficiency"]
        * costs.at["biomass boiler", "capital_cost"],
        marginal_cost=costs.at["biomass boiler", "efficiency"]
        * costs.at["biomass boiler", "marginal_cost"]
        + costs.at["biomass boiler", "pelletizing cost"]
        + costs.at["solid biomass", "fuel"],
        lifetime=costs.at["biomass boiler", "lifetime"],
    )


def add_carriers(network: pypsa.Network, config: dict, costs: pd.DataFrame):
    """Add the various carriers to the network based on the config file

    Args:
        network (pypsa.Network): the pypsa network
        config (dict): the config file
        costs (pd.DataFrame): the costs dataframe
    """

    network.add("Carrier", "AC")
    if config.get("heat_coupling", False):
        network.add("Carrier", "heat")
    for carrier in config["Techs"]["vre_techs"]:
        network.add("Carrier", carrier)
        if carrier == "hydroelectricity":
            network.add("Carrier", "hydro_inflow")
    for carrier in config["Techs"]["store_techs"]:
        network.add("Carrier", carrier)
        if carrier == "battery":
            network.add("Carrier", "battery discharger")

    # add fuel carriers, emissions in # in t_CO2/MWht
    if config["add_gas"]:
        network.add("Carrier", "gas", co2_emissions=costs.at["gas", "co2_emissions"])
    if config["add_coal"]:
        network.add("Carrier", "coal", co2_emissions=costs.at["coal", "co2_emissions"])
    if "CCGT-CCS" in config["Techs"]["conv_techs"]:
        network.add("Carrier", "gas ccs", co2_emissions=costs.at["gas ccs", "co2_emissions"])
    if "coal-CCS" in config["Techs"]["conv_techs"]:
        network.add("Carrier", "coal ccs", co2_emissions=costs.at["coal ccs", "co2_emissions"])


def add_co2_capture_support(
    network: pypsa.Network, nodes: pd.Index, prov_centroids: gpd.GeoDataFrame
):
    """Add the necessary CO2 capture carriers & stores to the network
    Args:
        network (pypsa.Network): the network object
        nodes (pd.Index): the nodes
        prov_centroids (gpd.GeoDataFrame): the x,y locations of the nodes
    """

    network.add("Carrier", "CO2", co2_emissions=0)
    network.add(
        "Bus",
        nodes,
        suffix=" CO2",
        x=prov_centroids.x,
        y=prov_centroids.y,
        carrier="CO2",
    )

    network.add("Store", nodes + " CO2", bus=nodes + " CO2", carrier="CO2")
    # normally taking away from carrier generates CO2, but here we are
    # adding CO2 stored, so the emissions will point the other way ?
    network.add("Carrier", "CO2 capture", co2_emissions=1)
    network.add(
        "Bus",
        nodes,
        suffix=" CO2 capture",
        x=prov_centroids.x,
        y=prov_centroids.y,
        carrier="CO2 capture",
    )

    network.add(
        "Store",
        nodes + " CO2 capture",
        bus=nodes + " CO2 capture",
        e_nom_extendable=True,
        carrier="CO2 capture",
    )


def add_conventional_generators(
    network: pypsa.Network,
    nodes: pd.Index,
    config: dict,
    prov_centroids: gpd.GeoDataFrame,
    costs: pd.DataFrame,
):
    """Add conventional generators to the network

    Args:
        network (pypsa.Network): the pypsa network object
        nodes (pd.Index): the nodes
        config (dict): the snakemake config
        prov_centroids (gpd.GeoDataFrame): the x,y locations of the nodes
        costs (pd.DataFrame): the costs data base
    """
    if config["add_gas"]:
        # add converter from fuel source
        network.add(
            "Bus",
            nodes,
            suffix=" gas",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="gas",
            location=nodes,
        )

        network.add(
            "Generator",
            nodes,
            suffix=" gas fuel",
            bus=nodes + " gas",
            carrier="gas",
            p_nom_extendable=True,
            p_nom=1e7,
            marginal_cost=costs.at["gas", "fuel"],
        )

        # gas prices identical per region, pipelines ignored
        network.add(
            "Store",
            nodes + " gas Store",
            bus=nodes + " gas",
            e_nom_extendable=True,
            carrier="gas",
            e_nom=1e7,
            e_cyclic=True,
        )

    # add gas will then be true
    gas_techs = ["OCGT", "CCGT"]
    for tech in gas_techs:
        if tech in config["Techs"]["conv_techs"]:
            network.add(
                "Link",
                nodes,
                suffix=f" {tech}",
                bus0=nodes + " gas",
                bus1=nodes,
                marginal_cost=costs.at[tech, "efficiency"]
                * costs.at[tech, "VOM"],  # NB: VOM is per MWel
                capital_cost=costs.at[tech, "efficiency"]
                * costs.at[tech, "capital_cost"],  # NB: capital cost is per MWel
                p_nom_extendable=True,
                efficiency=costs.at[tech, "efficiency"],
                lifetime=costs.at[tech, "lifetime"],
                carrier=f"gas {tech}",
            )

    if "CCGT-CCS" in config["Techs"]["conv_techs"]:
        network.add(
            "Generator",
            nodes,
            suffix=" CCGT-CCS",
            bus=nodes,
            carrier="gas ccs",
            p_nom_extendable=True,
            efficiency=costs.at["CCGT-CCS", "efficiency"],
            marginal_cost=costs.at["CCGT-CCS", "marginal_cost"],
            capital_cost=costs.at["CCGT-CCS", "efficiency"]
            * costs.at["CCGT-CCS", "capital_cost"],  # NB: capital cost is per MWel
            lifetime=costs.at["CCGT-CCS", "lifetime"],
            p_max_pu=0.9,  # planned and forced outages
        )

    if config["add_coal"]:
        ramps = config.get(
            "fossil_ramps", {"coal": {"ramp_limit_up": np.nan, "ramp_limit_down": np.nan}}
        )
        ramps = ramps.get("coal", {"ramp_limit_up": np.nan, "ramp_limit_down": np.nan})
        ramps = {k: v * config["snapshots"]["frequency"] for k, v in ramps.items()}
        # this is the non sector-coupled approach
        # for industry may have an issue in that coal feeds to chem sector
        # no coal in Beijing - political decision
        network.add(
            "Generator",
            nodes[nodes != "Beijing"],
            suffix=" coal power",
            bus=nodes[nodes != "Beijing"],
            carrier="coal",
            p_nom_extendable=True,
            efficiency=costs.at["coal", "efficiency"],
            marginal_cost=costs.at["coal", "marginal_cost"],
            capital_cost=costs.at["coal", "efficiency"]
            * costs.at["coal", "capital_cost"],  # NB: capital cost is per MWel
            lifetime=costs.at["coal", "lifetime"],
            ramp_limit_up=ramps["ramp_limit_up"],
            ramp_limit_down=ramps["ramp_limit_down"],
            p_max_pu=0.9,  # planned and forced outages
        )

        if "coal-CCS" in config["Techs"]["conv_techs"]:
            network.add(
                "Generator",
                nodes,
                suffix=" coal-CCS",
                bus=nodes,
                carrier="coal ccs",
                p_nom_extendable=True,
                efficiency=costs.at["coal ccs", "efficiency"],
                marginal_cost=costs.at["coal ccs", "marginal_cost"],
                capital_cost=costs.at["coal ccs", "efficiency"]
                * costs.at["coal ccs", "capital_cost"],  # NB: capital cost is per MWel
                lifetime=costs.at["coal ccs", "lifetime"],
                p_max_pu=0.9,  # planned and forced outages
            )


def add_H2(network: pypsa.Network, config: dict, nodes: pd.Index, costs: pd.DataFrame):
    """Add H2 generators, storage and links to the network - currently all or nothing

    Args:
        network (pypsa.Network): network object too which H2 comps will be added
        config (dict): the config (snakemake config)
        nodes (pd.Index): the buses
        costs (pd.DataFrame): the cost database
    """
    # TODO, does it make sense?
    if config.get("heat_coupling", False):
        network.add(
            "Link",
            name=nodes + " H2 Electrolysis",
            bus0=nodes,
            bus1=nodes + " H2",
            bus2=nodes + " central heat",
            p_nom_extendable=True,
            carrier="H2 Electrolysis",
            efficiency=costs.at["electrolysis", "efficiency"],
            efficiency2=costs.at["electrolysis", "efficiency-heat"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            lifetime=costs.at["electrolysis", "lifetime"],
        )
    else:
        network.add(
            "Link",
            name=nodes + " H2 Electrolysis",
            bus0=nodes,
            bus1=nodes + " H2",
            p_nom_extendable=True,
            carrier="H2 Electrolysis",
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            lifetime=costs.at["electrolysis", "lifetime"],
        )

    if "fuel cell" in config["Techs"]["vre_techs"]:
        network.add(
            "Link",
            name=nodes + " H2 Fuel Cell",
            bus0=nodes + " H2",
            bus1=nodes,
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            capital_cost=costs.at["fuel cell", "efficiency"]
            * costs.at["fuel cell", "capital_cost"],
            lifetime=costs.at["fuel cell", "lifetime"],
            carrier="H2 fuel cell",
        )
    if "H2 turbine" in config["Techs"]["vre_techs"]:
        network.add(
            "Link",
            name=nodes + " H2 turbine",
            bus0=nodes + " H2",
            bus1=nodes,
            p_nom_extendable=True,
            efficiency=costs.at["H2 turbine", "efficiency"],
            capital_cost=costs.at["H2 turbine", "efficiency"]
            * costs.at["H2 turbine", "capital_cost"],
            lifetime=costs.at["H2 turbine", "lifetime"],
            carrier="H2 turbine",
        )

    H2_under_nodes_ = pd.Index(config["H2"]["geo_storage_nodes"])
    H2_type1_nodes_ = nodes.difference(H2_under_nodes_)
    H2_under_nodes = H2_under_nodes_.intersection(nodes)
    H2_type1_nodes = H2_type1_nodes_.intersection(nodes)
    if not (
        H2_under_nodes_.shape == H2_under_nodes.shape
        and H2_type1_nodes_.shape == H2_type1_nodes.shape
    ):
        logger.warning("Some H2 storage nodes are not in the network buses")

    network.add(
        "Store",
        H2_under_nodes + " H2 Store",
        bus=H2_under_nodes + " H2",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=costs.at["hydrogen storage underground", "capital_cost"],
        lifetime=costs.at["hydrogen storage underground", "lifetime"],
    )

    # TODO harmonize with remind (add if in techs)
    network.add(
        "Store",
        H2_type1_nodes + " H2 Store",
        bus=H2_type1_nodes + " H2",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=costs.at["hydrogen storage tank type 1 including compressor", "capital_cost"],
        lifetime=costs.at["hydrogen storage tank type 1 including compressor", "lifetime"],
    )
    if config["add_methanation"]:
        cost_year = snakemake.wildcards["planning_horizons"]
        network.add(
            "Link",
            nodes + " Sabatier",
            bus0=nodes + " H2",
            bus1=nodes + " gas",
            carrier="Sabatier",
            p_nom_extendable=True,
            efficiency=costs.at["methanation", "efficiency"],
            capital_cost=costs.at["methanation", "efficiency"]
            * costs.at["methanation", "capital_cost"]
            + costs.at["direct air capture", "capital_cost"]
            * costs.at["gas", "co2_emissions"]
            * costs.at["methanation", "efficiency"],
            # TODO fix me
            lifetime=costs.at["methanation", "lifetime"],
            marginal_cost=(400 - 5 * (int(cost_year) - 2020))
            * costs.at["gas", "co2_emissions"]
            * costs.at["methanation", "efficiency"],
        )

    if config["Techs"]["hydrogen_lines"]:
        edge_path = config["edge_paths"].get(config["scenario"]["topology"], None)
        if edge_path is None:
            raise ValueError(f"No grid found for topology {config['scenario']['topology']}")
        else:
            edges_ = pd.read_csv(
                edge_path, sep=",", header=None, names=["bus0", "bus1", "p_nom"]
            ).fillna(0)
            edges = edges_[edges_["bus0"].isin(nodes) & edges_["bus1"].isin(nodes)]
            if edges_.shape[0] != edges.shape[0]:
                logger.warning("Some edges are not in the network buses")

        # fix this to use map with x.y
        lengths = config["lines"]["line_length_factor"] * np.array(
            [
                haversine(
                    [network.buses.at[bus0, "x"], network.buses.at[bus0, "y"]],
                    [network.buses.at[bus1, "x"], network.buses.at[bus1, "y"]],
                )
                for bus0, bus1 in edges[["bus0", "bus1"]].values
            ]
        )

        # TODO harmonize with remind (add if in techs)
        cc = costs.at["H2 (g) pipeline", "capital_cost"] * lengths

        # === h2 pipeline with losses ====
        # NB this only works if there is an equalising constraint, which is hidden in solve_ntwk
        network.add(
            "Link",
            edges["bus0"] + "-" + edges["bus1"] + " H2 pipeline",
            suffix=" positive",
            bus0=edges["bus0"].values + " H2",
            bus1=edges["bus1"].values + " H2",
            bus2=edges["bus0"].values,
            carrier="H2 pipeline",
            p_nom_extendable=True,
            p_nom=0,
            p_nom_min=0,
            p_min_pu=0,
            efficiency=config["transmission_efficiency"]["H2 pipeline"]["efficiency_static"]
            * config["transmission_efficiency"]["H2 pipeline"]["efficiency_per_1000km"]
            ** (lengths / 1000),
            efficiency2=-config["transmission_efficiency"]["H2 pipeline"]["compression_per_1000km"]
            * lengths
            / 1e3,
            length=lengths,
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
            capital_cost=cc,
        )

        network.add(
            "Link",
            edges["bus0"] + "-" + edges["bus1"] + " H2 pipeline",
            suffix=" reversed",
            carrier="H2 pipeline",
            bus0=edges["bus1"].values + " H2",
            bus1=edges["bus0"].values + " H2",
            bus2=edges["bus1"].values,
            p_nom_extendable=True,
            p_nom=0,
            p_nom_min=0,
            p_min_pu=0,
            efficiency=config["transmission_efficiency"]["H2 pipeline"]["efficiency_static"]
            * config["transmission_efficiency"]["H2 pipeline"]["efficiency_per_1000km"]
            ** (lengths / 1000),
            efficiency2=-config["transmission_efficiency"]["H2 pipeline"]["compression_per_1000km"]
            * lengths
            / 1e3,
            length=lengths,
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
            capital_cost=0,
        )


# TODO harmonize with remind
def add_voltage_links(network: pypsa.Network, config: dict):
    """Add HVDC/AC links (no KVL)

    Args:
        network (pypsa.Network): the network object
        config (dict): the snakemake config

    Raises:
        ValueError: Invalid Edge path in config options
    """

    represented_hours = network.snapshot_weightings.sum()[0]
    n_years = represented_hours / 8760.0

    # determine topology
    edge_path = config["edge_paths"].get(config["scenario"]["topology"], None)
    if edge_path is None:
        raise ValueError(f"No grid found for topology {config['scenario']['topology']}")
    else:
        edges_ = pd.read_csv(
            edge_path, sep=",", header=None, names=["bus0", "bus1", "p_nom"]
        ).fillna(0)
        edges = edges_[edges_["bus0"].isin(PROV_NAMES) & edges_["bus1"].isin(PROV_NAMES)]
        if edges_.shape[0] != edges.shape[0]:
            logger.warning("Some edges are not in the network")
    # fix this to use map with x.y
    lengths = config["lines"]["line_length_factor"] * np.array(
        [
            haversine(
                [network.buses.at[bus0, "x"], network.buses.at[bus0, "y"]],
                [network.buses.at[bus1, "x"], network.buses.at[bus1, "y"]],
            )
            for bus0, bus1 in edges[["bus0", "bus1"]].values
        ]
    )

    # get for backward compatibility
    security_config = config.get("security", {"line_security_margin": 70})
    line_margin = security_config.get("line_security_margin", 70) / 100

    line_cost = (
        lengths * costs.at["HVDC overhead", "capital_cost"] * FOM_LINES * n_years
    ) + costs.at["HVDC inverter pair", "capital_cost"]  # /MW

    # ==== lossy transport model (split into 2) ====
    # NB this only works if there is an equalising constraint, which is hidden in solve_ntwk
    if config["line_losses"]:
        network.add(
            "Link",
            edges["bus0"] + "-" + edges["bus1"],
            bus0=edges["bus0"].values,
            bus1=edges["bus1"].values,
            suffix=" positive",
            p_nom_extendable=True,
            p_nom=edges["p_nom"].values,
            p_nom_min=edges["p_nom"].values,
            p_min_pu=0,
            p_max_pu=line_margin,
            efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]
            * config["transmission_efficiency"]["DC"]["efficiency_per_1000km"] ** (lengths / 1000),
            length=lengths,
            capital_cost=line_cost,
            carrier="AC",  # Fake - actually DC
        )
        # 0 len for reversed in case line limits are specified in km. Limited in constraints to fwdcap
        network.add(
            "Link",
            edges["bus0"] + "-" + edges["bus1"],
            bus0=edges["bus1"].values,
            bus1=edges["bus0"].values,
            suffix=" reversed",
            p_nom_extendable=True,
            p_nom=edges["p_nom"].values,
            p_nom_min=edges["p_nom"].values,
            p_min_pu=0,
            efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]
            * config["transmission_efficiency"]["DC"]["efficiency_per_1000km"] ** (lengths / 1000),
            length=0,
            capital_cost=0,
            carrier="AC",
        )
    # lossless transport model
    else:
        network.add(
            "Link",
            edges["bus0"] + "-" + edges["bus1"],
            p_nom=edges["p_nom"].values,
            p_nom_min=edges["p_nom"].values,
            bus0=edges["bus0"].values,
            bus1=edges["bus1"].values,
            p_nom_extendable=True,
            p_min_pu=-1,
            length=lengths,
            capital_cost=line_cost,
            carrier="AC",
        )


def add_wind_and_solar(
    network: pypsa.Network,
    techs: list,
    paths: os.PathLike,
    year: int,
    costs: pd.DataFrame,
):
    """
    Adds wind and solar generators for each grade of renewable energy technology

    Args:
        network (pypsa.Network): The PyPSA network to which the generators will be added
        techs (list): A list of renewable energy technologies to add.
            (e.g., ["solar", "onwind", "offwind"])
        paths (os.PathLike): file paths containing renewable profiles (snakemake.input)
        year (int): planning year
        costs (pd.DataFrame): cost parameters for each technology
    Raises:
        ValueError: for unsupported technologies or missing paths.
    """

    unsupported = set(techs).difference({"solar", "onwind", "offwind"})
    if unsupported:
        raise ValueError(f"Carrier(s) {unsupported} not wind or solar pv")
    prof_paths = {f"profile_{tech}": paths[f"profile_{tech}"] for tech in techs}
    if len(prof_paths) != len(techs):
        raise ValueError(f"Paths do not correspond to techs  ({prof_paths} vs {techs})")

    for tech in techs:
        # load the renewable profiles
        logger.info(f"Attaching {tech} to network")
        with xr.open_dataset(prof_paths[f"profile_{tech}"]) as ds:
            if ds.indexes["bus"].empty:
                continue
            if "year" in ds.indexes:
                ds = ds.sel(year=ds.year.min(), drop=True)

            timestamps = pd.DatetimeIndex(ds.time)
            def shift_weather_to_planning_yr(t):
                """Shift weather data to planning year."""
                return t.replace(year=int(year))
            timestamps = timestamps.map(shift_weather_to_planning_yr)
            ds = ds.assign_coords(time=timestamps)

            mask = ds.time.isin(network.snapshots)
            ds = ds.sel(time=mask)

            if not len(ds.time) == len(network.snapshots):
                err = f"{len(ds.time)} and {len(network.snapshots)}"
                raise ValueError("Mismatch in profile and network timestamps " + err)
            ds = ds.stack(bus_bin=["bus", "bin"])

        # remove low potential bins
        cutoff = config.get("renewable_potential_cutoff", 0)
        ds = ds.where(ds["p_nom_max"] > cutoff, drop=True)

        # bins represent renewable generation grades
        def flatten(t):
            """Flatten tuple to string with ' grade' separator."""
            return " grade".join(map(str, t))
        buses = ds.indexes["bus_bin"].get_level_values("bus")
        bus_bins = ds.indexes["bus_bin"].map(flatten)

        p_nom_max = ds["p_nom_max"].to_pandas()
        p_nom_max.index = p_nom_max.index.map(flatten)

        p_max_pu = ds["profile"].to_pandas()
        p_max_pu.columns = p_max_pu.columns.map(flatten)

        # add renewables
        network.add(
            "Generator",
            bus_bins,
            suffix=f" {tech}",
            bus=buses,
            carrier=tech,
            p_nom_extendable=True,
            p_nom_max=p_nom_max,
            capital_cost=costs.at[tech, "capital_cost"],
            marginal_cost=costs.at[tech, "marginal_cost"],
            p_max_pu=p_max_pu,
            lifetime=costs.at[tech, "lifetime"],
        )


def add_heat_coupling(
    network: pypsa.Network,
    config: dict,
    nodes: pd.Index,
    prov_centroids: gpd.GeoDataFrame,
    costs: pd.DataFrame,
    planning_year: int,
    paths: dict,
):
    """Add the heat-coupling links and generators to the network

    Args:
        network (pypsa.Network): the network object
        config (dict): the config
        nodes (pd.Index): the node names. Defaults to pd.Index.
        prov_centroids (gpd.GeoDataFrame): the node locations.
        costs (pd.DataFrame): the costs dataframe for emissions
        paths (dict): the paths to the data files
    """
    central_fraction = pd.read_hdf(paths["central_fraction"])
    with pd.HDFStore(paths["heat_demand_profile"], mode="r") as store:
        heat_demand = store["heat_demand_profiles"]
        # TODO fix this if not working
        heat_demand.index = heat_demand.index.tz_localize(None)
        heat_demand = heat_demand.loc[network.snapshots]

    network.add(
        "Bus",
        nodes,
        suffix=" decentral heat",
        x=prov_centroids.x,
        y=prov_centroids.y,
        carrier="heat",
        location=nodes,
    )

    network.add(
        "Bus",
        nodes,
        suffix=" central heat",
        x=prov_centroids.x,
        y=prov_centroids.y,
        carrier="heat",
        location=nodes,
    )

    network.add(
        "Load",
        nodes,
        suffix=" decentral heat",
        bus=nodes + " decentral heat",
        p_set=heat_demand[nodes].multiply(1 - central_fraction[nodes]),
    )

    network.add(
        "Load",
        nodes,
        suffix=" central heat",
        bus=nodes + " central heat",
        p_set=heat_demand[nodes].multiply(central_fraction[nodes]),
    )

    if "heat pump" in config["Techs"]["vre_techs"]:
        logger.info(f"loading cop profiles from {paths['cop_name']}")
        with pd.HDFStore(paths["cop_name"], mode="r") as store:
            ashp_cop = store["ashp_cop_profiles"]
            ashp_cop.index = ashp_cop.index.tz_localize(None)
            ashp_cop = shift_profile_to_planning_year(
                ashp_cop, snakemake.wildcards.planning_horizons
            )
            gshp_cop = store["gshp_cop_profiles"]
            gshp_cop.index = gshp_cop.index.tz_localize(None)
            gshp_cop = shift_profile_to_planning_year(
                gshp_cop, snakemake.wildcards.planning_horizons
            )

        for cat in [" decentral ", " central "]:
            network.add(
                "Link",
                nodes,
                suffix=cat + "heat pump",
                bus0=nodes,
                bus1=nodes + cat + "heat",
                carrier="heat pump",
                efficiency=(
                    ashp_cop.loc[network.snapshots, nodes]
                    if config["time_dep_hp_cop"]
                    else costs.at[cat.lstrip() + "air-sourced heat pump", "efficiency"]
                ),
                capital_cost=costs.at[cat.lstrip() + "air-sourced heat pump", "efficiency"]
                * costs.at[cat.lstrip() + "air-sourced heat pump", "capital_cost"],
                marginal_cost=costs.at[cat.lstrip() + "air-sourced heat pump", "efficiency"]
                * costs.at[cat.lstrip() + "air-sourced heat pump", "marginal_cost"],
                p_nom_extendable=True,
                lifetime=costs.at[cat.lstrip() + "air-sourced heat pump", "lifetime"],
            )

        network.add(
            "Link",
            nodes,
            suffix=" ground heat pump",
            bus0=nodes,
            bus1=nodes + " decentral heat",
            carrier="heat pump",
            efficiency=(
                gshp_cop.loc[network.snapshots, nodes]
                if config["time_dep_hp_cop"]
                else costs.at["decentral ground-sourced heat pump", "efficiency"]
            ),
            marginal_cost=costs.at[cat.lstrip() + "ground-sourced heat pump", "efficiency"]
            * costs.at[cat.lstrip() + "ground-sourced heat pump", "marginal_cost"],
            capital_cost=costs.at[cat.lstrip() + "ground-sourced heat pump", "efficiency"]
            * costs.at["decentral ground-sourced heat pump", "capital_cost"],
            p_nom_extendable=True,
            lifetime=costs.at["decentral ground-sourced heat pump", "lifetime"],
        )

    if "water tanks" in config["Techs"]["store_techs"]:
        for cat in [" decentral ", " central "]:
            network.add(
                "Bus",
                nodes,
                suffix=cat + "water tanks",
                x=prov_centroids.x,
                y=prov_centroids.y,
                carrier="water tanks",
                location=nodes,
            )

            network.add(
                "Link",
                nodes + cat + "water tanks charger",
                bus0=nodes + cat + "heat",
                bus1=nodes + cat + "water tanks",
                carrier="water tanks",
                efficiency=costs.at["water tank charger", "efficiency"],
                p_nom_extendable=True,
            )

            network.add(
                "Link",
                nodes + cat + "water tanks discharger",
                bus0=nodes + cat + "water tanks",
                bus1=nodes + cat + "heat",
                carrier="water tanks",
                efficiency=costs.at["water tank discharger", "efficiency"],
                p_nom_extendable=True,
            )
            # [HP] 180 day time constant for centralised, 3 day for decentralised
            tes_tau = config["water_tanks"]["tes_tau"][cat.strip()]
            network.add(
                "Store",
                nodes + cat + "water tank",
                bus=nodes + cat + "water tanks",
                carrier="water tanks",
                e_cyclic=True,
                e_nom_extendable=True,
                standing_loss=1 - np.exp(-1 / (24.0 * tes_tau)),
                capital_cost=costs.at[cat.lstrip() + "water tank storage", "capital_cost"],
                lifetime=costs.at[cat.lstrip() + "water tank storage", "lifetime"],
            )

    if "resistive heater" in config["Techs"]["vre_techs"]:
        for cat in [" decentral ", " central "]:
            network.add(
                "Link",
                nodes + cat + "resistive heater",
                bus0=nodes,
                bus1=nodes + cat + "heat",
                carrier="resistive heater",
                efficiency=costs.at[cat.lstrip() + "resistive heater", "efficiency"],
                capital_cost=costs.at[cat.lstrip() + "resistive heater", "efficiency"]
                * costs.at[cat.lstrip() + "resistive heater", "capital_cost"],
                marginal_cost=costs.at[cat.lstrip() + "resistive heater", "efficiency"]
                * costs.at[cat.lstrip() + "resistive heater", "marginal_cost"],
                p_nom_extendable=True,
                lifetime=costs.at[cat.lstrip() + "resistive heater", "lifetime"],
            )

    if (
        "H2 CHP" in config["Techs"]["vre_techs"]
        and config["add_H2"]
        and config.get("heat_coupling", False)
    ):
        network.add(
            "Bus",
            nodes,
            suffix=" central H2 CHP",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="H2",
            location=nodes,
        )
        network.add(
            "Link",
            name=nodes + " central H2 CHP",
            bus0=nodes + " H2",
            bus1=nodes,
            bus2=nodes + " central heat",
            p_nom_extendable=True,
            efficiency=costs.at["central hydrogen CHP", "efficiency"],
            efficiency2=costs.at["central hydrogen CHP", "efficiency"]
            / costs.at["central hydrogen CHP", "c_b"],
            capital_cost=costs.at["central hydrogen CHP", "efficiency"]
            * costs.at["central hydrogen CHP", "capital_cost"],
            lifetime=costs.at["central hydrogen CHP", "lifetime"],
            carrier="H2 CHP",
        )

    if "CHP gas" in config["Techs"]["conv_techs"]:
        # TODO apply same as for coal (include Cb)
        # OCGT CHP
        network.add(
            "Link",
            nodes,
            suffix=" CHP gas",
            bus0=nodes + " gas",
            bus1=nodes,
            bus2=nodes + " central heat",
            p_nom_extendable=True,
            marginal_cost=costs.at["central gas CHP", "efficiency"]
            * costs.at["central gas CHP", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["central gas CHP", "efficiency"]
            * costs.at["central gas CHP", "capital_cost"],  # NB: capital cost is per MWel
            efficiency=costs.at["central gas CHP", "efficiency"],
            efficiency2=config["chp_parameters"]["eff_th"],
            lifetime=costs.at["central gas CHP", "lifetime"],
            carrier="CHP gas",
        )

    if "CHP coal" in config["Techs"]["conv_techs"]:
        logger.info("Adding CHP coal to network")

        network.add(
            "Bus",
            nodes,
            suffix=" coal fuel",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="coal",
            location=nodes,
        )

        network.add(
            "Generator",
            nodes + " coal fuel",
            bus=nodes + " coal fuel",
            carrier="coal",
            p_nom_extendable=False,
            p_nom=1e8,
            marginal_cost=costs.at["coal", "fuel"],
        )

        # NOTE generator | boiler is a key word for the constraint
        network.add(
            "Link",
            name=nodes,
            suffix=" CHP coal generator",
            bus0=nodes + " coal fuel",
            bus1=nodes,
            p_nom_extendable=True,
            marginal_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "capital_cost"],  # NB: capital cost is per MWel
            efficiency=costs.at["central coal CHP", "efficiency"],
            c_b=costs.at["central coal CHP", "c_b"],
            p_nom_ratio=1.0,
            lifetime=costs.at["central coal CHP", "lifetime"],
            carrier="CHP coal",
        )

        network.add(
            "Link",
            nodes,
            suffix=" central CHP coal boiler",
            bus0=nodes + " coal fuel",
            bus1=nodes + " central heat",
            carrier="CHP coal",
            p_nom_extendable=True,
            marginal_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
            efficiency=costs.at["central coal CHP", "efficiency"]
            / costs.at["central coal CHP", "c_v"],
            lifetime=costs.at["central coal CHP", "lifetime"],
        )

    if "coal boiler" in config["Techs"]["conv_techs"]:
        for cat in ["decentral", "central"]:
            network.add(
                "Link",
                nodes + f" {cat} coal boiler",
                p_nom_extendable=True,
                bus0=nodes + " coal fuel",
                bus1=nodes + f" {cat} heat",
                efficiency=costs.at[f"{cat} coal boiler", "efficiency"],
                marginal_cost=costs.at[f"{cat} coal boiler", "efficiency"]
                * costs.at[f"{cat} coal boiler", "VOM"],
                capital_cost=costs.at[f"{cat} coal boiler", "efficiency"]
                * costs.at[f"{cat} coal boiler", "capital_cost"],
                lifetime=costs.at[f"{cat} coal boiler", "lifetime"],
                carrier=f"coal boiler {cat}",
            )

    if "gas boiler" in config["Techs"]["conv_techs"]:
        for cat in ["decentral", "central"]:
            network.add(
                "Link",
                nodes + cat + "gas boiler",
                p_nom_extendable=True,
                bus0=nodes + " gas",
                bus1=nodes + f" {cat} heat",
                efficiency=costs.at[f"{cat} gas boiler", "efficiency"],
                marginal_cost=costs.at[f"{cat} gas boiler", "VOM"],
                capital_cost=costs.at[f"{cat} gas boiler", "efficiency"]
                * costs.at[f"{cat} gas boiler", "capital_cost"],
                lifetime=costs.at[f"{cat} gas boiler", "lifetime"],
                carrier=f"gas boiler {cat}",
            )

    if "solar thermal" in config["Techs"]["vre_techs"]:
        # this is the amount of heat collected in W per m^2, accounting
        # for efficiency
        with pd.HDFStore(paths["solar_thermal_name"], mode="r") as store:
            # 1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
            solar_thermal = config["solar_cf_correction"] * store["solar_thermal_profiles"] / 1e3

        solar_thermal.index = solar_thermal.index.tz_localize(None)
        solar_thermal = shift_profile_to_planning_year(solar_thermal, planning_year)
        solar_thermal = solar_thermal.loc[network.snapshots]

        for cat in [" decentral ", " central "]:
            network.add(
                "Generator",
                nodes,
                suffix=cat + "solar thermal",
                bus=nodes + cat + "heat",
                carrier="solar thermal",
                p_nom_extendable=True,
                capital_cost=costs.at[cat.lstrip() + "solar thermal", "capital_cost"],
                p_max_pu=solar_thermal[nodes].clip(1.0e-4),
                lifetime=costs.at[cat.lstrip() + "solar thermal", "lifetime"],
            )


def add_hydro(
    network: pypsa.Network,
    config: dict,
    nodes: pd.Index,
    prov_shapes: gpd.GeoDataFrame,
    costs: pd.DataFrame,
    planning_horizons: int,
):
    """Add the hydropower plants (dams) to the network.
    Due to the spillage/basin calculations these have real locations not just nodes.
    WARNING: the node is assigned based on the damn province name (turbine link)
            NOT future proof

    Args:
        network (pypsa.Network): the network object
        config (dict): the yaml config
        nodes (pd.Index): the buses
        prov_shapes (gpd.GeoDataFrame): the province shapes GDF
        costs (pd.DataFrame): the costs dataframe
        planning_horizons (int): the year
    """

    logger.info("\tAdding dam cascade")

    # load dams
    df = pd.read_csv(config["hydro_dams"]["dams_path"], index_col=0)
    points = df.apply(lambda row: Point(row.Lon, row.Lat), axis=1)
    dams = gpd.GeoDataFrame(df, geometry=points, crs=CRS)
    # store all info, then filter by selected nodes
    dam_provinces = dams.Province
    all_dams = dams.index.values
    dams = dams[dams.Province.isin(nodes)]

    logger.debug(f"Hydro dams in {nodes} provinces: {dams.index}")

    hourly_rng = pd.date_range(
        config["hydro_dams"]["inflow_date_start"],
        config["hydro_dams"]["inflow_date_end"],
        freq="1h",  # THIS IS THE INFLOW RES
        inclusive="left",
    )
    # TODO implement inflow calc, understand resolution (seems daily!)
    inflow = pd.read_pickle(config["hydro_dams"]["inflow_path"])
    # select inflow year
    hourly_rng = hourly_rng[hourly_rng.year == INFLOW_DATA_YR]
    inflow = inflow.loc[inflow.index.year == INFLOW_DATA_YR]
    inflow = inflow.reindex(hourly_rng, fill_value=0)
    inflow.columns = all_dams  # TODO dangerous
    # select only the dams in the network
    inflow = inflow.loc[:, inflow.columns.map(dam_provinces).isin(nodes)]
    inflow = shift_profile_to_planning_year(inflow, planning_horizons)
    inflow = inflow.loc[network.snapshots]
    # m^3/KWh -> m^3/MWh
    water_consumption_factor = dams.loc[:, "Water_consumption_factor_avg"] * 1e3

    #######
    # ### Add hydro stations as buses
    network.add(
        "Bus",
        dams.index,
        suffix=" station",
        carrier="stations",
        x=dams["geometry"].to_crs("+proj=cea").centroid.to_crs(prov_shapes.crs).x,
        y=dams["geometry"].to_crs("+proj=cea").centroid.to_crs(prov_shapes.crs).y,
        location=dams["Province"],
    )

    dam_buses = network.buses[network.buses.carrier == "stations"]

    # ===== add hydro reservoirs as stores ======
    initial_capacity = pd.read_pickle(config["hydro_dams"]["reservoir_initial_capacity_path"])
    effective_capacity = pd.read_pickle(config["hydro_dams"]["reservoir_effective_capacity_path"])
    initial_capacity.index = all_dams
    effective_capacity.index = all_dams
    initial_capacity = initial_capacity / water_consumption_factor
    effective_capacity = effective_capacity / water_consumption_factor

    # select relevant dams in nodes
    effective_capacity = effective_capacity.loc[
        effective_capacity.index.map(dam_provinces).isin(nodes)
    ]
    initial_capacity = initial_capacity.loc[initial_capacity.index.map(dam_provinces).isin(nodes)]

    network.add(
        "Store",
        dams.index,
        suffix=" reservoir",
        bus=dam_buses.index,
        e_nom=effective_capacity,
        e_initial=initial_capacity,
        e_cyclic=True,
        # TODO fix all config["costs"]
        marginal_cost=config["hydro"]["marginal_cost"]["reservoir"],
    )

    # add hydro turbines to link stations to provinces
    network.add(
        "Link",
        dams.index,
        suffix=" turbines",
        bus0=dam_buses.index,
        bus1=dams["Province"],
        carrier="hydroelectricity",
        p_nom=10 * dams["installed_capacity_10MW"],
        capital_cost=(
            costs.at["hydro", "capital_cost"] if config["hydro"]["hydro_capital_cost"] else 0
        ),
        efficiency=1,
        location=dams["Province"],
        p_nom_extendable=False,
    )

    # ===  add rivers to link station to station
    dam_edges = pd.read_csv(config["hydro_dams"]["damn_flows_path"], delimiter=",")
    in_nodes = dam_edges.bus0.map(dam_provinces).isin(nodes) & dam_edges.end_bus.map(
        dam_provinces
    ).isin(nodes)
    dam_edges = dam_edges[in_nodes]

    # === normal flow ====
    for row in dam_edges.iterrows():
        bus0 = row[1].bus0 + " turbines"
        bus2 = row[1].end_bus + " station"
        network.links.at[bus0, "bus2"] = bus2
        network.links.at[bus0, "efficiency2"] = 1.0

    # === spillage ====
    # TODO WHY EXTENDABLE - weather year?
    for row in dam_edges.iterrows():
        bus0 = row[1].bus0 + " station"
        bus1 = row[1].end_bus + " station"
        network.add(
            "Link",
            f"{bus0}-{bus1}" + " spillage",
            bus0=bus0,
            bus1=bus1,
            p_nom_extendable=True,
        )

    dam_ends = [
        dam
        for dam in np.unique(dams.index.values)
        if dam not in dam_edges["bus0"]
        or dam not in dam_edges["end_bus"]
        or (dam in dam_edges["end_bus"].values & dam not in dam_edges["bus0"])
    ]
    # need some kind of sink to absorb spillage (e,g ocean).
    # here hack by flowing to existing bus with 0 efficiency (lose)
    # TODO make more transparent -> generator with neg sign and 0 c0st
    for bus0 in dam_ends:
        network.add(
            "Link",
            bus0 + " spillage",
            bus0=bus0 + " station",
            bus1=bus0 + " station",
            p_nom_extendable=True,
            efficiency=0.0,
        )

    # add inflow as generators
    # only feed into hydro stations which are the first of a cascade
    inflow_stations = [
        dam for dam in np.unique(dams.index.values) if dam not in dam_edges["end_bus"].values
    ]

    for inflow_station in inflow_stations:
        # p_nom = 1 and p_max_pu & p_min_pu = p_pu, compulsory inflow
        p_nom = (inflow / water_consumption_factor)[inflow_station].max()
        p_pu = (inflow / water_consumption_factor)[inflow_station] / p_nom
        p_pu.index = network.snapshots
        network.add(
            "Generator",
            inflow_station + " inflow",
            bus=inflow_station + " station",
            carrier="hydro_inflow",
            p_max_pu=p_pu.clip(1.0e-6),
            # p_min_pu=p_pu.clip(1.0e-6),
            p_nom=p_nom,
        )

        # p_nom*p_pu = XXX m^3 then use turbines efficiency to convert to power

    # ======= add other existing hydro power (not lattitude resolved) ===
    hydro_p_nom = pd.read_hdf(config["hydro_dams"]["p_nom_path"]).loc[nodes]
    hydro_p_max_pu = (
        pd.read_hdf(
            config["hydro_dams"]["p_max_pu_path"],
            key=config["hydro_dams"]["p_max_pu_key"],
        ).tz_localize(None)
    )[nodes]

    hydro_p_max_pu = shift_profile_to_planning_year(hydro_p_max_pu, planning_horizons)
    # sort buses (columns) otherwise stuff will break
    hydro_p_max_pu.sort_index(axis=1, inplace=True)

    hydro_p_max_pu = hydro_p_max_pu.loc[snapshots]
    hydro_p_max_pu.index = network.snapshots

    logger.info("\tAdding extra hydro capacity (regionally aggregated)")

    network.add(
        "Generator",
        nodes,
        suffix=" hydroelectricity",
        bus=nodes,
        carrier="hydroelectricity",
        p_nom=hydro_p_nom,
        p_nom_min=hydro_p_nom,
        p_nom_extendable=False,
        p_max_pu=hydro_p_max_pu,
        capital_cost=(
            costs.at["hydro", "capital_cost"] if config["hydro"]["hydro_capital_cost"] else 0
        ),
    )

def cycling_shift(df, steps=1):
    """
    Cyclic shift on index of pd.Series|pd.DataFrame by number of steps.
    """
    df = df.copy()
    new_index = np.roll(df.index, steps)
    df.values[:] = df.reindex(index=new_index).values
    return df


def attach_EV_REMIND(
    n: pypsa.Network,
    avail_profile: pd.DataFrame,
    dsm_profile: pd.DataFrame,
    p_set: pd.DataFrame,
    nodes: pd.Index,
    options: dict,
    type: str,
) -> None:
    """
    Attach EV components (passenger or freight) to a PyPSA network.

    Logic
    -----
    - DSM OFF: Direct charging mode, p_set = charging demand
    - DSM ON : Storage mode, p_set = driving demand
               (load satisfied by store discharge; charging constrained by availability)

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network to modify
    avail_profile : pd.DataFrame
        Charging availability profile (snapshots × nodes)
    dsm_profile : pd.DataFrame
        DSM state-of-charge profile (snapshots × nodes)
    p_set : pd.DataFrame
        EV demand profile (snapshots × nodes)
    nodes : pd.Index
        Node names (typically province names)
    options : dict
        Required keys:
        - dsm (bool)
        - annual_consumption (float, MWh/year per vehicle)
        - charge_rate (float, MW per vehicle)
        - share_charger (float)
        - battery_size (float, MWh per vehicle)
        - dsm_availability (float)
    type : str
        "pass" or "freight"
    """
    if type not in ("pass", "freight"):
        raise ValueError("type must be 'pass' or 'freight'")

    dsm_enabled = options["dsm"]

    # --- 1. Compute EV numbers from demand ---
    total_energy = p_set.sum().sum()
    annual_consumption_per_ev = options["annual_consumption"]

    total_number_evs = max(total_energy / max(annual_consumption_per_ev, 1e-6), 0.0)
    node_energy_ratio = p_set.sum() / total_energy
    number_evs = node_energy_ratio * total_number_evs

    charge_power = (number_evs * options["charge_rate"] * options["share_charger"]).clip(lower=0.001)
    battery_energy = (number_evs * options["battery_size"]).clip(lower=0.001)

    logger.info(
        f"EV {type} - DSM {'ON' if dsm_enabled else 'OFF'}, "
        f"Total energy: {total_energy:.1f} MWh, "
        f"Total EVs: {int(total_number_evs):,}"
    )

    # --- 2. Add buses ---
    ev_load_bus = nodes + f" EV_load_{type}"
    n.add("Carrier", f"EV_load_{type}")
    n.add("Bus", ev_load_bus, location=nodes,
          carrier=f"EV_load_{type}", unit="MWh_el")

    if dsm_enabled:
        carrier_name = f"EV_{type}_battery"
        ev_battery_bus = nodes + " " + carrier_name
        n.add("Carrier", carrier_name)
        n.add("Bus", ev_battery_bus, location=nodes,
              carrier=carrier_name, unit="MWh_el")

    # --- 3. Add load ---
    n.add(
        "Load",
        nodes,
        suffix=f" land transport EV {type}",
        bus=ev_load_bus,
        carrier=f"land_transport_EV_{type}",
        p_set=p_set.loc[n.snapshots, nodes],
    )

    # --- 4. Add components depending on DSM ---
    if dsm_enabled:
        # Charger: Grid -> Battery
        n.add(
            "Link",
            nodes,
            suffix=f" BEV {type} charger",
            bus0=nodes,
            bus1=ev_battery_bus,
            p_nom=charge_power * options["dsm_availability"],
            carrier=f"BEV_{type}_charger",
            p_max_pu=avail_profile.loc[n.snapshots, nodes].clip(0, 1),
            efficiency=1.0,
        )

        # Discharger: Battery -> EV load
        n.add(
            "Link",
            nodes,
            suffix=f" BEV {type} discharger",
            bus0=ev_battery_bus,
            bus1=ev_load_bus,
            carrier=f"BEV_{type}_discharger",
            efficiency=1.0,
            p_nom_extendable=True,
        )

        # Battery Store
        n.add(
            "Store",
            nodes,
            suffix=f" EV_{type}_battery",
            bus=ev_battery_bus,
            carrier=f"EV_{type}_battery",
            e_cyclic=True,
            e_nom=battery_energy * options["dsm_availability"],
            e_max_pu=1.0,
            e_min_pu=dsm_profile.loc[n.snapshots, nodes].clip(0, 1),
        )

        logger.info(f"✅ DSM-enabled EV {type}: {int(total_number_evs):,} vehicles with storage")

    else:
        # Direct charger: Grid -> EV load
        n.add(
            "Link",
            nodes,
            suffix=f" BEV {type} charger",
            bus0=nodes,
            bus1=ev_load_bus,
            carrier=f"BEV_{type}_charger",
            efficiency=1.0,
            p_nom=charge_power,
            p_max_pu=1.0,
        )

        logger.info(f"✅ Direct EV {type}: {int(total_number_evs):,} vehicles, direct charging")

    logger.info(f"EV {type} setup complete: {'DSM storage' if dsm_enabled else 'Direct charging'} mode")



def prepare_network(
    config: dict,
    costs: pd.DataFrame,
    snapshots: pd.date_range,
    biomass_potential: pd.DataFrame = None,
    paths: dict = None,
) -> pypsa.Network:
    """Prepares/makes the network object for overnight mode according to config &
    at 1 node per region/province

    Args:
        config (dict): the snakemake config
        costs (pd.DataFrame): the costs dataframe (anualised capex and marginal costs)
        snapshots (pd.date_range): the snapshots for the network
        biomass_potential (Optional, pd.DataFrame): biomass potential dataframe. Defaults to None.
        paths (Optional, dict): the paths to the data files. Defaults to None.

    Returns:
        pypsa.Network: the pypsa network object
    """

    # determine whether gas/coal to be added depending on specified conv techs
    config["add_gas"] = (
        True
        if [tech for tech in config["Techs"]["conv_techs"] if ("gas" in tech or "CGT" in tech)]
        else False
    )
    config["add_coal"] = (
        True if [tech for tech in config["Techs"]["conv_techs"] if "coal" in tech] else False
    )

    planning_horizons = snakemake.wildcards["planning_horizons"]

    # Build the Network object, which stores all other objects
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.snapshot_weightings[:] = config["snapshots"]["frequency"]
    # load graph
    nodes = pd.Index(PROV_NAMES)
    # toso soft code
    countries = ["CN"] * len(nodes)

    # TODO check crs projection correct
    # load provinces
    prov_shapes = read_province_shapes(paths["province_shape"])
    prov_centroids = prov_shapes.to_crs("+proj=cea").centroid.to_crs(CRS)

    # add AC buses
    network.add(
        "Bus", nodes, x=prov_centroids.x, y=prov_centroids.y, location=nodes, country=countries
    )

    # add carriers
    add_carriers(network, config, costs)

    # load electricity demand data
    with pd.HDFStore(paths["elec_load"], mode="r") as store:
        load = store["load"].loc[network.snapshots, PROV_NAMES]

    network.add("Load", nodes, bus=nodes, p_set=load[nodes])

    # EV components setup
    dsm_profile = pd.read_csv(paths["dsm_profile"], index_col=0, parse_dates=True)

    # Passenger EVs
    if config.get("EV_pass", {}).get("enable", False):
        logger.info("adding passenger EV components")

        charging_demand_pass = pd.read_csv(paths["transport_demand_passenger"], index_col=0, parse_dates=True)
        driving_demand_pass  = pd.read_csv(paths["driving_demand_passenger"], index_col=0, parse_dates=True)
        avail_profile_pass   = pd.read_csv(paths["avail_profile_passenger"], index_col=0, parse_dates=True)

        options_pass = config["EV_pass"]
        p_set_pass = driving_demand_pass if options_pass["dsm"] else charging_demand_pass

        attach_EV_REMIND(
            n=network,
            avail_profile=avail_profile_pass,
            dsm_profile=dsm_profile,
            p_set=p_set_pass,
            nodes=nodes,
            options=options_pass,
            type="pass",
        )

    # Freight EVs
    if config.get("EV_freight", {}).get("enable", False):
        logger.info("adding freight EV components")

        charging_demand_freight = pd.read_csv(paths["transport_demand_freight"], index_col=0, parse_dates=True)
        driving_demand_freight  = pd.read_csv(paths["driving_demand_freight"], index_col=0, parse_dates=True)
        avail_profile_freight   = pd.read_csv(paths["avail_profile_freight"], index_col=0, parse_dates=True)

        options_freight = config["EV_freight"]
        p_set_freight = driving_demand_freight if options_freight["dsm"] else charging_demand_freight

        attach_EV_REMIND(
            n=network,
            avail_profile=avail_profile_freight,
            dsm_profile=dsm_profile,
            p_set=p_set_freight,
            nodes=nodes,
            options=options_freight,
            type="freight",
        )

    ws_carriers = [c for c in config["Techs"]["vre_techs"] if c.find("wind") >= 0 or c == "solar"]
    add_wind_and_solar(network, ws_carriers, paths, planning_horizons, costs)

    add_conventional_generators(network, nodes, config, prov_centroids, costs)

    # nuclear is brownfield
    if "nuclear" in config["Techs"]["vre_techs"]:
        nuclear_nodes = pd.Index(NUCLEAR_EXTENDABLE)
        network.add(
            "Generator",
            nuclear_nodes,
            suffix=" nuclear",
            p_nom_extendable=True,
            p_max_pu=config["nuclear_reactors"]["p_max_pu"],
            p_min_pu=config["nuclear_reactors"]["p_min_pu"],
            bus=nuclear_nodes,
            carrier="nuclear",
            efficiency=costs.at["nuclear", "efficiency"],
            capital_cost=costs.at["nuclear", "capital_cost"],  # NB: capital cost is per MWel
            marginal_cost=costs.at["nuclear", "marginal_cost"],
            lifetime=costs.at["nuclear", "lifetime"],
        )

    # TODO add coal CC? no retrofit option

    if "PHS" in config["Techs"]["store_techs"]:
        # pure pumped hydro storage, fixed, 6h energy by default, no inflow
        cc = costs.at["PHS", "capital_cost"]

        network.add(
            "StorageUnit",
            nodes,
            suffix=" PHS",
            bus=nodes,
            carrier="PHS",
            p_nom_extendable=True,
            max_hours=config["hydro"]["PHS_max_hours"],
            efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
            efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
            cyclic_state_of_charge=True,
            capital_cost=cc,
            marginal_cost=0.0,
        )

    if config["add_hydro"]:
        logger.info("Adding hydro to network")
        add_hydro(network, config, nodes, prov_centroids, costs, planning_horizons)

    if config["add_H2"]:
        logger.info("Adding H2 buses to network")
        # do beore heat coupling to avoid warning
        network.add(
            "Bus",
            nodes,
            suffix=" H2",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="H2",
            location=nodes,
        )

    if config.get("heat_coupling", False):
        logger.info("Adding heat and CHP to the network")
        add_heat_coupling(network, config, nodes, prov_centroids, costs, planning_horizons, paths)

        if config["add_biomass"]:
            logger.info("Adding biomass to network")
            add_co2_capture_support(network, nodes, prov_centroids)
            add_biomass_chp(
                network,
                costs,
                nodes,
                biomass_potential[nodes],
                prov_centroids,
                add_beccs="beccs" in config["Techs"]["vre_techs"],
            )

    if config["add_H2"]:
        logger.info("Adding H2 to network")
        add_H2(network, config, nodes, costs)

    if "battery" in config["Techs"]["store_techs"]:
        network.add(
            "Bus",
            nodes,
            suffix=" battery",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="battery",
            location=nodes,
        )

        # TODO Why no standing loss?: test with
        min_charge = config["electricity"].get("min_charge", {"battery": 0})
        min_charge = min_charge.get("battery", 0)
        network.add(
            "Store",
            nodes + " battery",
            bus=nodes + " battery",
            e_cyclic=True,
            e_nom_extendable=True,
            e_min_pu=min_charge,
            capital_cost=costs.at["battery storage", "capital_cost"],
            lifetime=costs.at["battery storage", "lifetime"],
        )

        # TODO understand/remove sources, data should not be in code
        # Sources:
        # [HP]: Henning, Palzer http://www.sciencedirect.com/science/article/pii/S1364032113006710
        # [B]: Budischak et al. http://www.sciencedirect.com/science/article/pii/S0378775312014759

        network.add(
            "Link",
            nodes + " battery charger",
            bus0=nodes,
            bus1=nodes + " battery",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["battery inverter", "efficiency"]
            * costs.at["battery inverter", "capital_cost"],
            p_nom_extendable=True,
            carrier="battery",
            lifetime=costs.at["battery inverter", "lifetime"],
        )

        network.add(
            "Link",
            nodes + " battery discharger",
            bus0=nodes + " battery",
            bus1=nodes,
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            marginal_cost=0.0,
            p_nom_extendable=True,
            carrier="battery discharger",
        )

    # ============= add lines =========
    # The lines are implemented according to the transport model (no KVL) and without losses.
    # see Neumann et al 10.1016/j.apenergy.2022.118859
    # TODO make not lossless optional (? - increases computing cost)

    if not config["no_lines"]:
        add_voltage_links(network, config)

    assign_locations(network)
    return network


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "prepare_networks",
            topology="current+FCG",
            # co2_pathway="exp175default",
            co2_pathway="SSP2-PkBudg1000-freeze",
            planning_horizons=2030,
            heating_demand="positive",
            configfiles="resources/tmp/remind_coupled.yaml",
        )

    configure_logging(snakemake)

    config = snakemake.config

    logger.info("Preparing network for scenario:")
    logger.info(config["scenario"])
    logger.info(config["co2_scenarios"])

    yr = int(snakemake.wildcards.planning_horizons)
    logger.info(f"Preparing network for {yr}")

    pathway = snakemake.wildcards.co2_pathway
    co2_opts = ConfigManager(config).fetch_co2_restriction(pathway, yr)

    # make snapshots (drop leap days) -> possibly do all the unpacking in the function
    snapshot_cfg = config["snapshots"]
    snapshots = make_periodic_snapshots(
        year=yr,
        freq=snapshot_cfg["freq"],
        start_day_hour=snapshot_cfg["start"],
        end_day_hour=snapshot_cfg["end"],
        bounds=snapshot_cfg["bounds"],
        # naive local timezone
        tz=None,
        end_year=(None if not snapshot_cfg["end_year_plus1"] else yr + 1),
    )

    # load costs
    n_years = config["snapshots"]["frequency"] * len(snapshots) / 8760.0
    tech_costs = snakemake.input["tech_costs"]
    input_paths = {k: v for k, v in snakemake.input.items()}
    cost_year = yr
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    # biomass
    if config["add_biomass"]:
        biomass_potential = pd.read_hdf(input_paths["biomass_potential"])
    else:
        biomass_potential = None

    network = prepare_network(
        snakemake.config, costs, snapshots, biomass_potential, paths=input_paths
    )
    sanitize_carriers(network, snakemake.config)

    outp = snakemake.output.network_name
    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    network.export_to_netcdf(outp, compression=compression)

    logger.info(f"Network for {yr} prepared and saved to {outp}")

    costs_outp = os.path.dirname(outp) + f"/costs_{yr}.csv"
    costs.to_csv(costs_outp)
