# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# for non-pathway network
# TODO fix timezones
# TODO WHY DO WE USE VRESUTILS ANNUITY IN ONE PLACE AND OUR OWN CALC ELSEWHERE?

import pypsa
from vresutils.costdata import annuity
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import logging

from _helpers import (
    configure_logging,
    mock_snakemake,
    shift_profile_to_planning_year,
    make_periodic_snapshots,
    assign_locations,
)
from functions import haversine, HVAC_cost_curve
from add_electricity import load_costs, sanitize_carriers
from readers import read_province_shapes
from prepare_network_common import calc_renewable_pu_avail
from constants import (
    PROV_NAMES,
    CRS,
    CO2_HEATING_2020,
    CO2_EL_2020,
    LOAD_CONVERSION_FACTOR,
    INFLOW_DATA_YR,
    NUCLEAR_EXTENDABLE,
    NON_LIN_PATH_SCALING,
    LINE_SECURITY_MARGIN,
    ECON_LIFETIME_LINES,
    FOM_LINES,
)

logger = logging.getLogger(__name__)


def add_carriers(network: pypsa.Network, config: dict, costs: pd.DataFrame):
    """add the various carriers to the network based on the config file

    Args:
        network (pypsa.Network): the pypsa network
        config (dict): the config file
        costs (pd.DataFrame): the costs dataframe
    """

    network.add("Carrier", "AC")
    if config["heat_coupling"]:
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


def add_conventional_generators(
    network: pypsa.Network,
    nodes: pd.Index,
    config: dict,
    prov_centroids: gpd.GeoDataFrame,
    costs: pd.DataFrame,
):
    """add conventional generators to the network

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

        # TODO why not centralised?
        network.add(
            "Store",
            nodes + " gas Store",
            bus=nodes + " gas",
            e_nom_extendable=True,
            carrier="gas",
            e_nom=1e7,
            e_cyclic=True,
        )

        network.add(
            "Link",
            nodes,
            suffix=" OCGT",
            bus0=nodes + " gas",
            bus1=nodes,
            marginal_cost=costs.at["OCGT", "efficiency"]
            * costs.at["OCGT", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["OCGT", "efficiency"]
            * costs.at["OCGT", "capital_cost"],  # NB: capital cost is per MWel
            p_nom_extendable=True,
            efficiency=costs.at["OCGT", "efficiency"],
            lifetime=costs.at["OCGT", "lifetime"],
            carrier="gas",
        )

    if config["add_coal"]:
        # this is the non sector-coupled approach
        # for industry may have an issue in that coal feeds to chem sector
        network.add(
            "Generator",
            nodes,
            suffix=" coal power",
            bus=nodes,
            carrier="coal",
            p_nom_extendable=True,
            efficiency=costs.at["coal", "efficiency"],
            marginal_cost=costs.at["coal", "marginal_cost"],
            capital_cost=costs.at["coal", "efficiency"]
            * costs.at["coal", "capital_cost"],  # NB: capital cost is per MWel
            lifetime=costs.at["coal", "lifetime"],
        )


def add_H2(network: pypsa.Network, config: dict, nodes: pd.Index, costs: pd.DataFrame):
    """add H2 generators, storage and links to the network - currently all or nothing

    Args:
        network (pypsa.Network): network object too which H2 comps will be added
        config (dict): the config (snakemake config)
        nodes (pd.Index): the buses
        costs (pd.DataFrame): the cost database
    """

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

    # TODO consider switching to turbines and making a switch for off
    # TODO understand MVs
    network.add(
        "Link",
        name=nodes + " H2 Fuel Cell",
        bus0=nodes + " H2",
        bus1=nodes,
        p_nom_extendable=True,
        efficiency=costs.at["fuel cell", "efficiency"],
        capital_cost=costs.at["fuel cell", "efficiency"] * costs.at["fuel cell", "capital_cost"],
        lifetime=costs.at["fuel cell", "lifetime"],
        carrier="H2 fuel cell",
    )

    H2_under_nodes = pd.Index(config["H2"]["geo_storage_nodes"])
    H2_type1_nodes = nodes.difference(H2_under_nodes)

    network.add(
        "Store",
        H2_under_nodes + " H2 Store",
        bus=H2_under_nodes + " H2",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=costs.at["hydrogen storage underground", "capital_cost"],
        lifetime=costs.at["hydrogen storage underground", "lifetime"],
    )

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
            edges = pd.read_csv(
                edge_path, sep=",", header=None, names=["bus0", "bus1", "p_nom"]
            ).fillna(0)

        # fix this to use map with x.y
        lengths = NON_LIN_PATH_SCALING * np.array(
            [
                haversine(
                    [network.buses.at[bus0, "x"], network.buses.at[bus0, "y"]],
                    [network.buses.at[bus1, "x"], network.buses.at[bus1, "y"]],
                )
                for bus0, bus1 in edges[["bus0", "bus1"]].values
            ]
        )

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


def add_voltage_links(network: pypsa.Network, config: dict):
    """add HVDC/AC links (no KVL)

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
        edges = pd.read_csv(
            edge_path, sep=",", header=None, names=["bus0", "bus1", "p_nom"]
        ).fillna(0)

    # fix this to use map with x.y
    lengths = NON_LIN_PATH_SCALING * np.array(
        [
            haversine(
                [network.buses.at[bus0, "x"], network.buses.at[bus0, "y"]],
                [network.buses.at[bus1, "x"], network.buses.at[bus1, "y"]],
            )
            for bus0, bus1 in edges[["bus0", "bus1"]].values
        ]
    )

    cc = (
        (config["line_cost_factor"] * lengths * [HVAC_cost_curve(len_) for len_ in lengths])
        * LINE_SECURITY_MARGIN
        * FOM_LINES
        * n_years
        * annuity(ECON_LIFETIME_LINES, config["costs"]["discountrate"])
    )

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
            efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]
            * config["transmission_efficiency"]["DC"]["efficiency_per_1000km"] ** (lengths / 1000),
            length=lengths,
            capital_cost=cc,
        )
        # 0 len for reversed in case line limits are specified in km
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
            capital_cost=cc,
        )


def add_heat_coupling(
    network: pypsa.Network,
    config: dict,
    nodes: pd.Index,
    prov_centroids: gpd.GeoDataFrame,
    costs: pd.DataFrame,
    planning_year: int,
):
    """add the heat-coupling links and generators to the network

    Args:
        network (pypsa.Network): the network object
        config (dict): the config
        nodes (pd.Index): the node names. Defaults to pd.Index.
        prov_centroids (gpd.GeoDataFrame): the node locations.
        costs (pd.DataFrame): the costs dataframe for emissions
    """

    central_fraction = pd.read_hdf(snakemake.input.central_fraction)
    with pd.HDFStore(snakemake.input.heat_demand_profile, mode="r") as store:
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
        p_set=heat_demand[nodes].multiply(1 - central_fraction),
    )

    network.add(
        "Load",
        nodes,
        suffix=" central heat",
        bus=nodes + " central heat",
        p_set=heat_demand[nodes].multiply(central_fraction),
    )

    if "heat pump" in config["Techs"]["vre_techs"]:

        with pd.HDFStore(snakemake.input.cop_name, mode="r") as store:
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

    if "H2 CHP" in config["Techs"]["vre_techs"] and config["add_H2"] and config["heat_coupling"]:
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
            carrier="H2 CHP",
            efficiency=costs.at["central hydrogen CHP", "efficiency"],
            efficiency2=costs.at["central hydrogen CHP", "efficiency"]
            / costs.at["central hydrogen CHP", "c_b"],
            capital_cost=costs.at["central hydrogen CHP", "efficiency"]
            * costs.at["central hydrogen CHP", "capital_cost"],
            lifetime=costs.at["central hydrogen CHP", "lifetime"],
        )
    if "gas boiler" in config["Techs"]["conv_techs"]:
        for cat in [" decentral ", " central "]:
            network.add(
                "Link",
                nodes + cat + "gas boiler",
                p_nom_extendable=True,
                bus0=nodes + " gas",
                bus1=nodes + cat + "heat",
                efficiency=costs.at[cat.lstrip() + "gas boiler", "efficiency"],
                marginal_cost=costs.at[cat.lstrip() + "gas boiler", "VOM"],
                capital_cost=costs.at[cat.lstrip() + "gas boiler", "efficiency"]
                * costs.at[cat.lstrip() + "gas boiler", "capital_cost"],
                lifetime=costs.at[cat.lstrip() + "gas boiler", "lifetime"],
            )

    if "CHP gas" in config["Techs"]["conv_techs"]:
        # TODO merge with gas ?
        network.add(
            "Bus",
            nodes,
            suffix=" CHP gas",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="gas",
            location=nodes,
        )

        network.add(
            "Generator",
            name=nodes + " CHP gas",
            bus=nodes + " CHP gas",
            carrier="gas",
            p_nom_extendable=True,
            marginal_cost=costs.at["gas", "marginal_cost"],
        )

        # TODO why is not combined cycle?
        # TODO efficiency to be understood - DK doc not clear
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
            efficiency=config["chp_parameters"]["eff_el"],
            efficiency2=config["chp_parameters"]["eff_th"],
            lifetime=costs.at["central gas CHP", "lifetime"],
        )

    if "CHP coal" in config["Techs"]["conv_techs"]:

        # TODO merge with normal coal?
        network.add(
            "Bus",
            nodes,
            suffix=" CHP coal",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="coal",
            location=nodes,
        )

        network.add(
            "Generator",
            name=nodes + " CHP coal",
            bus=nodes + " CHP coal",
            carrier="coal",
            p_nom_extendable=True,
            marginal_cost=costs.at["coal", "marginal_cost"],
        )

        network.add(
            "Link",
            name=nodes,
            suffix=" CHP coal",
            bus0=nodes + " CHP coal",
            bus1=nodes,
            bus2=nodes + " central heat",
            p_nom_extendable=True,
            marginal_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "capital_cost"],  # NB: capital cost is per MWel
            efficiency=config["chp_parameters"]["eff_el"],
            efficiency2=config["chp_parameters"]["eff_th"],
            lifetime=costs.at["central coal CHP", "lifetime"],
        )

    if "solar thermal" in config["Techs"]["vre_techs"]:

        # this is the amount of heat collected in W per m^2, accounting
        # for efficiency
        with pd.HDFStore(snakemake.input.solar_thermal_name, mode="r") as store:
            # 1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
            solar_thermal = config["solar_cf_correction"] * store["solar_thermal_profiles"] / 1e3

        solar_thermal.index = solar_thermal.index.tz_localize(None)
        solar_thermal = shift_profile_to_planning_year(solar_thermal, planning_year)
        solar_thermal = solar_thermal.loc[network.snapshots]

        for cat in [" decentral "]:
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

    # load dams
    df = pd.read_csv(config["hydro_dams"]["dams_path"], index_col=0)
    points = df.apply(lambda row: Point(row.Lon, row.Lat), axis=1)
    dams = gpd.GeoDataFrame(df, geometry=points, crs=CRS)

    hourly_rng = pd.date_range(
        config["hydro_dams"]["inflow_date_start"],
        config["hydro_dams"]["inflow_date_end"],
        freq=config["snapshots"]["freq"],
        inclusive="left",
    )
    # TODO understand where inflow is calculated
    inflow = pd.read_pickle(config["hydro_dams"]["inflow_path"]).reindex(hourly_rng, fill_value=0)
    inflow.columns = dams.index
    inflow = inflow.loc[str(INFLOW_DATA_YR)]
    inflow = shift_profile_to_planning_year(inflow, INFLOW_DATA_YR)

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
    )

    dam_buses = network.buses[network.buses.carrier == "stations"]

    # ===== add hydro reservoirs as stores ======
    initial_capacity = pd.read_pickle(config["hydro_dams"]["reservoir_initial_capacity_path"])
    effective_capacity = pd.read_pickle(config["hydro_dams"]["reservoir_effective_capacity_path"])
    initial_capacity.index = dams.index
    effective_capacity.index = dams.index
    initial_capacity = initial_capacity / water_consumption_factor
    effective_capacity = effective_capacity / water_consumption_factor

    network.add(
        "Store",
        dams.index,
        suffix=" reservoir",
        bus=dam_buses.index,
        e_nom=effective_capacity,
        e_initial=initial_capacity,
        e_cyclic=True,
        # TODO fix all config["costs"]
        marginal_cost=config["costs"]["marginal_cost"]["hydro"],
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

    # === normal flow ====
    for row in dam_edges.iterrows():
        bus0 = row[1].bus0 + " turbines"
        bus2 = row[1].end_bus + " station"
        network.links.at[bus0, "bus2"] = bus2
        network.links.at[bus0, "efficiency2"] = 1.0

    # TODO WHY EXTENDABLE - weather year?
    for row in dam_edges.iterrows():
        bus0 = row[1].bus0 + " station"
        bus1 = row[1].end_bus + " station"
        network.add(
            "Link",
            "{}-{}".format(bus0, bus1) + " spillage",
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
            bus1="Tibet",
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

    # TODO clarify that accounting for hydro is working


# TODO fix timezones/centralsie, think Shanghai won't work on its own
def generate_periodic_profiles(
    dt_index=None,
    col_tzs=pd.Series(index=PROV_NAMES, data=len(PROV_NAMES) * ["Shanghai"]),
    weekly_profile=range(24 * 7),
):
    """Give a 24*7 long list of weekly hourly profiles, generate this
    for each country for the period dt_index, taking account of time
    zones and Summer Time."""

    weekly_profile = pd.Series(weekly_profile, range(24 * 7))
    # TODO fix, no longer take into accoutn summer time
    # ALSO ADD A TODO in base_network
    week_df = pd.DataFrame(index=dt_index, columns=col_tzs.index)
    for ct in col_tzs.index:
        week_df[ct] = [24 * dt.weekday() + dt.hour for dt in dt_index.tz_localize(None)]
        week_df[ct] = week_df[ct].map(weekly_profile)
    return week_df


def prepare_network(config: dict) -> pypsa.Network:
    """Prepares/makes the network object for overnight mode according to config &
    at 1 node per region/province

    Args:
        config (dict): the snakemake config

    Returns:
        pypsa.Network: the pypsa network object
    """

    # determine whether gas/coal to be added depending on specified conv techs
    config["add_gas"] = (
        True if [tech for tech in config["Techs"]["conv_techs"] if "gas" in tech] else False
    )
    config["add_coal"] = (
        True if [tech for tech in config["Techs"]["conv_techs"] if "coal" in tech] else False
    )

    planning_horizons = snakemake.wildcards["planning_horizons"]

    # Build the Network object, which stores all other objects
    network = pypsa.Network()
    # load graph
    nodes = pd.Index(PROV_NAMES)

    # make snapshots (drop leap days) -> possibly do all the unpacking in the function
    snapshot_cfg = config["snapshots"]
    snapshots = make_periodic_snapshots(
        year=planning_horizons,
        freq=snapshot_cfg["freq"],
        start_day_hour=snapshot_cfg["start"],
        end_day_hour=snapshot_cfg["end"],
        bounds=snapshot_cfg["bounds"],
        tz=snapshot_cfg["timezone"],
        end_year=(None if not snapshot_cfg["end_year_plus1"] else planning_horizons + 1),
    )
    network.set_snapshots(snapshots)

    network.snapshot_weightings[:] = config["snapshots"]["frequency"]
    represented_hours = network.snapshot_weightings.sum()[0]
    n_years = represented_hours / 8760.0

    # load costs
    tech_costs = snakemake.input.tech_costs
    cost_year = planning_horizons
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    # TODO check crs projection correct
    # load provinces
    prov_shapes = read_province_shapes(snakemake.input.province_shape)
    prov_centroids = prov_shapes.to_crs("+proj=cea").centroid.to_crs(CRS)

    # add AC buses
    network.add("Bus", nodes, x=prov_centroids.x, y=prov_centroids.y, location=nodes)

    # add carriers
    add_carriers(network, config, costs)

    # load datasets calculated by build_renewable_profiles
    ds_solar = xr.open_dataset(snakemake.input.profile_solar)
    ds_onwind = xr.open_dataset(snakemake.input.profile_onwind)
    ds_offwind = xr.open_dataset(snakemake.input.profile_offwind)

    # == shift datasets from reference to planning year, sort columns to match network bus order ==
    solar_p_max_pu = calc_renewable_pu_avail(ds_solar, planning_horizons, snapshots)
    onwind_p_max_pu = calc_renewable_pu_avail(ds_onwind, planning_horizons, snapshots)
    offwind_p_max_pu = calc_renewable_pu_avail(ds_offwind, planning_horizons, snapshots)

    # TODO SOFT CODE BASE YEAR
    if config["scenario"]["co2_reduction"] is None:
        pass
    elif isinstance(config["scenario"]["co2_reduction"], dict):
        logger.info("Adding CO2 constraint based on scenario")
        pathway = snakemake.wildcards["pathway"]
        reduction = float(config["scenario"]["co2_reduction"][pathway][str(planning_horizons)])
        co2_limit = (CO2_EL_2020 + CO2_HEATING_2020) * (1 - reduction)
        network.add(
            "GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit,
        )
    elif not isinstance(config["scenario"]["co2_reduction"], tuple):
        logger.info("Adding CO2 constraint based on scenario")
        # TODO fix hard coded
        co2_limit = (CO2_EL_2020 + CO2_HEATING_2020) * (
            1 - float(config["scenario"]["co2_reduction"])
        )  # Chinese 2020 CO2 emissions of electric and heating sector

        network.add(
            "GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit,
        )
    else:
        logger.error(f"Unhandled CO2 config {config["scenario"]["co2_reduction"]}.")
        raise ValueError(f"Unhandled CO2 config {config["scenario"]["co2_reduction"]}")

    # load electricity demand data
    demand_path = snakemake.input.elec_load.replace("{planning_horizons}", f"{cost_year}")
    with pd.HDFStore(demand_path, mode="r") as store:
        load = LOAD_CONVERSION_FACTOR * store["load"]  # TODO add unit
        load = load.loc[network.snapshots]
    load.columns = PROV_NAMES

    network.add("Load", nodes, bus=nodes, p_set=load[nodes])

    # add renewables
    network.add(
        "Generator",
        nodes,
        suffix=" onwind",
        bus=nodes,
        carrier="onwind",
        p_nom_extendable=True,
        p_nom_max=ds_onwind["p_nom_max"].to_pandas(),
        capital_cost=costs.at["onwind", "capital_cost"],
        marginal_cost=costs.at["onwind", "marginal_cost"],
        p_max_pu=onwind_p_max_pu,
        lifetime=costs.at["onwind", "lifetime"],
    )

    offwind_nodes = ds_offwind["bus"].to_pandas().index
    network.add(
        "Generator",
        offwind_nodes,
        suffix=" offwind",
        bus=offwind_nodes,
        carrier="offwind",
        p_nom_extendable=True,
        p_nom_max=ds_offwind["p_nom_max"].to_pandas(),
        capital_cost=costs.at["offwind", "capital_cost"],
        marginal_cost=costs.at["offwind", "marginal_cost"],
        p_max_pu=offwind_p_max_pu,
        lifetime=costs.at["offwind", "lifetime"],
    )

    network.add(
        "Generator",
        nodes,
        suffix=" solar",
        bus=nodes,
        carrier="solar",
        p_nom_extendable=True,
        p_nom_max=ds_solar["p_nom_max"].to_pandas(),
        capital_cost=costs.at["solar", "capital_cost"],
        marginal_cost=costs.at["solar", "marginal_cost"],
        p_max_pu=solar_p_max_pu,
        lifetime=costs.at["solar", "lifetime"],
    )

    add_conventional_generators(network, nodes, config, prov_centroids, costs)

    # nuclear is brownfield
    if "nuclear" in config["Techs"]["vre_techs"]:
        nuclear_p_nom = pd.read_csv(config["nuclear_reactors"]["pp_path"], index_col=0)
        nuclear_p_nom = pd.Series(nuclear_p_nom.squeeze())

        nuclear_nodes = pd.Index(NUCLEAR_EXTENDABLE)
        network.add(
            "Generator",
            nuclear_nodes,
            suffix=" nuclear",
            p_nom_extendable=True,
            p_min_pu=0.7,
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
        hydrocapa_df = pd.read_csv("resources/data/hydro/PHS_p_nom.csv", index_col=0)
        phss = hydrocapa_df.index[hydrocapa_df["MW"] > 0].intersection(nodes)
        if config["hydro"]["hydro_capital_cost"]:
            cc = costs.at["PHS", "capital_cost"]
        else:
            cc = 0.0

        network.add(
            "StorageUnit",
            phss,
            suffix=" PHS",
            bus=phss,
            carrier="PHS",
            p_nom_extendable=False,
            p_nom=hydrocapa_df.loc[phss]["MW"],
            p_nom_min=hydrocapa_df.loc[phss]["MW"],
            max_hours=config["hydro"]["PHS_max_hours"],
            efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
            efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
            cyclic_state_of_charge=True,
            capital_cost=cc,
            marginal_cost=0.0,
        )

    if config["add_hydro"]:
        add_hydro(network, config, nodes, prov_centroids, costs, planning_horizons)

    if config["add_H2"]:
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

    if config["heat_coupling"]:
        add_heat_coupling(network, config, nodes, prov_centroids, costs, planning_horizons)

    if config["add_H2"]:
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

        # TODO Why no standing loss?
        network.add(
            "Store",
            nodes + " battery",
            bus=nodes + " battery",
            e_cyclic=True,
            e_nom_extendable=True,
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
            pathway="exp175",
            co2_reduction="0.0",
            planning_horizons=2060,
            heating_demand="positive",
        )
    configure_logging(snakemake)

    network = prepare_network(snakemake.config)
    sanitize_carriers(network, snakemake.config)
    sanitize_carriers(network, snakemake.config)

    network.export_to_netcdf(snakemake.output.network_name)

    outp = {snakemake.output.network_name}
    logger.info(f"Network for {snakemake.wildcards.planning_horizons} prepared and saved to {outp}")
