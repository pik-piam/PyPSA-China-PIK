# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# for pathway network

import pypsa
import geopandas as gpd
import pandas as pd
import numpy as np
import os

import xarray as xr
from vresutils.costdata import annuity
from shapely.geometry import Point
from logging import getLogger

from constants import (
    PROV_NAMES,
    CRS,
    TIMEZONE,
    LOAD_CONVERSION_FACTOR,
    YEAR_HRS,
    INFLOW_DATA_YR,
    NUCLEAR_EXTENDABLE,
    NON_LIN_PATH_SCALING,
    LINE_SECURITY_MARGIN,
    FOM_LINES,
    ECON_LIFETIME_LINES,
)
from functions import HVAC_cost_curve
from _helpers import configure_logging, mock_snakemake, ConfigManager
from _pypsa_helpers import (
    make_periodic_snapshots,
    shift_profile_to_planning_year,
)
from add_electricity import load_costs, sanitize_carriers
from functions import haversine
from readers_geospatial import read_province_shapes

logger = getLogger(__name__)
logger.setLevel("INFO")


def add_buses(
    network: pypsa.Network,
    nodes: list | pd.Index,
    suffix: str,
    carrier: str,
    prov_centroids: gpd.GeoSeries,
):
    """Add buses

    Args:
        network (pypsa.Network): _description_
        nodes (list | pd.Index): _description_
        suffix (str): _description_
        carrier (str): _description_
        prov_centroids (gpd.GeoSeries): _description_
    """

    network.add(
        "Bus",
        nodes,
        suffix=suffix,
        x=prov_centroids.x,
        y=prov_centroids.y,
        carrier=carrier,
        location=nodes,
    )


def add_carriers(network: pypsa.Network, config: dict, costs: pd.DataFrame):
    """ad the various carriers to the network based on the config file

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

    if "coal power plant" in config["Techs"]["conv_techs"] and config["Techs"]["coal_ccs_retrofit"]:
        network.add("Carrier", "coal cc", co2_emissions=0.034)


def add_co2_constraints_prices(network: pypsa.Network, co2_control: dict):
    """Add co2 constraints or prices

    Args:
        network (pypsa.Network): the network to which prices or constraints are to be added
        co2_control (dict): the config


    Raises:
        ValueError: unrecognised co2 control option
    """

    if co2_control["control"] is None:
        pass
    elif co2_control["control"].startswith("budget"):
        co2_limit = co2_control["co2_pr_or_limit"]
        logger.info("Adding CO2 constraint based on scenario {co2_limit}")
        network.add(
            "GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit,
        )
    else:
        logger.error(f"Unhandled CO2 control config {co2_control} due to unknown control.")
        raise ValueError(f"Unhandled CO2 config {config['scenario']['co2_reduction']}")


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
        techs (list): A list of renewable energy technologies to add (e.g., ["solar", "onwind", "offwind"])
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
            # shift weather year to planning year
            timestamps = timestamps.map(lambda t: t.replace(year=int(year)))
            ds = ds.assign_coords(time=timestamps)

            mask = ds.time.isin(network.snapshots)
            ds = ds.sel(time=mask)

            if not len(ds.time) == len(network.snapshots):
                raise ValueError(
                    f"Mismatch in profile and network timestamps {len(ds.time)} and {len(network.snapshots)}"
                )
            ds = ds.stack(bus_bin=["bus", "bin"])

        # bins represent renewable generation grades
        flatten = lambda t: " grade".join(map(str, t))
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


def prepare_network(config: dict, costs: pd.DataFrame, paths: dict) -> pypsa.Network:
    """Prepares/makes the network object for myopic mode according to config &
    at 1 node per region/province

    Args:
        config (dict): the snakemake config
        costs (pd.DataFrame): the costs dataframe (anualised capex and marginal costs)
        paths (dict): dictionary of paths to input data

    Returns:
        pypsa.Network: the pypsa network object
    """

    # derive the config
    config["add_gas"] = (
        True if [tech for tech in config["Techs"]["conv_techs"] if "gas" in tech] else False
    )
    config["add_coal"] = (
        True if [tech for tech in config["Techs"]["conv_techs"] if "coal" in tech] else False
    )

    planning_horizons = snakemake.wildcards["planning_horizons"]
    # empty network object
    network = pypsa.Network()
    # load graph
    nodes = pd.Index(PROV_NAMES)

    # set times
    # make snapshots (drop leap days)
    snapshot_cfg = config["snapshots"]
    snapshots = make_periodic_snapshots(
        year=snakemake.wildcards.planning_horizons,
        freq=snapshot_cfg["freq"],
        start_day_hour=snapshot_cfg["start"],
        end_day_hour=snapshot_cfg["end"],
        bounds=snapshot_cfg["bounds"],
        # naive local tz
        tz=None,
        end_year=(
            None
            if not snapshot_cfg["end_year_plus1"]
            else snakemake.wildcards.planning_horizons + 1
        ),
    )
    network.set_snapshots(snapshots.values)

    network.snapshot_weightings[:] = config["snapshots"]["frequency"]
    represented_hours = network.snapshot_weightings.sum()[0]
    # TODO: what about leap years?
    n_years = represented_hours / YEAR_HRS

    prov_shapes = read_province_shapes(snakemake.input.province_shape)
    prov_centroids = prov_shapes.to_crs("+proj=cea").centroid.to_crs(CRS)

    # TODO split by carrier, make transparent
    # add buses
    for suffix in config["bus_suffix"]:
        carrier = config["bus_carrier"][suffix]
        add_buses(network, nodes, suffix, carrier, prov_centroids)

    add_carriers(network, config, costs)

    # ===== add load demand data =======
    demand_path = snakemake.input.elec_load.replace("{planning_horizons}", cost_year)
    with pd.HDFStore(demand_path, mode="r") as store:
        load = store["load"].loc[network.snapshots]  # MWh !!

    load.columns = PROV_NAMES

    network.add("Load", nodes, bus=nodes, p_set=load[nodes])

    ws_carriers = [c for c in config["Techs"]["vre_techs"] if c.find("wind") >= 0 or c == "solar"]
    add_wind_and_solar(network, ws_carriers, paths, planning_horizons, costs)

    if config["heat_coupling"]:

        central_fraction = pd.read_hdf(snakemake.input.central_fraction)
        with pd.HDFStore(snakemake.input.heat_demand_profile, mode="r") as store:
            heat_demand = store["heat_demand_profiles"]
            # TODO fix this possilby not working
            heat_demand.index = heat_demand.index.tz_localize(None)
            heat_demand = heat_demand.loc[network.snapshots]

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

    # ====== add gas techs ======
    if [tech for tech in config["Techs"]["conv_techs"] if "gas" in tech]:

        # add converter from fuel source
        network.add(
            "Generator",
            nodes,
            suffix=" gas fuel",
            bus=nodes + " gas",
            carrier="gas",
            p_nom_extendable=False,
            p_nom=1e8,
            marginal_cost=costs.at["OCGT", "fuel"],
        )

        network.add(
            "Store",
            nodes + " gas Store",
            bus=nodes + " gas",
            e_nom_extendable=False,
            e_nom=1e8,
            e_cyclic=True,
            carrier="gas",
        )

    if "OCGT gas" in config["Techs"]["conv_techs"]:
        network.add(
            "Link",
            nodes,
            suffix=" OCGT",
            bus0=nodes + " gas",
            bus1=nodes,
            carrier="OCGT gas",
            marginal_cost=costs.at["OCGT", "efficiency"]
            * costs.at["OCGT", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["OCGT", "efficiency"]
            * costs.at["OCGT", "capital_cost"],  # NB: capital cost is per MWel
            p_nom_extendable=True,
            efficiency=costs.at["OCGT", "efficiency"],
            lifetime=costs.at["OCGT", "lifetime"],
        )

    if "gas boiler" in config["Techs"]["conv_techs"] and config["heat_coupling"]:
        for cat in [" decentral "]:
            network.add(
                "Link",
                nodes + cat + "gas boiler",
                p_nom_extendable=True,
                bus0=nodes + " gas",
                bus1=nodes + cat + "heat",
                carrier="gas boiler",
                efficiency=costs.at[cat.lstrip() + "gas boiler", "efficiency"],
                marginal_cost=costs.at[cat.lstrip() + "gas boiler", "efficiency"]
                * costs.at[cat.lstrip() + "gas boiler", "VOM"],
                capital_cost=costs.at[cat.lstrip() + "gas boiler", "efficiency"]
                * costs.at[cat.lstrip() + "gas boiler", "capital_cost"],
                lifetime=costs.at[cat.lstrip() + "gas boiler", "lifetime"],
            )

    # TODO missing second bus?
    if "CHP gas" in config["Techs"]["conv_techs"]:
        network.add(
            "Link",
            nodes,
            suffix=" central CHP gas generator",
            bus0=nodes + " gas",
            bus1=nodes,
            carrier="CHP gas",
            p_nom_extendable=True,
            marginal_cost=costs.at["central gas CHP", "efficiency"]
            * costs.at["central gas CHP", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["central gas CHP", "efficiency"]
            * costs.at["central gas CHP", "capital_cost"],  # NB: capital cost is per MWel
            efficiency=costs.at["central gas CHP", "efficiency"],
            p_nom_ratio=1.0,
            c_b=costs.at["central gas CHP", "c_b"],
            lifetime=costs.at["central gas CHP", "lifetime"],
        )

        network.add(
            "Link",
            nodes,
            suffix=" central CHP gas boiler",
            bus0=nodes + " gas",
            bus1=nodes + " central heat",
            carrier="CHP gas",
            p_nom_extendable=True,
            marginal_cost=costs.at["central gas CHP", "efficiency"]
            * costs.at["central gas CHP", "VOM"],  # NB: VOM is per MWel
            efficiency=costs.at["central gas CHP", "efficiency"]
            / costs.at["central gas CHP", "c_v"],
            lifetime=costs.at["central gas CHP", "lifetime"],
        )

    # TODO separate retrofit in config from coal power plant
    if "coal power plant" in config["Techs"]["conv_techs"] and config["Techs"]["coal_ccs_retrofit"]:
        network.add(
            "Generator",
            nodes,
            suffix=" coal cc",
            bus=nodes,
            carrier="coal cc",
            p_nom_extendable=True,
            efficiency=costs.at["coal", "efficiency"],
            marginal_cost=costs.at["coal", "marginal_cost"],
            capital_cost=costs.at["coal", "capital_cost"]
            + costs.at["retrofit", "capital_cost"],  # NB: capital cost is per MWel
            lifetime=costs.at["coal", "lifetime"],
        )
        # TODO FIXME harcoded
        for year in range(int(planning_horizons) - 25, 2021, 5):
            network.add(
                "Generator",
                nodes,
                suffix=" coal-" + str(year) + "-retrofit",
                bus=nodes,
                carrier="coal cc",
                p_nom_extendable=True,
                capital_cost=costs.at["coal", "capital_cost"]
                + costs.at["retrofit", "capital_cost"]
                + 2021
                - year,
                efficiency=costs.at["coal", "efficiency"],
                lifetime=costs.at["coal", "lifetime"],
                build_year=year,
                marginal_cost=costs.at["coal", "marginal_cost"],
            )

    # ===== add coal techs =====
    if [tech for tech in config["Techs"]["conv_techs"] if "coal" in tech]:
        # TODO check if this is needed (added Ivan), add for gas too, also why is it node resolved?
        # network.add(
        #     "Bus",
        #     nodes,
        #     suffix=" coal fuel",
        #     x=prov_centroids.x,
        #     y=prov_centroids.y,
        #     carrier="coal",
        # )

        network.add(
            "Generator",
            nodes + " coal fuel",
            bus=nodes + " coal",
            carrier="coal",
            p_nom_extendable=False,
            p_nom=1e8,
            marginal_cost=costs.at["coal", "marginal_cost"],
        )

    if "coal boiler" in config["Techs"]["conv_techs"]:
        for cat in [" decentral ", " central "]:
            network.add(
                "Link",
                nodes + cat + "coal boiler",
                p_nom_extendable=True,
                bus0=nodes + " coal",
                bus1=nodes + cat + "heat",
                carrier="coal boiler",
                efficiency=costs.at[cat.lstrip() + "coal boiler", "efficiency"],
                marginal_cost=costs.at[cat.lstrip() + "coal boiler", "efficiency"]
                * costs.at[cat.lstrip() + "coal boiler", "VOM"],
                capital_cost=costs.at[cat.lstrip() + "coal boiler", "efficiency"]
                * costs.at[cat.lstrip() + "coal boiler", "capital_cost"],
                lifetime=costs.at[cat.lstrip() + "coal boiler", "lifetime"],
            )
    # TODO missing second bus?
    if "CHP coal" in config["Techs"]["conv_techs"]:
        network.add(
            "Link",
            nodes,
            suffix=" central CHP coal generator",
            bus0=nodes + " coal",
            bus1=nodes,
            carrier="CHP coal",
            p_nom_extendable=True,
            marginal_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "capital_cost"],  # NB: capital cost is per MWel
            efficiency=costs.at["central coal CHP", "efficiency"],
            p_nom_ratio=1.0,
            c_b=costs.at["central coal CHP", "c_b"],
            lifetime=costs.at["central coal CHP", "lifetime"],
        )

        network.add(
            "Link",
            nodes,
            suffix=" central CHP coal boiler",
            bus0=nodes + " coal",
            bus1=nodes + " central heat",
            carrier="CHP coal",
            p_nom_extendable=True,
            marginal_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
            efficiency=costs.at["central coal CHP", "efficiency"]
            / costs.at["central coal CHP", "c_v"],
            lifetime=costs.at["central coal CHP", "lifetime"],
        )

    if config["add_biomass"]:
        network.add(
            "Bus",
            nodes,
            suffix=" biomass",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="biomass",
        )

        biomass_potential = pd.read_hdf(snakemake.input.biomass_potential)
        biomass_potential.index = biomass_potential.index + " biomass"
        network.add(
            "Store",
            nodes + " biomass",
            bus=nodes + " biomass",
            e_nom_extendable=False,
            e_nom=biomass_potential,
            e_initial=biomass_potential,
            carrier="biomass",
        )

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
        # TODO rmemoe hard coded
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
            # 4187.0095385594495TWh equates to 0.79*(5.24/3.04) Gt CO2  # tCO2/MWh
            # TODO centralise
            efficiency3=0.32522269504651985,
            capital_cost=costs.at["biomass CHP", "efficiency"]
            * costs.at["biomass CHP", "capital_cost"],
            marginal_cost=costs.at["biomass CHP", "efficiency"]
            * costs.at["biomass CHP", "marginal_cost"]
            + costs.at["solid biomass", "fuel"],
            lifetime=costs.at["biomass CHP", "lifetime"],
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

    if config["add_hydro"]:

        # load dams
        df = pd.read_csv(config["hydro_dams"]["dams_path"], index_col=0)
        points = df.apply(lambda row: Point(row.Lon, row.Lat), axis=1)
        dams = gpd.GeoDataFrame(df, geometry=points, crs=CRS)

        hourly_rng = pd.date_range(
            config["hydro_dams"]["inflow_date_start"],
            config["hydro_dams"]["inflow_date_end"],
            freq="1h",
            inclusive="left",
        )
        inflow = pd.read_pickle(config["hydro_dams"]["inflow_path"]).reindex(
            hourly_rng, fill_value=0
        )
        inflow.columns = dams.index
        # convert to naive local timezone
        inflow.index = inflow.index.tz_localize("UTC").tz_convert(TIMEZONE).tz_localize(None)
        inflow = inflow.loc[str(INFLOW_DATA_YR)]
        inflow = shift_profile_to_planning_year(inflow, planning_horizons)
        inflow = inflow.loc[network.snapshots]

        water_consumption_factor = (
            dams.loc[:, "Water_consumption_factor_avg"] * 1e3
        )  # m^3/KWh -> m^3/MWh

        ###
        # # Add hydro stations as buses
        network.add(
            "Bus",
            dams.index,
            suffix=" station",
            carrier="stations",
            x=dams["geometry"].to_crs("+proj=cea").centroid.to_crs(CRS).x,
            y=dams["geometry"].to_crs("+proj=cea").centroid.to_crs(CRS).y,
            location=dams["Province"],
        )

        dam_buses = network.buses[network.buses.carrier == "stations"]

        # ===== add hydro reservoirs as stores ======
        initial_capacity = pd.read_pickle(config["hydro_dams"]["reservoir_initial_capacity_path"])
        effective_capacity = pd.read_pickle(
            config["hydro_dams"]["reservoir_effective_capacity_path"]
        )
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
            marginal_cost=config["hyro"]["marginal_cost"]["reservoir"]  # EUR/MWh"
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
            capital_cost=costs.at["hydro", "capital_cost"],
            efficiency=1,
        )

        # add rivers to link station to station
        bus0s = [
            0,
            21,
            11,
            19,
            22,
            29,
            8,
            40,
            25,
            1,
            7,
            4,
            10,
            15,
            12,
            20,
            26,
            6,
            3,
            39,
        ]
        bus1s = [
            5,
            11,
            19,
            22,
            32,
            8,
            40,
            25,
            35,
            2,
            4,
            10,
            9,
            12,
            20,
            23,
            6,
            17,
            14,
            16,
        ]

        for bus0, bus2 in list(zip(dams.index[bus0s], dam_buses.iloc[bus1s].index)):

            # normal flow
            network.links.at[bus0 + " turbines", "bus2"] = bus2
            network.links.at[bus0 + " turbines", "efficiency2"] = 1.0

        # spillage
        for bus0, bus1 in list(zip(dam_buses.iloc[bus0s].index, dam_buses.iloc[bus1s].index)):
            network.add(
                "Link",
                "{}-{}".format(bus0, bus1) + " spillage",
                bus0=bus0,
                bus1=bus1,
                p_nom_extendable=True,
            )

        dam_ends = [
            dam
            for dam in range(len(dams.index))
            if (dam in bus1s and dam not in bus0s) or (dam not in bus0s + bus1s)
        ]

        for bus0 in dam_buses.iloc[dam_ends].index:
            network.add(
                "Link",
                bus0 + " spillage",
                bus0=bus0,
                bus1="Tibet",
                p_nom_extendable=True,
                efficiency=0.0,
            )

        # == add inflow as generators
        # only feed into hydro stations which are the first of a cascade
        inflow_stations = [dam for dam in range(len(dams.index)) if dam not in bus1s]

        for inflow_station in inflow_stations:

            # p_nom = 1 and p_max_pu & p_min_pu = p_pu, compulsory inflow

            p_nom = (inflow / water_consumption_factor).iloc[:, inflow_station].max()
            p_pu = (inflow / water_consumption_factor).iloc[:, inflow_station] / p_nom

            network.add(
                "Generator",
                dams.index[inflow_station] + " inflow",
                bus=dam_buses.iloc[inflow_station].name,
                carrier="hydro_inflow",
                p_max_pu=p_pu.clip(1.0e-6),
                p_min_pu=p_pu.clip(1.0e-6),
                p_nom=p_nom,
            )

            # p_nom*p_pu = XXX m^3 then use turbines efficiency to convert to power

        # ======= add other existing hydro power
        hydro_p_nom = pd.read_hdf(config["hydro_dams"]["p_nom_path"])
        hydro_p_max_pu = pd.read_hdf(
            config["hydro_dams"]["p_max_pu_path"],
            key=config["hydro_dams"]["p_max_pu_key"],
        ).tz_localize(None)

        hydro_p_max_pu = shift_profile_to_planning_year(hydro_p_max_pu, planning_horizons)
        # sort buses (columns) otherwise stuff will break
        hydro_p_max_pu.sort_index(axis=1, inplace=True)

        hydro_p_max_pu = hydro_p_max_pu.loc[snapshots]
        hydro_p_max_pu.index = network.snapshots

        network.add(
            "Generator",
            nodes,
            suffix=" hydroelectricity",
            bus=nodes,
            carrier="hydroelectricity",
            p_nom=hydro_p_nom,
            p_nom_min=hydro_p_nom,
            p_nom_extendable=False,
            capital_cost=costs.at["hydro", "capital_cost"],
            p_max_pu=hydro_p_max_pu,
        )

    if config["add_H2"]:

        network.add(
            "Bus",
            nodes,
            suffix=" H2",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="H2",
        )

        network.add(
            "Link",
            nodes + " H2 Electrolysis",
            bus0=nodes,
            bus1=nodes + " H2",
            bus2=nodes + " central heat",
            p_nom_extendable=True,
            carrier="H2",
            efficiency=costs.at["electrolysis", "efficiency"],
            efficiency2=costs.at["electrolysis", "efficiency-heat"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            lifetime=costs.at["electrolysis", "lifetime"],
        )

        network.add(
            "Link",
            nodes + " central H2 CHP",
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

        # TODO fix hard coded
        H2_under_nodes = pd.Index(
            [
                "Sichuan",
                "Chongqing",
                "Hubei",
                "Jiangxi",
                "Anhui",
                "Jiangsu",
                "Shandong",
                "Guangdong",
            ]
        )
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
            capital_cost=costs.at[
                "hydrogen storage tank type 1 including compressor", "capital_cost"
            ],
            lifetime=costs.at["hydrogen storage tank type 1 including compressor", "lifetime"],
        )

    if config["add_methanation"]:
        network.add(
            "Link",
            nodes + " Sabatier",
            bus0=nodes + " H2",
            bus1=nodes + " gas",
            p_nom_extendable=True,
            carrier="Sabatier",
            efficiency=costs.at["methanation", "efficiency"],
            capital_cost=costs.at["methanation", "efficiency"]
            * costs.at["methanation", "capital_cost"]
            + costs.at["direct air capture", "capital_cost"]
            * costs.at["gas", "co2_emissions"]
            * costs.at["methanation", "efficiency"],
            # TODO fix hardcoded
            marginal_cost=(400 - 5 * (int(cost_year) - 2020))
            * costs.at["gas", "co2_emissions"]
            * costs.at["methanation", "efficiency"],
            lifetime=costs.at["methanation", "lifetime"],
        )

    if "nuclear" in config["Techs"]["vre_techs"]:
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

    if "heat pump" in config["Techs"]["vre_techs"]:

        with pd.HDFStore(snakemake.input.cop_name, mode="r") as store:
            ashp_cop = store["ashp_cop_profiles"]
            ashp_cop.index = ashp_cop.index.tz_localize(None)
            ashp_cop = shift_profile_to_planning_year(ashp_cop, planning_horizons)
            gshp_cop = store["gshp_cop_profiles"]
            gshp_cop.index = gshp_cop.index.tz_localize(None)
            gshp_cop = shift_profile_to_planning_year(gshp_cop, planning_horizons)
            ashp_cop = ashp_cop.loc[snapshots]
            gshp_cop = gshp_cop.loc[snapshots]

        for cat in [" decentral ", " central "]:
            network.add(
                "Link",
                nodes,
                suffix=cat + "heat pump",
                bus0=nodes,
                bus1=nodes + cat + "heat",
                carrier="heat pump",
                efficiency=(
                    ashp_cop[nodes]
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
            # TODO not valid for decentral
            network.add(
                "Link",
                nodes,
                suffix=cat + " ground heat pump",
                bus0=nodes,
                bus1=nodes + cat + "heat",
                carrier="heat pump",
                efficiency=(
                    gshp_cop[nodes]
                    if config["time_dep_hp_cop"]
                    else costs.at["decentral ground-sourced heat pump", "efficiency"]
                ),
                capital_cost=costs.at[cat.lstrip() + "ground-sourced heat pump", "efficiency"]
                * costs.at["decentral ground-sourced heat pump", "capital_cost"],
                marginal_cost=costs.at[cat.lstrip() + "ground-sourced heat pump", "efficiency"]
                * costs.at[cat.lstrip() + "ground-sourced heat pump", "marginal_cost"],
                p_nom_extendable=True,
                lifetime=costs.at["decentral ground-sourced heat pump", "lifetime"],
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

    if "solar thermal" in config["Techs"]["vre_techs"]:
        # this is the amount of heat collected in W per m^2, accounting
        # for efficiency
        with pd.HDFStore(snakemake.input.solar_thermal_name, mode="r") as store:
            # 1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
            solar_thermal = config["solar_cf_correction"] * store["solar_thermal_profiles"] / 1e3

        solar_thermal.index = solar_thermal.index.tz_localize(None)
        solar_thermal = shift_profile_to_planning_year(solar_thermal, planning_horizons)
        solar_thermal = solar_thermal.loc[snapshots]

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

    if "water tanks" in config["Techs"]["store_techs"]:
        for cat in [" decentral ", " central "]:
            network.add(
                "Bus",
                nodes,
                suffix=cat + "water tanks",
                x=prov_centroids.x,
                y=prov_centroids.y,
                carrier="water tanks",
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

    if "battery" in config["Techs"]["store_techs"]:
        network.add(
            "Bus",
            nodes,
            suffix=" battery",
            x=prov_centroids.x,
            y=prov_centroids.y,
            carrier="battery",
        )

        network.add(
            "Store",
            nodes + " battery",
            bus=nodes + " battery",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=costs.at["battery storage", "capital_cost"],
            lifetime=costs.at["battery storage", "lifetime"],
        )

        network.add(
            "Link",
            nodes + " battery charger",
            bus0=nodes,
            bus1=nodes + " battery",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["battery inverter", "capital_cost"],
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
            carrier="battery discharger",
            p_nom_extendable=True,
        )

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

    # ============= add lines =========
    # The lines are implemented according to the transport model (no KVL) with losses.
    # This requires two directions
    # see Neumann et al 10.1016/j.apenergy.2022.118859
    # TODO make lossless optional (speed up)

    if not config["no_lines"]:
        edges = pd.read_csv(snakemake.input.edges, header=None)

        lengths = NON_LIN_PATH_SCALING * np.array(
            [
                haversine(
                    [network.buses.at[name0, "x"], network.buses.at[name0, "y"]],
                    [network.buses.at[name1, "x"], network.buses.at[name1, "y"]],
                )
                for name0, name1 in edges[[0, 1]].values
            ]
        )

        cc = (
            (config["line_cost_factor"] * lengths * [HVAC_cost_curve(len_) for len_ in lengths])
            * LINE_SECURITY_MARGIN
            * FOM_LINES
            * n_years
            * annuity(ECON_LIFETIME_LINES, config["costs"]["discountrate"])
        )

        network.add(
            "Link",
            edges[0] + "-" + edges[1],
            bus0=edges[0].values,
            bus1=edges[1].values,
            suffix=" positive",
            p_nom_extendable=True,
            p_min_pu=0,
            efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]
            * config["transmission_efficiency"]["DC"]["efficiency_per_1000km"] ** (lengths / 1000),
            length=lengths,
            capital_cost=cc,
        )

        network.add(
            "Link",
            edges[1] + "-" + edges[0],
            bus0=edges[1].values,
            bus1=edges[0].values,
            suffix=" reversed",
            p_nom_extendable=True,
            p_min_pu=0,
            efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]
            * config["transmission_efficiency"]["DC"]["efficiency_per_1000km"] ** (lengths / 1000),
            length=lengths,
            capital_cost=0,
        )

    if config["Techs"]["hydrogen_lines"]:
        edges = pd.read_csv(snakemake.input.edges, header=None)
        lengths = NON_LIN_PATH_SCALING * np.array(
            [
                haversine(
                    [network.buses.at[name0, "x"], network.buses.at[name0, "y"]],
                    [network.buses.at[name1, "x"], network.buses.at[name1, "y"]],
                )
                for name0, name1 in edges[[0, 1]].values
            ]
        )

        cc = costs.at["H2 (g) pipeline", "capital_cost"] * lengths

        network.add(
            "Link",
            edges[0] + "-" + edges[1] + " H2 pipeline",
            suffix=" positive",
            bus0=edges[0].values + " H2",
            bus1=edges[1].values + " H2",
            bus2=edges[0].values,
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
            edges[1] + "-" + edges[0] + " H2 pipeline",
            suffix=" reversed",
            bus0=edges[1].values + " H2",
            bus1=edges[0].values + " H2",
            bus2=edges[1].values,
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
    return network


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "prepare_base_networks",
            opts="ll",
            topology="current+FCG",
            co2_pathway="exp175default",
            planning_horizons="2030",
            heating_demand="positive",
        )
    configure_logging(snakemake)

    yr = int(snakemake.wildcards.planning_horizons)
    logger.info(f"Preparing network for {yr}")

    pathway = snakemake.wildcards.co2_pathway
    config = snakemake.config
    input_paths = {k: v for k, v in snakemake.input.items()}

    co2_opts = ConfigManager(config).fetch_co2_restriction(pathway, yr)
    if not co2_opts["control"].startswith("budget"):
        raise ValueError("Only budget CO2 control is currently supported in myopic")

    # TODO pass this to prep network
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
    tech_costs = snakemake.input.tech_costs
    cost_year = snakemake.wildcards["planning_horizons"]
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    network = prepare_network(snakemake.config, costs, input_paths)
    add_co2_constraints_prices(network, co2_opts)
    sanitize_carriers(network, snakemake.config)

    outp = snakemake.output.network_name
    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    network.export_to_netcdf(outp, compression=compression)

    msg = f"Network for {snakemake.wildcards.planning_horizons} prepared and saved to {outp}"
    logger.info(msg)

    costs_outp = os.path.dirname(outp) + f"/costs_{yr}.csv"
    costs.to_csv(costs_outp)
