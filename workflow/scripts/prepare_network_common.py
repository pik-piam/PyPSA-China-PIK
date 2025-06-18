# Shared functions for building the China Regional resolution networks
import pypsa
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import Point

import logging

from _pypsa_helpers import shift_profile_to_planning_year
from functions import HVAC_cost_curve, haversine
from vresutils.costdata import annuity

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


def add_hydro(
    network: pypsa.Network,
    config: dict,
    nodes: pd.Index,
    prov_shapes: gpd.GeoDataFrame,
    costs: pd.DataFrame,
    planning_horizons: int,
    fake_hydro_at_node: bool = False,
):

    # load dams
    df = pd.read_csv(config["hydro_dams"]["dams_path"], index_col=0)
    points = df.apply(lambda row: Point(row.Lon, row.Lat), axis=1)
    dams = gpd.GeoDataFrame(df, geometry=points, crs=CRS)

    hourly_rng = pd.date_range(
        config["hydro_dams"]["inflow_date_start_path"],
        config["hydro_dams"]["inflow_date_end_path"],
        freq=config["freq"],
        inclusive="left",
    )
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
        capital_cost=costs.at["hydro", "capital_cost"],
        efficiency=1,
    )

    # TODO fix hard coded
    # ===  add rivers to link station to station
    bus0s = [0, 21, 11, 19, 22, 29, 8, 40, 25, 1, 7, 4, 10, 15, 12, 20, 26, 6, 3, 39]
    bus1s = [5, 11, 19, 22, 32, 8, 40, 25, 35, 2, 4, 10, 9, 12, 20, 23, 6, 17, 14, 16]

    # normal flow
    for bus0, bus2 in list(zip(dams.index[bus0s], dam_buses.iloc[bus1s].index)):
        network.links.at[bus0 + " turbines", "bus2"] = bus2
        network.links.at[bus0 + " turbines", "efficiency2"] = 1.0

    #  spillage
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

    # add inflow as generators
    # only feed into hydro stations which are the first of a cascade
    inflow_stations = [dam for dam in range(len(dams.index)) if dam not in bus1s]

    for inflow_station in inflow_stations:

        # p_nom = 1 and p_max_pu & p_min_pu = p_pu, compulsory inflow
        p_nom = (inflow / water_consumption_factor).iloc[:, inflow_station].max()
        p_pu = (inflow / water_consumption_factor).iloc[:, inflow_station] / p_nom
        p_pu.index = network.snapshots
        network.add(
            "Generator",
            dams.index[inflow_station] + " inflow",
            bus=dam_buses.iloc[inflow_station].name,
            carrier="hydro_inflow",
            p_max_pu=p_pu.clip(1.0e-6),
            # p_min_pu=p_pu.clip(1.0e-6),
            p_nom=p_nom,
        )

        # p_nom*p_pu = XXX m^3 then use turbines efficiency to convert to power

    # ====== add other existing hydro power =======
    hydro_p_nom = pd.read_hdf(config["hydro_dams"]["p_nom_path"]).tz_localize(None)
    hydro_p_max_pu = pd.read_hdf(
        config["hydro_dams"]["p_max_pu_path"], key=config["hydro_dams"]["p_max_pu_key"]
    ).tz_localize(None)

    hydro_p_max_pu = shift_profile_to_planning_year(hydro_p_max_pu, planning_horizons)
    # sort buses (columns) otherwise stuff will break
    hydro_p_max_pu.sort_index(axis=1, inplace=True)
    # TODO check this respects hours/is still needed
    hydro_p_max_pu = hydro_p_max_pu.loc[network.snapshots]
    hydro_p_max_pu.index = network.snapshots
    network.add(
        "Generator",
        nodes,
        suffix=" hydroelectricity",
        bus=nodes,
        carrier="hydroelectricity",
        p_nom=hydro_p_nom,
        capital_cost=costs.at["hydro", "capital_cost"],
        p_max_pu=hydro_p_max_pu,
    )

    if fake_hydro_at_node:
        # ===  add "fake" hydro at network node (and not real location) ===
        # this allows to introduce capital cost in relevant bus
        # WARNING NOT ROBUST if nodes not the same as province
        if config["add_hydro"] and config["hydro"]["hydro_capital_cost"]:
            hydro_cc = costs.at["hydro", "capital_cost"]
            network.add(
                "StorageUnit",
                dams.index,
                suffix=" hydro dummy",
                bus=dams["Province"],
                carrier="hydroelectricity",
                p_nom=10 * dams["installed_capacity_10MW"],
                p_max_pu=0.0,
                p_min_pu=0.0,
                capital_cost=hydro_cc,
            )


def calc_renewable_pu_avail(
    renewable_ds: xr.Dataset, planning_year: int, snapshots: pd.Index
) -> pd.DataFrame:
    """calaculate the renewable per unit availability

    Args:
        renewable_ds (xr.Dataset): the renewable dataset from build_renewable_potential
        planning_year (int): the investment year
        snapshots (pd.Index): the network snapshots
    """
    profile = renewable_ds["profile"]
    if "year" in profile.dims:
        profile = profile.isel(year=0)
    if "bin" in profile.dims:
        profile = profile.isel(bin=0)  # 只用第一个 bin
    rnwable_p_max_pu = profile.transpose("time", "bus").to_pandas()
    rnwable_p_max_pu = shift_profile_to_planning_year(rnwable_p_max_pu, planning_year)
    if not (snapshots.isin(rnwable_p_max_pu.index)).all():
        err = "Snapshots do not match renewable data profile data:"
        err += f"\n\tmissing {snapshots.difference(rnwable_p_max_pu.index)}.\n"
        tip = "You may may need to regenerate your cutout或adapt the snapshots"
        raise ValueError(err + tip)
    rnwable_p_max_pu = rnwable_p_max_pu.loc[snapshots]
    return rnwable_p_max_pu.sort_index(axis=1)


def add_HV_links(network: pypsa.Network, config: dict, n_years: int):
    """add high voltage connections as links in the lossy transport model (see Neumann et al)

    Args:
        network (pypsa.Network): the pypsa network
        config (dict): the configuration dictionary
        n_years (int): the number of years for discounting
    """

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
