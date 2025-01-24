# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# for pathway network

from vresutils.costdata import annuity
import pypsa
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr


from constants import (
    PROV_NAMES,
    CRS,
    YEAR_HRS,
    LOAD_CONVERSION_FACTOR,
    INFLOW_DATA_YR,
    LINE_SECURITY_MARGIN,
    FOM_LINES,
    NON_LIN_PATH_SCALING,
    ECON_LIFETIME_LINES,
    CO2_EL_2020,
    CO2_HEATING_2020,
)
from _helpers import (
    configure_logging,
    override_component_attrs,
    mock_snakemake,
    make_periodic_snapshots,
)
from functions import HVAC_cost_curve
from readers import read_province_shapes
from add_electricity import load_costs
from functions import haversine
from prepare_base_network import add_buses, shift_profile_to_planning_year, add_carriers

from logging import getLogger, DEBUG  # INFO

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def prepare_network(config):

    # derive from the config
    config["add_gas"] = (
        True if [tech for tech in config["Techs"]["conv_techs"] if "gas" in tech] else False
    )
    config["add_coal"] = (
        True if [tech for tech in config["Techs"]["conv_techs"] if "coal" in tech] else False
    )

    if "overrides" in snakemake.input.keys():
        overrides = override_component_attrs(snakemake.input.overrides)
        network = pypsa.Network(override_component_attrs=overrides)
    else:
        network = pypsa.Network()

    # set times
    planning_horizons = snakemake.wildcards["planning_horizons"]
    # make snapshots (drop leap days)
    snapshot_cfg = config["snapshots"]
    snapshots = make_periodic_snapshots(
        year=snakemake.wildcards.planning_horizons,
        freq=snapshot_cfg["freq"],
        start_day_hour=snapshot_cfg["start"],
        end_day_hour=snapshot_cfg["end"],
        bounds=snapshot_cfg["bounds"],
        tz=snapshot_cfg["timezone"],
        end_year=(
            None
            if not snapshot_cfg["end_year_plus1"]
            else snakemake.wildcards.planning_horizons + 1
        ),
    )

    network.set_snapshots(snapshots.values)
    network.snapshot_weightings[:] = config["snapshots"]["frequency"]
    represented_hours = network.snapshot_weightings.sum()[0]
    n_years = represented_hours / YEAR_HRS

    # load graph
    nodes = pd.Index(PROV_NAMES)
    pathway = snakemake.wildcards["pathway"]

    tech_costs = snakemake.input.tech_costs
    cost_year = snakemake.wildcards["planning_horizons"]
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    # load data sets
    ds_solar = xr.open_dataset(snakemake.input.profile_solar)
    ds_onwind = xr.open_dataset(snakemake.input.profile_onwind)
    ds_offwind = xr.open_dataset(snakemake.input.profile_offwind)

    # == shift datasets  from reference to planning year, sort columns to match network bus order ==
    solar_p_max_pu = ds_solar["profile"].transpose("time", "bus").to_pandas()
    solar_p_max_pu = shift_profile_to_planning_year(solar_p_max_pu, planning_horizons)
    solar_p_max_pu = solar_p_max_pu.loc[snapshots]
    solar_p_max_pu.sort_index(axis=1, inplace=True)

    onwind_p_max_pu = ds_onwind["profile"].transpose("time", "bus").to_pandas()
    onwind_p_max_pu = shift_profile_to_planning_year(onwind_p_max_pu, planning_horizons)
    onwind_p_max_pu = onwind_p_max_pu.loc[snapshots]
    onwind_p_max_pu.sort_index(axis=1, inplace=True)

    offwind_p_max_pu = ds_offwind["profile"].transpose("time", "bus").to_pandas()
    offwind_p_max_pu = shift_profile_to_planning_year(offwind_p_max_pu, planning_horizons)
    offwind_p_max_pu = offwind_p_max_pu.loc[snapshots]
    offwind_p_max_pu.sort_index(axis=1, inplace=True)

    tech_costs = snakemake.input.tech_costs
    cost_year = snakemake.wildcards["planning_horizons"]
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    prov_shapes = read_province_shapes(snakemake.input.province_shape)
    prov_centroids = prov_shapes.to_crs("+proj=cea").centroid.to_crs(CRS)

    # add buses
    for suffix in config["bus_suffix"]:
        carrier = config["bus_carrier"][suffix]
        add_buses(network, nodes, suffix, carrier, prov_centroids)

    # add carriers
    add_carriers(network, config, costs)

    # add global constraint
    if not isinstance(config["scenario"]["co2_reduction"], tuple):

        if config["scenario"]["co2_reduction"] is not None:

            # extra co2
            co2_limit = (CO2_EL_2020 + CO2_HEATING_2020) * (
                1 - config["scenario"]["co2_reduction"][pathway][planning_horizons]
            )  # Chinese 2020 CO2 emissions of electric and heating sector

            network.add(
                "GlobalConstraint",
                "co2_limit",
                type="primary_energy",
                carrier_attribute="co2_emissions",
                sense="<=",
                constant=co2_limit,
            )

    # load demand data
    demand_path = snakemake.input.elec_load.replace("{planning_horizons}", cost_year)
    with pd.HDFStore(demand_path, mode="r") as store:
        load = LOAD_CONVERSION_FACTOR * store["load"]  # TODO add unit
        load = load.loc[network.snapshots]

    load.columns = PROV_NAMES

    network.add("Load", nodes, bus=nodes, p_set=load[nodes])

    if config["heat_coupling"]:

        central_fraction = pd.read_hdf(snakemake.input.central_fraction)
        with pd.HDFStore(snakemake.input.heat_demand_profile, mode="r") as store:
            heat_demand = store["heat_demand_profiles"]
            heat_demand = heat_demand.tz_localize(None)
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

    if config["add_gas"]:
        # add converter from fuel source
        network.add(
            "Generator",
            nodes,
            suffix=" gas fuel",
            bus=nodes + " gas",
            carrier="gas",
            p_nom_extendable=True,
            marginal_cost=costs.at["OCGT", "fuel"],
        )

        network.add("Carrier", "biogas")
        network.add(
            "Store",
            nodes + " gas Store",
            bus=nodes + " gas",
            e_nom_extendable=True,
            carrier="biogas",
        )

    # TODO Clarify this
    if config["add_coal"]:
        network.add(
            "Generator",
            nodes + " coal fuel",
            bus=nodes + " coal",
            carrier="coal",
            p_nom_extendable=True,
            efficiency=costs.at["coal", "efficiency"],
            marginal_cost=costs.at["coal", "marginal_cost"],
            capital_cost=costs.at["coal", "efficiency"]
            * costs.at["coal", "capital_cost"],  # NB: capital cost is per MWel
            lifetime=costs.at["coal", "lifetime"],
        )

    # TODO decide readd?
    # network.add(
    #     "Generator",
    #     nodes,
    #     suffix=" nuclear",
    #     p_nom_extendable=True,
    #     # p_nom=nuclear_p_nom,
    #     # p_nom_max=nuclear_p_nom * 2,
    #     # p_nom_min=nuclear_p_nom,
    #     bus=nodes,
    #     carrier="nuclear",
    #     efficiency=costs.at["nuclear", "efficiency"],
    #     # NB: capital cost is per MWel, for nuclear already per MWel
    #     capital_cost=costs.at["nuclear", "capital_cost"],
    #     marginal_cost=costs.at["nuclear", "marginal_cost"],
    # )

    if config["add_hydro"]:

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
        inflow = pd.read_pickle(config["hydro_dams"]["inflow_path"]).reindex(
            hourly_rng, fill_value=0
        )
        inflow.columns = dams.index
        inflow = inflow.loc[str(INFLOW_DATA_YR)]
        inflow = shift_profile_to_planning_year(inflow, INFLOW_DATA_YR)

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

        # add rivers to link station to station
        bus0s = [0, 21, 11, 19, 22, 29, 8, 40, 25, 1, 7, 4, 10, 15, 12, 20, 26, 6, 3, 39]
        bus1s = [5, 11, 19, 22, 32, 8, 40, 25, 35, 2, 4, 10, 9, 12, 20, 23, 6, 17, 14, 16]

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
            p_pu.index = network.snapshots
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

        # TODO clarify what this is and where it comes from
        # ======= add other existing hydro power
        hydro_p_nom = pd.read_hdf(config["hydro_dams"]["p_nom_path"])
        hydro_p_max_pu = pd.read_hdf(
            config["hydro_dams"]["p_max_pu_path"], key=config["hydro_dams"]["p_max_pu_key"]
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

    # add components
    network.add(
        "Generator",
        nodes,
        suffix=" onwind",
        bus=nodes,
        carrier="onwind",
        p_nom_extendable=False,
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
        p_nom_extendable=False,
        p_nom_max=ds_offwind["p_nom_max"].to_pandas()[offwind_nodes],
        capital_cost=costs.at["offwind", "capital_cost"],
        marginal_cost=costs.at["offwind", "marginal_cost"],
        p_max_pu=offwind_p_max_pu[offwind_nodes],
        lifetime=costs.at["offwind", "lifetime"],
    )

    network.add(
        "Generator",
        nodes,
        suffix=" solar",
        bus=nodes,
        carrier="solar",
        p_nom_extendable=False,
        p_nom_max=ds_solar["p_nom_max"].to_pandas(),
        capital_cost=costs.at["solar", "capital_cost"],
        marginal_cost=costs.at["solar", "marginal_cost"],
        p_max_pu=solar_p_max_pu,
        lifetime=costs.at["solar", "lifetime"],
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

        date_range = pd.date_range(
            "2025-01-01 00:00",
            "2025-12-31 23:00",
            freq=config["snapshots"]["freq"],
            tz="Asia/shanghai",
        )
        date_range = date_range.map(lambda t: t.replace(year=2020))

        solar_thermal.index = solar_thermal.index.tz_localize("Asia/shanghai")
        solar_thermal = solar_thermal.loc[date_range].set_index(network.snapshots)

        for cat in [" central "]:
            network.add(
                "Generator",
                nodes,
                suffix=cat + "solar thermal",
                bus=nodes + cat + "heat",
                carrier="solar thermal",
                p_nom_extendable=False,
                capital_cost=costs.at[cat.lstrip() + "solar thermal", "capital_cost"],
                p_max_pu=solar_thermal[nodes].clip(1.0e-4),
                lifetime=costs.at[cat.lstrip() + "solar thermal", "lifetime"],
            )

    if "coal boiler" in config["Techs"]["conv_techs"]:
        for cat in [" decentral "]:
            network.add(
                "Link",
                nodes + cat + "coal boiler",
                p_nom_extendable=True,
                bus0=nodes + " coal",
                bus1=nodes + cat + "heat",
                carrier="coal boiler",
                efficiency=costs.at[cat.lstrip() + "coal boiler", "efficiency"],
                marginal_cost=costs.at[cat.lstrip() + "coal boiler", "VOM"],
                capital_cost=costs.at[cat.lstrip() + "coal boiler", "efficiency"]
                * costs.at[cat.lstrip() + "coal boiler", "capital_cost"],
                lifetime=costs.at[cat.lstrip() + "coal boiler", "lifetime"],
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

            network.add(
                "Store",
                nodes + cat + "water tank",
                bus=nodes + cat + "water tanks",
                carrier="water tanks",
                e_cyclic=True,
                e_nom_extendable=True,
                standing_loss=1
                - np.exp(
                    -1 / (24.0 * (config["tes_tau"] if cat == " decentral " else 180.0))
                ),  # [HP] 180 day time constant for centralised
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
        "Split bidirectional links into two unidirectional links to include transmission losses."

        edges_existing = pd.read_csv(snakemake.input.edges_existing, header=None)

        lengths = NON_LIN_PATH_SCALING * np.array(
            [
                haversine(
                    [network.buses.at[name0, "x"], network.buses.at[name0, "y"]],
                    [network.buses.at[name1, "x"], network.buses.at[name1, "y"]],
                )
                for name0, name1 in edges_existing[[0, 1]].values
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
            edges_existing[0] + "-" + edges_existing[1],
            bus0=edges_existing[0].values,
            bus1=edges_existing[1].values,
            suffix=" ext positive",
            p_nom_extendable=False,
            p_nom=edges_existing[2].values,
            p_nom_min=edges_existing[2].values,
            p_min_pu=0,
            efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]
            * config["transmission_efficiency"]["DC"]["efficiency_per_1000km"] ** (lengths / 1000),
            length=lengths,
            build_year=2020,
            lifetime=70,
            capital_cost=cc,
        )

        network.add(
            "Link",
            edges_existing[1] + "-" + edges_existing[0],
            bus0=edges_existing[1].values,
            bus1=edges_existing[0].values,
            suffix=" ext reversed",
            p_nom_extendable=False,
            p_nom=edges_existing[2].values,
            p_nom_min=edges_existing[2].values,
            p_min_pu=0,
            efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]
            * config["transmission_efficiency"]["DC"]["efficiency_per_1000km"] ** (lengths / 1000),
            length=lengths,
            lifetime=70,
            build_year=2020,
            capital_cost=0,
        )

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

    return network


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "prepare_base_networks_2020",
            opts="ll",
            topology="current+Neighbor",
            pathway="exponential175",
            planning_horizons="2020",
            heating_demand="positive",
        )
    configure_logging(snakemake, level="DEBUG")

    network = prepare_network(snakemake.config)

    network.export_to_netcdf(snakemake.output.network_name)

    logger.info(f"Network prepared and saved to {snakemake.output.network_name}")
