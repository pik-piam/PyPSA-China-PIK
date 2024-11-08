# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# for non-pathway network

from vresutils.costdata import annuity
from _helpers import configure_logging
import pypsa
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
import pytz
from math import radians, cos, sin, asin, sqrt
from functools import partial
import pyproj
from shapely.ops import transform
import xarray as xr
from functions import pro_names, HVAC_cost_curve
from add_electricity import load_costs


# This function follows http://toblerity.org/shapely/manual.html
def area_from_lon_lat_poly(geometry):
    """For shapely geometry in lon-lat coordinates,
    returns area in km^2."""

    project = partial(
        pyproj.transform, pyproj.Proj(init="epsg:4326"), pyproj.Proj(proj="aea")  # Source: Lon-Lat
    )  # Target: Albers Equal Area Conical https://en.wikipedia.org/wiki/Albers_projection

    new_geometry = transform(project, geometry)

    # default area is in m^2
    return new_geometry.area / 1e6


def haversine(p1, p2):
    """Calculate the great circle distance in km between two points on
    the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def generate_periodic_profiles(
    dt_index=None,
    col_tzs=pd.Series(index=pro_names, data=len(pro_names) * ["Shanghai"]),
    weekly_profile=range(24 * 7),
):
    """Give a 24*7 long list of weekly hourly profiles, generate this
    for each country for the period dt_index, taking account of time
    zones and Summer Time."""

    weekly_profile = pd.Series(weekly_profile, range(24 * 7))

    week_df = pd.DataFrame(index=dt_index, columns=col_tzs.index)
    for ct in col_tzs.index:
        week_df[ct] = [
            24 * dt.weekday() + dt.hour
            for dt in dt_index.tz_convert(pytz.timezone("Asia/{}".format(col_tzs[ct])))
        ]
        week_df[ct] = week_df[ct].map(weekly_profile)
    return week_df


def shift_df(df, hours=1):
    """Works both on Series and DataFrame"""
    df = df.copy()
    df.values[:] = np.concatenate([df.values[-hours:], df.values[:-hours]])
    return df


def transport_degree_factor(
    temperature,
    deadband_lower=15,
    deadband_upper=20,
    lower_degree_factor=0.5,
    upper_degree_factor=1.6,
):
    """Work out how much energy demand in vehicles increases due to heating and cooling.
    There is a deadband where there is no increase.
    Degree factors are % increase in demand compared to no heating/cooling fuel consumption.
    Returns per unit increase in demand for each place and time"""

    dd = temperature.copy()

    dd[(temperature > deadband_lower) & (temperature < deadband_upper)] = 0.0

    dd[temperature < deadband_lower] = (
        lower_degree_factor / 100.0 * (deadband_lower - temperature[temperature < deadband_lower])
    )

    dd[temperature > deadband_upper] = (
        upper_degree_factor / 100.0 * (temperature[temperature > deadband_upper] - deadband_upper)
    )

    return dd


def prepare_data(network, date_range, planning_horizons):

    ##############
    # Heating
    ##############

    # copy forward the daily average heat demand into each hour, so it can be multipled by the intraday profile

    with pd.HDFStore(snakemake.input.heat_demand_name, mode="r") as store:
        # the ffill converts daily values into hourly values
        h = store["heat_demand_profiles"]
        h_n = h[~h.index.duplicated(keep="first")].iloc[:-1, :]
        heat_demand_hdh = h_n.reindex(index=network.snapshots, method="ffill")

    with pd.HDFStore(snakemake.input.cop_name, mode="r") as store:
        ashp_cop = store["ashp_cop_profiles"].loc[date_range].set_index(network.snapshots)
        gshp_cop = store["gshp_cop_profiles"].loc[date_range].set_index(network.snapshots)

    with pd.HDFStore(snakemake.input.energy_totals_name, mode="r") as store:
        space_heating_per_hdd = store["space_heating_per_hdd"]
        hot_water_per_day = store["hot_water_per_day"]

    intraday_profiles = pd.read_csv("data/heating/heat_load_profile_DK_AdamJensen.csv", index_col=0)
    intraday_year_profiles = generate_periodic_profiles(
        dt_index=heat_demand_hdh.index.tz_localize("UTC"),
        weekly_profile=(
            list(intraday_profiles["weekday"]) * 5 + list(intraday_profiles["weekend"]) * 2
        ),
    ).tz_localize(None)

    space_heat_demand = intraday_year_profiles.mul(heat_demand_hdh).mul(space_heating_per_hdd)
    water_heat_demand = intraday_year_profiles.mul(hot_water_per_day / 24.0)

    heat_demand_calculate = (
        space_heat_demand + water_heat_demand
    )  # only consider heat demand at first

    heat_demand = (
        heat_demand_calculate
        / (heat_demand_calculate.sum().sum() * snakemake.config["frequency"])
        * snakemake.config["heating_demand"][planning_horizons]
        * 1e9
    )

    ###############
    # CO2
    ###############

    # tCO2
    represented_hours = network.snapshot_weightings.sum()[0]
    Nyears = represented_hours / 8760.0
    with pd.HDFStore(snakemake.input.co2_totals_name, mode="r") as store:
        co2_totals = Nyears * store["co2"]

    return heat_demand, space_heat_demand, water_heat_demand, ashp_cop, gshp_cop, co2_totals


def prepare_network(config):
    # add CHP definition
    override_component_attrs = pypsa.descriptors.Dict(
        {k: v.copy() for k, v in pypsa.components.component_attrs.items()}
    )
    override_component_attrs["Link"].loc["bus2"] = [
        "string",
        np.nan,
        np.nan,
        "2nd bus",
        "Input (optional)",
    ]
    override_component_attrs["Link"].loc["efficiency2"] = [
        "static or series",
        "per unit",
        1.0,
        "2nd bus efficiency",
        "Input (optional)",
    ]
    override_component_attrs["Link"].loc["p2"] = [
        "series",
        "MW",
        0.0,
        "2nd bus output",
        "Output",
    ]

    # Build the Network object, which stores all other objects
    network = pypsa.Network(override_component_attrs=override_component_attrs)

    # load graph
    nodes = pd.Index(pro_names)
    edges = pd.read_csv("data/edges.txt", sep=",", header=None)
    edges_current = pd.read_csv("data/edges_current.csv", header=None)
    edges_current_FCG = pd.read_csv("data/edges_current_FCG.csv", header=None)

    # set times
    planning_horizons = snakemake.wildcards["planning_horizons"]
    if int(planning_horizons) % 4 != 0:
        snapshots = pd.date_range(
            str(planning_horizons) + "-01-01 00:00",
            str(planning_horizons) + "-12-31 23:00",
            freq=config["freq"],
        )
    else:
        snapshots = pd.date_range(
            str(planning_horizons) + "-01-01 00:00",
            str(planning_horizons) + "-02-28 23:00",
            freq=config["freq"],
        ).append(
            pd.date_range(
                str(planning_horizons) + "-03-01 00:00",
                str(planning_horizons) + "-12-31 23:00",
                freq=config["freq"],
            )
        )

    network.set_snapshots(snapshots)

    network.snapshot_weightings[:] = config["frequency"]
    represented_hours = network.snapshot_weightings.sum()[0]
    Nyears = represented_hours / 8760.0

    tech_costs = snakemake.input.tech_costs
    cost_year = snakemake.wildcards.planning_horizons
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, Nyears)

    date_range = pd.date_range("2020-01-01 00:00", "2020-02-28 23:00", freq=config["freq"]).append(
        pd.date_range("2020-03-01 00:00", "2020-12-31 23:00", freq=config["freq"])
    )

    heat_demand, space_heat_demand, water_heat_demand, ashp_cop, gshp_cop, co2_totals = (
        prepare_data(network, date_range, planning_horizons)
    )

    ds_solar = xr.open_dataset(snakemake.input.profile_solar)
    ds_onwind = xr.open_dataset(snakemake.input.profile_onwind)
    ds_offwind = xr.open_dataset(snakemake.input.profile_offwind)

    solar_p_max_pu = (
        ds_solar["profile"]
        .transpose("time", "bus")
        .to_pandas()
        .loc[date_range]
        .set_index(network.snapshots)
    )
    onwind_p_max_pu = (
        ds_onwind["profile"]
        .transpose("time", "bus")
        .to_pandas()
        .loc[date_range]
        .set_index(network.snapshots)
    )
    offwind_p_max_pu = (
        ds_offwind["profile"]
        .transpose("time", "bus")
        .to_pandas()
        .loc[date_range]
        .set_index(network.snapshots)
    )

    pro_shapes = gpd.GeoDataFrame.from_file("data/province_shapes/CHN_adm1.shp")
    pro_shapes = pro_shapes.to_crs(4326)
    pro_shapes.index = pro_names

    # add buses
    network.madd(
        "Bus",
        nodes,
        x=pro_shapes["geometry"].centroid.x,
        y=pro_shapes["geometry"].centroid.y,
    )

    # add carriers
    network.add("Carrier", "onwind")
    network.add("Carrier", "offwind")
    network.add("Carrier", "solar")
    if config["add_solar_thermal"]:
        network.add("Carrier", "solar thermal")
    if config["add_PHS"]:
        network.add("Carrier", "PHS")
    if config["add_hydro"]:
        network.add("Carrier", "hydro")
    if config["add_H2_storage"]:
        network.add("Carrier", "H2")
    if config["add_battery_storage"]:
        network.add("Carrier", "battery")
    if config["heat_coupling"]:
        network.add("Carrier", "heat")
        network.add("Carrier", "water tanks")

    if not isinstance(config["scenario"]["co2_reduction"], tuple):

        if config["scenario"]["co2_reduction"] is not None:

            co2_limit = (
                5.59 * 1e9 * (1 - config["scenario"]["co2_reduction"])
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
    with pd.HDFStore(
        f"data/load/load_{planning_horizons}_weatheryears_1979_2016_TWh.h5", mode="r"
    ) as store:
        load = 1e6 * store["load"].loc[network.snapshots]

    load.columns = pro_names

    network.madd("Load", nodes, bus=nodes, p_set=load[nodes])

    # add renewables
    network.madd(
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
    network.madd(
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

    network.madd(
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

    # add conventionals
    if config["add_gas"]:
        # add converter from fuel source
        network.add(
            "Carrier", "gas", co2_emissions=costs.at["gas", "co2_emissions"]
        )  # in t_CO2/MWht
        network.madd(
            "Bus",
            nodes,
            suffix=" gas",
            x=pro_shapes["geometry"].centroid.x,
            y=pro_shapes["geometry"].centroid.y,
            carrier="gas",
        )

        network.madd(
            "Generator",
            nodes,
            suffix=" gas fuel",
            bus=nodes,
            carrier="gas",
            p_nom_extendable=True,
            marginal_cost=costs.at["OCGT", "fuel"],
        )

        network.madd(
            "Store", nodes + " gas Store", bus=nodes + " gas", e_nom_extendable=True, carrier="gas"
        )

        network.madd(
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
        )

        if config["add_chp"]:
            network.madd(
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

    if config["add_coal"]:
        network.add("Carrier", "coal", co2_emissions=costs.at["coal", "co2_emissions"])
        network.madd(
            "Generator",
            nodes,
            suffix=" coal",
            bus=nodes,
            carrier="coal",
            p_nom_extendable=True,
            efficiency=costs.at["coal", "efficiency"],
            marginal_cost=costs.at["coal", "efficiency"] * costs.at["coal", "marginal_cost"],
            capital_cost=costs.at["coal", "efficiency"]
            * costs.at["coal", "capital_cost"],  # NB: capital cost is per MWel
            lifetime=costs.at["coal", "lifetime"],
        )

        if config["add_chp"]:
            network.madd(
                "Bus",
                nodes,
                suffix=" CHP coal",
                x=pro_shapes["geometry"].centroid.x,
                y=pro_shapes["geometry"].centroid.y,
                carrier="coal",
            )

            network.madd(
                "Generator",
                nodes + " CHP coal fuel",
                bus=nodes + " CHP coal",
                carrier="coal",
                p_nom_extendable=True,
                marginal_cost=costs.at["coal", "fuel"],
            )

            network.madd(
                "Link",
                nodes,
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

    if config["add_nuclear"]:
        network.add("Carrier", "uranium")
        nuclear_p_nom = pd.read_hdf("data/p_nom/nuclear_p_nom.h5")
        network.madd(
            "Generator",
            nodes,
            suffix=" nuclear",
            p_nom_extendable=True,
            p_nom=nuclear_p_nom,
            p_nom_max=nuclear_p_nom * 2,
            p_nom_min=nuclear_p_nom,
            bus=nodes,
            carrier="uranium",
            efficiency=costs.at["nuclear", "efficiency"],
            capital_cost=costs.at["nuclear", "efficiency"]
            * costs.at["nuclear", "capital_cost"],  # NB: capital cost is per MWel
            marginal_cost=costs.at["nuclear", "efficiency"] * costs.at["nuclear", "marginal_cost"],
        )

    if config["add_PHS"]:
        # pure pumped hydro storage, fixed, 6h energy by default, no inflow
        hydrocapa_df = pd.read_csv("data/hydro/PHS_p_nom.csv", index_col=0)
        phss = hydrocapa_df.index[hydrocapa_df["MW"] > 0].intersection(nodes)
        if config["hydro"]["hydro_capital_cost"]:
            cc = costs.at["PHS", "capital_cost"]
        else:
            cc = 0.0

        network.madd(
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

        #######
        df = pd.read_csv("data/hydro/dams_large.csv", index_col=0)
        points = df.apply(lambda row: Point(row.Lon, row.Lat), axis=1)
        dams = gpd.GeoDataFrame(df, geometry=points)
        dams.crs = {"init": "epsg:4326"}

        hourly_rng = pd.date_range("1979-01-01", "2017-01-01", freq="1H", closed="left")
        inflow = pd.read_pickle(
            "data/hydro/daily_hydro_inflow_per_dam_1979_2016_m3.pickle"
        ).reindex(hourly_rng, fill_value=0)
        inflow.columns = dams.index

        water_consumption_factor = (
            dams.loc[:, "Water_consumption_factor_avg"] * 1e3
        )  # m^3/KWh -> m^3/MWh

        #######
        # ### Add hydro stations as buses
        network.madd(
            "Bus",
            dams.index,
            suffix=" station",
            carrier="stations",
            x=dams["geometry"].centroid.x,
            y=dams["geometry"].centroid.y,
        )

        dam_buses = network.buses[network.buses.carrier == "stations"]

        # ### add hydro reservoirs as stores

        initial_capacity = pd.read_pickle("data/hydro/reservoir_initial_capacity.pickle")
        effective_capacity = pd.read_pickle("data/hydro/reservoir_effective_capacity.pickle")
        initial_capacity.index = dams.index
        effective_capacity.index = dams.index
        initial_capacity = initial_capacity / water_consumption_factor
        effective_capacity = effective_capacity / water_consumption_factor

        network.madd(
            "Store",
            dams.index,
            suffix=" reservoir",
            bus=dam_buses.index,
            e_nom=effective_capacity,
            e_initial=initial_capacity,
            e_cyclic=True,
            marginal_cost=config["costs"]["marginal_cost"]["hydro"],
        )

        ### add hydro turbines to link stations to provinces
        network.madd(
            "Link",
            dams.index,
            suffix=" turbines",
            bus0=dam_buses.index,
            bus1=dams["Province"],
            p_nom=10 * dams["installed_capacity_10MW"],
            efficiency=1,
        )
        # p_nom * efficiency = 10 * dams['installed_capacity_10MW']

        ### add rivers to link station to station
        bus0s = [0, 21, 11, 19, 22, 29, 8, 40, 25, 1, 7, 4, 10, 15, 12, 20, 26, 6, 3, 39]
        bus1s = [5, 11, 19, 22, 32, 8, 40, 25, 35, 2, 4, 10, 9, 12, 20, 23, 6, 17, 14, 16]

        for bus0, bus2 in list(zip(dams.index[bus0s], dam_buses.iloc[bus1s].index)):

            # normal flow
            network.links.at[bus0 + " turbines", "bus2"] = bus2
            network.links.at[bus0 + " turbines", "efficiency2"] = 1.0

        ### spillage
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

        # for bus0 in dam_buses.iloc[dam_ends].index:
        #     network.add('Link',
        #                 bus0 + ' spillage',
        #                 bus0=bus0,
        #                 bus1='Tibet',
        #                 p_nom_extendable=True,
        #                 efficiency=0.0)

        #### add inflow as generators
        # only feed into hydro stations which are the first of a cascade
        inflow_stations = [dam for dam in range(len(dams.index)) if not dam in bus1s]

        for inflow_station in inflow_stations:

            # p_nom = 1 and p_max_pu & p_min_pu = p_pu, compulsory inflow
            date_range = pd.date_range(
                "2016-01-01 00:00", "2016-02-28 23:00", freq=config["freq"]
            ).append(pd.date_range("2016-03-01 00:00", "2016-12-31 23:00", freq=config["freq"]))

            p_nom = (
                (inflow.loc[date_range] / water_consumption_factor).iloc[:, inflow_station].max()
            )
            p_pu = (inflow.loc[date_range] / water_consumption_factor).iloc[
                :, inflow_station
            ] / p_nom
            p_pu.index = network.snapshots
            network.add(
                "Generator",
                dams.index[inflow_station] + " inflow",
                bus=dam_buses.iloc[inflow_station].name,
                carrier="hydro_inflow",
                p_max_pu=p_pu.clip(1.0e-6),
                # p_min_pu=p_pu.clip(1.e-6),
                p_nom=p_nom,
            )

            # p_nom*p_pu = XXX m^3 then use turbines efficiency to convert to power

        # ### add fake hydro just to introduce capital cost
        if config["add_hydro"] and config["hydro"]["hydro_capital_cost"]:
            hydro_cc = costs.at["hydro", "capital_cost"]

            network.madd(
                "StorageUnit",
                dams.index,
                suffix=" hydro dummy",
                bus=dams["Province"],
                carrier="hydro",
                p_nom=10 * dams["installed_capacity_10MW"],
                p_max_pu=0.0,
                p_min_pu=0.0,
                capital_cost=hydro_cc,
            )

        # else: hydro_cc=0.

    if config["add_H2_storage"]:

        network.madd(
            "Bus",
            nodes,
            suffix=" H2",
            x=pro_shapes["geometry"].centroid.x,
            y=pro_shapes["geometry"].centroid.y,
            carrier="H2",
        )

        network.madd(
            "Link",
            nodes + " H2 Electrolysis",
            bus0=nodes,
            bus1=nodes + " H2",
            p_nom_extendable=True,
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "efficiency"]
            * costs.at["electrolysis", "capital_cost"],
            lifetime=costs.at["electrolysis", "lifetime"],
        )

        network.madd(
            "Link",
            nodes + " H2 Fuel Cell",
            bus0=nodes + " H2",
            bus1=nodes,
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            capital_cost=costs.at["fuel cell", "efficiency"]
            * costs.at["fuel cell", "capital_cost"],
            lifetime=costs.at["fuel cell", "lifetime"],
        )

        network.madd(
            "Store",
            nodes + " H2 Store",
            bus=nodes + " H2",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at["hydrogen storage tank", "capital_cost"],
            lifetime=costs.at["hydrogen storage tank", "lifetime"],
        )

    if config["add_methanation"]:
        network.madd(
            "Link",
            nodes + " Sabatier",
            bus0=nodes + " H2",
            bus1=nodes + " gas",
            p_nom_extendable=True,
            efficiency=costs.at["methanation", "efficiency"],
            capital_cost=costs.at["methanation", "efficiency"]
            * costs.at["methanation", "capital_cost"],
            lifetime=costs.at["methanation", "lifetime"],
        )

    if config["add_battery_storage"]:

        network.madd(
            "Bus",
            nodes,
            suffix=" battery",
            x=pro_shapes["geometry"].centroid.x,
            y=pro_shapes["geometry"].centroid.y,
            carrier="battery",
        )

        network.madd(
            "Store",
            nodes + " battery",
            bus=nodes + " battery",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=costs.at["battery storage", "capital_cost"],
            lifetime=costs.at["battery storage", "lifetime"],
        )

        network.madd(
            "Link",
            nodes + " battery charger",
            bus0=nodes,
            bus1=nodes + " battery",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["battery inverter", "efficiency"]
            * costs.at["battery inverter", "capital_cost"],
            p_nom_extendable=True,
            lifetime=costs.at["battery inverter", "lifetime"],
        )

        network.madd(
            "Link",
            nodes + " battery discharger",
            bus0=nodes + " battery",
            bus1=nodes,
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            marginal_cost=0.0,
            p_nom_extendable=True,
        )

    # Sources:
    # [HP]: Henning, Palzer http://www.sciencedirect.com/science/article/pii/S1364032113006710
    # [B]: Budischak et al. http://www.sciencedirect.com/science/article/pii/S0378775312014759

    if config["heat_coupling"]:

        # urban = nodes

        # NB: must add costs of central heating afterwards (EUR 400 / kWpeak, 50a, 1% FOM from Fraunhofer ISE)

        # central are urban nodes with district heating
        # central = nodes ^ urban

        central_fraction = pd.read_hdf("data/heating/DH1.h5")

        network.madd(
            "Bus",
            nodes,
            suffix=" decentral heat",
            x=pro_shapes["geometry"].centroid.x,
            y=pro_shapes["geometry"].centroid.y,
            carrier="heat",
        )

        network.madd(
            "Bus",
            nodes,
            suffix=" central heat",
            x=pro_shapes["geometry"].centroid.x,
            y=pro_shapes["geometry"].centroid.y,
            carrier="heat",
        )

        network.madd(
            "Load",
            nodes,
            suffix=" decentral heat",
            bus=nodes + " decentral heat",
            p_set=heat_demand[nodes].multiply(1 - central_fraction),
        )

        network.madd(
            "Load",
            nodes,
            suffix=" central heat",
            bus=nodes + " central heat",
            p_set=heat_demand[nodes].multiply(central_fraction),
        )

        if config["add_heat_pumps"]:

            for cat in [" decentral ", " central "]:
                network.madd(
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
                    capital_cost=costs.at[cat.lstrip() + "air-sourced heat pump", "capital_cost"],
                    p_nom_extendable=True,
                    lifetime=costs.at[cat.lstrip() + "air-sourced heat pump", "lifetime"],
                )

            network.madd(
                "Link",
                nodes,
                suffix=" ground heat pump",
                bus0=nodes,
                bus1=nodes + " decentral heat",
                carrier="heat pump",
                efficiency=(
                    gshp_cop[nodes]
                    if config["time_dep_hp_cop"]
                    else costs.at["decentral ground-sourced heat pump", "efficiency"]
                ),
                capital_cost=costs.at["decentral ground-sourced heat pump", "capital_cost"],
                p_nom_extendable=True,
                lifetime=costs.at["decentral ground-sourced heat pump", "lifetime"],
            )

        if config["add_thermal_storage"]:

            for cat in [" decentral ", " central "]:
                network.madd(
                    "Bus",
                    nodes,
                    suffix=cat + "water tanks",
                    x=pro_shapes["geometry"].centroid.x,
                    y=pro_shapes["geometry"].centroid.y,
                    carrier="water tanks",
                )

                network.madd(
                    "Link",
                    nodes + cat + "water tanks charger",
                    bus0=nodes + cat + "heat",
                    bus1=nodes + cat + "water tanks",
                    carrier="water tanks",
                    efficiency=costs.at["water tank charger", "efficiency"],
                    p_nom_extendable=True,
                )

                network.madd(
                    "Link",
                    nodes + cat + "water tanks discharger",
                    bus0=nodes + cat + "water tanks",
                    bus1=nodes + cat + "heat",
                    carrier="water tanks",
                    efficiency=costs.at["water tank discharger", "efficiency"],
                    p_nom_extendable=True,
                )

                network.madd(
                    "Store",
                    nodes + cat + "water tank",
                    bus=nodes + cat + "water tanks",
                    carrier="water tanks",
                    e_cyclic=True,
                    e_nom_extendable=True,
                    standing_loss=1
                    - np.exp(
                        -1 / (24.0 * (config["tes_tau"] if cat == " decentral " else 180.0))
                    ),  # [HP] 180 day time constant for centralised, 3 day for decentralised
                    capital_cost=costs.at[cat.lstrip() + "water tank storage", "capital_cost"]
                    / (1.17e-3 * 40),
                    lifetime=costs.at[cat.lstrip() + "water tank storage", "lifetime"],
                )  # conversion from EUR/m^3 to EUR/MWh for 40 K diff and 1.17 kWh/m^3/K

        if config["add_boilers"]:
            if config["add_resistive_heater"]:
                network.add("Carrier", "resistive heater")

                for cat in [" decentral ", " central "]:
                    network.madd(
                        "Link",
                        nodes + cat + "resistive heater",
                        bus0=nodes,
                        bus1=nodes + cat + "heat",
                        carrier="resistive heater",
                        efficiency=costs.at[cat.lstrip() + "resistive heater", "efficiency"],
                        capital_cost=costs.at[cat.lstrip() + "resistive heater", "efficiency"]
                        * costs.at[cat.lstrip() + "resistive heater", "capital_cost"],
                        p_nom_extendable=True,
                        lifetime=costs.at[cat.lstrip() + "resistive heater", "lifetime"],
                    )

            if config["add_gas"]:
                for cat in [" decentral ", " central "]:
                    network.madd(
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

        if config["add_solar_thermal"]:

            # this is the amount of heat collected in W per m^2, accounting
            # for efficiency
            with pd.HDFStore(snakemake.input.solar_thermal_name, mode="r") as store:
                # 1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
                solar_thermal = (
                    config["solar_cf_correction"] * store["solar_thermal_profiles"] / 1e3
                )

            date_range = pd.date_range(
                "2020-01-01 00:00", "2020-02-28 23:00", freq=config["freq"]
            ).append(pd.date_range("2020-03-01 00:00", "2020-12-31 23:00", freq=config["freq"]))

            solar_thermal = solar_thermal.loc[date_range].set_index(network.snapshots)

            for cat in [" decentral "]:
                network.madd(
                    "Generator",
                    nodes,
                    suffix=cat + "solar thermal collector",
                    bus=nodes + cat + "heat",
                    carrier="solar thermal",
                    p_nom_extendable=True,
                    capital_cost=costs.at[cat.lstrip() + "solar thermal", "capital_cost"],
                    p_max_pu=solar_thermal[nodes].clip(1.0e-4),
                    lifetime=costs.at[cat.lstrip() + "solar thermal", "lifetime"],
                )

    # add lines

    if not config["no_lines"]:

        if config["scenario"]["topology"] == "FCG":
            lengths = 1.25 * np.array(
                [
                    haversine(
                        [network.buses.at[name0, "x"], network.buses.at[name0, "y"]],
                        [network.buses.at[name1, "x"], network.buses.at[name1, "y"]],
                    )
                    for name0, name1 in edges.values
                ]
            )

            # if config['line_volume_limit_max'] is not None:
            #     cc = Nyears * 0.01  # Set line costs to ~zero because we already restrict the line volume
            # else:
            cc = (
                (config["line_cost_factor"] * lengths * [HVAC_cost_curve(l) for l in lengths])
                * 1.5
                * 1.02
                * Nyears
                * annuity(40.0, config["costs"]["discountrate"])
            )

            network.madd(
                "Link",
                edges[0] + "-" + edges[1],
                bus0=edges[0].values,
                bus1=edges[1].values,
                p_nom_extendable=True,
                p_min_pu=-1,
                length=lengths,
                capital_cost=cc,
            )

        if config["scenario"]["topology"] == "current":
            lengths = 1.25 * np.array(
                [
                    haversine(
                        [network.buses.at[name0, "x"], network.buses.at[name0, "y"]],
                        [network.buses.at[name1, "x"], network.buses.at[name1, "y"]],
                    )
                    for name0, name1 in edges_current[[0, 1]].values
                ]
            )

            # if config['line_volume_limit_max'] is not None:
            #     cc = Nyears * 0.01  # Set line costs to ~zero because we already restrict the line volume
            # else:
            cc = (
                (config["line_cost_factor"] * lengths * [HVAC_cost_curve(l) for l in lengths])
                * 1.5
                * 1.02
                * Nyears
                * annuity(40.0, config["costs"]["discountrate"])
            )

            network.madd(
                "Link",
                edges_current[0] + "-" + edges_current[1],
                bus0=edges_current[0].values,
                bus1=edges_current[1].values,
                p_nom_extendable=True,
                p_min_pu=-1,
                p_nom=edges_current[2].values,
                p_nom_min=edges_current[2].values,
                length=lengths,
                capital_cost=cc,
            )

        if config["scenario"]["topology"] == "current+FCG":
            lengths = 1.25 * np.array(
                [
                    haversine(
                        [network.buses.at[name0, "x"], network.buses.at[name0, "y"]],
                        [network.buses.at[name1, "x"], network.buses.at[name1, "y"]],
                    )
                    for name0, name1 in edges_current_FCG[[0, 1]].values
                ]
            )

            # if config['line_volume_limit_max'] is not None:
            #     cc = Nyears * 0.01  # Set line costs to ~zero because we already restrict the line volume
            # else:
            cc = (
                (config["line_cost_factor"] * lengths * [HVAC_cost_curve(l) for l in lengths])
                * 1.5
                * 1.02
                * Nyears
                * annuity(40.0, config["costs"]["discountrate"])
            )

            network.madd(
                "Link",
                edges_current_FCG[0] + "-" + edges_current_FCG[1],
                bus0=edges_current_FCG[0].values,
                bus1=edges_current_FCG[1].values,
                p_nom_extendable=True,
                p_min_pu=-1,
                p_nom=edges_current_FCG[2].values,
                p_nom_min=edges_current_FCG[2].values,
                length=lengths,
                capital_cost=cc,
            )

    return network


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_networks",
            opts="ll",
            topology="current+FCG",
            pathway="non-pathway",
            co2_reduction="0.0",
            planning_horizons=2060,
        )
    configure_logging(snakemake)

    population = pd.read_hdf(snakemake.input.population_name)

    network = prepare_network(snakemake.config)

    network.export_to_netcdf(snakemake.output.network_name)
