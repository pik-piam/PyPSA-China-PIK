# SPDX-FileCopyrightText: : 2024 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# for pathway network

from vresutils.costdata import annuity
from _helpers import configure_logging,override_component_attrs
import pypsa
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import xarray as xr
from functions import pro_names, HVAC_cost_curve
from add_electricity import load_costs

def haversine(p1,p2):
    """Calculate the great circle distance in km between two points on
    the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def add_buses(network,nodes,suffix,carrier,pro_centroid_x,pro_centroid_y):

    network.madd('Bus',
                 nodes,
                 suffix=suffix,
                 x=pro_centroid_x,
                 y=pro_centroid_y,
                 carrier=carrier,
                 )


def prepare_network(config):

    if "overrides" in snakemake.input.keys():
        overrides = override_component_attrs(snakemake.input.overrides)
        network = pypsa.Network(override_component_attrs=overrides)
    else:
        network = pypsa.Network()

    # set times
    planning_horizons = snakemake.wildcards['planning_horizons']
    if int(planning_horizons) % 4 != 0:
        snapshots = pd.date_range(str(planning_horizons)+'-01-01 00:00', str(planning_horizons)+'-12-31 23:00', freq=config['freq'],tz='Asia/shanghai')
    else:
        snapshots = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'],tz='Asia/shanghai')
        snapshots = snapshots.map(lambda t: t.replace(year=int(planning_horizons)))

    network.set_snapshots(snapshots)
    network.snapshot_weightings[:] = config['frequency']
    represented_hours = network.snapshot_weightings.sum()[0]
    Nyears= represented_hours/8760.

    #load graph
    nodes = pd.Index(pro_names)
    pathway = snakemake.wildcards['pathway']

    tech_costs = snakemake.input.tech_costs
    cost_year = snakemake.wildcards.planning_horizons
    costs = load_costs(tech_costs,config['costs'],config['electricity'],cost_year, Nyears)

    date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'],tz='Asia/shanghai')
    date_range = date_range.map(lambda t: t.replace(year=2020))

    ds_solar = xr.open_dataset(snakemake.input.profile_solar)
    ds_onwind = xr.open_dataset(snakemake.input.profile_onwind)
    ds_offwind = xr.open_dataset(snakemake.input.profile_offwind)

    solar_p_max_pu = ds_solar['profile'].transpose('time', 'bus').to_pandas()
    solar_p_max_pu.index = solar_p_max_pu.index.tz_localize('Asia/shanghai')
    solar_p_max_pu = solar_p_max_pu.loc[date_range].set_index(network.snapshots)
    onwind_p_max_pu = ds_onwind['profile'].transpose('time', 'bus').to_pandas()
    onwind_p_max_pu.index = onwind_p_max_pu.index.tz_localize('Asia/shanghai')
    onwind_p_max_pu = onwind_p_max_pu.loc[date_range].set_index(network.snapshots)
    offwind_p_max_pu = ds_offwind['profile'].transpose('time', 'bus').to_pandas()
    offwind_p_max_pu.index = offwind_p_max_pu.index.tz_localize('Asia/shanghai')
    offwind_p_max_pu = offwind_p_max_pu.loc[date_range].set_index(network.snapshots)

    def rename_province(label):
        rename = {
            "Nei Mongol": "InnerMongolia",
            "Ningxia Hui": "Ningxia",
            "Xinjiang Uygur": "Xinjiang",
            "Xizang": "Tibet"
        }

        for old, new in rename.items():
            if old == label:
                label = new
        return label

    pro_shapes = gpd.GeoDataFrame.from_file(snakemake.input.province_shape)
    pro_shapes = pro_shapes.to_crs(4326)
    pro_shapes.index = pro_shapes.NAME_1.map(rename_province)
    pro_centroid_x = pro_shapes.to_crs('+proj=cea').centroid.to_crs(pro_shapes.crs).x
    pro_centroid_y = pro_shapes.to_crs('+proj=cea').centroid.to_crs(pro_shapes.crs).y

    # add buses
    for suffix in config["bus_suffix"]:
        carrier = config["bus_carrier"][suffix]
        add_buses(network, nodes, suffix, carrier, pro_centroid_x, pro_centroid_y)

    # add carriers
    if config["heat_coupling"]:
        network.add("Carrier", "heat")
    for carrier in config["Techs"]["vre_techs"]:
        network.add("Carrier", carrier)
    for carrier in config["Techs"]["store_techs"]:
        if carrier == 'battery':
            network.add("Carrier", "battery")
            network.add("Carrier", "battery discharger")
        else:
            network.add("Carrier", carrier)
    for carrier in config["Techs"]["conv_techs"]:
        if "gas" in carrier:
            network.add("Carrier", carrier, co2_emissions=costs.at['gas', 'co2_emissions'])  # in t_CO2/MWht
        if "coal" in carrier:
            network.add("Carrier", carrier, co2_emissions=costs.at['coal', 'co2_emissions'])
    if config["add_gas"]:
        network.add("Carrier", "gas", co2_emissions=costs.at['gas', 'co2_emissions'])  # in t_CO2/MWht
    if config["add_coal"]:
        network.add("Carrier", "coal", co2_emissions=costs.at['coal', 'co2_emissions'])

    # add global constraint
    if not isinstance(config['scenario']['co2_reduction'], tuple):

        if config['scenario']['co2_reduction'] is not None:

            co2_limit = (5.288987673 + 0.628275682)*1e9  * (1 - config['scenario']['co2_reduction'][pathway][planning_horizons]) # Chinese 2020 CO2 emissions of electric and heating sector

            network.add("GlobalConstraint",
                        "co2_limit",
                        type="primary_energy",
                        carrier_attribute="co2_emissions",
                        sense="<=",
                        constant=co2_limit)

    #load demand data
    with pd.HDFStore(snakemake.input.elec_load, mode='r') as store:
        load = 1e6 * store['load']
        load.index = load.index.tz_localize('Asia/shanghai')
        load = load.loc[network.snapshots]

    load.columns = pro_names

    network.madd("Load", nodes, bus=nodes, p_set=load[nodes])

    if config["heat_coupling"]:

        central_fraction = pd.read_hdf(snakemake.input.central_fraction)
        with pd.HDFStore(snakemake.input.heat_demand_profile, mode='r') as store:
            heat_demand = store['heat_demand_profiles']
            heat_demand = heat_demand.loc[network.snapshots]

        network.madd("Load",
                     nodes,
                     suffix=" decentral heat",
                     bus=nodes + " decentral heat",
                     p_set=heat_demand[nodes].multiply(1-central_fraction))

        network.madd("Load",
                     nodes,
                     suffix=" central heat",
                     bus=nodes + " central heat",
                     p_set=heat_demand[nodes].multiply(central_fraction))

    if config["add_gas"]:
        # add converter from fuel source
        network.madd("Generator",
                     nodes,
                     suffix=' gas fuel',
                     bus=nodes + " gas",
                     carrier="gas",
                     p_nom_extendable=False,
                     p_nom=1e8,
                     marginal_cost=costs.at['OCGT', 'fuel'])

        network.madd("Store",
                     nodes + " gas Store",
                     bus=nodes + " gas",
                     e_nom_extendable=False,
                     e_nom=1e8,
                     e_cyclic=True,
                     carrier="gas")

    if config["add_coal"]:
        network.madd("Generator",
                     nodes + " coal fuel",
                     bus=nodes + " coal",
                     carrier="coal",
                     p_nom_extendable=False,
                     p_nom=1e8,
                     marginal_cost=costs.at['coal', 'fuel'])

    if config["add_biomass"]:
        network.madd('Bus',
                     nodes,
                     suffix=" biomass",
                     x=pro_centroid_x,
                     y=pro_centroid_y,
                     carrier="biomass",
                     )

        biomass_potential = pd.read_hdf(snakemake.input.biomass_potental)
        network.madd("Store",
                     nodes + " biomass",
                     bus =nodes + " biomass",
                     e_nom_extendable=False,
                     e_nom=biomass_potential,
                     e_initial=biomass_potential,
                     carrier='biomass'
        )

        network.add("Carrier", "CO2", co2_emissions=0)
        network.madd('Bus',
                     nodes,
                     suffix=" CO2",
                     x=pro_centroid_x,
                     y=pro_centroid_y,
                     carrier="CO2",
                     )

        network.madd("Store",
                     nodes + " CO2",
                     bus =nodes + " CO2",
                     carrier='CO2'
        )

        network.add("Carrier", "CO2 capture", co2_emissions=1)
        network.madd('Bus',
                     nodes,
                     suffix=" CO2 capture",
                     x=pro_centroid_x,
                     y=pro_centroid_y,
                     carrier="CO2 capture",
        )

        network.madd("Store",
                     nodes + " CO2 capture",
                     bus =nodes + " CO2 capture",
                     e_nom_extendable=True,
                     carrier='CO2 capture'
        )

        network.madd("Link",
                     nodes + " central biomass CHP capture",
                     bus0=nodes + " CO2",
                     bus1=nodes + " CO2 capture",
                     bus2=nodes,
                     p_nom_extendable=True,
                     carrier='CO2 capture',
                     efficiency=costs.at["biomass CHP capture", "capture_rate"],
                     efficiency2=-1*costs.at["biomass CHP capture", "capture_rate"]*costs.at["biomass CHP capture", "electricity-input"],
                     capital_cost=costs.at["biomass CHP capture", "capture_rate"]*costs.at["biomass CHP capture", "capital_cost"],
                     lifetime=costs.at["biomass CHP capture", "lifetime"]
        )

        network.madd("Link",
                     nodes + " central biomass CHP",
                     bus0=nodes + " biomass",
                     bus1=nodes,
                     bus2=nodes + " central heat",
                     bus3=nodes + " CO2",
                     p_nom_extendable=True,
                     carrier="biomass",
                     efficiency=costs.at["biomass CHP", "efficiency"],
                     efficiency2=costs.at["biomass CHP", "efficiency-heat"],
                     efficiency3=0.32522269504651985, # 4187.0095385594495TWh equates to 0.79*(5.24/3.04) Gt CO2  # tCO2/MWh
                     capital_cost=costs.at["biomass CHP", "efficiency"] * costs.at[
                         "biomass CHP", "capital_cost"],
                     marginal_cost=costs.at["biomass CHP", "efficiency"] * costs.at[
                         "biomass CHP", "marginal_cost"] + costs.at['solid biomass', 'fuel'],
                     lifetime=costs.at["biomass CHP", "lifetime"]
        )

        network.madd("Link",
                     nodes + " decentral biomass boiler",
                     bus0=nodes + " biomass",
                     bus1=nodes + " decentral heat",
                     p_nom_extendable=True,
                     carrier="biomass",
                     efficiency=costs.at["biomass boiler", "efficiency"],
                     capital_cost=costs.at["biomass boiler", "efficiency"] * costs.at[
                         "biomass boiler", "capital_cost"],
                     marginal_cost=costs.at["biomass boiler", "efficiency"] * costs.at[
                         "biomass boiler", "marginal_cost"] + costs.at["biomass boiler", "pelletizing cost"] + costs.at['solid biomass', 'fuel'],
                     lifetime=costs.at["biomass boiler", "lifetime"]
        )


    if config['add_hydro']:

        #######
        df = pd.read_csv('data/hydro/dams_large.csv', index_col=0)
        points = df.apply(lambda row: Point(row.Lon, row.Lat), axis=1)
        dams = gpd.GeoDataFrame(df, geometry=points, crs=4236)

        hourly_rng = pd.date_range('1979-01-01', '2017-01-01', freq='1H', inclusive='left')
        inflow = pd.read_pickle('data/hydro/daily_hydro_inflow_per_dam_1979_2016_m3.pickle').reindex(hourly_rng, fill_value=0)
        inflow.columns = dams.index

        water_consumption_factor = dams.loc[:, 'Water_consumption_factor_avg'] * 1e3 # m^3/KWh -> m^3/MWh


        #######
        # ### Add hydro stations as buses
        network.madd('Bus',
            dams.index,
            suffix=' station',
            carrier='stations',
            x=dams['geometry'].to_crs('+proj=cea').centroid.to_crs(pro_shapes.crs).x,
            y=dams['geometry'].to_crs('+proj=cea').centroid.to_crs(pro_shapes.crs).y)

        dam_buses = network.buses[network.buses.carrier=='stations']


        # ### add hydro reservoirs as stores

        initial_capacity = pd.read_pickle('data/hydro/reservoir_initial_capacity.pickle')
        effective_capacity = pd.read_pickle('data/hydro/reservoir_effective_capacity.pickle')
        initial_capacity.index = dams.index
        effective_capacity.index = dams.index
        initial_capacity = initial_capacity/water_consumption_factor
        effective_capacity=effective_capacity/water_consumption_factor

        network.madd('Store',
            dams.index,
            suffix=' reservoir',
            bus=dam_buses.index,
            e_nom=effective_capacity,
            e_initial=initial_capacity,
            e_cyclic=True,
            marginal_cost=config['costs']['marginal_cost']['hydro'])

        ### add hydro turbines to link stations to provinces
        network.madd('Link',
                    dams.index,
                    suffix=' turbines',
                    bus0=dam_buses.index,
                    bus1=dams['Province'],
                    carrier="hydroelectricity",
                    p_nom=10 * dams['installed_capacity_10MW'],
                    capital_cost=costs.at['hydro', 'capital_cost'],
                    efficiency= 1)


        ### add rivers to link station to station
        bus0s = [0, 21, 11, 19, 22, 29, 8, 40, 25, 1, 7, 4, 10, 15, 12, 20, 26, 6, 3, 39]
        bus1s = [5, 11, 19, 22, 32, 8, 40, 25, 35, 2, 4, 10, 9, 12, 20, 23, 6, 17, 14, 16]

        for bus0, bus2 in list(zip(dams.index[bus0s], dam_buses.iloc[bus1s].index)):

            # normal flow
            network.links.at[bus0 + ' turbines', 'bus2'] = bus2
            network.links.at[bus0 + ' turbines', 'efficiency2'] = 1.

        ### spillage
        for bus0, bus1 in list(zip(dam_buses.iloc[bus0s].index, dam_buses.iloc[bus1s].index)):
            network.add('Link',
                       "{}-{}".format(bus0,bus1) + ' spillage',
                       bus0=bus0,
                       bus1=bus1,
                       p_nom=1e8,
                       p_nom_extendable=False)

        dam_ends = [dam for dam in range(len(dams.index)) if (dam in bus1s and dam not in bus0s) or (dam not in bus0s+bus1s)]

        for bus0 in dam_buses.iloc[dam_ends].index:
            network.add('Link',
                        bus0 + ' spillage',
                        bus0=bus0,
                        bus1='Tibet',
                        p_nom_extendable=False,
                        p_nom=1e8,
                        efficiency=0.0)

        #### add inflow as generators
        # only feed into hydro stations which are the first of a cascade
        inflow_stations = [dam for dam in range(len(dams.index)) if not dam in bus1s ]

        for inflow_station in inflow_stations:

            # p_nom = 1 and p_max_pu & p_min_pu = p_pu, compulsory inflow

            date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'])
            date_range = date_range.map(lambda t: t.replace(year=2016))

            p_nom = (inflow.loc[date_range]/water_consumption_factor).iloc[:,inflow_station].max()
            p_pu = (inflow.loc[date_range]/water_consumption_factor).iloc[:,inflow_station] / p_nom
            p_pu.index = network.snapshots
            network.add('Generator',
                       dams.index[inflow_station] + ' inflow',
                       bus=dam_buses.iloc[inflow_station].name,
                       carrier='hydro_inflow',
                       p_max_pu=p_pu.clip(1.e-6),
                       p_min_pu=p_pu.clip(1.e-6),
                       p_nom=p_nom)

            # p_nom*p_pu = XXX m^3 then use turbines efficiency to convert to power

        ## add otehr existing hydro power
        hydro_p_nom = pd.read_hdf("data/p_nom/hydro_p_nom.h5")
        hydro_p_max_pu = pd.read_hdf("data/p_nom/hydro_p_max_pu.h5", key="hydro_p_max_pu")

        date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'], tz='Asia/shanghai')
        date_range = date_range.map(lambda t: t.replace(year=2020))
        hydro_p_max_pu = hydro_p_max_pu.loc[date_range]
        hydro_p_max_pu.index = network.snapshots

        network.madd('Generator',
                    nodes,
                    suffix=' hydroelectricity',
                    bus=nodes,
                    carrier="hydroelectricity",
                    p_nom=hydro_p_nom,
                    capital_cost=costs.at['hydro','capital_cost'],
                    p_max_pu=hydro_p_max_pu)

    if config['add_H2']:

        network.madd("Bus",
                     nodes,
                     suffix=" H2",
                     x=pro_centroid_x,
                     y=pro_centroid_y,
                     carrier="H2")

        network.madd("Link",
                    nodes + " H2 Electrolysis",
                    bus0=nodes,
                    bus1=nodes + " H2",
                    bus2=nodes + " central heat",
                    p_nom_extendable=True,
                    carrier="H2",
                    efficiency=costs.at["electrolysis","efficiency"],
                    efficiency2=costs.at["electrolysis","efficiency-heat"],
                    capital_cost=costs.at["electrolysis","capital_cost"],
                    lifetime=costs.at["electrolysis","lifetime"])

        network.madd("Link",
                     nodes + " central H2 CHP",
                     bus0=nodes + " H2",
                     bus1=nodes,
                     bus2=nodes + " central heat",
                     p_nom_extendable=True,
                     carrier="H2 CHP",
                     efficiency=costs.at["central hydrogen CHP","efficiency"],
                     efficiency2=costs.at["central hydrogen CHP","efficiency"]/costs.at["central hydrogen CHP","c_b"],
                     capital_cost=costs.at["central hydrogen CHP","efficiency"]*costs.at["central hydrogen CHP","capital_cost"],
                     lifetime=costs.at["central hydrogen CHP","lifetime"]
                     )

        H2_under_nodes = pd.Index(['Sichuan','Chongqing','Hubei','Jiangxi','Anhui','Jiangsu','Shandong','Guangdong'])
        H2_type1_nodes = nodes.difference(H2_under_nodes)

        network.madd("Store",
                     H2_under_nodes + " H2 Store",
                     bus=H2_under_nodes + " H2",
                     e_nom_extendable=True,
                     e_cyclic=True,
                     capital_cost=costs.at["hydrogen storage underground","capital_cost"],
                     lifetime=costs.at["hydrogen storage underground","lifetime"])

        network.madd("Store",
                     H2_type1_nodes + " H2 Store",
                     bus=H2_type1_nodes + " H2",
                     e_nom_extendable=True,
                     e_cyclic=True,
                     capital_cost=costs.at["hydrogen storage tank type 1 including compressor","capital_cost"],
                     lifetime=costs.at["hydrogen storage tank type 1 including compressor","lifetime"])

    if config['add_methanation']:
        network.madd("Link",
                     nodes + " Sabatier",
                     bus0=nodes+" H2",
                     bus1=nodes+" gas",
                     p_nom_extendable=True,
                     carrier="Sabatier",
                     efficiency=costs.at["methanation","efficiency"],
                     capital_cost=costs.at["methanation","efficiency"] * costs.at["methanation","capital_cost"] + costs.at["direct air capture","capital_cost"]*costs.at['gas', 'co2_emissions']*costs.at["methanation","efficiency"],
                     marginal_cost=(400-5*(int(cost_year)-2020))*costs.at['gas', 'co2_emissions']*costs.at["methanation","efficiency"],
                     lifetime=costs.at["methanation","lifetime"])

    # add components
    network.madd("Generator",
                 nodes,
                 suffix=' onwind',
                 bus=nodes,
                 carrier="onwind",
                 p_nom_extendable=True,
                 p_nom_max=ds_onwind['p_nom_max'].to_pandas(),
                 capital_cost = costs.at['onwind','capital_cost'],
                 marginal_cost=costs.at['onwind','marginal_cost'],
                 p_max_pu=onwind_p_max_pu,
                 lifetime=costs.at['onwind','lifetime'])

    offwind_nodes = ds_offwind['bus'].to_pandas().index
    network.madd("Generator",
                 offwind_nodes,
                 suffix=' offwind',
                 bus=offwind_nodes,
                 carrier="offwind",
                 p_nom_extendable=True,
                 p_nom_max=ds_offwind['p_nom_max'].to_pandas(),
                 capital_cost = costs.at['offwind','capital_cost'],
                 marginal_cost=costs.at['offwind','marginal_cost'],
                 p_max_pu=offwind_p_max_pu,
                 lifetime=costs.at['offwind', 'lifetime'])

    network.madd("Generator",
                 nodes,
                 suffix=' solar',
                 bus=nodes,
                 carrier="solar",
                 p_nom_extendable=True,
                 p_nom_max=ds_solar['p_nom_max'].to_pandas(),
                 capital_cost = costs.at['solar','capital_cost'],
                 marginal_cost=costs.at['solar','marginal_cost'],
                 p_max_pu=solar_p_max_pu,
                 lifetime=costs.at['solar', 'lifetime'])

    if "nuclear" in config["Techs"]["vre_techs"]:
        nuclear_extendable=["Liaoning","Shandong","Jiangsu","Zhejiang","Fujian","Guangdong","Hainan","Guangxi"]
        nuclear_nodes = pd.Index(nuclear_extendable)
        network.madd("Generator",
                     nuclear_nodes,
                     suffix=' nuclear',
                     p_nom_extendable=True,
                     p_min_pu = 0.7,
                     bus=nuclear_nodes,
                     carrier="nuclear",
                     efficiency=costs.at['nuclear','efficiency'],
                     capital_cost = costs.at['nuclear','capital_cost'], #NB: capital cost is per MWel
                     marginal_cost= costs.at['nuclear','marginal_cost'],
                     lifetime=costs.at['nuclear', 'lifetime'])

    if "heat pump" in config["Techs"]["vre_techs"]:

        date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'], tz='Asia/shanghai')
        date_range = date_range.map(lambda t: t.replace(year=2020))

        with pd.HDFStore(snakemake.input.cop_name, mode='r') as store:
            ashp_cop = store['ashp_cop_profiles']
            ashp_cop.index = ashp_cop.index.tz_localize('Asia/shanghai')
            ashp_cop = ashp_cop.loc[date_range].set_index(network.snapshots)
            gshp_cop = store['gshp_cop_profiles']
            gshp_cop.index = gshp_cop.index.tz_localize('Asia/shanghai')
            gshp_cop = gshp_cop.loc[date_range].set_index(network.snapshots)

        for cat in [' decentral ', ' central ']:
            network.madd("Link",
                         nodes,
                         suffix=cat + "heat pump",
                         bus0=nodes,
                         bus1=nodes + cat + "heat",
                         carrier='heat pump',
                         efficiency=ashp_cop[nodes] if config["time_dep_hp_cop"] else costs.at[cat.lstrip()+"air-sourced heat pump",'efficiency'],
                         capital_cost=costs.at[cat.lstrip()+'air-sourced heat pump','efficiency'] * costs.at[cat.lstrip()+'air-sourced heat pump','capital_cost'],
                         marginal_cost=costs.at[cat.lstrip()+'air-sourced heat pump','efficiency'] * costs.at[cat.lstrip()+'air-sourced heat pump','marginal_cost'],
                         p_nom_extendable=True,
                         lifetime=costs.at[cat.lstrip()+'air-sourced heat pump','lifetime'])

            network.madd("Link",
                         nodes,
                         suffix=cat + " ground heat pump",
                         bus0=nodes,
                         bus1=nodes + cat + "heat",
                         carrier='heat pump',
                         efficiency=gshp_cop[nodes] if config["time_dep_hp_cop"] else costs.at['decentral ground-sourced heat pump','efficiency'],
                         capital_cost=costs.at[cat.lstrip()+'ground-sourced heat pump','efficiency'] * costs.at['decentral ground-sourced heat pump','capital_cost'],
                         marginal_cost=costs.at[cat.lstrip() + 'ground-sourced heat pump', 'efficiency'] * costs.at[
                             cat.lstrip() + 'ground-sourced heat pump', 'marginal_cost'],
                         p_nom_extendable=True,
                         lifetime=costs.at['decentral ground-sourced heat pump','lifetime'])

    if "resistive heater" in config["Techs"]["vre_techs"]:
        for cat in [" decentral ", " central "]:
            network.madd("Link",
                         nodes + cat + "resistive heater",
                         bus0=nodes,
                         bus1=nodes + cat + "heat",
                         carrier="resistive heater",
                         efficiency=costs.at[cat.lstrip()+'resistive heater','efficiency'],
                         capital_cost=costs.at[cat.lstrip()+'resistive heater','efficiency']*costs.at[cat.lstrip()+'resistive heater','capital_cost'],
                         marginal_cost=costs.at[cat.lstrip()+'resistive heater','efficiency']*costs.at[cat.lstrip()+'resistive heater','marginal_cost'],
                         p_nom_extendable=True,
                         lifetime=costs.at[cat.lstrip()+'resistive heater','lifetime'])

    if "solar thermal" in config["Techs"]["vre_techs"]:
        #this is the amount of heat collected in W per m^2, accounting
        #for efficiency
        with pd.HDFStore(snakemake.input.solar_thermal_name, mode='r') as store:
            #1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
            solar_thermal = config['solar_cf_correction'] * store['solar_thermal_profiles']/1e3

        date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'],tz='Asia/shanghai')
        date_range = date_range.map(lambda t: t.replace(year=2020))

        solar_thermal.index = solar_thermal.index.tz_localize('Asia/shanghai')
        solar_thermal = solar_thermal.loc[date_range].set_index(network.snapshots)

        for cat in [" decentral ", " central "]:
            network.madd("Generator",
                         nodes,
                         suffix=cat + "solar thermal",
                         bus=nodes + cat + "heat",
                         carrier="solar thermal",
                         p_nom_extendable=True,
                         capital_cost=costs.at[cat.lstrip()+'solar thermal','capital_cost'],
                         p_max_pu=solar_thermal[nodes].clip(1.e-4),
                         lifetime=costs.at[cat.lstrip()+'solar thermal','lifetime'])

    if "coal boiler" in config["Techs"]["conv_techs"]:
        for cat in [" decentral ", " central "]:
            network.madd("Link",
                         nodes + cat + "coal boiler",
                         p_nom_extendable=True,
                         bus0=nodes + " coal",
                         bus1=nodes + cat + "heat",
                         carrier="coal boiler",
                         efficiency=costs.at[cat.lstrip()+'coal boiler','efficiency'],
                         marginal_cost=costs.at[cat.lstrip()+'coal boiler','efficiency']*costs.at[cat.lstrip() + 'coal boiler', 'VOM'],
                         capital_cost=costs.at[cat.lstrip()+'coal boiler','efficiency']*costs.at[cat.lstrip()+'coal boiler','capital_cost'],
                         lifetime=costs.at[cat.lstrip()+'coal boiler','lifetime'])

    if "gas boiler" in config["Techs"]["conv_techs"]:
        for cat in [" decentral "]:
            network.madd("Link",
                         nodes + cat + "gas boiler",
                         p_nom_extendable=True,
                         bus0=nodes + " gas",
                         bus1=nodes + cat + "heat",
                         carrier="gas boiler",
                         efficiency=costs.at[cat.lstrip()+'gas boiler','efficiency'],
                         marginal_cost=costs.at[cat.lstrip()+'gas boiler','efficiency']*costs.at[cat.lstrip() + 'gas boiler', 'VOM'],
                         capital_cost=costs.at[cat.lstrip()+'gas boiler','efficiency']*costs.at[cat.lstrip()+'gas boiler','capital_cost'],
                         lifetime=costs.at[cat.lstrip()+'gas boiler','lifetime'])

    if "OCGT gas" in config["Techs"]["conv_techs"]:
        network.madd("Link",
                     nodes,
                     suffix=" OCGT",
                     bus0=nodes + " gas",
                     bus1=nodes,
                     carrier="OCGT gas",
                     marginal_cost=costs.at["OCGT",'efficiency'] * costs.at["OCGT", 'VOM'], #NB: VOM is per MWel
                     capital_cost=costs.at["OCGT",'efficiency'] * costs.at["OCGT", 'capital_cost'], #NB: capital cost is per MWel
                     p_nom_extendable=True,
                     efficiency=costs.at["OCGT", 'efficiency'],
                     lifetime=costs.at["OCGT", 'lifetime'])

    if "CHP gas" in config["Techs"]["conv_techs"]:
        network.madd("Link",
                     nodes,
                     suffix=" central CHP gas generator",
                     bus0=nodes + " gas",
                     bus1=nodes,
                     carrier="CHP gas",
                     p_nom_extendable=True,
                     marginal_cost=costs.at['central gas CHP', 'efficiency'] * costs.at[
                         'central gas CHP', 'VOM'],  # NB: VOM is per MWel
                     capital_cost=costs.at['central gas CHP', 'efficiency'] * costs.at[
                         'central gas CHP', 'capital_cost'],  # NB: capital cost is per MWel
                     efficiency=costs.at['central gas CHP', 'efficiency'],
                     p_nom_ratio=1.0,
                     c_b=costs.at['central gas CHP', 'c_b'],
                     lifetime=costs.at['central gas CHP', 'lifetime'])

        network.madd("Link",
                     nodes,
                     suffix=" central CHP gas boiler",
                     bus0=nodes + " gas",
                     bus1=nodes + " central heat",
                     carrier="CHP gas",
                     p_nom_extendable=True,
                     marginal_cost=costs.at['central gas CHP', 'efficiency'] * costs.at[
                         'central gas CHP', 'VOM'],  # NB: VOM is per MWel
                     efficiency=costs.at['central gas CHP', 'efficiency']/costs.at['central gas CHP', 'c_v'],
                     lifetime=costs.at['central gas CHP', 'lifetime'])

    if "coal power plant" in config["Techs"]["conv_techs"]:
            network.add("Carrier", "coal cc", co2_emissions=0.034)
            network.madd("Generator",
                        nodes,
                        suffix=' coal cc',
                        bus=nodes,
                        carrier="coal cc",
                        p_nom_extendable=True,
                        efficiency=costs.at['coal', 'efficiency'],
                        marginal_cost= costs.at['coal', 'marginal_cost'],
                        capital_cost=costs.at['coal', 'capital_cost'] + costs.at['retrofit', 'capital_cost'], #NB: capital cost is per MWel
                        lifetime=costs.at['coal', 'lifetime'])

            for year in range(int(planning_horizons)-25,2021,5):
                network.madd("Generator",
                             nodes,
                             suffix=' coal-' + str(year) + "-retrofit",
                             bus=nodes,
                             carrier="coal cc",
                             p_nom_extendable=True,
                             capital_cost=costs.at['coal', 'capital_cost'] + costs.at['retrofit', 'capital_cost'] + 2021 - year,
                             efficiency=costs.at['coal', 'efficiency'],
                             lifetime=costs.at['coal', 'lifetime'],
                             build_year=year,
                             marginal_cost=costs.at['coal', 'marginal_cost'])

    if "CHP coal" in config["Techs"]["conv_techs"]:
        network.madd("Link",
                 nodes,
                 suffix=" central CHP coal generator",
                 bus0=nodes + " coal",
                 bus1=nodes,
                 carrier="CHP coal",
                 p_nom_extendable=True,
                 marginal_cost=costs.at['central coal CHP', 'efficiency'] * costs.at['central coal CHP', 'VOM'],#NB: VOM is per MWel
                 capital_cost=costs.at['central coal CHP', 'efficiency'] * costs.at['central coal CHP', 'capital_cost'],#NB: capital cost is per MWel
                 efficiency=costs.at['central coal CHP', 'efficiency'],
                 p_nom_ratio=1.0,
                 c_b=costs.at['central coal CHP', 'c_b'],
                 lifetime=costs.at['central coal CHP', 'lifetime'])

        network.madd("Link",
                   nodes,
                   suffix=" central CHP coal boiler",
                   bus0=nodes + " coal",
                   bus1=nodes + " central heat",
                   carrier="CHP coal",
                   p_nom_extendable=True,
                   marginal_cost=costs.at['central coal CHP', 'efficiency'] * costs.at[
                       'central coal CHP', 'VOM'],  # NB: VOM is per MWel
                   efficiency=costs.at['central coal CHP', 'efficiency']/costs.at['central coal CHP', 'c_v'],
                   lifetime=costs.at['central coal CHP', 'lifetime'])

    if "water tanks" in config["Techs"]["store_techs"]:
        for cat in [' decentral ', ' central ']:
            network.madd("Bus",
                         nodes,
                         suffix=cat + "water tanks",
                         x=pro_centroid_x,
                         y=pro_centroid_y,
                         carrier="water tanks")

            network.madd("Link",
                         nodes + cat + "water tanks charger",
                         bus0=nodes + cat + "heat",
                         bus1=nodes + cat + "water tanks",
                         carrier="water tanks",
                         efficiency=costs.at['water tank charger','efficiency'],
                         p_nom_extendable=True)

            network.madd("Link",
                         nodes + cat + "water tanks discharger",
                         bus0=nodes + cat + "water tanks",
                         bus1=nodes + cat + "heat",
                         carrier="water tanks",
                         efficiency=costs.at['water tank discharger','efficiency'],
                         p_nom_extendable=True)

            network.madd("Store",
                         nodes + cat + "water tank",
                         bus=nodes + cat + "water tanks",
                         carrier="water tanks",
                         e_cyclic=True,
                         e_nom_extendable=True,
                         standing_loss=1-np.exp(-1/(24.* (config["tes_tau"] if cat==' decentral ' else 180.))),  # [HP] 180 day time constant for centralised, 3 day for decentralised
                         capital_cost=costs.at[cat.lstrip()+'water tank storage','capital_cost'],
                         lifetime=costs.at[cat.lstrip()+'water tank storage','lifetime'])

    if "battery" in config["Techs"]["store_techs"]:
        network.madd("Bus",
                     nodes,
                     suffix=" battery",
                     x=pro_centroid_x,
                     y=pro_centroid_y,
                     carrier="battery")

        network.madd("Store",
                     nodes + " battery",
                     bus=nodes + " battery",
                     e_cyclic=True,
                     e_nom_extendable=True,
                     capital_cost=costs.at['battery storage','capital_cost'],
                     lifetime=costs.at['battery storage','lifetime'])

        network.madd("Link",
                     nodes + " battery charger",
                     bus0=nodes,
                     bus1=nodes + " battery",
                     efficiency=costs.at['battery inverter','efficiency']**0.5,
                     capital_cost=costs.at['battery inverter','capital_cost'],
                     p_nom_extendable=True,
                     carrier="battery",
                     lifetime=costs.at['battery inverter','lifetime'] )

        network.madd("Link",
                     nodes + " battery discharger",
                     bus0=nodes + " battery",
                     bus1=nodes,
                     efficiency=costs.at['battery inverter','efficiency']**0.5,
                     marginal_cost=0.,
                     carrier="battery discharger",
                     p_nom_extendable=True)

    if "PHS" in config["Techs"]["store_techs"]:
        # pure pumped hydro storage, fixed, 6h energy by default, no inflow
        hydrocapa_df = pd.read_csv('data/hydro/PHS_p_nom.csv', index_col=0)
        phss = hydrocapa_df.index[hydrocapa_df['MW'] > 0].intersection(nodes)
        if config['hydro']['hydro_capital_cost']:
            cc=costs.at['PHS','capital_cost']
        else:
            cc=0.

        network.madd("StorageUnit",
                     phss,
                     suffix=" PHS",
                     bus=phss,
                     carrier="PHS",
                     p_nom_extendable=False,
                     p_nom=hydrocapa_df.loc[phss]['MW'],
                     p_nom_min=hydrocapa_df.loc[phss]['MW'],
                     max_hours=config['hydro']['PHS_max_hours'],
                     efficiency_store=np.sqrt(costs.at['PHS','efficiency']),
                     efficiency_dispatch=np.sqrt(costs.at['PHS','efficiency']),
                     cyclic_state_of_charge=True,
                     capital_cost = cc,
                     marginal_cost=0.)

    #add lines

    if not config['no_lines']:
        edges = pd.read_csv(snakemake.input.edges, header=None)

        lengths = 1.25 * np.array([haversine([network.buses.at[name0,"x"],network.buses.at[name0,"y"]],
                                  [network.buses.at[name1,"x"],network.buses.at[name1,"y"]]) for name0,name1 in edges[[0,1]].values])

        cc = (config['line_cost_factor'] * lengths * [HVAC_cost_curve(l) for l in
                                                          lengths]) * 1.5 * 1.02 * Nyears * annuity(40.,config['costs']['discountrate'])

        network.madd("Link",
                     edges[0] + '-' + edges[1],
                     bus0=edges[0].values,
                     bus1=edges[1].values,
                     suffix =" positive",
                     p_nom_extendable=True,
                     p_min_pu=0,
                     efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]* config["transmission_efficiency"]["DC"]["efficiency_per_1000km"]**(lengths/1000),
                     length=lengths,
                     capital_cost=cc)

        network.madd("Link",
                     edges[1] + '-' + edges[0],
                     bus0=edges[1].values,
                     bus1=edges[0].values,
                     suffix=" reversed",
                     p_nom_extendable=True,
                     p_min_pu=0,
                     efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]* config["transmission_efficiency"]["DC"]["efficiency_per_1000km"]**(lengths/1000),
                     length=lengths,
                     capital_cost=0)

    if config['hydrogen_lines']:
        edges = pd.read_csv(snakemake.input.edges, header=None)
        lengths = 1.25 * np.array([haversine([network.buses.at[name0,"x"],network.buses.at[name0,"y"]],
                                  [network.buses.at[name1,"x"],network.buses.at[name1,"y"]]) for name0,name1 in edges[[0,1]].values])

        cc = (costs.at['H2 (g) pipeline','capital_cost'] * lengths)

        network.madd("Link",
                     edges[0] + '-' + edges[1] + " H2 pipeline",
                     suffix=" positive",
                     bus0=edges[0].values + " H2",
                     bus1=edges[1].values + " H2",
                     bus2=edges[0].values,
                     p_nom_extendable=True,
                     p_nom=0,
                     p_nom_min=0,
                     p_min_pu=0,
                     efficiency=config["transmission_efficiency"]["H2 pipeline"]["efficiency_static"]* config["transmission_efficiency"]["H2 pipeline"]["efficiency_per_1000km"]**(lengths/1000),
                     efficiency2=-config["transmission_efficiency"]["H2 pipeline"]["compression_per_1000km"]*lengths/1e3,
                     length=lengths,
                     lifetime=costs.at['H2 (g) pipeline','lifetime'],
                     capital_cost=cc)

        network.madd("Link",
                     edges[1] + '-' + edges[0] + " H2 pipeline",
                     suffix=" reversed",
                     bus0=edges[1].values + " H2",
                     bus1=edges[0].values + " H2",
                     bus2=edges[1].values,
                     p_nom_extendable=True,
                     p_nom=0,
                     p_nom_min=0,
                     p_min_pu=0,
                     efficiency=config["transmission_efficiency"]["H2 pipeline"]["efficiency_static"]* config["transmission_efficiency"]["H2 pipeline"]["efficiency_per_1000km"]**(lengths/1000),
                     efficiency2=-config["transmission_efficiency"]["H2 pipeline"][
                         "compression_per_1000km"] * lengths / 1e3,
                     length=lengths,
                     lifetime=costs.at['H2 (g) pipeline','lifetime'],
                     capital_cost=0)
    return network

if __name__ == '__main__':

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('prepare_base_networks',
                                   opts='ll',
                                   topology='current+Neighbor',
                                   pathway='exponential175',
                                   planning_horizons="2025")
    configure_logging(snakemake)

    network = prepare_network(snakemake.config)

    network.export_to_netcdf(snakemake.output.network_name)
