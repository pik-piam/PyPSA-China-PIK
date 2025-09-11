"""
Script to conditionally add sector coupling components (EV, heat, etc.) to PyPSA networks.

This script serves as a bridge between pure electricity-only networks and full sector-coupled
networks. It allows the workflow to run with or without sector coupling by selectively adding
transport (EV) and heating components based on configuration flags.

"""

# SPDX-FileCopyrightText: : 2025 The PyPSA-China Authors
# SPDX-License-Identifier: MIT

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def attach_EV_components(
    n: pypsa.Network,
    avail_profile: pd.DataFrame,
    dsm_profile: pd.DataFrame,
    p_set: pd.DataFrame,
    nodes: pd.Index,
    options: dict,
    type: str,
) -> None:
    """
    Attach electric vehicle components to a PyPSA network for energy system modeling.

    Adds EV buses, loads, chargers, and optionally battery storage components to model
    either direct charging or demand-side management (DSM) with storage flexibility.
    The function scales EV fleet size based on annual energy demand and creates
    appropriate network components for passenger or freight vehicles.

    Two modeling modes:
    - DSM OFF: Direct charging mode where p_set represents charging demand
    - DSM ON: Storage mode where p_set represents driving demand, with charging
              constrained by availability and load satisfied by battery discharge

    Args:
        n (pypsa.Network): PyPSA network object to modify in-place.
        avail_profile (pd.DataFrame): EV charging availability profile (0-1) with
            snapshots as index and nodes as columns.
        dsm_profile (pd.DataFrame): DSM state-of-charge profile (0-1) with
            snapshots as index and nodes as columns.
        p_set (pd.DataFrame): EV energy demand profile in MWh with snapshots as
            index and nodes as columns.
        nodes (pd.Index): Network node names (typically province/region names).
        options (dict): EV configuration parameters containing:
            - dsm (bool): Enable demand-side management with battery storage
            - annual_consumption (float): Annual energy consumption per vehicle in MWh/year
            - charge_rate (float): Maximum charging power per vehicle in MW
            - share_charger (float): Fraction of vehicles with access to chargers (0-1)
            - battery_size (float): Battery capacity per vehicle in MWh
            - dsm_availability (float): Fraction of battery capacity available for DSM (0-1)
        type (str): Vehicle type identifier, either "passenger" or "freight".

    Raises:
        ValueError: If type is not "passenger" or "freight".
    """
    if type not in ("passenger", "freight"):
        raise ValueError("type must be 'passenger' or 'freight'")

    dsm_enabled = options["dsm"]

    # --- 1. Compute EV numbers from demand ---
    total_energy = p_set.sum().sum()
    annual_consumption_per_ev = options["annual_consumption"]

    total_number_evs = max(total_energy / max(annual_consumption_per_ev, 1e-6), 0.0)
    node_energy_ratio = p_set.sum() / total_energy
    number_evs = node_energy_ratio * total_number_evs

    charge_power = (number_evs * options["charge_rate"] * options["share_charger"]).clip(
        lower=0.001
    )
    battery_energy = (number_evs * options["battery_size"]).clip(lower=0.001)

    logger.info(
        f"EV {type} - DSM {'ON' if dsm_enabled else 'OFF'}, "
        f"Total energy: {total_energy:.1f} MWh, "
        f"Total EVs: {int(total_number_evs):,}"
    )

    # --- 2. Add buses ---
    ev_load_bus = nodes + f" EV_load_{type}"
    n.add("Carrier", f"EV_load_{type}")
    n.add("Bus", ev_load_bus, location=nodes, carrier=f"EV_load_{type}", unit="MWh_el")

    if dsm_enabled:
        carrier_name = f"EV_{type}_battery"
        ev_battery_bus = nodes + " " + carrier_name
        n.add("Carrier", carrier_name)
        n.add("Bus", ev_battery_bus, location=nodes, carrier=carrier_name, unit="MWh_el")

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

    logger.info(
        f"EV {type} setup complete: {'DSM storage' if dsm_enabled else 'Direct charging'} mode"
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_sectors",
            planning_horizons="2030",
        )

    configure_logging(snakemake)

    logger.info("Preparing sector-coupled network")

    # Load base network
    network = pypsa.Network(snakemake.input.network)
    nodes = network.buses.query("carrier == 'AC'").index

    # Add transport (EV) components
    logger.info("Adding EV components to sector-coupled network")

    # Load DSM profile
    dsm_profile = pd.read_csv(snakemake.input.dsm_profile, index_col=0, parse_dates=True)

    # Passenger EVs
    if snakemake.config.get("transport", {}).get("passenger_bev", {}).get("on", True):
        logger.info("Adding passenger EV components")

        charging_demand_pass = pd.read_csv(
            snakemake.input.transport_demand_passenger, index_col=0, parse_dates=True
        )
        driving_demand_pass = pd.read_csv(
            snakemake.input.driving_demand_passenger, index_col=0, parse_dates=True
        )
        avail_profile_pass = pd.read_csv(
            snakemake.input.avail_profile_passenger, index_col=0, parse_dates=True
        )

        options_pass = snakemake.config["transport"]["passenger_bev"]
        p_set_pass = driving_demand_pass if options_pass["dsm"] else charging_demand_pass

        attach_EV_components(
            n=network,
            avail_profile=avail_profile_pass,
            dsm_profile=dsm_profile,
            p_set=p_set_pass,
            nodes=nodes,
            options=options_pass,
            type="passenger",
        )

    # Freight EVs
    if snakemake.config.get("transport", {}).get("freight_bev", {}).get("on", True):
        logger.info("Adding freight EV components")

        charging_demand_freight = pd.read_csv(
            snakemake.input.transport_demand_freight, index_col=0, parse_dates=True
        )
        driving_demand_freight = pd.read_csv(
            snakemake.input.driving_demand_freight, index_col=0, parse_dates=True
        )
        avail_profile_freight = pd.read_csv(
            snakemake.input.avail_profile_freight, index_col=0, parse_dates=True
        )

        options_freight = snakemake.config["transport"]["freight_bev"]
        p_set_freight = (
            driving_demand_freight if options_freight["dsm"] else charging_demand_freight
        )

        attach_EV_components(
            n=network,
            avail_profile=avail_profile_freight,
            dsm_profile=dsm_profile,
            p_set=p_set_freight,
            nodes=nodes,
            options=options_freight,
            type="freight",
        )

    # Add heat sector components (future extension)
    if snakemake.config.get("sectors", {}).get("heat_coupling", False):
        logger.info("Heat coupling components would be added here")

    # Save modified network
    logger.info(f"Saving sector-coupled network to {snakemake.output.network}")
    network.export_to_netcdf(snakemake.output.network)
