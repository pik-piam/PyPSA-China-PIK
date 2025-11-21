"""Add basic EV loads and chargers to a PyPSA network."""

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def add_carrier_if_missing(n: pypsa.Network, carrier_name: str):
    """Add a carrier to the network if it doesn't already exist.
    
    Args:
        n (pypsa.Network): PyPSA network to modify.
        carrier_name (str): Name of the carrier to add.
    """
    if carrier_name not in n.carriers.index:
        n.add("Carrier", carrier_name)
        logger.debug("Carrier '%s' added to network.", carrier_name)


def attach_simple_ev(
    n: pypsa.Network, p_set: pd.DataFrame, nodes: pd.Index, options: dict, ev_type: str
):
    """
    Attach electric vehicle demand and charging links to the PyPSA network.

    This function implements an EV demand model with a fixed charging profile.  
    For each node, it creates:
      • an EV load bus,
      • a Load component representing the EV charging demand time series,
      • and a charger Link connecting the AC bus to the EV load bus.

    Args:
        n (pypsa.Network): Network to modify in place.
        p_set (pd.DataFrame): EV charging demand time series (MW), indexed by
            snapshots and with nodes as columns.
        nodes (pd.Index): AC buses where EV components are added.
        options (dict): EV configuration parameters, including:
            - annual_consumption (float): annual energy demand per vehicle (MWh/year)
            - charge_rate (float): charger power rating per vehicle (MW)
            - share_charger (float): fraction of vehicles that can charge simultaneously
        ev_type (str): EV category for component naming (e.g., "passenger", "freight").

    Returns:
        None: Modifies the network in place.
    """

    total_energy = p_set.sum().sum()
    total_number_evs = total_energy / max(options["annual_consumption"], 1e-6)
    node_ratio = p_set.sum() / max(total_energy, 1e-6)
    number_evs = node_ratio * total_number_evs
    charge_power = (
        number_evs * options["charge_rate"] * options["share_charger"]
    ).clip(lower=0.001)

    logger.info("EV %s: %s vehicles (direct charging)", ev_type, f"{int(total_number_evs):,}")
    logger.debug(
        "EV %s: total energy %.2f MWh, annual consumption %.3f MWh/veh",
        ev_type,
        total_energy,
        options["annual_consumption"],
    )
    logger.debug(
        "EV %s: sample node power (MW)\n%s",
        ev_type,
        charge_power.head().to_string(),
    )

    ev_load_carrier = f"EV_{ev_type}_load"
    add_carrier_if_missing(n, ev_load_carrier)
    ev_load_bus = nodes + f" EV_{ev_type}_load"
    n.add("Bus", nodes, suffix=f" EV_{ev_type}_load", carrier=ev_load_carrier)

    n.add(
        "Load",
        nodes,
        suffix=f" EV_{ev_type}_load",
        bus=ev_load_bus,
        carrier=ev_load_carrier,
        p_set=p_set.loc[n.snapshots, nodes],
    )

    charger_carrier = f"EV_{ev_type}_charger"
    add_carrier_if_missing(n, charger_carrier)
    n.add(
        "Link",
        nodes,
        suffix=f" EV_{ev_type}_charger",
        bus0=nodes,
        bus1=ev_load_bus,
        carrier=charger_carrier,
        p_nom=charge_power,
        efficiency=1.0,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("add_sectors", planning_horizons="2030")
    configure_logging(snakemake)

    network = pypsa.Network(snakemake.input.network)
    nodes = network.buses.query("carrier == 'AC'").index

    ev_cfg = snakemake.config.get("sectors", {}).get("electric_vehicles", {})
    if not ev_cfg.get("enabled", False):
        logger.info("EV sector disabled; exporting original network.")
        network.export_to_netcdf(snakemake.output.network)
        raise SystemExit(0)

    transport_cfg = ev_cfg.get("transport", {})
    logger.info("Transport configuration: %s", transport_cfg)
    
    passenger_cfg = transport_cfg.get("passenger_bev", {})
    if passenger_cfg.get("enable", False):
        charging = pd.read_csv(
            snakemake.input.transport_demand_passenger, index_col=0, parse_dates=True
        )
        attach_simple_ev(network, charging, nodes, passenger_cfg, "passenger")
        logger.info(
            "Passenger EV load added for %d nodes; snapshots %d",
            len(charging.columns),
            len(charging),
        )
    else:
        logger.info("Passenger EV disabled; skipping.")

    freight_cfg = transport_cfg.get("freight_bev", {})
    if freight_cfg.get("enable", False):
        charging = pd.read_csv(
            snakemake.input.transport_demand_freight, index_col=0, parse_dates=True
        )
        attach_simple_ev(network, charging, nodes, freight_cfg, "freight")
        logger.info(
            "Freight EV load added for %d nodes; snapshots %d",
            len(charging.columns),
            len(charging),
        )
    else:
        logger.info("Freight EV disabled; skipping.")

    network.export_to_netcdf(snakemake.output.network)
    logger.info(
        "Network with EV sectors exported to %s", snakemake.output.network
    )
