"""
Simplified version: Add electric vehicle (EV) sector coupling components to PyPSA network
Only supports direct charging mode, no DSM functionality
"""

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def add_carrier_if_missing(n: pypsa.Network, carrier_name: str):
    """Add carrier if it doesn't exist"""
    if carrier_name not in n.carriers.index:
        n.add("Carrier", carrier_name)
        logger.debug(f"Carrier '{carrier_name}' added to network.")


def attach_EV_components(
    n: pypsa.Network,
    p_set: pd.DataFrame,
    nodes: pd.Index,
    options: dict,
    ev_type: str,
):
    """Add EV components to network, only supports direct charging mode"""

    if ev_type not in ("passenger", "freight"):
        raise ValueError("ev_type must be 'passenger' or 'freight'")

    # --- 1. Scale calculation ---
    total_energy = p_set.sum().sum()
    total_number_evs = total_energy / max(options["annual_consumption"], 1e-6)
    node_ratio = p_set.sum() / max(total_energy, 1e-6)
    number_evs = node_ratio * total_number_evs

    charge_power = (number_evs * options["charge_rate"] * options["share_charger"]).clip(
        lower=0.001
    )

    logger.info(f"EV {ev_type}: {int(total_number_evs):,} vehicles (direct charging mode)")

    # --- 2. EV load bus ---
    ev_load_carrier = f"EV_{ev_type}_load"
    add_carrier_if_missing(n, ev_load_carrier)
    ev_load_bus = nodes + f" EV_{ev_type}_load"
    n.add("Bus", nodes, suffix=f" EV_{ev_type}_load", carrier=ev_load_carrier)

    # --- 3. EV load ---
    n.add(
        "Load",
        nodes,
        suffix=f" EV_{ev_type}_load",
        bus=ev_load_bus,
        carrier=ev_load_carrier,
        p_set=p_set.loc[n.snapshots, nodes],
    )

    # --- 4. Direct charger: AC -> Load ---
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

    # Passenger EVs
    if snakemake.config.get("transport", {}).get("passenger_bev", {}).get("on", True):
        charging = pd.read_csv(
            snakemake.input.transport_demand_passenger, index_col=0, parse_dates=True
        )
        opts = snakemake.config["transport"]["passenger_bev"]
        attach_EV_components(network, charging, nodes, opts, "passenger")

    # Freight EVs
    if snakemake.config.get("transport", {}).get("freight_bev", {}).get("on", True):
        charging = pd.read_csv(
            snakemake.input.transport_demand_freight, index_col=0, parse_dates=True
        )
        opts = snakemake.config["transport"]["freight_bev"]
        attach_EV_components(network, charging, nodes, opts, "freight")

    network.export_to_netcdf(snakemake.output.network)
