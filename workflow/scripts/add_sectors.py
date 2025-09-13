"""
核心逻辑版：向 PyPSA 网络添加电动车 (EV) 部门耦合组件
自动注册所需的 carrier，避免警告
"""

import logging
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def add_carrier_if_missing(n: pypsa.Network, carrier_name: str):
    """如果 carrier 不存在则添加"""
    if carrier_name not in n.carriers.index:
        n.add("Carrier", carrier_name)
        logger.debug(f"Carrier '{carrier_name}' added to network.")


def attach_EV_components(
    n: pypsa.Network,
    avail_profile: pd.DataFrame,
    dsm_profile: pd.DataFrame,
    p_set: pd.DataFrame,
    nodes: pd.Index,
    options: dict,
    ev_type: str,
):
    """向网络添加 EV 组件，支持直充模式和 DSM 模式"""

    if ev_type not in ("passenger", "freight"):
        raise ValueError("ev_type must be 'passenger' or 'freight'")

    dsm_enabled = options["dsm"]

    # --- 1. 规模计算 ---
    total_energy = p_set.sum().sum()
    total_number_evs = total_energy / max(options["annual_consumption"], 1e-6)
    node_ratio = p_set.sum() / max(total_energy, 1e-6)
    number_evs = node_ratio * total_number_evs

    charge_power = (number_evs * options["charge_rate"] * options["share_charger"]).clip(lower=0.001)
    battery_energy = (number_evs * options["battery_size"]).clip(lower=0.001)

    logger.info(f"EV {ev_type}: DSM {'ON' if dsm_enabled else 'OFF'}, {int(total_number_evs):,} vehicles")

    # --- 2. EV 负荷总线 ---
    ev_load_carrier = f"EV_{ev_type}_load"
    add_carrier_if_missing(n, ev_load_carrier)
    ev_load_bus = nodes + f" EV_{ev_type}_load"
    n.add("Bus", nodes, suffix=f" EV_{ev_type}_load", carrier=ev_load_carrier)

    # --- 3. DSM 电池总线（可选） ---
    if dsm_enabled:
        ev_batt_carrier = f"EV_{ev_type}_battery"
        add_carrier_if_missing(n, ev_batt_carrier)
        ev_batt_bus = nodes + f" EV_{ev_type}_battery"
        n.add("Bus", nodes, suffix=f" EV_{ev_type}_battery", carrier=ev_batt_carrier)

    # --- 4. EV 负荷 ---
    n.add(
        "Load",
        nodes,
        suffix=f" EV_{ev_type}_load",
        bus=ev_load_bus,
        carrier=ev_load_carrier,
        p_set=p_set.loc[n.snapshots, nodes],
    )

    # --- 5. 充电/放电组件 ---
    if dsm_enabled:
        # Charger: AC -> Battery
        charger_carrier = f"EV_{ev_type}_charger"
        add_carrier_if_missing(n, charger_carrier)
        n.add(
            "Link",
            nodes,
            suffix=f" EV_{ev_type}_charger",
            bus0=nodes,
            bus1=ev_batt_bus,
            carrier=charger_carrier,
            p_nom=charge_power * options["dsm_availability"],
            p_max_pu=avail_profile.loc[n.snapshots, nodes],
            efficiency=1.0,
        )

        # Discharger: Battery -> Load
        discharger_carrier = f"EV_{ev_type}_discharger"
        add_carrier_if_missing(n, discharger_carrier)
        n.add(
            "Link",
            nodes,
            suffix=f" EV_{ev_type}_discharger",
            bus0=ev_batt_bus,
            bus1=ev_load_bus,
            carrier=discharger_carrier,
            p_nom=charge_power,
            efficiency=1.0,
        )

        # Battery
        n.add(
            "Store",
            nodes,
            suffix=f" EV_{ev_type}_battery",
            bus=ev_batt_bus,
            carrier=ev_batt_carrier,
            e_nom=battery_energy * options["dsm_availability"],
            e_cyclic=True,
            e_max_pu=1.0,
            e_min_pu=dsm_profile.loc[n.snapshots, nodes],
        )
    else:
        # Direct charger: AC -> Load
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
    dsm_profile = pd.read_csv(snakemake.input.dsm_profile, index_col=0, parse_dates=True)

    # Passenger EVs
    if snakemake.config.get("transport", {}).get("passenger_bev", {}).get("on", True):
        charging = pd.read_csv(snakemake.input.transport_demand_passenger, index_col=0, parse_dates=True)
        driving = pd.read_csv(snakemake.input.driving_demand_passenger, index_col=0, parse_dates=True)
        avail = pd.read_csv(snakemake.input.avail_profile_passenger, index_col=0, parse_dates=True)
        opts = snakemake.config["transport"]["passenger_bev"]
        p_set = driving if opts["dsm"] else charging
        attach_EV_components(network, avail, dsm_profile, p_set, nodes, opts, "passenger")

    # Freight EVs
    if snakemake.config.get("transport", {}).get("freight_bev", {}).get("on", True):
        charging = pd.read_csv(snakemake.input.transport_demand_freight, index_col=0, parse_dates=True)
        driving = pd.read_csv(snakemake.input.driving_demand_freight, index_col=0, parse_dates=True)
        avail = pd.read_csv(snakemake.input.avail_profile_freight, index_col=0, parse_dates=True)
        opts = snakemake.config["transport"]["freight_bev"]
        p_set = driving if opts["dsm"] else charging
        attach_EV_components(network, avail, dsm_profile, p_set, nodes, opts, "freight")

    network.export_to_netcdf(snakemake.output.network)
