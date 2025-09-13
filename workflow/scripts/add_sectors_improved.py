"""
改进的部门耦合组件添加脚本

基于add_existing_baseyear.py和prepare_network.py的成功模式，
重新设计EV和其他部门组件的添加逻辑，确保网络可以成功求解。
"""

# SPDX-FileCopyrightText: : 2025 The PyPSA-China Authors
# SPDX-License-Identifier: MIT

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def ensure_carriers_exist(n: pypsa.Network, config: dict):
    """确保所有必要的载体都已注册到网络中

    载体的颜色和显示名称通过plot_config.yaml管理，这里只注册载体名称

    Args:
        n (pypsa.Network): PyPSA网络对象
        config (dict): 配置字典
    """
    # 基础载体列表
    carriers_to_add = [
        "EVpassenger",
        "EVfreight",
        "landEVpassenger",
        "landEVfreight",
        "BEVpassengercharger",
        "BEVfreightcharger",
    ]

    # DSM相关载体
    if config.get("transport", {}).get("passenger_bev", {}).get("dsm", False):
        carriers_to_add.extend(
            [
                "EVpassengerbattery",
                "BEVpassengerdischarger",
            ]
        )

    if config.get("transport", {}).get("freight_bev", {}).get("dsm", False):
        carriers_to_add.extend(
            [
                "EVfreightbattery",
                "BEVfreightdischarger",
            ]
        )

    # 添加缺失的载体（不设置颜色和名称，让plot_config.yaml管理）
    for carrier_name in carriers_to_add:
        if carrier_name not in n.carriers.index:
            n.add("Carrier", carrier_name)
            logger.debug(f"Added carrier: {carrier_name}")


def attach_EV_components_improved(
    n: pypsa.Network,
    avail_profile: pd.DataFrame,
    dsm_profile: pd.DataFrame,
    p_set: pd.DataFrame,
    nodes: pd.Index,
    options: dict,
    config: dict,
    type: str,
) -> None:
    """
    改进的EV组件添加函数，基于成功的网络组件添加模式

    Args:
        n (pypsa.Network): PyPSA网络对象
        avail_profile (pd.DataFrame): EV充电可用性配置文件
        dsm_profile (pd.DataFrame): DSM状态配置文件
        p_set (pd.DataFrame): EV能源需求配置文件
        nodes (pd.Index): 网络节点名称
        options (dict): EV配置参数
        config (dict): 全局配置
        type (str): 车辆类型 ("passenger" 或 "freight")
    """
    if type not in ("passenger", "freight"):
        raise ValueError("type must be 'passenger' or 'freight'")

    dsm_enabled = options["dsm"]

    # 获取节点的地理位置信息（从现有AC总线复制）
    ac_buses = n.buses.query("carrier == 'AC'")
    if ac_buses.empty:
        raise ValueError("No AC buses found in network")

    prov_centroids_x = ac_buses.loc[nodes, "x"]
    prov_centroids_y = ac_buses.loc[nodes, "y"]
    countries = (
        ac_buses.loc[nodes, "country"].values
        if "country" in ac_buses.columns
        else ["CN"] * len(nodes)
    )

    # --- 1. 计算EV数量和参数 ---
    total_energy = p_set.sum().sum()
    annual_consumption_per_ev = max(options["annual_consumption"], 1e-6)

    total_number_evs = max(total_energy / annual_consumption_per_ev, 0.0)
    node_energy_ratio = p_set.sum() / max(total_energy, 1e-6)
    number_evs = node_energy_ratio * total_number_evs

    # 确保最小值以避免数值问题
    charge_power = (number_evs * options["charge_rate"] * options["share_charger"]).clip(
        lower=0.001
    )
    battery_energy = (number_evs * options["battery_size"]).clip(lower=0.001)

    logger.info(
        f"EV {type} - DSM {'ON' if dsm_enabled else 'OFF'}, "
        f"Total energy: {total_energy:.1f} MWh, "
        f"Total EVs: {int(total_number_evs):,}"
    )

    # --- 2. 添加EV负载总线（简化命名，遵循现有模式） ---
    ev_load_suffix = f" EV{type}"  # 简化：Beijing EVpassenger
    n.add(
        "Bus",
        nodes,
        suffix=ev_load_suffix,
        x=prov_centroids_x.values,
        y=prov_centroids_y.values,
        location=nodes,
        country=countries,
        carrier=f"EV{type}",
        unit="MWh_el",
    )
    ev_load_bus = nodes + ev_load_suffix

    # --- 3. 添加DSM电池总线（如果启用） ---
    if dsm_enabled:
        carrier_name = f"EV{type}battery"
        ev_battery_suffix = f" EV{type}battery"  # 简化：Beijing EVpassengerbattery
        n.add(
            "Bus",
            nodes,
            suffix=ev_battery_suffix,  # 使用suffix参数
            x=prov_centroids_x.values,
            y=prov_centroids_y.values,
            location=nodes,
            country=countries,
            carrier=carrier_name,  # carrier保留下划线用于内部识别
            unit="MWh_el",
        )
        ev_battery_bus = nodes + ev_battery_suffix  # 用于后续引用

    # --- 4. 添加EV负载（类似于prepare_network中的电力负载添加方式） ---
    n.add(
        "Load",
        nodes,
        suffix=f" land transport EV {type}",
        bus=ev_load_bus,
        carrier=f"landEV{type}",
        p_set=p_set.loc[n.snapshots, nodes],
    )

    # --- 5. 添加组件（基于add_existing_baseyear的Link添加模式） ---
    if dsm_enabled:
        # 充电器: 电网 -> 电池（类似于CHP gas的添加方式）
        n.add(
            "Link",
            nodes,
            suffix=f" BEV {type} charger",
            bus0=nodes,  # AC电网总线
            bus1=ev_battery_bus,  # EV电池总线
            carrier=f"BEV{type}charger",
            p_nom=charge_power * options["dsm_availability"],
            p_nom_min=0.0,
            p_nom_extendable=False,
            p_max_pu=avail_profile.loc[n.snapshots, nodes].clip(0, 1),
            efficiency=1.0,
            marginal_cost=0.0,
            location=nodes,
        )

        # 放电器: 电池 -> EV负载
        n.add(
            "Link",
            nodes,
            suffix=f" BEV {type} discharger",
            bus0=ev_battery_bus,
            bus1=ev_load_bus,
            carrier=f"BEV{type}discharger",
            p_nom_extendable=True,
            p_nom_min=0.0,
            efficiency=1.0,
            marginal_cost=0.0,
            location=nodes,
        )

        # 电池储存（类似于PHS储存的添加方式）
        n.add(
            "Store",
            nodes,
            suffix=f" EV{type}battery",
            bus=ev_battery_bus,
            carrier=f"EV{type}battery",
            e_nom=battery_energy * options["dsm_availability"],
            e_nom_min=0.0,
            e_nom_extendable=False,
            e_cyclic=True,
            e_max_pu=1.0,
            e_min_pu=dsm_profile.loc[n.snapshots, nodes].clip(0, 1),
            marginal_cost=0.0,
            standing_loss=0.0,  # 避免不必要的损失
        )

        logger.info(f"✅ DSM-enabled EV {type}: {int(total_number_evs):,} vehicles with storage")

    else:
        # 直接充电器: 电网 -> EV负载（类似于简单的Generator添加）
        n.add(
            "Link",
            nodes,
            suffix=f" BEV {type} charger",
            bus0=nodes,  # AC电网总线
            bus1=ev_load_bus,  # EV负载总线
            carrier=f"BEV{type}charger",
            p_nom=charge_power,
            p_nom_min=0.0,
            p_nom_extendable=False,
            p_max_pu=1.0,
            efficiency=1.0,
            marginal_cost=0.0,
            location=nodes,
        )

        logger.info(f"✅ Direct EV {type}: {int(total_number_evs):,} vehicles, direct charging")

    logger.info(
        f"EV {type} setup complete: {'DSM storage' if dsm_enabled else 'Direct charging'} mode"
    )


def validate_network_consistency(n: pypsa.Network):
    """验证网络的一致性，确保所有组件都正确连接

    Args:
        n (pypsa.Network): 要验证的网络
    """
    issues = []

    # 检查所有Link的bus0和bus1是否存在
    for link_name, link in n.links.iterrows():
        if link.bus0 not in n.buses.index:
            issues.append(f"Link {link_name}: bus0 '{link.bus0}' not found in buses")
        if link.bus1 not in n.buses.index:
            issues.append(f"Link {link_name}: bus1 '{link.bus1}' not found in buses")
        # 只检查非空的bus2字段
        if (
            hasattr(link, "bus2")
            and pd.notna(link.bus2)
            and link.bus2 != ""
            and link.bus2 not in n.buses.index
        ):
            issues.append(f"Link {link_name}: bus2 '{link.bus2}' not found in buses")

    # 检查所有Load的bus是否存在
    for load_name, load in n.loads.iterrows():
        if load.bus not in n.buses.index:
            issues.append(f"Load {load_name}: bus '{load.bus}' not found in buses")

    # 检查所有Store的bus是否存在
    for store_name, store in n.stores.iterrows():
        if store.bus not in n.buses.index:
            issues.append(f"Store {store_name}: bus '{store.bus}' not found in buses")

    # 检查载体是否存在（排除空载体）
    all_carriers = set()
    for comp_name in ["buses", "generators", "loads", "links", "stores"]:
        comp = getattr(n, comp_name)
        if "carrier" in comp.columns:
            carriers = comp.carrier.dropna().unique()
            # 排除空字符串载体
            carriers = [c for c in carriers if c != ""]
            all_carriers.update(carriers)

    missing_carriers = all_carriers - set(n.carriers.index)
    for carrier in missing_carriers:
        issues.append(f"Carrier '{carrier}' used but not defined in n.carriers")

    # 统计信息（不作为错误）
    empty_bus2_count = len(n.links[n.links.bus2 == ""])
    empty_carrier_count = len(n.links[n.links.carrier == ""])

    logger.info("Network validation statistics:")
    logger.info(f"  - Links with empty bus2: {empty_bus2_count}")
    logger.info(f"  - Links with empty carrier: {empty_carrier_count}")

    if issues:
        logger.error("Network validation issues found:")
        for issue in issues:
            logger.error(f"  - {issue}")
        raise ValueError(f"Network validation failed with {len(issues)} issues")
    else:
        logger.info("✅ Network validation passed")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_sectors",
            planning_horizons="2030",
        )

    configure_logging(snakemake)

    logger.info("Starting improved sector-coupled network preparation")

    # 加载基础网络
    network = pypsa.Network(snakemake.input.network)
    nodes = network.buses.query("carrier == 'AC'").index

    logger.info(f"Base network loaded with {len(nodes)} AC buses")

    # 确保载体存在
    ensure_carriers_exist(network, snakemake.config)

    # 添加交通（EV）组件
    logger.info("Adding improved EV components to network")

    # 加载DSM配置文件
    dsm_profile = pd.read_csv(snakemake.input.dsm_profile, index_col=0, parse_dates=True)

    # 乘用车EV
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

        attach_EV_components_improved(
            n=network,
            avail_profile=avail_profile_pass,
            dsm_profile=dsm_profile,
            p_set=p_set_pass,
            nodes=nodes,
            options=options_pass,
            config=snakemake.config,
            type="passenger",
        )

    # 货运EV
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

        attach_EV_components_improved(
            n=network,
            avail_profile=avail_profile_freight,
            dsm_profile=dsm_profile,
            p_set=p_set_freight,
            nodes=nodes,
            options=options_freight,
            config=snakemake.config,
            type="freight",
        )

    # 验证网络一致性
    validate_network_consistency(network)

    # 保存修改后的网络
    logger.info(f"Saving improved sector-coupled network to {snakemake.output.network}")

    compression = snakemake.config.get("io", {}).get("nc_compression", None)
    network.export_to_netcdf(snakemake.output.network, compression=compression)

    # 输出网络统计信息
    logger.info("Network statistics:")
    logger.info(f"  - Buses: {len(network.buses)}")
    logger.info(f"  - Generators: {len(network.generators)}")
    logger.info(f"  - Loads: {len(network.loads)}")
    logger.info(f"  - Links: {len(network.links)}")
    logger.info(f"  - Stores: {len(network.stores)}")
    logger.info(f"  - Carriers: {len(network.carriers)}")

    logger.info("✅ Improved sector-coupled network preparation completed successfully")
