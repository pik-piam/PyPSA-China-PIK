"""Helper functions for pypsa network handling"""

import os
import pandas as pd
import numpy as np
import logging
import pytz
import xarray as xr
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

import pypsa

# get root logger
logger = logging.getLogger()

# 类型别名定义
ComponentName = str
ConstraintType = str
DualValue = Union[float, np.ndarray, xr.DataArray, List[float], Dict[str, float]]

# 组件映射常量
COMPONENT_MAPPING: Dict[str, str] = {
    'generator': 'generators',
    'generators': 'generators',
    'link': 'links',
    'links': 'links',
    'line': 'lines',
    'lines': 'lines',
    'store': 'stores',
    'stores': 'stores',
    'storageunit': 'storage_units',
    'storageunits': 'storage_units',
    'storage_units': 'storage_units',
    'bus': 'buses',
    'buss': 'buses',
    'buses': 'buses',
    'globalconstraint': 'global_constraints',
    'globalconstraints': 'global_constraints',
    'global_constraints': 'global_constraints',
}


def get_location_and_carrier(
    n: pypsa.Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """Get component location and carrier.

    Args:
        n (pypsa.Network): the network object
        c (str): component name
        port (str, optional): port name. Defaults to "".
        nice_names (bool, optional): use nice names. Defaults to True.

    Returns:
        list[pd.Series]: list of location and carrier series
    """

    # bus = f"bus{port}"
    bus, carrier = pypsa.statistics.get_bus_and_carrier(n, c, port, nice_names=nice_names)
    location = bus.map(n.buses.location).rename("location")
    return [location, carrier]


def assign_locations(n: pypsa.Network):
    """Assign location based on the node location

    Args:
        n (pypsa.Network): the pypsa network object
    """
    for c in n.iterate_components(n.one_port_components):
        c.df["location"] = c.df.bus.map(n.buses.location)

    for c in n.iterate_components(n.branch_components):
        # use bus1 and bus2
        c.df["_loc1"] = c.df.bus0.map(n.buses.location)
        c.df["_loc2"] = c.df.bus1.map(n.buses.location)
        # if only one of buses is in the ntwk node list, make it a loop to the location
        c.df["_loc2"] = c.df.apply(lambda row: row._loc1 if row._loc2 == "" else row._loc2, axis=1)
        c.df["_loc1"] = c.df.apply(lambda row: row._loc2 if row._loc1 == "" else row._loc1, axis=1)
        # add location to loops. Links between nodes have ambiguos location
        c.df["location"] = c.df.apply(
            lambda row: row._loc1 if row._loc1 == row._loc2 else "", axis=1
        )
        c.df.drop(columns=["_loc1", "_loc2"], inplace=True)


def aggregate_p(n: pypsa.Network) -> pd.Series:
    """Make a single series for generators, storage units, loads, and stores power,
    summed over all carriers

    Args:
        n (pypsa.Network): the network object

    Returns:
        pd.Series: the aggregated p data
    """
    return pd.concat(
        [
            n.generators_t.p.sum().groupby(n.generators.carrier).sum(),
            n.storage_units_t.p.sum().groupby(n.storage_units.carrier).sum(),
            n.stores_t.p.sum().groupby(n.stores.carrier).sum(),
            -n.loads_t.p.sum().groupby(n.loads.carrier).sum(),
        ]
    )


def calc_lcoe(
    n: pypsa.Network, grouper=pypsa.statistics.get_carrier_and_bus_carrier, **kwargs
) -> pd.DataFrame:
    """calculate the LCOE for the network: (capex+opex)/supply.

    Args:
        n (pypsa.Network): the network for which LCOE is to be calaculated
        grouper (function | list, optional): function to group the data in network.statistics.
                Overwritten if groupby is passed in kwargs.
                Defaults to pypsa.statistics.get_carrier_and_bus_carrier.
        **kwargs: other arguments to be passed to network.statistics
    Returns:
        pd.DataFrame: The LCOE for the network  with or without brownfield CAPEX. MV and delta

    """
    if "groupby" in kwargs:
        grouper = kwargs.pop("groupby")

    rev = n.statistics.revenue(groupby=grouper, **kwargs)
    opex = n.statistics.opex(groupby=grouper, **kwargs)
    capex = n.statistics.expanded_capex(groupby=grouper, **kwargs)
    tot_capex = n.statistics.capex(groupby=grouper, **kwargs)
    supply = n.statistics.supply(groupby=grouper, **kwargs)

    profits = pd.concat(
        [opex, capex, tot_capex, rev, supply],
        axis=1,
        keys=["OPEX", "CAPEX", "CAPEX_wBROWN", "Revenue", "supply"],
    ).fillna(0)
    profits["rev-costs"] = profits.apply(lambda row: row.Revenue - row.CAPEX - row.OPEX, axis=1)
    profits["LCOE"] = profits.apply(lambda row: (row.CAPEX + row.OPEX) / row.supply, axis=1)
    profits["LCOE_wbrownfield"] = profits.apply(
        lambda row: (row.CAPEX_wBROWN + row.OPEX) / row.supply, axis=1
    )
    profits["MV"] = profits.apply(lambda row: row.Revenue / row.supply, axis=1)
    profits["profit_pu"] = profits["rev-costs"] / profits.supply
    profits.sort_values("profit_pu", ascending=False, inplace=True)

    return profits[profits.supply > 0]


# TODO is thsi really good? useful?
# TODO make a standard apply/str op instead ofmap in add_electricity.sanitize_carriers
def rename_techs(label: str, nice_names: dict | pd.Series = None) -> str:
    """Rename technology labels for better readability. Removes some prefixes
        and renames if certain conditions  defined in function body are met.

    Args:
        label (str): original technology label
        nice_names (dict, optional): nice names that will overwrite defaults

    Returns:
        str: renamed tech label
    """

    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "land transport fuel cell",
        "land transport oil": "land transport oil",
        "oil shipping": "shipping oil",
        # "CC": "CC"
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new
    # import here to not mess with snakemake
    from constants import NICE_NAMES_DEFAULT

    names_new = NICE_NAMES_DEFAULT.copy()
    names_new.update(nice_names)
    for old, new in names_new.items():
        if old == label:
            label = new
    return label


def aggregate_costs(
    n: pypsa.Network,
    flatten=False,
    opts: dict = None,
    existing_only=False,
) -> pd.Series | pd.DataFrame:
    """LEGACY FUNCTION used in pypsa heating plots - unclear what it does
    
    Args:
        n (pypsa.Network): the network object
        flatten (bool, optional):merge capex and marginal ? Defaults to False.
        opts (dict, optional): options for the function. Defaults to None.
        existing_only (bool, optional): use _nom instead of nom_opt. Defaults to False."""

    components = dict(
        Link=("p_nom", "p0"),
        Generator=("p_nom", "p"),
        StorageUnit=("p_nom", "p"),
        Store=("e_nom", "p"),
        Line=("s_nom", None),
        Transformer=("s_nom", None),
    )

    costs = {}
    for c, (p_nom, p_attr) in zip(
        n.iterate_components(components.keys(), skip_empty=True), components.values()
    ):
        if not existing_only:
            p_nom += "_opt"
        costs[(c.list_name, "capital")] = (
            (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        )
        if p_attr is not None:
            p = c.dynamic[p_attr].sum()
            if c.name == "StorageUnit":
                p = p.loc[p > 0]
            costs[(c.list_name, "marginal")] = (p * c.df.marginal_cost).groupby(c.df.carrier).sum()
    costs = pd.concat(costs)

    if flatten:
        assert opts is not None
        conv_techs = opts["conv_techs"]

        costs = costs.reset_index(level=0, drop=True)
        costs = costs["capital"].add(
            costs["marginal"].rename({t: t + " marginal" for t in conv_techs}), fill_value=0.0
        )

    return costs


def calc_atlite_heating_timeshift(date_range: pd.date_range, use_last_ts=False) -> int:
    """Imperfect function to calculate the heating time shift for atlite
    Atlite is in xarray, which does not have timezone handling. Adapting the UTC ERA5 data
    to the network local time, is therefore limited to a single shift, which is based on the first
    entry of the time range. For a whole year, in the northern Hemisphere -> winter

    Args:
        date_range (pd.date_range): the date range for which the shift is calc
        use_last_ts (bool, optional): use last instead of first. Defaults to False.

    Returns:
        int: a single timezone shift to utc in hours
    """
    # import constants here to not interfere with snakemake
    from constants import TIMEZONE

    idx = 0 if not use_last_ts else -1
    return pytz.timezone(TIMEZONE).utcoffset(date_range[idx]).total_seconds() / 3600


def is_leap_year(year: int) -> bool:
    """Determine whether a year is a leap year.
    Args:
        year (int): the year
    Returns:
        bool: True if leap year, False otherwise"""
    year = int(year)
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def load_network_for_plots(
    network_file: os.PathLike,
    tech_costs: os.PathLike,
    config: dict,
    cost_year: int,
    combine_hydro_ps=True,
) -> pypsa.Network:
    """load network object (LEGACY FUNCTION for heat plot)

    Args:
        network_file (os.PathLike): the path to the network file
        tech_costs (os.PathLike): the path to the costs file
        config (dict): the snamekake config
        cost_year (int): the year for the costs
        combine_hydro_ps (bool, optional): combine the hydro & PHS carriers. Defaults to True.

    Returns:
        pypsa.Network: the network object
    """

    from add_electricity import update_transmission_costs, load_costs

    n = pypsa.Network(network_file)

    n.loads["carrier"] = n.loads.bus.map(n.buses.carrier) + " load"
    n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)

    n.links["carrier"] = n.links.bus0.map(n.buses.carrier) + "-" + n.links.bus1.map(n.buses.carrier)
    n.lines["carrier"] = "AC line"
    n.transformers["carrier"] = "AC transformer"

    # n.lines['s_nom'] = n.lines['s_nom_min']
    # n.links['p_nom'] = n.links['p_nom_min']

    if combine_hydro_ps:
        n.storage_units.loc[n.storage_units.carrier.isin({"PHS", "hydro"}), "carrier"] = "hydro+PHS"

    # if the carrier was not set on the heat storage units
    # bus_carrier = n.storage_units.bus.map(n.buses.carrier)
    # n.storage_units.loc[bus_carrier == "heat","carrier"] = "water tanks"

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, Nyears)
    update_transmission_costs(n, costs)

    return n


def make_periodic_snapshots(
    year: int,
    freq: int,
    start_day_hour="01-01 00:00:00",
    end_day_hour="12-31 23:00",
    bounds="both",
    end_year: int = None,
    tz: str = None,
) -> pd.date_range:
    """Centralised function to make regular snapshots.
    REMOVES LEAP DAYS

    Args:
        year (int): start time stamp year (end year if end_year None)
        freq (int): snapshot frequency in hours
        start_day_hour (str, optional): Day and hour. Defaults to "01-01 00:00:00".
        end_day_hour (str, optional): _description_. Defaults to "12-31 23:00".
        bounds (str, optional):  bounds behaviour (pd.data_range) . Defaults to "both".
        tz (str, optional): timezone (UTC, None or a timezone). Defaults to None (naive).
        end_year (int, optional): end time stamp year. Defaults to None (use year).

    Returns:
        pd.date_range: the snapshots for the network
    """
    if not end_year:
        end_year = year

    # do not apply freq yet or get inconsistencies with leap years
    snapshots = pd.date_range(
        f"{int(year)}-{start_day_hour}",
        f"{int(end_year)}-{end_day_hour}",
        freq="1h",
        inclusive=bounds,
        tz=tz,
    )
    if is_leap_year(int(year)):
        snapshots = snapshots[~((snapshots.month == 2) & (snapshots.day == 29))]
    freq_hours = int("".join(filter(str.isdigit, str(freq))))
    return snapshots[::freq_hours]  # every freq hour


def shift_profile_to_planning_year(data: pd.DataFrame, planning_yr: int | str) -> pd.DataFrame:
    """Shift the profile to the planning year - this harmonises weather and network timestamps
       which is needed for pandas loc operations
    Args:
        data (pd.DataFrame): profile data, for 1 year
        planning_yr (int): planning year
    Returns:
        pd.DataFrame: shifted profile data
    Raises:
        ValueError: if the profile data crosses years
    """

    years = data.index.year.unique()
    if not len(years) == 1:
        raise ValueError(f"Data should be for one year only but got {years}")

    ref_year = years[0]
    # remove all planning year leap days
    if is_leap_year(ref_year):  # and not is_leap_year(planning_yr):
        data = data.loc[~((data.index.month == 2) & (data.index.day == 29))]

    # TODO CONSIDER CHANGING METHOD TO REINDEX inex = daterange w new year method = FORWARDFILL
    data.index = data.index.map(lambda t: t.replace(year=int(planning_yr)))

    return data


def update_p_nom_max(n: pypsa.Network) -> None:
    # if extendable carriers (solar/onwind/...) have capacity >= 0,
    # e.g. existing assets from the OPSD project are included to the network,
    # the installed capacity might exceed the expansion limit.
    # Hence, we update the assumptions.

    n.generators.p_nom_max = n.generators[["p_nom_min", "p_nom_max"]].max(1)


def process_dual_variables(network: pypsa.Network) -> pypsa.Network:
    """
    处理网络模型的对偶变量并将其添加到网络对象中。
    
    该函数解析模型的对偶变量，将其映射到相应的网络组件，并添加为组件的属性。
    对于无法直接映射的复杂对偶变量，将其存储在network.duals字典中。
    
    Args:
        network: 包含模型和对偶变量的PyPSA网络对象
        
    Returns:
        pypsa.Network: 处理后的网络对象，包含对偶变量属性
        
    Raises:
        AttributeError: 如果网络对象没有model或dual属性
        ValueError: 如果对偶变量格式不正确
        
    Example:
        >>> n = pypsa.Network("network.nc")
        >>> n = process_dual_variables(n)
        >>> # 现在可以访问对偶变量
        >>> shadow_price = n.generators.mu_p_nom_upper
        >>> nodal_balance = n.buses.mu_nodal_balance
    """
    if not hasattr(network, "model") or not hasattr(network.model, "dual"):
        raise AttributeError("Network object must have 'model' and 'model.dual' attributes")
    
    if network.model.dual is None or len(network.model.dual) == 0:
        logger.info("No dual variables found in network model")
        return network
    
    # 获取所有对偶变量
    all_duals = pd.Series(network.model.dual)
    processed_duals: Dict[str, Any] = {}
    
    for dual_name, dual_value in all_duals.items():
        try:
            _process_single_dual_variable(network, dual_name, dual_value, processed_duals)
        except Exception as e:
            logger.warning(f"Failed to process dual variable '{dual_name}': {str(e)}")
            continue
    
    # 将处理过的对偶变量添加到网络对象
    network.duals = processed_duals
    logger.info(f"Successfully processed {len(processed_duals)} dual variables")
    
    return network


def _process_single_dual_variable(
    network: pypsa.Network, 
    dual_name: str, 
    dual_value: DualValue, 
    processed_duals: Dict[str, Any]
) -> None:
    """
    处理单个对偶变量。
    
    Args:
        network: PyPSA网络对象
        dual_name: 对偶变量名称
        dual_value: 对偶变量值
        processed_duals: 存储无法直接映射的对偶变量的字典
    """
    parts = dual_name.split("-")
    if len(parts) < 2:
        logger.warning(f"Invalid dual variable name format: {dual_name}")
        return
    
    component_type = parts[0].lower()
    constraint_type = "-".join(parts[1:])
    
    # 获取正确的组件名称
    component_name = COMPONENT_MAPPING.get(component_type, component_type)
    if not hasattr(network, component_name):
        logger.warning(f"Component '{component_name}' not found in network")
        return
    
    # 生成属性名称
    attr_name = f"mu_{constraint_type.replace('[', '_').replace(']', '')}"
    component_obj = getattr(network, component_name)
    
    # 根据对偶变量类型进行处理
    if np.isscalar(dual_value):
        _add_scalar_dual(component_obj, attr_name, dual_value)
    elif len(np.shape(dual_value)) == 1:
        _add_1d_dual(component_obj, attr_name, dual_value, processed_duals, dual_name)
    else:
        _add_multidimensional_dual(component_obj, attr_name, dual_value, processed_duals, dual_name)


def _add_scalar_dual(component_obj: pd.DataFrame, attr_name: str, value: float) -> None:
    """为组件添加标量对偶变量。"""
    component_obj[attr_name] = pd.Series(value, index=component_obj.index)
    logger.debug(f"Added scalar dual variable: {attr_name} = {value}")


def _add_1d_dual(
    component_obj: pd.DataFrame, 
    attr_name: str, 
    value: Union[np.ndarray, xr.DataArray], 
    processed_duals: Dict[str, Any], 
    dual_name: str
) -> None:
    """为组件添加一维对偶变量。"""
    if isinstance(value, xr.DataArray) and hasattr(value, 'coords'):
        _add_xarray_dual(component_obj, attr_name, value)
    elif len(value) == len(component_obj):
        component_obj[attr_name] = pd.Series(value, index=component_obj.index)
        logger.debug(f"Added 1D dual variable: {attr_name}")
    else:
        processed_duals[f"{component_obj.name}_{attr_name}"] = value
        logger.warning(f"Could not match indices for {attr_name}, stored in processed_duals")


def _add_xarray_dual(component_obj: pd.DataFrame, attr_name: str, value: xr.DataArray) -> None:
    """处理xarray类型的对偶变量。"""
    orig_indices = value.coords[value.dims[0]].values
    value_dict = {str(idx): val for idx, val in zip(orig_indices, value.values)}
    
    series = pd.Series(index=component_obj.index)
    for idx in component_obj.index:
        if idx in value_dict:
            series[idx] = value_dict[idx]
        elif str(idx) in value_dict:
            series[idx] = value_dict[str(idx)]
    
    component_obj[attr_name] = series
    logger.debug(f"Added xarray dual variable: {attr_name}")


def _add_multidimensional_dual(
    component_obj: pd.DataFrame, 
    attr_name: str, 
    value: np.ndarray, 
    processed_duals: Dict[str, Any], 
    dual_name: str
) -> None:
    """处理多维对偶变量。"""
    processed_duals[f"{component_obj.name}_{attr_name}"] = value
    logger.warning(f"Multi-dimensional dual variable '{dual_name}' stored in processed_duals")


def export_duals_to_csv_by_year(
    network: pypsa.Network, 
    current_year: Union[int, str], 
    output_base_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    将网络模型的对偶变量导出为按年份组织的CSV文件。
    
    该函数将网络模型中的所有对偶变量导出为CSV文件，按年份组织在指定目录下。
    支持多种数据类型，包括标量、数组、xarray对象等。
    
    Args:
        network: 包含模型和对偶变量的PyPSA网络对象
        current_year: 当前模拟的年份
        output_base_dir: 输出基础目录。如果为None，将尝试从网络文件路径推断
        
    Raises:
        AttributeError: 如果网络对象没有model或dual属性
        OSError: 如果无法创建输出目录或写入文件
        
    Example:
        >>> n = pypsa.Network("network.nc")
        >>> export_duals_to_csv_by_year(n, 2025, "/path/to/results/dual")
        >>> # 文件将保存在 /path/to/results/dual/dual_values_raw_2025/
    """
    if not hasattr(network, "model") or not hasattr(network.model, "dual"):
        raise AttributeError("Network object must have 'model' and 'model.dual' attributes")
    
    if network.model.dual is None or len(network.model.dual) == 0:
        logger.info("No dual variables to export")
        return
    
    # 确定输出目录
    output_dir = _determine_output_directory(network, current_year, output_base_dir)
    logger.info(f"Exporting {len(network.model.dual)} dual variables to '{output_dir}'")
    
    # 导出所有对偶变量
    for dual_name, dual_value in network.model.dual.items():
        _export_single_dual_variable(dual_name, dual_value, output_dir)


def _determine_output_directory(
    network: pypsa.Network, 
    current_year: Union[int, str], 
    output_base_dir: Optional[Union[str, Path]]
) -> Path:
    """
    确定输出目录路径。
    
    Args:
        network: PyPSA网络对象
        current_year: 当前年份
        output_base_dir: 可选的输出基础目录
        
    Returns:
        Path: 完整的输出目录路径
    """
    if output_base_dir is None:
        # 尝试从网络文件路径推断结果目录
        network_path = getattr(network, '_path', None)
        if network_path and 'postnetworks' in str(network_path):
            results_dir = Path(network_path).parent.parent
        else:
            results_dir = Path.cwd() / 'results'
        
        output_base_dir = results_dir / 'dual'
    else:
        output_base_dir = Path(output_base_dir)
    
    output_dir = output_base_dir / f"dual_values_raw_{current_year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def _export_single_dual_variable(
    dual_name: str, 
    dual_value: DualValue, 
    output_dir: Path
) -> None:
    """
    导出单个对偶变量到CSV文件。
    
    Args:
        dual_name: 对偶变量名称
        dual_value: 对偶变量值
        output_dir: 输出目录路径
    """
    # 清理文件名
    safe_name = _sanitize_filename(dual_name)
    filepath = output_dir / f"{safe_name}.csv"
    
    try:
        if np.isscalar(dual_value):
            _export_scalar_dual(dual_name, dual_value, filepath)
        elif hasattr(dual_value, 'to_pandas'):
            _export_xarray_dual(dual_value, filepath)
        elif isinstance(dual_value, np.ndarray):
            _export_numpy_dual(dual_value, filepath)
        elif isinstance(dual_value, (list, tuple)):
            pd.Series(dual_value).to_csv(filepath, header=False)
        elif isinstance(dual_value, dict):
            pd.Series(dual_value).to_csv(filepath)
        else:
            _export_generic_dual(dual_value, filepath)
            
    except Exception as e:
        logger.warning(f"Failed to export dual variable '{dual_name}': {str(e)}")


def _sanitize_filename(filename: str) -> str:
    """清理文件名，移除或替换特殊字符。"""
    invalid_chars = ['[', ']', '-', '.', ':', '/', '\\', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def _export_scalar_dual(dual_name: str, value: float, filepath: Path) -> None:
    """导出标量对偶变量。"""
    pd.Series([value], index=[dual_name]).to_csv(filepath, header=False)


def _export_xarray_dual(value: xr.DataArray, filepath: Path) -> None:
    """导出xarray类型的对偶变量。"""
    try:
        value.to_pandas().to_csv(filepath)
    except Exception:
        if hasattr(value, 'values'):
            pd.Series(value.values.flatten()).to_csv(filepath, header=False)
        else:
            pd.Series(value).to_csv(filepath, header=False)


def _export_numpy_dual(value: np.ndarray, filepath: Path) -> None:
    """导出numpy数组类型的对偶变量。"""
    if value.ndim == 0:
        pd.Series([value.item()]).to_csv(filepath, header=False)
    elif value.ndim == 1:
        pd.Series(value).to_csv(filepath, header=False)
    else:
        pd.Series(value.flatten()).to_csv(filepath, header=False)


def _export_generic_dual(value: Any, filepath: Path) -> None:
    """导出通用类型的对偶变量。"""
    try:
        pd.Series(value).to_csv(filepath, header=False)
    except Exception as e:
        logger.warning(f"Could not convert dual value to pandas format: {str(e)}")