"""Helper functions for pypsa network handling"""

import os
import pandas as pd
import numpy as np
import logging
import pytz
import xarray as xr
from typing import Optional, Any, Union
from pathlib import Path

import pypsa

# get root logger
logger = logging.getLogger()

# Type aliases
ComponentName = str
ConstraintType = str
DualValue = float | np.ndarray | xr.DataArray | list[float] | dict[str, float] | pd.Series | pd.DataFrame

# Simplified component mapping - only essential mappings
COMPONENT_MAPPING: dict[str, str] = {
    'generator': 'generators',
    'link': 'links',
    'line': 'lines',
    'store': 'stores',
    'storageunit': 'storage_units',
    'bus': 'buses',
    'globalconstraint': 'global_constraints',
    'load': 'loads',
    'transformer': 'transformers',
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
        pd.DataFrame: The LCOE for the network with or without brownfield CAPEX, MV and delta

    """
    if "groupby" in kwargs:
        grouper = kwargs.pop("groupby")

    # store marginal costs we will manipulate to merge gas costs
    original_marginal_costs = n.links.marginal_cost.copy()
    # TODO remve the != Inner Mongolia gas, there for backward compat with a bug
    gas_links = n.links.query("carrier.str.contains('gas') & bus0 != 'Inner Mongolia gas'").index
    if not gas_links.empty:
        # 检查gas fuel发电机是否存在
        gas_fuel_generators = n.links.loc[gas_links, "bus0"] + " fuel"
        existing_gas_fuel = gas_fuel_generators[gas_fuel_generators.isin(n.generators.index)]
        if not existing_gas_fuel.empty:
            fuel_costs = n.generators.loc[existing_gas_fuel].marginal_cost.values
            # eta is applied by statistics
            n.links.loc[existing_gas_fuel.index.str.replace(" fuel", ""), "marginal_cost"] += fuel_costs
    # TODO same with BECCS? & other links?

    rev = n.statistics.revenue(groupby=grouper, **kwargs)
    opex = n.statistics.opex(groupby=grouper, **kwargs)
    capex = n.statistics.expanded_capex(groupby=grouper, **kwargs)
    tot_capex = n.statistics.capex(groupby=grouper, **kwargs)
    supply = n.statistics.supply(groupby=grouper, **kwargs)
    # restore original marginal costs
    n.links.marginal_cost = original_marginal_costs

    # incase no grouper was specified, get different levels
    if grouper is None:
        supply = supply.groupby(level=[0, 1]).sum()

    outputs = pd.concat(
        [opex, capex, tot_capex, rev, supply],
        axis=1,
        keys=["OPEX", "CAPEX", "CAPEX_wBROWN", "Revenue", "supply"],
    ).fillna(0)
    outputs["rev-costs"] = outputs.apply(lambda row: row.Revenue - row.CAPEX - row.OPEX, axis=1)
    outputs["LCOE"] = (outputs.CAPEX + outputs.OPEX) / outputs.supply
    outputs["LCOE_wbrownfield"] = (outputs.CAPEX_wBROWN + outputs.OPEX) / outputs.supply
    outputs["MV"] = outputs.apply(lambda row: row.Revenue / row.supply, axis=1)
    outputs["profit_pu"] = outputs["rev-costs"] / outputs.supply
    outputs.sort_values("profit_pu", ascending=False, inplace=True)

    return outputs[outputs.supply > 0]


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
            costs["marginal"].rename({t: t + " marginal" for t in conv_techs}),
            fill_value=0.0,
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


def mock_solve(n: pypsa.Network) -> pypsa.Network:
    """Mock the solving step for tests

    Args:
        n (pypsa.Network): the network object
    """
    for c in n.iterate_components(components=["Generator", "Link", "Store", "LineType"]):
        opt_cols = [col for col in c.df.columns if col.endswith("opt")]
        base_cols = [col.split("_opt")[0] for col in opt_cols]
        c.df[opt_cols] = c.df[base_cols]
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
    Process dual variables from network model and add them to network object.
    
    This function parses dual variables from the model, maps them to corresponding network components,
    and adds them as component attributes. Complex dual variables that cannot be directly mapped
    are stored in the network.duals dictionary.
    
    Args:
        network: PyPSA network object containing model and dual variables
        
    Returns:
        pypsa.Network: Processed network object with dual variable attributes
        
    Raises:
        AttributeError: If network object lacks 'model' or 'dual' attributes
        ValueError: If dual variable format is incorrect
        
    Example:
        >>> n = pypsa.Network("network.nc")
        >>> n = process_dual_variables(n)
        >>> # Now can access dual variables
        >>> shadow_price = n.generators.mu_p_nom_upper
        >>> nodal_balance = n.buses.mu_nodal_balance
    """
    if not hasattr(network, "model") or not hasattr(network.model, "dual"):
        raise AttributeError("Network object must have 'model' and 'model.dual' attributes")
    
    if network.model.dual is None or len(network.model.dual) == 0:
        logger.info("No dual variables found in network model")
        return network
    
    # Get all dual variables
    all_duals = pd.Series(network.model.dual)
    processed_duals: dict[str, Any] = {}
    
    for dual_name, dual_value in all_duals.items():
        try:
            _process_single_dual_variable(network, dual_name, dual_value, processed_duals)
        except Exception as e:
            logger.warning(f"Failed to process dual variable '{dual_name}': {str(e)}")
            continue
    
    # Add processed dual variables to network object
    network.duals = processed_duals
    logger.info(f"Successfully processed {len(processed_duals)} dual variables")
    
    return network


def _process_single_dual_variable(
    network: pypsa.Network,
    dual_name: str,
    dual_value: DualValue,
    processed_duals_storage: dict[str, Any],
) -> None:
    """Processes a single dual variable and attempts to map it to a network component."""
    
    # Safely partition the dual name into component type and constraint type
    component_type_raw, _, constraint_type = dual_name.partition("-")
    
    if not constraint_type:
        logger.warning(f"Invalid dual name format (no '-' found): '{dual_name}'. Skipping.")
        return

    component_type = component_type_raw.lower()
    component_name = COMPONENT_MAPPING.get(component_type)

    # Fallback to using the raw component type if not found in mapping
    if component_name is None:
        component_name = component_type
        if not hasattr(network, component_name):
            logger.warning(f"Component type '{component_type}' from '{dual_name}' not mapped and not found as attribute in network. Skipping.")
            return

    if not hasattr(network, component_name):
        logger.warning(f"Network attribute '{component_name}' not found for dual '{dual_name}'. Skipping.")
        return
    
    component_obj = getattr(network, component_name)
    
    # Generate attribute name, simplifying square brackets
    attr_name = f"mu_{constraint_type.replace('[', '_').replace(']', '')}"
    
    # Process based on dual value type
    try:
        if np.isscalar(dual_value):
            _add_scalar_dual(component_obj, attr_name, float(dual_value))
        elif isinstance(dual_value, xr.DataArray):
            _add_xarray_dual(component_obj, attr_name, dual_value)
        elif isinstance(dual_value, np.ndarray) and dual_value.ndim == 1:
            _add_1d_numpy_dual(component_obj, attr_name, dual_value)
        else:
            # Store complex or unhandled types directly in network.duals
            processed_duals_storage[dual_name] = dual_value
            logger.debug(f"Stored complex dual '{dual_name}' (type: {type(dual_value)}) in network.duals.")
    except Exception as e:
        # Catch any error during processing and store the original value
        logger.warning(f"Error processing dual '{dual_name}' (type: {type(dual_value)}): {e}. Storing original value in network.duals.")
        processed_duals_storage[dual_name] = dual_value


def _add_1d_numpy_dual(
    component_obj: pd.DataFrame,
    attr_name: str,
    value: np.ndarray,
) -> None:
    """Adds a 1D numpy array dual variable to the component DataFrame."""
    if len(value) == len(component_obj):
        component_obj[attr_name] = pd.Series(value, index=component_obj.index)
        logger.debug(f"Added 1D numpy dual: '{attr_name}'.")
    else:
        raise ValueError(f"Index length mismatch for '{attr_name}': data length {len(value)}, component length {len(component_obj)}.")


def _add_xarray_dual(
    component_obj: pd.DataFrame,
    attr_name: str,
    value: xr.DataArray,
) -> None:
    """Adds an xarray DataArray dual variable, attempting robust alignment."""
    try:
        # Attempt to convert to pandas Series, which handles basic index alignment
        value_series = value.to_pandas()
        
        # Reindex to match the component's index, NaNs for mismatches
        aligned_series = value_series.reindex(component_obj.index)
        
        component_obj[attr_name] = aligned_series
        logger.debug(f"Added aligned xarray dual: '{attr_name}'.")
        
    except Exception as e:
        # If any error occurs during alignment, raise it to be caught by the caller
        raise ValueError(f"Failed to align xarray dual '{attr_name}': {e}") from e


def _add_scalar_dual(component_obj: pd.DataFrame, attr_name: str, value: float) -> None:
    """Adds a scalar dual variable to the component DataFrame."""
    # Assign scalar to all rows, ensuring it's a float
    component_obj[attr_name] = float(value)
    logger.debug(f"Added scalar dual: '{attr_name}'.")


def _sanitize_filename(filename: str) -> str:
    """Sanitizes a string for use as a filename, replacing invalid characters with underscores."""
    invalid_chars = ['[', ']', '-', '.', ':', '/', '\\', '*', '?', '"', '<', '>', '|']
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    # Remove consecutive underscores resulting from replacements
    safe_name = '_'.join(filter(None, safe_name.split('_')))
    return safe_name


# --- Simple dual export function ---
def export_duals_simple(dual_data: dict, output_dir: Path) -> None:
    """
    Simple export of dual variables to CSV files.
    
    Args:
        dual_data: Dictionary of dual variables {name: value}
        output_dir: Directory to save CSV files
    """
    if not dual_data:
        logger.info("No dual variables to export.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting {len(dual_data)} dual variables to '{output_dir}'")
    
    for dual_name, dual_value in dual_data.items():
        safe_name = _sanitize_filename(dual_name)
        filepath = output_dir / f"{safe_name}.csv"
        
        try:
            # Convert to pandas Series for consistent CSV output with proper indexing
            if isinstance(dual_value, (dict, pd.Series)):
                # Preserve original index if available
                if hasattr(dual_value, 'index'):
                    pd.Series(dual_value).to_csv(filepath, header=True)
                else:
                    pd.Series(dual_value).to_csv(filepath, header=True)
            elif isinstance(dual_value, (list, tuple)):
                pd.Series(dual_value, name=dual_name).to_csv(filepath, header=True)
            elif np.isscalar(dual_value):
                pd.Series([dual_value], name=dual_name).to_csv(filepath, header=True)
            elif isinstance(dual_value, np.ndarray):
                # Try to preserve original shape and create meaningful index
                if dual_value.ndim == 1:
                    pd.Series(dual_value, name=dual_name).to_csv(filepath, header=True)
                else:
                    pd.Series(dual_value.flatten(), name=dual_name).to_csv(filepath, header=True)
            elif isinstance(dual_value, xr.DataArray):
                # Preserve xarray coordinates as index if possible
                try:
                    if hasattr(dual_value, 'coords') and dual_value.coords:
                        # Try to use coordinates as index
                        series = dual_value.to_pandas()
                        series.to_csv(filepath, header=True)
                    else:
                        pd.Series(dual_value.values.flatten(), name=dual_name).to_csv(filepath, header=True)
                except:
                    pd.Series(dual_value.values.flatten(), name=dual_name).to_csv(filepath, header=True)
            else:
                # Fallback for other types
                pd.Series([str(dual_value)], name=dual_name).to_csv(filepath, header=True)
                
        except Exception as e:
            logger.warning(f"Failed to export dual variable '{dual_name}': {str(e)}")


