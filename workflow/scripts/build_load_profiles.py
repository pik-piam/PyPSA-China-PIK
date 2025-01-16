import logging
import atlite
import pandas as pd
import scipy as sp
import numpy as np
from scipy.optimize import curve_fit

from _helpers import configure_logging, is_leap_year, mock_snakemake
from constants import (
    PROV_NAMES,
    TIMEZONE,
    HEATING_START_TEMP,
    HEATING_LIN_SLOPE,
    HEATING_OFFET,
    UNIT_HOT_WATER_START_YEAR,
    UNIT_HOT_WATER_END_YEAR,
    START_YEAR,
    END_YEAR,
    REF_YEAR,
)

logger = logging.getLogger(__name__)
idx = pd.IndexSlice
nodes = pd.Index(PROV_NAMES)


# TODO check this is compatible w naive timestamps!
def generate_periodic_profiles(
    dt_index: pd.DatetimeIndex = None,
    col_tzs=pd.Series(index=PROV_NAMES, data=len(PROV_NAMES) * ["Shanghai"]),
    weekly_profile=range(24 * 7),
) -> pd.DataFrame:
    """Give a 24*7 long list of weekly hourly profiles, generate this
    for each country for the period dt_index, taking account of time
    zones and Summer Time.

    Args:
        dt_index (DatetimeIndex, optional): _description_. Defaults to None.
        col_tzs (pd.Series, optional): _description_. Defaults to pd.Series(index=PROV_NAMES, data=len(PROV_NAMES) * ["Shanghai"]).
        weekly_profile (_type_, optional): _description_. Defaults to range(24 * 7).

    Returns:
        pd.DataFrame: _description_
    """
    # TODO fix this profile timezone
    weekly_profile = pd.Series(weekly_profile, range(24 * 7))

    week_df = pd.DataFrame(index=dt_index, columns=col_tzs.index)
    for ct in col_tzs.index:
        week_df[ct] = [24 * dt.weekday() + dt.hour for dt in dt_index.tz_localize(None)]
        week_df[ct] = week_df[ct].map(weekly_profile)
    return week_df


def make_heat_demand_projections(
    planning_year: int, projection_name: str, ref_year=REF_YEAR
) -> float:
    """Make projections for heating demand

    Args:
        projection_name (str): name of projection
        planning_year (int): year to project to
        ref_year (int, optional): reference year. Defaults to REF_YEAR.

    Returns:
        float: scaling factor relative to base year for heating demand
    """
    years = np.array([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2060])

    if projection_name == "positive":

        def func(x, a, b, c, d):
            """Cubic polynomial fit to proj"""
            return a * x**3 + b * x**2 + c * x + d

        # heated area projection in China
        # 2060: 6.02 * 36.52 * 1e4 north population * floor_space_per_capita in city
        heated_area = np.array(
            [2742, 21263, 64645, 110766, 252056, 435668, 672205, 988209, 2198504]
        )  # 10000 m2

        # Perform curve fitting
        popt, pcov = curve_fit(func, years, heated_area)
        factor = func(int(planning_year), *popt) / func(REF_YEAR, *popt)

    elif projection_name == "constant":
        factor = 1.0

    else:
        raise ValueError(f"Invalid heating demand projection {projection_name}")
    return factor


def build_daily_heat_demand_profiles(
    atlite_heating_hr_shift: int, heat_demand_config: dict, switch_month_day: bool = True
) -> pd.DataFrame:
    """build the heat demand profile according to forecast demans

    Args:
        atlite_heating_hr_shift (int): the hour shift for heating demand, needed due to imperfect
            timezone handling in atlite
        heat_demand_config (dict): the heat demand configuration
        switch_month_day (bool, optional): whether to switch month and day in the heat_demand_config. Defaults to True.
    Returns:
        pd.DataFrame: daily heating demand with April to Sept forced to 0
    """
    with pd.HDFStore(snakemake.input.population_map, mode="r") as store:
        pop_map = store["population_gridcell_map"]

    cutout = atlite.Cutout(snakemake.input.cutout)

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    hd = cutout.heat_demand(
        matrix=pop_matrix,
        index=index,
        threshold=heat_demand_config["heating_start_temp"],
        a=heat_demand_config["heating_lin_slope"],
        constant=heat_demand_config["heating_offet"],
        hour_shift=atlite_heating_hr_shift,
    )

    daily_hd = hd.to_pandas().divide(pop_map.sum())
    # input given as dd-mm but loc as yyyy-mm-dd
    if switch_month_day:
        start_day = "{}-{}".format(*heat_demand_config["start_day"].split("-")[::-1])
        end_day = "{}-{}".format(*heat_demand_config["end_day"].split("-")[::-1])
    else:
        start_day = heat_demand_config["start_day"]
        end_day = heat_demand_config["end_day"]
    daily_hd.loc[f"{REF_YEAR}-{start_day}":f"{REF_YEAR}-{end_day}"] = 0

    return daily_hd


# TODO separate the two functions (day and yearly)
def build_hot_water_per_day(planning_horizons: int | str) -> np.array:

    with pd.HDFStore(snakemake.input.population, mode="r") as store:
        population_count = store["population"]

    unit_hot_water_start_yr = UNIT_HOT_WATER_START_YEAR
    unit_hot_water_end_yr = UNIT_HOT_WATER_END_YEAR

    if snakemake.wildcards["heating_demand"] == "positive":

        def func(x, a, b):
            return a * x + b

        x = np.array([START_YEAR, END_YEAR])
        y = np.array([unit_hot_water_start_yr, unit_hot_water_end_yr])
        popt, pcov = curve_fit(func, x, y)

        unit_hot_water = func(int(planning_horizons), *popt)

    elif snakemake.wildcards["heating_demand"] == "constant":
        unit_hot_water = unit_hot_water_start_yr

    elif snakemake.wildcards["heating_demand"] == "mean":

        def lin_func(x: np.array, a: float, b: float) -> np.array:
            return a * x + b

        x = np.array([START_YEAR, END_YEAR])
        y = np.array([unit_hot_water_start_yr, unit_hot_water_end_yr])
        popt, pcov = curve_fit(lin_func, x, y)

        unit_hot_water = (lin_func(int(planning_horizons), *popt) + UNIT_HOT_WATER_START_YEAR) / 2
    else:
        raise ValueError(f"Invalid heating demand type {snakemake.wildcards['heating_demand']}")
        # MWh per day
    hot_water_per_day = unit_hot_water * population_count / 365.0

    return hot_water_per_day


# TODO make indep of model years
def build_heat_demand_profile(
    daily_hd: pd.DataFrame,
    hot_water_per_day: pd.DataFrame,
    date_range,
    planning_horizons: int | str,
) -> tuple[pd.DataFrame, pd.DataFrame, object, pd.DataFrame]:
    """_summary_

    Args:
        daily_hd (DataFrame): _description_
        hot_water_per_day (DataFrame): _description_
        date_range (_type_): _description_
        planning_horizons (int | str): the planning year

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, object, pd.DataFrame]:
            heat, space_heat, space_heating_per_hdd, water_heat demands
    """

    h = daily_hd
    h_n = h[~h.index.duplicated(keep="first")].iloc[:-1, :]
    h_n.index = h_n.index.tz_localize(TIMEZONE)

    date_range_ref_yr = pd.date_range(
        f"{REF_YEAR}-01-01 00:00", f"{REF_YEAR}-12-31 23:00", freq=date_range.freq, tz=TIMEZONE
    )

    heat_demand_hdh = h_n.reindex(index=date_range_ref_yr, method="ffill")
    planning_year = date_range.year[0]
    if not is_leap_year(int(planning_year)):
        heat_demand_hdh = heat_demand_hdh.loc[
            ~((heat_demand_hdh.index.month == 2) & (heat_demand_hdh.index.day == 29))
        ]
    elif is_leap_year(int(planning_year)) and not is_leap_year(REF_YEAR):
        logger.warning("Leap year detected in planning year, but not in reference year")

    heat_demand_hdh.index = date_range

    intraday_profiles = pd.read_csv(snakemake.input.intraday_profiles, index_col=0)
    intraday_year_profiles = generate_periodic_profiles(
        dt_index=heat_demand_hdh.index,
        weekly_profile=(
            list(intraday_profiles["weekday"]) * 5 + list(intraday_profiles["weekend"]) * 2
        ),
    )

    space_heat_demand_total = pd.read_csv(snakemake.input.space_heat_demand, index_col=0)
    space_heat_demand_total = space_heat_demand_total * 1e6
    space_heat_demand_total = space_heat_demand_total.squeeze()

    factor = make_heat_demand_projections(
        planning_horizons, snakemake.wildcards["heating_demand"], ref_year=REF_YEAR
    )

    space_heating_per_hdd = (space_heat_demand_total * factor) / (
        heat_demand_hdh.sum() * snakemake.config["frequency"]
    )

    space_heat_demand = intraday_year_profiles.mul(heat_demand_hdh).mul(space_heating_per_hdd)
    water_heat_demand = intraday_year_profiles.mul(hot_water_per_day / 24.0)

    heat_demand = space_heat_demand + water_heat_demand

    return heat_demand, space_heat_demand, space_heating_per_hdd, water_heat_demand


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "build_load_profiles",
            heating_demand="positive",
            planning_horizons="2020",
            pathway="exponential-175",
            topology="Current+Neigbor",
        )

    configure_logging(snakemake, logger=logger)

    atlite_hour_shift = snakemake.config["atlite"]["hour_shift_heating"]
    start_day = snakemake.config["heat_demand"]["start_day"]
    end_day
    daily_hd = build_daily_heat_demand_profiles(atlite_heating_hr_shift=atlite_hour_shift)

    config = snakemake.config
    planning_horizons = snakemake.wildcards["planning_horizons"]
    hot_water_per_day = build_hot_water_per_day(planning_horizons)

    # why?
    # TODO cebtralise and align with settings
    date_range = pd.date_range(
        f"{planning_horizons}-01-01 00:00",
        f"{planning_horizons}-12-31 23:00",
        freq=snakemake.config["freq"],
        tz=TIMEZONE,
    )

    heat_demand, space_heat_demand, space_heating_per_hdd, water_heat_demand = (
        build_heat_demand_profile(
            daily_hd,
            hot_water_per_day,
            date_range,
            planning_horizons,
        )
    )

    with pd.HDFStore(snakemake.output.heat_demand_profile, mode="w", complevel=4) as store:
        store["heat_demand_profiles"] = heat_demand

    with pd.HDFStore(snakemake.output.energy_totals_name, mode="w") as store:
        store["space_heating_per_hdd"] = space_heating_per_hdd
        store["hot_water_per_day"] = hot_water_per_day

    logger.info("Heat demand profiles successfully built")
