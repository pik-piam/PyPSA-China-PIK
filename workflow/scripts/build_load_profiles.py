"""Functions for the rules to build the hourly heat and load demand profiles
- electricity load profiles are based on scaling an hourly base year profile to yearly future
    projections
- daily heating demand is based on the degree day approx (from atlite) & upscaled hourly based
  on an intraday profile (for Denmark by default, see snakefile)
"""

import logging
import os
from collections.abc import Iterable

import atlite
import numpy as np
import pandas as pd
import scipy as sp
from _helpers import configure_logging, get_cutout_params, mock_snakemake
from _pypsa_helpers import (
    calc_atlite_heating_timeshift,
    make_periodic_snapshots,
    shift_profile_to_planning_year,
)

# TODO switch from hardocded REF_YEAR to a base year?
from constants import (
    PROV_NAMES,
    REF_YEAR,
    REGIONAL_GEO_TIMEZONES,
    TIMEZONE,
)
from readers import read_yearly_load_projections
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)
idx = pd.IndexSlice
nodes = pd.Index(PROV_NAMES)

TWH2MWH = 1e6


def _extend_iea_projections(iea: pd.Series, planning_horizons) -> pd.Series:
    """Extend IEA projections to cover all planning horizons.

    Args:
        iea (pd.Series): IEA projections for hot water demand.
        planning_horizons (list): List of years to project.

    Returns:
        pd.Series: Extended IEA projections.
    """
    last_projection = iea.sort_index().iloc[-1]
    missing = set(planning_horizons) - set(iea.index)
    for year in missing:
        iea.loc[year] = last_projection
    return iea


def _read_iea_hot_water(path: os.PathLike) -> pd.Series:
    """Read the IEA hot water data

    Args:
        path (os.PathLike): path to the data

    Returns:
        pd.DataFrame: the data in MWH
    """

    iea = pd.read_csv(path, usecols=["period", "enduse", "TWH"])
    iea = iea.query("enduse == 'water_heating'")
    iea.set_index("period", inplace=True)
    return iea["TWH"] * TWH2MWH


# TODO this is really stupid since there are 5 hours shifts on the heating demand due to sun time
def downscale_time_data(
    dt_index: pd.DatetimeIndex,
    weekly_profile: Iterable,
    regional_tzs: pd.Series,
) -> pd.DataFrame:
    """Make hourly resolved data profiles based on exogenous weekdays and weekend profiles.
    This fn takes into account that the profiles are in local time and that regions may
     have different timezones.

    Args:
        dt_index (DatetimeIndex): the snapshots (in network local naive time) but hourly res.
        weekly_profile (Iterable): the weekly profile as a list of 7*24 entries.
        regional_tzs (pd.Series, optional): regional geographical timezones for profiles.
            Defaults to pd.Series(index=PROV_NAMES, data=list(REGIONAL_GEO_TIMEZONES.values())).

    Returns:
        pd.DataFrame: Regionally resolved profiles for each snapshot hour rperesented by dt_index
    """
    weekly_profile = pd.Series(weekly_profile, range(24 * 7))
    # make a dataframe with timestamps localized to the network TIMEZONE timestamps
    all_times = pd.DataFrame(
        dict(zip(PROV_NAMES, [dt_index.tz_localize(TIMEZONE)] * len(PROV_NAMES))),
        index=dt_index.tz_localize(TIMEZONE),
        columns=PROV_NAMES,
    )
    # then localize to regional time. _dt ensures index is not changed
    week_hours = all_times.apply(
        lambda col: col.dt.tz_convert(regional_tzs[col.name]).tz_localize(None)
    )
    # then convert into week hour & map to the intraday heat demand profile (based on local time)
    return week_hours.apply(lambda col: col.dt.weekday * 24 + col.dt.hour).apply(
        lambda col: col.map(weekly_profile)
    )


# TODO ADD CALIBRATION to make ti closer to actual demand for provinces that have
def build_daily_heat_demand_profiles(
    cutout: atlite.Cutout,
    pop_map: pd.DataFrame,
    heat_demand_config: dict,
    atlite_heating_hr_shift: int,
    switch_month_day: bool = True,
) -> pd.DataFrame:
    """Build the heat demand profile according to forecast demands

    Args:
        cutout (atlite.Cutout): the weather cutout object
        pop_map (pd.DataFrame): the population raster map
        heat_demand_config (dict): the heat demand configuration
        atlite_heating_hr_shift (int): the hour shift for heating demand, needed due to imperfect
            timezone handling in atlite
        switch_month_day (bool, optional): whether to switch month & day from heat_demand_config.
            Defaults to True.

    Returns:
        pd.DataFrame: regional daily heating demand with April to Sept forced to 0
    """
    atlite_year = get_cutout_params(snakemake.config)["weather_year"]

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    # TODO clarify a bit here, maybe the po_matrix should be normalised earlier?
    # unclear whether it's per cap or not
    total_hd = cutout.heat_demand(
        matrix=pop_matrix,
        index=index,
        threshold=heat_demand_config["heating_start_temp"],
        a=heat_demand_config["heating_lin_slope"],
        constant=heat_demand_config["heating_offet"],
        # hack to bring it back to local from UTC
        hour_shift=atlite_heating_hr_shift,
    )

    regonal_daily_hd = total_hd.to_pandas().divide(pop_map.sum())
    # input given as dd-mm but loc as yyyy-mm-dd
    if switch_month_day:
        start_day = "{}-{}".format(*heat_demand_config["start_day"].split("-")[::-1])
        end_day = "{}-{}".format(*heat_demand_config["end_day"].split("-")[::-1])
    else:
        start_day = heat_demand_config["start_day"]
        end_day = heat_demand_config["end_day"]
    regonal_daily_hd.loc[f"{atlite_year}-{start_day}":f"{atlite_year}-{end_day}"] = 0

    # drop leap day
    regonal_daily_hd.index = pd.to_datetime(regonal_daily_hd.index)
    regonal_daily_hd = regonal_daily_hd[
        ~((regonal_daily_hd.index.month == 2) & (regonal_daily_hd.index.day == 29))
    ]
    return regonal_daily_hd


def downscale_by_pop(total: pd.Series, population: pd.Series) -> pd.Series:
    """simple downscale by population

    Args:
        total (pd.Series): the value to downsacale
        population (pd.Series): population by node

    Returns:
        pd.Series: downscaled to nodal resolution by population
    """

    return total * population / population.sum()


# TODO return dict would be nice
def build_heat_demand_profile(
    daily_hd: pd.DataFrame,
    hot_water_per_day: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    intraday_profiles: pd.Series,
    planning_horizons: int | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Downscale the daily heat demand to hourly heat demand using pre-defined intraday profiles

    Args:
        daily_hd (DataFrame): the day resolved heat demand for each region (atlite time axis)
        hot_water_per_day (DataFrame): the day resolved hot water demand for each region
        snapshots (pd.date_range): the snapshots for the planning year
        planning_horizons (int | str): the planning year

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: space_heat_demand, domestic hot water demand
    """
    # drop leap day
    daily_hd = daily_hd[~((daily_hd.index.month == 2) & (daily_hd.index.day == 29))]
    # hourly resolution regional demand (but wrong data, it's just ffill)
    heat_demand_hourly = shift_profile_to_planning_year(
        daily_hd, planning_yr=planning_horizons
    ).reindex(index=snapshots, method="ffill")

    # ===== downscale to hourly =======
    intraday_year_profiles = (
        downscale_time_data(
            dt_index=heat_demand_hourly.index,
            weekly_profile=(
                list(intraday_profiles["weekday"]) * 5 + list(intraday_profiles["weekend"]) * 2
            ),
            regional_tzs=pd.Series(index=PROV_NAMES, data=list(REGIONAL_GEO_TIMEZONES.values())),
        )
        / 24
    )
    # intraday_year_profiles /= intraday_year_profiles.sum()

    space_heat_demand = intraday_year_profiles.mul(heat_demand_hourly)
    water_heat_demand = intraday_year_profiles.mul(hot_water_per_day)

    return space_heat_demand, water_heat_demand


def scale_degree_day_to_reference(
    heating_dd: pd.DataFrame, reg_yrly_tot: pd.DataFrame
) -> pd.DataFrame:
    """Scale the heating degree days to the reference region

    Args:
        heating_dd (pd.Series): the heating degree days for each region (atlite)
        reg_yrly_tot (DataFrame): the reference data per region

    Returns:
        pd.Series: the scaled heating degree days
    """
    norm_daily_hd = heating_dd / heating_dd.sum()
    return norm_daily_hd.mul(reg_yrly_tot)


def prepare_hourly_load_data(
    hourly_load_p: os.PathLike,
    prov_codes_p: os.PathLike,
) -> pd.DataFrame:
    """Read the hourly electricity demand data and prepare it for use in the model

    Args:
        hourly_load_p (os.PathLike, optional): raw elec data from zenodo, see readme in data.
        prov_codes_p (os.PathLike, optional): province mapping for data.

    Returns:
        pd.DataFrame: the hourly demand data with the right province names, in TWh/hr
    """
    TO_TWh = 1e-6
    hourly = pd.read_csv(hourly_load_p)
    hourly_TWh = hourly.drop(columns=["Time Series"]) * TO_TWh
    prov_codes = pd.read_csv(prov_codes_p)
    prov_codes.set_index("Code", inplace=True)
    hourly_TWh.columns = hourly_TWh.columns.map(prov_codes["Full name"])
    return hourly_TWh


def project_elec_demand(
    hourly_demand_base_yr_MWh: pd.DataFrame,
    yearly_projections_MWh: pd.DataFrame,
    year=2020,
) -> pd.DataFrame:
    """Project the hourly demand to the future years

    Args:
        hourly_demand_base_yr_MWh (pd.DataFrame): the hourly demand in the base year
        yearly_projections_MWh (pd.DataFrame): the yearly projections

    Returns:
        pd.DataFrame: the projected hourly demand
    """
    hourly_load_profile = hourly_demand_base_yr_MWh.loc[:, PROV_NAMES]
    # normalise the hourly load
    hourly_load_profile /= hourly_load_profile.sum(axis=0)

    yearly_projections_MWh = yearly_projections_MWh.T.loc[int(year), PROV_NAMES]
    hourly_load_projected = yearly_projections_MWh.multiply(hourly_load_profile)

    # TODO fix this to use timestamps
    if len(hourly_load_projected) == 8784:
        # rm feb 29th
        hourly_load_projected.drop(hourly_load_projected.index[1416:1440], inplace=True)
    elif len(hourly_load_projected) != 8760:
        raise ValueError("The length of the hourly load is not 8760 or 8784 (leap year, dropped)")

    snapshots = make_periodic_snapshots(
        year=year,
        freq="1h",
        start_day_hour="01-01 00:00:00",
        end_day_hour="12-31 23:00",
        bounds="both",
    )

    hourly_load_projected.index = snapshots
    return hourly_load_projected


# TODO replace with REMIND/IEA/OTHER
def project_heat_demand(planning_year: int, projection_name: str, ref_year=REF_YEAR) -> float:
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

        # TODO soft code
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


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_load_profiles",
            heating_demand="positive",
            planning_horizons="2060",
            # co2_pathway="remind_ssp2NPI",
            co2_pathway="exp175default",
            topology="Current+Neigbor",
        )

    configure_logging(snakemake, logger=logger)

    planning_horizons = int(snakemake.wildcards["planning_horizons"])
    config = snakemake.config

    date_range = make_periodic_snapshots(
        year=planning_horizons,
        freq="1h",
        start_day_hour=config["snapshots"]["start"],
        end_day_hour=config["snapshots"]["end"],
        bounds=config["snapshots"]["bounds"],
        tz=None,
        end_year=(None if not config["snapshots"]["end_year_plus1"] else planning_horizons + 1),
    )

    # project the electric load based on the demand
    conversion = snakemake.params.elec_load_conversion  # to MWHr
    hrly_MWh_load = prepare_hourly_load_data(
        snakemake.input.hrly_regional_ac_load, snakemake.input.province_codes
    )

    yearly_projs = read_yearly_load_projections(snakemake.input.elec_load_projs, conversion)
    projected_demand = project_elec_demand(hrly_MWh_load, yearly_projs, planning_horizons)

    with pd.HDFStore(snakemake.output.elec_load_hrly, mode="w", complevel=4) as store:
        store["load"] = projected_demand

    if config.get("heat_coupling", False):

        # load heat data
        with pd.HDFStore(snakemake.input.population_map, mode="r") as store:
            pop_map = store["population_gridcell_map"]
        with pd.HDFStore(snakemake.input.population, mode="r") as store:
            population_count = store["population"]
        intraday_profiles = pd.read_csv(snakemake.input.intraday_profiles, index_col=0)

        space_heat_demand_total = (
            pd.read_csv(snakemake.input.space_heat_demand, index_col=0) * TWH2MWH
        )
        space_heat_demand_total = space_heat_demand_total.squeeze()
        hot_water_total = _read_iea_hot_water(snakemake.input.hot_water_demand)
        hot_water_total = _extend_iea_projections(
            hot_water_total, config["scenario"]["planning_horizons"]
        )
        atlite_hour_shift = calc_atlite_heating_timeshift(date_range, use_last_ts=False)
        cutout = atlite.Cutout(snakemake.input.cutout)

        reg_daily_hd = build_daily_heat_demand_profiles(
            cutout,
            pop_map,
            config["heat_demand"],
            atlite_heating_hr_shift=atlite_hour_shift,
            switch_month_day=True,
        )

        # Scale the degree day to reference values
        daily_heat_demand = scale_degree_day_to_reference(reg_daily_hd, space_heat_demand_total)
        # ==== SCALE TO FUTURE DEMAND ======
        # TODO soft-code ref/base year or find a better variable name
        demand_growth_factor = project_heat_demand(
            planning_horizons, snakemake.wildcards["heating_demand"], ref_year=REF_YEAR
        )
        daily_heat_demand = daily_heat_demand * demand_growth_factor

        # distribute evenly across year
        hot_water_per_day = (
            downscale_by_pop(hot_water_total.loc[planning_horizons], population_count) / 365
        )

        space_heat_demand, domestic_hot_water = build_heat_demand_profile(
            daily_heat_demand,
            hot_water_per_day,
            date_range,
            intraday_profiles=intraday_profiles,
            planning_horizons=planning_horizons,
        )

        with pd.HDFStore(snakemake.output.heat_demand_profile, mode="w", complevel=4) as store:
            store["heat_demand_profiles"] = space_heat_demand
            store["hot_water_demand"] = domestic_hot_water

        with pd.HDFStore(snakemake.output.energy_totals_name, mode="w") as store:
            store["space_heating_per_hdd"] = daily_heat_demand
            store["hot_water_per_day"] = hot_water_per_day

        logger.info("Heat demand profiles successfully built")
    else:
        with pd.HDFStore(snakemake.output.heat_demand_profile, mode="w") as store:
            store["skipped_not_heat_coupled"] = pd.Series(["Skipped, heat coupling is disabled"])
        with pd.HDFStore(snakemake.output.energy_totals_name, mode="w") as store:
            store["skipped_not_heat_coupled"] = pd.Series(["Skipped, heat coupling is disabled"])
