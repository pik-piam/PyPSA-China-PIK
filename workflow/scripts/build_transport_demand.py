# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build land transport demand per clustered model region including efficiency
improvements due to drivetrain changes, time series for electric vehicle
availability and demand-side management constraints.
"""

import logging

import numpy as np
import pandas as pd

# from remind_coupling.disaggregate_data import get_ev_provincial_shares
from _helpers import configure_logging
from _pypsa_helpers import generate_periodic_profiles, make_periodic_snapshots
from constants import PROV_NAMES, TIMEZONE

logger = logging.getLogger(__name__)


def build_transport_demand(
    traffic_fn: str,
    nodes: list,
    sector_load_data: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    nyears: int,
) -> pd.DataFrame:
    """
    Build transport demand time series (unit: MWh), given provincial annual totals.

    Args:
        traffic_fn (str): Path to driving ratio data (CSV with 'count' column).
        nodes (list): List of nodes (provinces).
        sector_load_data (pd.DataFrame): Annual transport load data by province (index) and year (columns).
        snapshots (pd.DatetimeIndex): Time series index.
        nyears (int): Number of simulated years.

    Returns:
        pd.DataFrame: Provincial transport demand time series (MWh), total per node = input annual total.
    """

    # 1. Load and normalize traffic profile
    traffic = pd.read_csv(traffic_fn, usecols=["count"]).squeeze("columns")
    traffic = traffic / traffic.sum()

    base_shape = generate_periodic_profiles(
        dt_index=snapshots,
        col_tzs=pd.Series(index=nodes, data=[TIMEZONE] * len(nodes)),
        weekly_profile=traffic.values,
    )

    # 2. Get annual totals
    target_year = str(snapshots[0].year)
    nodal_totals = sector_load_data[target_year]

    # 3. Build time series per province
    result = pd.DataFrame(index=snapshots, columns=nodes, dtype=float)
    for node in nodes:
        total = nodal_totals[node]
        shape = base_shape[node] / base_shape[node].sum()  # normalize
        result[node] = shape * total

    return result


# TODO: transport_degree_factor function was removed as temperature correction
#       was not being used. If needed in future, restore from git history.


def bev_dsm_profile(
    snapshots: pd.DatetimeIndex, nodes: list, reset_hour: int, restriction_value: float
) -> pd.DataFrame:
    """
    Creates a weekly repeating profile where DSM restrictions are applied at a specific
    hour each day.

    Args:
        snapshots (pd.DatetimeIndex): Time index for the simulation period.
        nodes (list): List of network nodes (provinces/regions).
        reset_hour (int): Hour of day (0-23) when DSM restrictions are applied.
        restriction_value (float): DSM restriction value (0-1, where 1 = full restriction).

    Returns:
        pd.DataFrame: DSM restriction profile with snapshots as index and nodes as columns.
            Values represent the restriction value at each time step.
    """
    dsm_week = np.zeros(24 * 7)
    idx = np.arange(0, 7) * 24 + reset_hour
    dsm_week[idx] = restriction_value

    repeat_count = len(snapshots) // 168
    remainder = len(snapshots) % 168
    repeated = np.tile(dsm_week, repeat_count)
    if remainder > 0:
        repeated = np.concatenate([repeated, dsm_week[:remainder]])

    profile = pd.DataFrame(repeated, index=snapshots, columns=["tmp"])
    for node in nodes:
        profile[node] = profile["tmp"]
    return profile.drop("tmp", axis=1)


def _compute_cv_eff_series(
    p_park_series: np.ndarray,
    dj_km: np.ndarray,
    e_kwh_per_km: float,
    p_eff_kw: float,
    reset_daily: bool,
    reset_hour: int = 0,
    initial_soc: float = 0.5,
) -> np.ndarray:
    """
    Compute causal electric vehicle charging availability profile using a fleet simulation model.

    This function implements a detailed causal model that tracks individual vehicle states
    over time to determine when EVs are available for charging. It considers driving patterns,
    energy consumption, charging requirements, and battery state-of-charge dynamics.

    The model simulates a fleet of J vehicles with different daily driving distances (dj_km)
    and tracks their charging needs based on 'actual' driving behavior and parking availability.

    Args:
        p_park_series (np.ndarray): Time series of parking probability (0-1) for each hour.
            Length N represents the simulation time horizon in hours.
        dj_km (np.ndarray): Daily driving distances for J vehicle samples (km).
            Represents the heterogeneity in driving patterns across the vehicle fleet.
        e_kwh_per_km (float): Energy consumption per kilometer (kWh/km).
        p_eff_kw (float): Charging power efficiency (kW). Maximum charging rate per vehicle.
        reset_daily (bool): If True, reset battery state-of-charge daily at reset_hour.
            If False, carry forward battery state between days.
        reset_hour (int, optional): Hour of day (0-23) when daily reset occurs. Defaults to 0.
        initial_soc (float, optional): Initial state of charge as fraction (0-1). Defaults to 0.5.

    Returns:
        np.ndarray: Charging availability profile of length N. Each element represents
            the expected fraction of the vehicle fleet available for charging at that hour.
            Values range from 0 (no vehicles available) to 1 (all vehicles available).
    """
    p_park_series = np.clip(p_park_series, 0.0, 1.0)
    p_run_series = 1.0 - p_park_series

    J = dj_km.shape[0]
    alpha = e_kwh_per_km / p_eff_kw  # charging hours per km

    N = len(p_park_series)
    cv_eff = np.zeros(N, dtype=float)

    # per-vehicle state
    Tj = np.zeros(J, dtype=float)

    cum_run_day = np.zeros(J, dtype=float)
    den_run_day = np.ones(J, dtype=float)

    for t in range(N):
        # new day start
        if (t - reset_hour) % 24 == 0:
            p_run_day = p_run_series[t : t + 24]
            den_run_day[:] = p_run_day.sum()
            cum_run_day[:] = 0.0
            if reset_daily:
                Tj[:] = initial_soc * dj_km * e_kwh_per_km / p_eff_kw
            else:
                Tj[:] = 0.0

        p_run_d_t = np.full(J, p_run_series[t])
        p_park_d_t = 1.0 - p_run_d_t

        # fraction of daily run already done
        S_d = np.where(den_run_day > 0.0, np.minimum(cum_run_day / den_run_day, 1.0), 0.0)

        # cumulative demand (h)
        H_cum = alpha * dj_km * S_d + 0.1

        # remaining demand
        Rj = np.maximum(H_cum, 0.0)

        wj = np.minimum(Rj, 1.0)

        # conditional expectation over parking vehicles
        N_t = (wj * p_park_d_t).mean()
        cv_eff[t] = N_t

        # update states
        Tj += p_park_d_t * wj
        cum_run_day += p_run_d_t

    return cv_eff


def build_availability_profile_via_cdf(
    snapshots: pd.DatetimeIndex,
    nodes: list,
    cdf_csv: str,
    cdf_column: str,
    e_mwh_per_100km: float,
    p_eff_kw: float,
    parking_ratio_csv: str,
    J: int = 200,
    reset_daily: bool = True,
    reset_hour: int = 0,
    initial_soc: float = 0.2,
) -> pd.DataFrame:
    """
    Build electric vehicle charging availability profile using driving patterns.

    Combines trip-distance cumulative distribution function (CDF) with parking ratios
    to calculate when EVs are available for charging. Uses a causal model that tracks
    individual vehicle states and charging needs based on daily driving patterns.

    Args:
        snapshots (pd.DatetimeIndex): Time index for the simulation period.
        nodes (list): List of network nodes (provinces/regions).
        cdf_csv (str): Path to CSV file containing trip distance CDF data.
        cdf_column (str): Column name in CDF file for the specific vehicle type.
        e_mwh_per_100km (float): Energy consumption per 100km in MWh.
        p_eff_kw (float): Charging power efficiency in kW.
        parking_ratio_csv (str): Path to CSV file with weekly parking availability ratios.
        J (int, optional): Number of vehicle samples for CDF interpolation. Defaults to 200.
        reset_daily (bool, optional): Whether to reset battery state daily. Defaults to True.
        reset_hour (int, optional): Hour of day (0-23) for daily reset. Defaults to 0.
        initial_soc (float, optional): Initial state of charge (0-1). Defaults to 0.5.

    Returns:
        pd.DataFrame: Charging availability profile with snapshots as index and nodes as columns.
            Values represent the fraction of EVs available for charging at each time step.
    """

    # 1. Parking profile (weekly, 168h)
    p_park_week = (
        pd.read_csv(parking_ratio_csv, usecols=["count"]).squeeze("columns").to_numpy(float)
    )
    p_park_week = np.clip(p_park_week, 0.0, 1.0)
    p_park_series = np.tile(p_park_week, int(np.ceil(len(snapshots) / 168)))[: len(snapshots)]

    # 2. Driving distance samples from CDF
    df = pd.read_csv(cdf_csv).dropna()
    q = df["CDF Percentile"].to_numpy(float) / 100.0
    d = df[cdf_column].to_numpy(float)
    qj = (np.arange(J) + 0.5) / J
    dj_km = np.interp(qj, q, d)

    # 3. Energy per km
    e_kwh_per_km = e_mwh_per_100km * 10.0  # (MWh/100km) → kWh/km

    # 4. Compute causal availability
    cv_series = _compute_cv_eff_series(
        p_park_series,
        dj_km,
        e_kwh_per_km,
        p_eff_kw,
        reset_daily=reset_daily,
        reset_hour=reset_hour,
        initial_soc=initial_soc,
    )

    # 5. Expand to all nodes
    return pd.DataFrame({node: cv_series for node in nodes}, index=snapshots)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_transport_demand", clusters=128)
    configure_logging(snakemake)

    nodes = list(PROV_NAMES)

    ev_pass_config = snakemake.params.EV_pass
    ev_freight_config = snakemake.params.EV_freight
    dsm_config = snakemake.params.DSM_config

    yr = int(snakemake.wildcards.planning_horizons)
    snapshots = make_periodic_snapshots(
        year=yr,
        freq="1h",
        start_day_hour="01-01 00:00:00",
        end_day_hour="12-31 23:00",
        bounds="both",
    )

    nyears = len(snapshots) / 8760.0

    # Load REMIND sectoral data (required for transport demand)
    logger.info("Using REMIND sectoral load data")
    sectoral_load = pd.read_csv(snakemake.input.sectoral_load)

    ev_passenger_load = sectoral_load[sectoral_load["sector"] == "ev_pass"].copy()
    ev_passenger_load = ev_passenger_load.drop("sector", axis=1)
    ev_passenger_load.set_index("province", inplace=True)

    ev_freight_load = sectoral_load[sectoral_load["sector"] == "ev_freight"].copy()
    ev_freight_load = ev_freight_load.drop("sector", axis=1)
    ev_freight_load.set_index("province", inplace=True)

    driving_demand_passenger = build_transport_demand(
        snakemake.input.driving_data_passenger, nodes, ev_passenger_load, snapshots, nyears
    )

    driving_demand_freight = build_transport_demand(
        snakemake.input.driving_data_freight, nodes, ev_freight_load, snapshots, nyears
    )

    charging_demand_passenger = build_transport_demand(
        snakemake.input.charging_data_passenger, nodes, ev_passenger_load, snapshots, nyears
    )

    charging_demand_freight = build_transport_demand(
        snakemake.input.charging_data_freight, nodes, ev_freight_load, snapshots, nyears
    )

    avail_profile_passenger = build_availability_profile_via_cdf(
        snapshots=snapshots,
        nodes=nodes,
        cdf_csv=snakemake.input.cdf_data_passenger,
        cdf_column=ev_pass_config["cdf_column"],
        e_mwh_per_100km=ev_pass_config["energy_per_100km"],
        p_eff_kw=ev_pass_config["charge_rate"] * 1000,  # MW -> kW
        parking_ratio_csv=snakemake.input.availability_data_passenger,
        reset_daily=dsm_config["reset_daily"],
        reset_hour=dsm_config["reset_hour"],
        initial_soc=ev_pass_config["initial_soc"],
    )

    avail_profile_freight = build_availability_profile_via_cdf(
        snapshots=snapshots,
        nodes=nodes,
        cdf_csv=snakemake.input.cdf_data_freight,
        cdf_column=ev_freight_config["cdf_column"],
        e_mwh_per_100km=ev_freight_config["energy_per_100km"],
        p_eff_kw=ev_freight_config["charge_rate"] * 1000,  # MW -> kW
        parking_ratio_csv=snakemake.input.availability_data_freight,
        reset_daily=dsm_config["reset_daily"],
        reset_hour=dsm_config["reset_hour"],
        initial_soc=ev_freight_config["initial_soc"],
    )

    dsm_profile = bev_dsm_profile(
        snapshots, nodes, dsm_config["reset_hour"], dsm_config["restriction_value"]
    )

    driving_demand_passenger.to_csv(snakemake.output.driving_demand_passenger)
    driving_demand_freight.to_csv(snakemake.output.driving_demand_freight)
    charging_demand_passenger.to_csv(snakemake.output.charging_demand_passenger)
    charging_demand_freight.to_csv(snakemake.output.charging_demand_freight)
    avail_profile_passenger.to_csv(snakemake.output.avail_profile_passenger)
    avail_profile_freight.to_csv(snakemake.output.avail_profile_freight)
    dsm_profile.to_csv(snakemake.output.dsm_profile)
