# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build land transport demand per clustered model region including efficiency
improvements due to drivetrain changes, time series for electric vehicle
availability and demand-side management constraints.
"""

import logging
import os
from pathlib import Path
from constants import TIMEZONE

import numpy as np
import pandas as pd
import xarray as xr

# from remind_coupling.disaggregate_data import get_ev_provincial_shares

from _helpers import set_scenario_config, configure_logging
from _pypsa_helpers import generate_periodic_profiles,make_periodic_snapshots

logger = logging.getLogger(__name__)

def build_transport_demand(
    traffic_fn: str,
    airtemp_fn: str,   # ⚠️ 如果完全不用温度修正，可以删掉这个参数
    nodes: list,
    sector_load_data: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    options: dict,
    nyears: int,
) -> pd.DataFrame:
    """
    Build transport demand time series (unit: MWh), given provincial annual totals.

    Parameters
    ----------
    traffic_fn : str
        Path to driving ratio data (CSV with 'count' column).
    airtemp_fn : str
        (unused if no temperature correction)
    nodes : list
        List of nodes (provinces).
    sector_load_data : pd.DataFrame
        Annual transport load data by province (index) and year (columns).
    snapshots : pd.DatetimeIndex
        Time series index.
    options : dict
        (unused if no temperature correction)
    nyears : int
        Number of simulated years.

    Returns
    -------
    pd.DataFrame
        Provincial transport demand time series (MWh), total per node = input annual total.
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


def transport_degree_factor(
    temperature,
    deadband_lower=15,
    deadband_upper=20,
    lower_degree_factor=0.5,
    upper_degree_factor=1.6,
):
    """
    Work out how much energy demand in vehicles increases due to heating and
    cooling.

    There is a deadband where there is no increase. Degree factors are %
    increase in demand compared to no heating/cooling fuel consumption.
    Returns per unit increase in demand for each place and time
    """

    dd = temperature.copy()

    dd[(temperature > deadband_lower) & (temperature < deadband_upper)] = 0.0

    dT_lower = deadband_lower - temperature[temperature < deadband_lower]
    dd[temperature < deadband_lower] = lower_degree_factor / 100 * dT_lower

    dT_upper = temperature[temperature > deadband_upper] - deadband_upper
    dd[temperature > deadband_upper] = upper_degree_factor / 100 * dT_upper

    return dd


def bev_availability_profile(fn, snapshots, nodes, options):
    """
    Generate EV charging availability profiles from weekly ratio data.

    Parameters
    ----------
    fn : str
        Path to availability data file (contains "count" column).
    snapshots : pd.DatetimeIndex
        Time index.
    nodes : list
        List of nodes.
    options : dict
        Additional options.

    Returns
    -------
    pd.DataFrame
        EV charging availability (0–1) for all nodes.
    """
    availability = pd.read_csv(fn, usecols=["count"]).squeeze("columns")

    # Normalize if values are not ratios
    if availability.max() > 1.0:
        availability = availability / availability.max()
    availability = availability.clip(0, 1).fillna(availability.mean())

    # Repeat weekly pattern over full horizon
    repeat_count = len(snapshots) // 168
    remainder = len(snapshots) % 168
    repeated = np.tile(availability.values, repeat_count)
    if remainder > 0:
        repeated = np.concatenate([repeated, availability.values[:remainder]])

    profile = pd.DataFrame(repeated, index=snapshots)
    profile = pd.concat([profile]*len(nodes), axis=1)
    profile.columns = nodes
    return profile



def bev_dsm_profile(snapshots, nodes, options):
    """
    Generate DSM restriction profile for EVs.

    Parameters
    ----------
    snapshots : pd.DatetimeIndex
        Time index.
    nodes : list
        List of nodes.
    options : dict
        Must include:
            - bev_dsm_restriction_time : int (hour of day)
            - bev_dsm_restriction_value : float (restriction value)

    Returns
    -------
    pd.DataFrame
        DSM restriction profile for all nodes.
    """
    dsm_week = np.zeros(24 * 7)
    idx = np.arange(0, 7) * 24 + options["bev_dsm_restriction_time"]
    dsm_week[idx] = options["bev_dsm_restriction_value"]

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
    carry_backlog: bool = False,
    reset_hour: int = 0,
    initial_soc: float = 0.5,
) -> np.ndarray:
    """
    Compute causal EV availability profile (Δt=1h).
    """
    p_park_series = np.clip(p_park_series, 0.0, 1.0)
    p_run_series = 1.0 - p_park_series

    J = dj_km.shape[0]
    alpha = e_kwh_per_km / p_eff_kw  # charging hours per km

    N = len(p_park_series)
    cv_eff = np.zeros(N, dtype=float)

    # per-vehicle state
    Tj = np.zeros(J, dtype=float)
    Bj = np.zeros(J, dtype=float)

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
        if carry_backlog:
            Rj = np.maximum(Bj + H_cum, 0.0)
        else:
            Rj = np.maximum(H_cum, 0.0)

        wj = np.minimum(Rj, 1.0)

        # conditional expectation over parking vehicles
        N_t = (wj * p_park_d_t).mean()
        cv_eff[t] = N_t

        # update states
        Tj += p_park_d_t * wj
        cum_run_day += p_run_d_t

        # update backlog at day end
        if carry_backlog and ((t - reset_hour) % 24 == 23 or t == N - 1):
            Bj = np.maximum(Bj + alpha * dj_km - Tj, 0.0)

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
    carry_backlog: bool = False,
    reset_hour: int = 0,
    initial_soc: float = 0.5,
) -> pd.DataFrame:
    """
    Build EV charging availability profile using trip-distance CDF and parking ratio.
    """

    # 1. Parking profile (weekly, 168h)
    p_park_week = (
        pd.read_csv(parking_ratio_csv, usecols=["count"])
        .squeeze("columns")
        .to_numpy(float)
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
        carry_backlog=carry_backlog,
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
    set_scenario_config(snakemake)

    from constants import PROV_NAMES

    nodes = list(PROV_NAMES)

    transport_config = snakemake.params.transport_demand
    
    yr = int(snakemake.wildcards.planning_horizons)
    snapshots = make_periodic_snapshots(
        year=yr,
        freq="1h",
        start_day_hour="01-01 00:00:00",
        end_day_hour="12-31 23:00",
        bounds="both",
    )
    
    temperature_correction = transport_config.get("temperature_correction", {})
    options = {
        "transport_heating_deadband_lower": temperature_correction.get("heating_deadband_lower", 15.0),
        "transport_heating_deadband_upper": temperature_correction.get("heating_deadband_upper", 20.0),
        "ICE_lower_degree_factor": temperature_correction.get("ice_lower_degree_factor", 0.375),
        "ICE_upper_degree_factor": temperature_correction.get("ice_upper_degree_factor", 1.6),
        "bev_dsm_restriction_time": transport_config.get("bev_dsm_restriction_time", 7),
        "bev_dsm_restriction_value": transport_config.get("bev_dsm_restriction_value", 0.8),
    }

    passenger_e_mwh_per_100km = transport_config.get("energy_per_100km_passenger", 0.015)
    freight_e_mwh_per_100km = transport_config.get("energy_per_100km_freight", 0.2)
    passenger_charging_power_kw = transport_config.get("passenger_charging_power_kw", 7.0)
    freight_charging_power_kw = transport_config.get("freight_charging_power_kw", 50.0)
    
    nyears = len(snapshots) / 8760.0

    sectoral_load = pd.read_csv(snakemake.input.sectoral_load)
    
    ev_passenger_load = sectoral_load[sectoral_load["sector"] == "ev_pass"].copy()
    ev_passenger_load = ev_passenger_load.drop("sector", axis=1)
    ev_passenger_load.set_index("province", inplace=True)
    
    ev_freight_load = sectoral_load[sectoral_load["sector"] == "ev_freight"].copy()
    ev_freight_load = ev_freight_load.drop("sector", axis=1)
    ev_freight_load.set_index("province", inplace=True)

    driving_demand_passenger = build_transport_demand(
        snakemake.input.driving_data_passenger,
        snakemake.input.temperature,
        nodes,
        ev_passenger_load,
        snapshots,
        options,
        nyears
    )

    driving_demand_freight = build_transport_demand(
        snakemake.input.driving_data_freight,
        snakemake.input.temperature,
        nodes,
        ev_freight_load,
        snapshots,
        options,
        nyears
    )

    charging_demand_passenger = build_transport_demand(
        snakemake.input.charging_data_passenger,
        snakemake.input.temperature,
        nodes,
        ev_passenger_load,
        snapshots,
        options,
        nyears
    )

    charging_demand_freight = build_transport_demand(
        snakemake.input.charging_data_freight,
        snakemake.input.temperature,
        nodes,
        ev_freight_load,
        snapshots,
        options,
        nyears
    )

    reset_hour_cfg = transport_config.get("reset_hour", 7)
    initial_soc_cfg = transport_config.get("initial_soc", 0.2)
    cdf_column_passenger = transport_config.get("cdf_column_passenger", "Private car")
    cdf_column_freight = transport_config.get("cdf_column_freight", "SPV")

    avail_profile_passenger = build_availability_profile_via_cdf(
        snapshots=snapshots,
        nodes=nodes,
        cdf_csv=snakemake.input.cdf_data_passenger,                 # CDF 文件
        cdf_column=cdf_column_passenger,                             # CDF 列
        e_mwh_per_100km=passenger_e_mwh_per_100km,                  # 能耗
        p_eff_kw=passenger_charging_power_kw,                       # 充电功率
        parking_ratio_csv=snakemake.input.availability_data_passenger, # 停车比例
        reset_daily=True,
        carry_backlog=False,
        reset_hour=reset_hour_cfg,
        initial_soc=initial_soc_cfg
    )

    avail_profile_freight = build_availability_profile_via_cdf(
        snapshots=snapshots,
        nodes=nodes,
        cdf_csv=snakemake.input.cdf_data_freight,                   # CDF 文件
        cdf_column=cdf_column_freight,                               # CDF 列
        e_mwh_per_100km=freight_e_mwh_per_100km,                    # 能耗
        p_eff_kw=freight_charging_power_kw,                         # 充电功率
        parking_ratio_csv=snakemake.input.availability_data_freight, # 停车比例
        reset_daily=True,
        carry_backlog=False,
        reset_hour=reset_hour_cfg,
        initial_soc=initial_soc_cfg
    )

    dsm_profile = bev_dsm_profile(snapshots, nodes, options)

    driving_demand_passenger.to_csv(snakemake.output.driving_demand_passenger)
    driving_demand_freight.to_csv(snakemake.output.driving_demand_freight)
    charging_demand_passenger.to_csv(snakemake.output.charging_demand_passenger)
    charging_demand_freight.to_csv(snakemake.output.charging_demand_freight)
    avail_profile_passenger.to_csv(snakemake.output.avail_profile_passenger)
    avail_profile_freight.to_csv(snakemake.output.avail_profile_freight)
    dsm_profile.to_csv(snakemake.output.dsm_profile)
