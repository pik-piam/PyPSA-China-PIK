# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build transport charging demand per clustered model region.
Simplified version that only generates charging demand time series.
"""

import logging

import pandas as pd

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


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_transport_demand", clusters=128)
    configure_logging(snakemake)

    nodes = list(PROV_NAMES)

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

    # Only generate charging demand (simplified)
    charging_demand_passenger = build_transport_demand(
        snakemake.input.charging_data_passenger, nodes, ev_passenger_load, snapshots, nyears
    )

    charging_demand_freight = build_transport_demand(
        snakemake.input.charging_data_freight, nodes, ev_freight_load, snapshots, nyears
    )

    # Save only charging demand files
    charging_demand_passenger.to_csv(snakemake.output.charging_demand_passenger)
    charging_demand_freight.to_csv(snakemake.output.charging_demand_freight)
