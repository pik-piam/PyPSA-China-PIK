# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT
""" 
Functions to download ERA5/SARAH data and build the atlite cutout for the atlite.
These functions linked to the build_cutout rule.
"""
import logging
import atlite
import geopandas as gpd
import pandas as pd

from _helpers import configure_logging, mock_snakemake, make_periodic_snapshots, get_cutout_params
from constants import TIMEZONE

logger = logging.getLogger(__name__)


def cutout_timespan(config: dict, weather_year: int) -> list:
    """build the cutout timespan. Note that the coutout requests are in UTC (TBC)

    Args:
        config (dict): the snakemake config
        weather_year (dict): the coutout weather year

    Returns:
        tuple: end and start of the cutout timespan
    """
    snapshot_cfg = config["snapshots"]
    # make snapshots for TZ and then convert to naive UTC for atlite
    snapshots = (
        make_periodic_snapshots(
            year=weather_year,
            freq=snapshot_cfg["freq"],
            start_day_hour=snapshot_cfg["start"],
            end_day_hour=snapshot_cfg["end"],
            bounds=snapshot_cfg["bounds"],
            # here we need to convert UTC to local
            tz=TIMEZONE,
            end_year=(None if not snapshot_cfg["end_year_plus1"] else weather_year + 1),
        )
        .tz_convert("UTC")
        .tz_localize(None)
    )

    return [snapshots[0], snapshots[-1]]


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_cutout", cutout="China-2020c")
    configure_logging(snakemake, logger=logger)

    config = snakemake.config
    # overwrite cutoutname if was wildcard, get params
    config["atlite"]["cutout_name"] = snakemake.wildcards.get(
        "cutout", config["atlite"]["cutout_name"]
    )
    cutout_params = get_cutout_params(config)
    time = cutout_timespan(snakemake.config, cutout_params["weather_year"])
    cutout_params["time"] = slice(*cutout_params.get("time", time))

    # determine bounds for cutout
    if {"x", "y", "bounds"}.isdisjoint(cutout_params):
        # Determine the bounds from bus regions with a buffer of two grid cells
        onshore = gpd.read_file(snakemake.input.regions_onshore)
        offshore = gpd.read_file(snakemake.input.regions_offshore)
        regions = pd.concat([onshore, offshore])

        d = max(cutout_params.get("dx", 0.25), cutout_params.get("dy", 0.25)) * 2
        cutout_params["bounds"] = regions.total_bounds + [-d, -d, d, d]
    # if specified x,y (else use bounds directly)
    elif {"x", "y"}.issubset(cutout_params):
        cutout_params["x"] = slice(*cutout_params["x"])
        cutout_params["y"] = slice(*cutout_params["y"])

    logging.info(f"Preparing cutout with parameters {cutout_params}.")
    cutout = atlite.Cutout(snakemake.output[0], **cutout_params)
    logging.info("You can check progress at https://cds.climate.copernicus.eu/requests?tab=all")
    cutout.prepare(monthly_requests=config["atlite"]["monthly_requests"])

    logger.info(f"Cutout successfully built at {snakemake.output[0]}.")
