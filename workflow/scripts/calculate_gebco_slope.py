#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2025 The PyPSA-China Authors
# SPDX-License-Identifier: MIT

"""
Calculate slope from GEBCO bathymetry/altimetry data.

This script:
1. Reprojects GEBCO data to Mollweide (ESRI:54009) for accurate slope calculation
2. Calculates slope in percent using gdaldem
3. Reprojects slope back to EPSG:4326 for compatibility with atlite

Requires: GDAL (gdalwarp, gdaldem)
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from _helpers import mock_snakemake

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_gdal_availability():
    """Check if GDAL tools are available in the system.

    Raises:
        RuntimeError: If required GDAL tools are not found.
    """
    required_tools = ["gdalwarp", "gdaldem"]
    missing_tools = []

    for tool in required_tools:
        if not shutil.which(tool):
            missing_tools.append(tool)

    if missing_tools:
        error_msg = (
            f"Required GDAL tools not found: {', '.join(missing_tools)}\n"
            f"Please install GDAL. If using conda: conda install -c conda-forge gdal\n"
            f"Or ensure GDAL is installed and available in your PATH."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"✓ GDAL tools available: {', '.join(required_tools)}")


def setup_proj_environment():
    """Set up PROJ_LIB environment variable for GDAL to find projection database."""
    if "PROJ_LIB" not in os.environ:
        if "CONDA_PREFIX" in os.environ:
            proj_lib = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
            if os.path.isdir(proj_lib):
                os.environ["PROJ_LIB"] = proj_lib
                logger.info(f"Set PROJ_LIB to: {proj_lib}")
            else:
                logger.warning(f"PROJ data directory not found at: {proj_lib}")
        else:
            logger.warning("CONDA_PREFIX not set, PROJ_LIB may not be configured")
    else:
        logger.info(f"PROJ_LIB already set to: {os.environ['PROJ_LIB']}")


def run_command(cmd: str, description: str):
    """Run a shell command and log the output.

    Args:
        cmd: Shell command to execute.
        description: Human-readable description of the command.

    Returns:
        CompletedProcess instance from subprocess.run.

    Raises:
        RuntimeError: If the command fails (non-zero return code).
    """
    logger.info(f"{description}...")
    logger.debug(f"Command: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ.copy())

    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        logger.error(f"Command: {cmd}")

        # Log stdout if present (some tools output errors to stdout)
        if result.stdout:
            logger.error(f"STDOUT:\n{result.stdout}")

        # Log stderr with better formatting
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")
        else:
            logger.error("No error output captured")

        raise RuntimeError(
            f"{description} failed: {result.stderr.strip() if result.stderr else 'Unknown error'}"
        )

    if result.stdout:
        logger.debug(f"STDOUT: {result.stdout}")
    if result.stderr:
        # GDAL often outputs progress info to stderr, so only warn if there's content
        logger.debug(f"STDERR: {result.stderr}")

    logger.info(f"✓ {description} completed successfully")
    return result


def calculate_slope(input_gebco, output_slope, threads=4, log_file=None):
    """Calculate slope from GEBCO data.

    The slope is calculated in an equal-area projection (Mollweide/ESRI:54009)
    to ensure accurate slope values, then reprojected to EPSG:4326 for use
    with atlite.

    Args:
        input_gebco: Path to input GEBCO GeoTIFF file.
        output_slope: Path to output slope NetCDF file.
        threads: Number of threads for gdalwarp. Default is 4.
        log_file: Path to log file for appending command output (optional).

    Raises:
        FileNotFoundError: If input GEBCO file does not exist.
        RuntimeError: If any GDAL command fails.
    """
    input_gebco = Path(input_gebco)
    output_slope = Path(output_slope)

    if not input_gebco.exists():
        raise FileNotFoundError(f"Input GEBCO file not found: {input_gebco}")

    # Create output directory if needed
    output_slope.parent.mkdir(parents=True, exist_ok=True)

    # Check GDAL availability
    check_gdal_availability()

    # Set up PROJ environment for GDAL
    setup_proj_environment()

    # Define temporary file paths
    mollweide_file = output_slope.parent / (output_slope.stem + "_mollweide.nc")
    slope_mollweide_file = output_slope.parent / (output_slope.stem + "_mollweide_slope.nc")

    logger.info("=" * 60)
    logger.info("GEBCO Slope Calculation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input:   {input_gebco}")
    logger.info(f"Output:  {output_slope}")
    logger.info(f"Threads: {threads}")
    logger.info("=" * 60)

    try:
        # Step 1: Reproject to Mollweide (equal-area projection)
        logger.info("Reprojecting to Mollweide...")
        cmd1 = (
            f"gdalwarp -multi -wo NUM_THREADS={threads} "
            "-of netCDF -co FORMAT=NC4 "
            "-s_srs 'EPSG:4326' -t_srs 'ESRI:54009'"
            f" {input_gebco} {mollweide_file}"
        )
        run_command(cmd1, "Step 1: Reproject to Mollweide (ESRI:54009)")
        logger.info("Calculating slope...")
        # Step 2: Calculate slope in percent
        cmd2 = f"gdaldem slope -p -of netCDF -co FORMAT=NC4 {mollweide_file} {slope_mollweide_file}"
        run_command(cmd2, "Step 2: Calculate slope (percent)")

        # Step 3: Reproject back to EPSG:4326 with compression
        cmd3 = (
            f"gdalwarp -multi -wo NUM_THREADS={threads} "
            f"-of netCDF -co FORMAT=NC4 "
            f"-co COMPRESS=DEFLATE -co ZLEVEL=1 "
            f"-s_srs 'ESRI:54009' -t_srs 'EPSG:4326' "
            f"{slope_mollweide_file} {output_slope}"
        )
        run_command(cmd3, "Step 3: Reproject to EPSG:4326")

        logger.info("=" * 60)
        logger.info("✓ Slope calculation completed successfully!")
        logger.info(f"Output: {output_slope}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    finally:
        # Clean up intermediate files
        logger.info("Cleaning up intermediate files...")
        for temp_file in [mollweide_file, slope_mollweide_file]:
            if temp_file.exists():
                temp_file.unlink()
                logger.info(f"  Removed: {temp_file.name}")


if __name__ == "__main__":
    # When called from snakemake
    if "snakemake" not in globals():
        snakemake = mock_snakemake("calculate_gebco_slope")

    calculate_slope(
        input_gebco=snakemake.input.gebco,
        output_slope=snakemake.output.slope,
        threads=snakemake.threads,
        log_file=snakemake.log[0] if snakemake.log else None,
    )
    logger.info("GEBCO slope calculation script finished.")
