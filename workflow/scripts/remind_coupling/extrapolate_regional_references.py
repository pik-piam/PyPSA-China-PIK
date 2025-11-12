"""Extrapolate reference data files for different sectors

This script provides a general framework to coordinate the extrapolation of
sector-specific reference data files. Each sector has its own specialized
module implementing the `extrapolate_reference` function.
"""

import importlib
import logging
from pathlib import Path

import setup
from _helpers import configure_logging

logger = logging.getLogger(__name__)


class SectorReferenceGenerator:
    """General framework for sector reference data extrapolation"""

    def __init__(self, config: dict = None):
        """
        Initialize the generator

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.sector_modules = {}
        self._load_sector_modules()

    def _load_sector_modules(self):
        """Load sector-specific modules for reference data extrapolation."""
        supported_sectors = {
            "ev": "ev_refshare_extrapolator",
        }

        for sector, module_name in supported_sectors.items():
            try:
                module = importlib.import_module(module_name)
                self.sector_modules[sector] = module
                logger.info(f"Loaded {sector} sector module")
            except ImportError as e:
                logger.warning(f"Could not load {sector} sector module: {e}")

    def extrapolate_references(self, years: list[int], input_files: dict[str, str], output_dir: str):
        """Extrapolate reference data for all available sectors.

        Args:
            years: List of target years for projections
            input_files: Dictionary mapping data types to file paths
            output_dir: Directory to save extrapolated reference files
        """
        logger.info("Extrapolating sector reference data")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for sector, module in self.sector_modules.items():
            try:
                logger.info(f"Extrapolating reference data for {sector}")

                if hasattr(module, "extrapolate_reference"):
                    module.extrapolate_reference(
                        years=years,
                        input_files=input_files,
                        output_dir=str(output_path),
                        config=self.config.get(sector, {}),
                    )
                else:
                    logger.error(f"Sector module {sector} missing extrapolate_reference function")

            except Exception as e:
                logger.error(f"Failed to extrapolate reference data for {sector}: {e}")

        logger.info("Sector reference data extrapolation completed")


def main():
    """Main function"""
    if "snakemake" not in globals():
        snakemake = setup._mock_snakemake(
            "generate_sector_references",
            config_files="resources/tmp/remind_coupled_cg.yaml",
        )
    else:
        snakemake = globals()["snakemake"]

    configure_logging(snakemake)
    logger.info("Starting sector reference data extrapolation")

    params = snakemake.params
    years = params.years
    config = params.gompertz_config

    input_files = {
        "historical_gdp": snakemake.input.historical_gdp,
        "historical_pop": snakemake.input.historical_pop,
        "historical_cars": snakemake.input.historical_cars,
        "ssp2_pop": snakemake.input.ssp2_pop,
        "ssp2_gdp": snakemake.input.ssp2_gdp,
    }

    output_dir = Path(snakemake.output.ev_passenger_reference).parent

    generator = SectorReferenceGenerator(config)
    generator.extrapolate_references(years, input_files, str(output_dir))

    logger.info("Sector reference data extrapolation finished")


if __name__ == "__main__":
    main()
