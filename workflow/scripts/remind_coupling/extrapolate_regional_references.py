"""Extrapolate regional disaggregation reference shares for different sectors.

This script generates provincial share/ratio data used to spatially disaggregate
REMIND national-level outputs to the provincial level. For example, EV passenger
shares indicate what fraction of national EV demand belongs to each province.

This provides a general framework to coordinate the extrapolation of sector-specific
reference share files. Each sector has its own specialized module implementing the 
`extrapolate_reference` function.
"""

import importlib
import logging
from pathlib import Path

import setup
from _helpers import configure_logging

logger = logging.getLogger(__name__)


class SectorReferenceGenerator:
    """General framework for generating sectoral disaggregation shares.
    
    Coordinates the extrapolation of provincial share/ratio data for different
    sectors (e.g., EV, heat). These shares are used to spatially disaggregate
    REMIND national outputs to provincial level.
    """

    def __init__(self, config: dict = None):
        """Initialize the generator.

        Args:
            config (dict): Configuration dictionary with sector-specific settings.
        """
        self.config = config or {}
        self.sector_modules = {}
        self._load_sector_modules()

    def _load_sector_modules(self):
        """Load sector-specific modules for generating disaggregation shares."""
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
        """Extrapolate provincial disaggregation shares for all available sectors.
        
        Generates reference share files that indicate what fraction of national-level
        sectoral demand/activity belongs to each province. For example, EV passenger
        shares show the provincial distribution of passenger EV demand.

        Args:
            years (list[int]): List of target years for projections (e.g., [2020, 2025, 2030]).
            input_files (dict[str, str]): Dictionary mapping data types to file paths 
                (e.g., historical GDP, population, sector-specific data).
            output_dir (str): Directory to save extrapolated reference share files.
        """
        logger.info("Extrapolating sectoral disaggregation shares")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for sector, module in self.sector_modules.items():
            try:
                logger.info(f"Extrapolating disaggregation shares for {sector} sector")

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
                logger.error(f"Failed to extrapolate disaggregation shares for {sector}: {e}")

        logger.info("Sectoral disaggregation shares extrapolation completed")


def main():
    """Main entry point for generating provincial disaggregation shares."""
    if "snakemake" not in globals():
        snakemake = setup._mock_snakemake(
            "generate_sector_references",
            config_files="resources/tmp/remind_coupled_cg.yaml",
        )
    else:
        snakemake = globals()["snakemake"]

    configure_logging(snakemake)
    logger.info("Starting generation of sectoral disaggregation shares")

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

    output_dir = Path(snakemake.output.ev_passenger_shares).parent

    generator = SectorReferenceGenerator(config)
    generator.extrapolate_references(years, input_files, str(output_dir))

    logger.info("Generation of sectoral disaggregation shares finished")


if __name__ == "__main__":
    main()
