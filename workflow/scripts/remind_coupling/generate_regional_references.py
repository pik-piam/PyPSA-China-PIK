"""Generate reference data files for different sectors

This script provides a general framework to coordinate the generation of 
sector-specific reference data files. Each sector has its own specialized 
module implementing the `generate_reference` function.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import importlib
import setup
from _helpers import configure_logging

logger = logging.getLogger(__name__)


class SectorReferenceGenerator:
    """General framework for sector reference data generation"""
    
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
        """Load sector-specific modules"""
        supported_sectors = {
            'ev': 'ev_refshare_generator',
            # 'industry': 'industry_refshare_generator',
            # 'residential': 'residential_refshare_generator',
            # 'commercial': 'commercial_refshare_generator',
        }
        
        for sector, module_name in supported_sectors.items():
            try:
                module = importlib.import_module(f"sector_modules.{module_name}")
                self.sector_modules[sector] = module
                logger.info(f"Loaded {sector} sector module")
            except ImportError as e:
                logger.warning(f"Could not load {sector} sector module: {e}")
    
    def generate_references(self, years: list[int], input_files: dict[str, str], output_dir: str):
        """
        Generate reference data for all available sectors
        
        Args:
            years (list[int]): List of years
            input_files (dict[str, str]): Dictionary of input files
            output_dir (str): Output directory
        """
        logger.info("Generating sector reference data")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for sector, module in self.sector_modules.items():
            try:
                logger.info(f"Generating reference data for {sector}")
                
                if hasattr(module, 'generate_reference'):
                    module.generate_reference(
                        years=years,
                        input_files=input_files,
                        output_dir=str(output_path),
                        config=self.config.get(sector, {})
                    )
                else:
                    logger.error(f"Sector module {sector} missing generate_reference function")
                    
            except Exception as e:
                logger.error(f"Failed to generate reference data for {sector}: {e}")
        
        logger.info("Sector reference data generation completed")


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
    logger.info("Starting sector reference data generation")
    
    params = snakemake.params
    years = params.years
    config = params.gompertz_config
    
    input_files = {
        'historical_gdp': snakemake.input.historical_gdp,
        'historical_pop': snakemake.input.historical_pop,
        'historical_cars': snakemake.input.historical_cars,
        'ssp2_pop': snakemake.input.ssp2_pop,
        'ssp2_gdp': snakemake.input.ssp2_gdp,
    }
    
    output_dir = Path(snakemake.output.ev_passenger_reference).parent
    
    generator = SectorReferenceGenerator(config)
    generator.generate_references(years, input_files, str(output_dir))
    
    logger.info("Sector reference data generation finished")


if __name__ == "__main__":
    main()
