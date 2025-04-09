# TODO missing docu
import logging
import pandas as pd
from os import PathLike

from _helpers import configure_logging, mock_snakemake


logger = logging.getLogger(__name__)


def estimate_co2_intensity_xing() -> float:
    """Estimate the biomass Co2 intensity from the Xing paper

    Returns:
        float: the biomass co2 intensity in t/MWhth
    """

    biomass_potential_tot = 3.04 # Gt
    embodied_co2_tot = 5.24 # Gt
    heat_content = 19*1000/3600 # GJ/t -> MWh_th/t
    unit_co2 = embodied_co2_tot / biomass_potential_tot # t CO2/t biomass
    co2_intens = unit_co2 / heat_content # t CO2/MWh_th

    return co2_intens

def read_xing_si_data(biomass_potentials_path: PathLike):
    """read and prepare the xing SI data

    Args:
        biomass_potentials_path (PathLike): the path to the Xing SI data (xlsx).
    """
    # data is indexed by province and county
    df = pd.read_excel(biomass_potentials_path, sheet_name="supplementary data 1")
    df = df.groupby("Province name").sum()

    df = df.rename(index={"Inner-Monglia": "InnerMongolia", "Anhui ": "Anhui"})
    df = df.add_suffix(" biomass")

    return df


# TODO fix hardcoded issues
def build_biomass_potential_xing(biomass_potentials_path: PathLike):
    """build potential from Xing et al. https://doi.org/10.1038/s41467-021-23282-x

    Args:
        biomass_potentials_path (PathLike, optional): the path to the Xing SI data (xlsx).
    """

    df = read_xing_si_data(biomass_potentials_path)

    # select only relevant part of potential
    df = df[df.columns[df.columns.str.contains("Agricultural residues burnt as waste")]].sum(axis=1)

    # convert t biomass yr-1 to MWh, heat content is from paper reference 92
    heat_content = 19  # GJ (t biomass−1)
    heat_content *= 1000/3600  # GJ/t -> MWh
    df = df * heat_content

    return df
    


if __name__ == "__main__":

    # for testing & standalone purposes, emulate snakemake
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_biomass_potential",
            # co2_pathway="exp175default",
            # topology="current+FCG",
            # heating_demand="positive",
        )

    configure_logging(snakemake)
   
    df = build_biomass_potential_xing(biomass_potentials_path=snakemake.input.biomass_feedstocks)
    df.to_hdf(snakemake.output.biomass_potential, key="biomass")
    
    logger.info("Biomass potentials successfully built")
