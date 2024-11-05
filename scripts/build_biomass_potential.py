# SPDX-FileCopyrightText: : 2024 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

import logging
from _helpers import configure_logging

import pandas as pd


logger = logging.getLogger(__name__)

def build_biomass_potential():

    df = pd.read_excel(snakemake.input.biomass_feedstocks,sheet_name="supplementary data 1")
    df = df.groupby("Province name").sum()
    df = df[df.columns[df.columns.str.contains("Agricultural residues burnt as waste")]].sum(axis=1)

    ## convert t biomass yr-1 to TWh
    heat_content = 19  # GJ/t
    df = df * heat_content * 2.7777 * 1e-1 # convert GJ to MWh

    ##rename
    df = df.rename(index={"Inner-Monglia": "InnerMongolia","Anhui ":"Anhui"})
    df = df.add_suffix(" biomass")

    df.to_hdf(snakemake.output.biomass_potential,key="biomass")


if __name__ == '__main__':

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_biomass_potential')
    configure_logging(snakemake)

    build_biomass_potential()
