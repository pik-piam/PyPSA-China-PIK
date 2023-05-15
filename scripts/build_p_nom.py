# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
import logging
from _helpers import configure_logging

import pandas as pd
from functions import pro_names

logger = logging.getLogger(__name__)

def csv_to_df(csv_name=None):    
    df = pd.read_csv(csv_name, index_col=0, header=0)    
    df = df.apply(pd.to_numeric)
    return df['MW'].reindex(pro_names)
    
def build_p_nom():
    coal_capacity = csv_to_df(csv_name="data/p_nom/coal_p_nom.csv") 
    CHP_capacity = csv_to_df(csv_name="data/p_nom/CHP_p_nom.csv")
    OCGT_capacity = csv_to_df(csv_name="data/p_nom/OCGT_p_nom.csv")    
    offwind_capacity = csv_to_df(csv_name="data/p_nom/offwind_p_nom.csv")    
    onwind_capacity = csv_to_df(csv_name="data/p_nom/onwind_p_nom.csv")   
    solar_capacity = csv_to_df(csv_name="data/p_nom/solar_p_nom.csv")
    nuclear_capacity = csv_to_df(csv_name="data/p_nom/nuclear_p_nom.csv")

    coal_capacity.name = "coal_capacity"  
    CHP_capacity.name = "CHP_capacity"    
    OCGT_capacity.name = "OCGT_capacity"   
    offwind_capacity.name = "offwind_capacity"   
    onwind_capacity.name = "onwind_capacity"   
    solar_capacity.name = "solar_capacity"
    nuclear_capacity.name = "nuclear_capacity"

    coal_capacity.to_hdf(snakemake.output.coal_capacity, key=coal_capacity.name)    
    CHP_capacity.to_hdf(snakemake.output.CHP_capacity, key=CHP_capacity.name)   
    OCGT_capacity.to_hdf(snakemake.output.OCGT_capacity, key=OCGT_capacity.name)
    offwind_capacity.to_hdf(snakemake.output.offwind_capacity, key=offwind_capacity.name)
    onwind_capacity.to_hdf(snakemake.output.onwind_capacity, key=onwind_capacity.name)
    solar_capacity.to_hdf(snakemake.output.solar_capacity, key=solar_capacity.name)
    nuclear_capacity.to_hdf(snakemake.output.nuclear_capacity, key=nuclear_capacity.name)
    
if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_p_nom')
    configure_logging(snakemake)
        
    build_p_nom()
