"""generic disaggregation development
Split steps into:

- ETL
- disagg (also an ETL op)

to be rebalanced with the remind_coupling package
"""

import logging
import os.path

import pandas as pd
import setup  # sets up paths
from _helpers import configure_logging
from generic_etl import ETLRunner
from readers import read_yearly_load_projections

# import needed for the capacity method to be registered
from rpycpl.disagg import SpatialDisaggregator
from rpycpl.etl import ETL_REGISTRY, Transformation, register_etl

logger = logging.getLogger(__name__)

@register_etl("disagg_multisector_ref")
def disagg_multisector_using_ref(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    reference_year: int | str,
) -> pd.DataFrame:
    """Spatially disaggregate multi-sector loads using reference data.

    Args:
        data (pd.DataFrame): Load data with columns [province, sector, year, value].
        reference_data (pd.DataFrame): Reference data for spatial disaggregation.
        reference_year (int | str): Year to use as baseline for disaggregation.

    Returns:
        pd.DataFrame: Disaggregated load data (Province x Sector x Year).
    """

    logger.info("Starting multi-sector load disaggregation")

    default_reference = reference_data[int(reference_year)]
    default_reference /= default_reference.sum()

    loads_data = data["loads"]
    all_results = []

    for sector, sector_data in loads_data.groupby("load"):
        year_results = []
        for year, year_data in sector_data.groupby("year"):
            ref = _get_sector_reference(sector, data, default_reference, year=year)

            disagg = SpatialDisaggregator().use_static_reference(
                year_data["value"], ref
            )
            disagg = disagg.squeeze()
            disagg.name = int(year)
            year_results.append(disagg)

        wide_df = pd.concat(year_results, axis=1)
        wide_df.insert(0, "province", wide_df.index)
        wide_df.insert(1, "sector", sector)
        wide_df.reset_index(drop=True, inplace=True)

        all_results.append(wide_df)

    result = pd.concat(all_results, ignore_index=True)
    return result


def _get_sector_reference(
    sector: str, data: dict, default_reference: pd.Series, year: int = None
) -> pd.Series:
    """Select sector-specific reference data or fallback to default.

    Args:
        sector (str): Sector name.
        data (dict): Input data dictionary containing reference series or DataFrames.
        default_reference (pd.Series): Default reference distribution.
        year (int, optional): Year to select from reference data.

    Returns:
        pd.Series: Reference distribution for the given sector.
    """
    if sector == "ac":
        return default_reference

    ref_data = data.get(f"{sector}_reference")
    if ref_data is None:
        return default_reference

    if isinstance(ref_data, pd.DataFrame):
        ref_data = ref_data.set_index(ref_data.columns[0]) if "Unnamed: 0" in ref_data.columns else ref_data
        ref_data = ref_data.astype(float)
        col = str(int(year)) if year is not None and str(int(year)) in ref_data.columns.astype(str) else ref_data.columns[-1]
        return ref_data[col]

    if isinstance(ref_data, pd.Series):
        return ref_data.astype(float)

    return default_reference


@register_etl("disagg_acload_ref")
def disagg_ac_using_ref(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    reference_year: int | str,
) -> pd.DataFrame:
    """Spatially Disaggregate the load using regional/nodal reference data
        (e.g. the projections from Hu2013 as in the Zhou et al PyPSA-China version)

    Args:
        data (pd.DataFrame): DataFrame containing the load data
        reference_data (pd.DataFrame): DataFrame containing the reference data
        reference_year (int | str): Year to use for disaggregation
    Returns:
        pd.DataFrame: Disaggregated load data (Region x Year)
    """

    regional_reference = reference_data[int(reference_year)]
    regional_reference /= regional_reference.sum()
    electricity_demand = data["loads"].query("load == 'ac'")
    electricity_demand.set_index("year", inplace=True)
    logger.info("Disaggregating load according to Hu et al. demand projections")
    disagg_load = SpatialDisaggregator().use_static_reference(
        electricity_demand.value, regional_reference
    )

    return disagg_load


def add_possible_techs_to_paidoff(paidoff: pd.DataFrame, tech_groups: pd.Series) -> pd.DataFrame:
    """Add possible PyPSA technologies to the paid off capacities DataFrame.
    The paidoff capacities are grouped in case the Remind-PyPSA tecg mapping is not 1:1
    but the network needs to add PyPSA techs.
    A constraint is added so the paid off caps per group are not exceeded.

    Args:
        paidoff (pd.DataFrame): DataFrame with paid off capacities
    Returns:
        pd.DataFrame: paid off techs with list of PyPSA technologies
    Example:
        >> tech_groups
            PyPSA_tech, group
            coal CHP, coal
            coal, coal
        >> add_possible_techs_to_paidoff(paidoff, tech_groups)
        >> paidoff
            tech_group, paid_off_capacity, techs
            coal, 1000, ['coal CHP', 'coal']
    """
    df = tech_groups.reset_index()
    possibilities = df.groupby("group").PyPSA_tech.apply(lambda x: list(x.unique()))
    paidoff["techs"] = paidoff.tech_group.map(possibilities)
    return paidoff



if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = setup._mock_snakemake(
            "disaggregate_remind_data",
            co2_pathway="SSP2-PkBudg1000-CHA-paidoff09",
            topology="current+FCG",
            config_files="resources/tmp/remind_coupled_cg.yaml",
            heating_demand="positive",
        )
    configure_logging(snakemake)
    logger.info("Running disaggregation script")
    logger.debug(f"Available ETL methods: {ETL_REGISTRY.keys()}")

    params = snakemake.params
    region = params.region
    config = params.etl_cfg
    if not config:
        raise ValueError("Aborting: No REMIND data ETL config provided")

    # ================ Load data ===============
    input_files = {k: v for k, v in snakemake.input.items() if not os.path.isdir(v)}
    readers = {"reference_load": read_yearly_load_projections, "default": pd.read_csv}

    # read files (and not directories)
    data = {
        k: readers[k](v) if k in readers else readers["default"](v) for k, v in input_files.items()
    }

    powerplant_data = [k for k in data if k.startswith("pypsa_powerplants_")]
    data["pypsa_capacities"] = {k.split("pypsa_powerplants_")[-1]: data[k] for k in powerplant_data}
    # group techs together for harmonization
    pypsa_tech_groups = (
        data["remind_tech_groups"].set_index("PyPSA_tech")["group"].drop_duplicates()
    )
    if not pypsa_tech_groups.index.is_unique:
        raise ValueError(
            "PyPSA tech groups are not unique. Check the remind_tech_groups.csv"
            " file for remind techs that appear in multiple pypsa techs"
        )
    for cap_df in data["pypsa_capacities"].values():
        cap_df["tech_group"] = cap_df.Tech.map(pypsa_tech_groups)
        cap_df.fillna({"tech_group": ""}, inplace=True)

    logger.info(f"Loaded data files {data.keys()}")
    missing = set(input_files.keys()) - set(data.keys())
    if missing:
        logger.warning(f"Warning: Missing data files {missing}")

    # ==== transform remind data =======
    steps = config.get("disagg", [])
    results = {}
    for step_dict in steps:
        step = Transformation(**step_dict)
        logger.info(f"Running ETL step: {step.name} with method {step.method}")
        if step.method == "disagg_acload_ref":
            result = ETLRunner.run(
                step,
                data,
                reference_data=data["reference_load"],
                reference_year=params["reference_load_year"],
            )
        elif step.method == "disagg_multisector_ref":
            # multi-sector disaggregation method
            logger.info("Using multi-sector disaggregation method")
            result = ETLRunner.run(
                step,
                data,
                reference_data=data["reference_load"],
                reference_year=params["reference_load_year"],
            )
        elif step.method == "harmonize_capacities":
            # TODO loop over years
            result = ETLRunner.run(
                step, data["pypsa_capacities"], remind_capacities=data["remind_caps"]
            )
        elif step.method == "calc_paid_off_capacity":
            result = ETLRunner.run(
                step, data["remind_caps"], harmonized_pypsa_caps=results["harmonize_model_caps"]
            )
        else:
            result = ETLRunner.run(step, data)

        results[step.name] = result

    # TODO export, fix index
    logger.info("\n\nExporting results")
    outp_files = dict(snakemake.output.items())
    logger.info(f"Output files: {outp_files}")
    if "disagg_load" in results:
        logger.info(f"Exporting disaggregated load to {outp_files['disagg_load']}")
        results["disagg_load"].to_csv(
            outp_files["disagg_load"],
            index=False
        )
    if "harmonize_model_caps" in results:
        logger.info("Exporting harmonized model capacities")
        for year, df in results["harmonize_model_caps"].items():
            logger.info(f"Exporting harmonized capacities for year {year}")
            df.to_csv(outp_files[f"caps_{year}"], index=False)

    if "available_cap" in results:
        logger.info("Exporting paid off capacities")
        paid_off = results["available_cap"].copy()
        paid_off = add_possible_techs_to_paidoff(paid_off, pypsa_tech_groups)
        paid_off.to_csv(outp_files["paid_off"], index=False)
