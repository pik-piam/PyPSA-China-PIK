import logging
import pandas as pd
import pypsa
from os import PathLike

from _helpers import rename_techs
from constants import NICE_NAMES_DEFAULT

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


def calculate_annuity(lifetime: int, discount_rate: float) -> float:
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6

    Args:
        lifetime (int): _description_
        discount_rate (float): _description_

    Returns:
        float: the annuity factor
    """
    r = discount_rate
    n = lifetime

    if isinstance(r, pd.Series):
        if r.any() < 0:
            raise ValueError("Discount rate must be positive")
        return pd.Series(1 / n, index=r.index).where(r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n))
    elif r < 0:
        raise ValueError("Discount rate must be positive")
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n


# TODO fix docstring and change file + IO
def load_costs(
    tech_costs: PathLike, cost_config: dict, elec_config: dict, cost_year: int, n_years: int
) -> pd.DataFrame:
    """_summary_

    Args:
        tech_costs (PathLike): the csv containing the costs
        cost_config (dict): the snakemake pypsa-china cost config
        elec_config (dict): the snakemake pypsa-china electricity config
        cost_year (int): the year for which the costs are retrived
        n_years (int): _description_

    Returns:
        pd.DataFrame: costs dataframe
    """

    # set all asset costs and other parameters
    # tech_costs = tech_costs.replace("{planning_horizons}", str(cost_year))
    costs = pd.read_csv(tech_costs, index_col=list(range(3))).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= cost_config["USD2013_to_EUR2013"]

    cost_year = float(cost_year)
    costs = (
        costs.loc[idx[:, cost_year, :], "value"]
        .unstack(level=2)
        .groupby("technology")
        .sum(min_count=1)
    )

    costs = costs.fillna(
        {
            "CO2 intensity": 0,
            "FOM": 0,
            "VOM": 0,
            "discount rate": cost_config["discountrate"],
            "efficiency": 1,
            "fuel": 0,
            "investment": 0,
            "lifetime": 25,
        }
    )

    costs["capital_cost"] = (
        (calculate_annuity(costs["lifetime"], costs["discount rate"]) + costs["FOM"] / 100.0)
        * costs["investment"]
        * n_years
    )

    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]

    costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    costs.at["OCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]
    costs.at["CCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]

    costs.at["solar", "capital_cost"] = 0.5 * (
        costs.at["solar-rooftop", "capital_cost"] + costs.at["solar-utility", "capital_cost"]
    )

    def costs_for_storage(store, link1, link2=None, max_hours=1.0):
        capital_cost = link1["capital_cost"] + max_hours * store["capital_cost"]
        if link2 is not None:
            capital_cost += link2["capital_cost"]
        return pd.Series(dict(capital_cost=capital_cost, marginal_cost=0.0, co2_emissions=0.0))

    max_hours = elec_config["max_hours"]
    costs.loc["battery"] = costs_for_storage(
        costs.loc["battery storage"], costs.loc["battery inverter"], max_hours=max_hours["battery"]
    )
    costs.loc["H2"] = costs_for_storage(
        costs.loc["hydrogen storage tank type 1"],
        costs.loc["fuel cell"],
        costs.loc["electrolysis"],
        max_hours=max_hours["H2"],
    )

    for attr in ("marginal_cost", "capital_cost"):
        overwrites = cost_config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites

    return costs


# TODO understand why this is in make_summary but not in the main optimisation
def update_transmission_costs(n, costs, length_factor=1.0):
    # TODO: line length factor of lines is applied to lines and links.
    # Separate the function to distinguish.

    n.lines["capital_cost"] = (
        n.lines["length"] * length_factor * costs.at["HVAC overhead", "capital_cost"]
    )

    if n.links.empty:
        return

    dc_b = n.links.carrier == "DC"

    # If there are no dc links, then the 'underwater_fraction' column
    # may be missing. Therefore we have to return here.
    if n.links.loc[dc_b].empty:
        return

    costs = (
        n.links.loc[dc_b, "length"]
        * length_factor
        * (
            (1.0 - n.links.loc[dc_b, "underwater_fraction"])
            * costs.at["HVDC overhead", "capital_cost"]
            + n.links.loc[dc_b, "underwater_fraction"] * costs.at["HVDC submarine", "capital_cost"]
        )
        + costs.at["HVDC inverter pair", "capital_cost"]
    )
    n.links.loc[dc_b, "capital_cost"] = costs


def add_missing_carriers(n: pypsa.Network, carriers: list | set) -> None:
    """Function to add missing carriers to the network without raising errors.

    Args:
        n (pypsa.Network): the pypsa network object
        carriers (list | set): a list of carriers that should be included
    """
    missing_carriers = set(carriers) - set(n.carriers.index)
    if len(missing_carriers) > 0:
        n.add("Carrier", missing_carriers)


def sanitize_carriers(n: pypsa.Network, config: dict) -> None:
    """Sanitize the carrier information in a PyPSA Network object.

    The function ensures that all unique carrier names are present in the network's
    carriers attribute, and adds nice names and colors for each carrier according
    to the provided configuration dictionary.

    Args:
        n (pypsa.Network): PyPSA Network object representing the electrical power system.
        config (dict): A dictionary containing configuration information, specifically the
        "plotting" key with "nice_names" and "tech_colors" keys for carriers.
    """
    # update default nice names w user settings
    nice_names = NICE_NAMES_DEFAULT.update(config["plotting"].get("nice_names", {}))
    for c in n.iterate_components():
        if "carrier" in c.df:
            add_missing_carriers(n, c.df.carrier)

    # sort the nice names to match carriers and fill missing with "ugly" names
    carrier_i = n.carriers.index
    nice_names = pd.Series(nice_names).reindex(carrier_i).fillna(carrier_i.to_series())
    # replace empty nice names with nice names
    n.carriers.nice_name.where(n.carriers.nice_name != "", nice_names, inplace=True)

    # TODO make less messy, avoid using map
    tech_colors = config["plotting"]["tech_colors"]
    colors = pd.Series(tech_colors).reindex(carrier_i)
    # try to fill missing colors with tech_colors after renaming
    missing_colors_i = colors[colors.isna()].index
    colors[missing_colors_i] = missing_colors_i.map(lambda x: rename_techs(x, nice_names)).map(
        tech_colors
    )
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = n.carriers.color.where(n.carriers.color != "", colors)


# TODO break up prep network
def attach_conventional_generators(
    n,
    costs,
    ppl,
    conventional_carriers,
    extendable_carriers,
    conventional_params,
    conventional_inputs,
    unit_commitment=None,
    fuel_price=None,
):
    raise NotImplementedError("This function is not yet implemented")


def attach_hydro(n, costs, ppl, profile_hydro, hydro_capacities, carriers, **params):
    raise NotImplementedError("This function is not yet implemented")


def attach_wind_and_solar(
    n: pypsa.Network,
    costs: pd.DataFrame,
    input_profiles: str,
    carriers: list | set,
    extendable_carriers: list | set,
    line_length_factor: float = 1.0,
    landfall_lengths: dict = None,
) -> None:
    raise NotImplementedError("This function is not yet implemented")
