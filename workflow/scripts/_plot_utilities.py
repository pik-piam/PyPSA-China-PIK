import pypsa
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from os import PathLike

import logging


def make_nice_tech_colors(tech_colors: dict, nice_names: dict) -> dict:
    """add the nice names to the tech_colors dict keys

    Args:
        tech_colors (dict): the tech colors (plot config)
        nice_names (dict): the nice names (plot config)

    Returns:
        dict: dict with names & nice names as keys
    """
    nn_colors = {nice: tech_colors[n] for n, nice in nice_names.items() if n in tech_colors}
    tech_colors.update(nn_colors)
    return tech_colors


def get_stat_colors(
    n: pypsa.Network,
    nice_tech_colors: dict,
    extra_colors: dict = None,
    capitalize_words: bool = True,
) -> pd.DataFrame:
    """Combine colors from different sources

    Args:
        n (pypsa.Network): the network
        nice_tech_colors (dict): the tech colors from make_nice_tech_colors
        extra_colors (dict, optional): Additional args for color. Defaults to None.
        capitalize_words (bool, optional): Capitalize the words. Defaults to True.

    Returns:
        pd.DataFrame: the colors
    """

    # carrier colors with either nice or internal name
    carrier_colors = pd.concat(
        [n.carriers["color"], n.carriers[["nice_name", "color"]].set_index("nice_name")]
    )
    carrier_colors = carrier_colors.groupby(level=0).first()
    extra_colors = pd.DataFrame(extra_colors.values(), index=extra_colors.keys(), columns=["color"])
    carrier_colors = pd.concat([carrier_colors, extra_colors]).squeeze()
    if capitalize_words:
        carrier_colors.rename(index=lambda x: x.title(), inplace=True)
        nice_tech_colors = {k.title(): v for k, v in nice_tech_colors.items()}

    return pd.concat([carrier_colors, pd.Series(nice_tech_colors)]).groupby(level=0).first()


def rename_techs(label):
    """From pypsa-Eur

    Args:
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    prefix_to_remove = [
        "central ",
        "decentral ",
    ]

    rename_if_contains_dict = {"water tanks": "hot water storage", "H2": "H2", "coal cc": "CC"}

    rename_if_contains = ["gas", "coal"]

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "hydro_inflow": "hydroelectricity",
        "stations": "hydroelectricity",
        "AC": "transmission lines",
        "CO2 capture": "biomass carbon capture",
        "CC": "coal carbon capture",
        "battery": "battery",
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename.items():
        if old == label:
            label = new
    return label


def fix_network_names_colors(n: pypsa.Network, config: dict):
    """Add missing attributes to network for older versions of the code
    This ensures compatibility with the newer pypsa eur plot functions
    where the results belonged to the old code

    Args:
        n (pypsa.Network): the network
        config (dict): the snakemake config dict
    """
    # incase an old version need to add missing info to network
    if (n.carriers.nice_name == "").sum() > 0:
        # deal with missing carriers
        n.add("Carrier", "AC")
        if config["add_hydro"]:
            n.add("Carrier", "stations")
            n.add("Carrier", "hydro_inflow")

        # deal with missign colors and nice_names
        nice_names = config["plotting"]["nice_names"]
        missing_names = n.carriers.index.difference(nice_names)
        nice_names.update(dict(zip(missing_names, missing_names)))

        n.carriers.nice_name = n.carriers.index.map(nice_names)
        t_colors = config["plotting"]["tech_colors"]
        n.carriers.color = n.carriers.index.map(t_colors)
        NAN_COLOR = "lightgrey"
        n.carriers.color.fillna(NAN_COLOR, inplace=True)


def rename_index(ds: pd.DataFrame) -> pd.DataFrame:
    """Plot utility function from pypsa-eur that combined the multii index into as str idx

    Args:
        ds (pd.DataFrame): the multiindexed data

    Returns:
        pd.DataFrame: data w plot_friendly index
    """
    specific = ds.index.map(lambda x: f"{x[1]}\n({x[0]})")
    generic = ds.index.get_level_values("carrier")
    duplicated = generic.duplicated(keep=False)
    index = specific.where(duplicated, generic)
    return ds.set_axis(index)


def aggregate_small_values(df: pd.DataFrame, threshold: float, column_name=None) -> pd.DataFrame:
    """
    Aggregate small values into "other" category. Works on a copy of the input df_
    Args:
        df (pd.DataFrame): the values to aggregate
        threshold (float): the value below which to aggregate
        column_name (str, optional): the column for which the threshold applies. Defaults to None.
            which applies the threshold everywhere
    Returns:
        pd.DataFrame: the costs, with those below threshold aggregated into "other"
    """
    df_ = df.copy()
    if column_name:
        to_drop = df_.where(df_[column_name] < threshold).isna()
    else:
        to_drop = df_.where(df_ < threshold).isna()

    df_.loc["other"] = df_.loc[to_drop == False].sum()
    drop_index = to_drop[to_drop == True]
    df_.drop(drop_index.index, inplace=True)
    return df_


def determine_plottable(n: pypsa.Network) -> pd.Series:
    """Determine whether links should be plotted

    Args:
        n (pypsa.Network): the pypsa network object
    """
    for c in n.iterate_components(n.branch_components):
        c.df["plottable"] = c.df.bus0.map(n.buses.location != "") & c.df.bus1.map(
            n.buses.location != ""
        )
    # n.links["plottable"] = n.links.bus0.map(n.buses.location != "") & n.links.bus1.map(
    #     n.buses.location != ""
    # )
    for c in n.iterate_components(n.one_port_components):
        c.df["plottable"] = n.buses.location != ""


def set_plot_style(
    style_config_file: PathLike = "./config/plotting_styles/default_style.mplstyle",
    base_styles=["classic"],
):
    """Set the plot style to base_style(s)

    Args:
        style_config_file (PathLike, optional): Extra style args.
            Defaults to "./config/plotting_styles/default_style.mplstyle".
        base_styles (list, optional): The styles to be applied (in order). Defaults to ["classic"].

    Raises:
        FileNotFoundError: _description_
    """
    style_config_file = os.path.abspath(style_config_file)
    if not os.path.exists(style_config_file):
        raise FileNotFoundError(f"Style config file {style_config_file} not found")

    # plt.style.use only overwrites the specified parts of the previous style -> possible to combine
    plt.style.use(base_styles)
    plt.style.use(style_config_file)


def aggregate_small_pie_vals(pie: pd.Series, threshold: float) -> pd.Series:
    """Aggregate small pie values into the "Other" category

    Args:
        pie (pd.Series): pies for netwrk plotting with (Bus, Carrier) index
        threshold (float): the cutoff

    Returns:
        pd.Series: carriers below threshold per Bus merged into "Other"
    """

    pie_df = pie.to_frame()
    pie_df["new_carrier"] = pie_df.apply(
        lambda x: "Other" if x.values < threshold else x.name[1], axis=1
    )
    pie_df["location"] = pie_df.index.get_level_values(0)
    return pie_df.set_index(["location", "new_carrier"]).groupby(level=[0, 1]).sum().squeeze()


if __name__ == "__main__":
    set_plot_style()
    plt.plot([1, 2, 3, 4])
    plt.show()
