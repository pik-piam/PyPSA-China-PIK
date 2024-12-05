import pypsa
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from os import PathLike


def get_stat_colors(
    df_stats: pd.DataFrame,
    n: pypsa.Network,
    plot_config: dict,
    nan_color="lightgrey",
    extra_colors: dict = None,
) -> pd.DataFrame:
    """Make several attempts to get colors for statistics from difference sources

    Args:
        df_stats (pd.DataFrame): the statistics output from n.statistics
        n (pypsa.Network): the network
        plot_config (dict): the plotting config
        nan_color (str, optional): _description_. Defaults to "grey".
        extra_colors (dict, optional): Additional args for color. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    color_series = n.carriers.color.copy().drop_duplicates(ignore_index=False)
    tech_colors = plot_config["tech_colors"].copy()
    nice_names = plot_config["nice_names"]
    if extra_colors:
        tech_colors.update(extra_colors)
    # fill in with tech colors
    missing_colors = df_stats.columns.difference(color_series).to_frame()
    missing_colors["color"] = missing_colors.index.map(tech_colors).values
    missing_colors.loc[missing_colors.color.isna(), "color"] = (
        missing_colors.loc[missing_colors.color.isna()].index.str.lower().map(tech_colors)
    )
    # in case the carrier has the nicename in index
    nice_n_colors = {v: tech_colors[k] for k, v in nice_names.items() if k in tech_colors}
    missing_colors.loc[missing_colors.color.isna(), "color"] = missing_colors.loc[
        missing_colors.color.isna()
    ].index.map(nice_n_colors)
    # fillna & add to
    missing_colors.fillna(value={"color": nan_color}, inplace=True)
    return pd.concat([color_series, missing_colors.color]).drop_duplicates(ignore_index=False)


# x = get_stat_colors(p,n, config["plotting"], extra_colors={"Load":"black"})


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

    df_.loc["other"] = df_.loc[to_drop].sum()
    drop_index = to_drop[to_drop == True]
    df_.drop(drop_index.index, inplace=True)
    return df_


def assign_location(n: pypsa.Network):
    """Add the node location name as a column to the component dataframes.
    This is needed because the bus names are of style LOCATION TYPE and cannot be directly grouped
    by province/location otherwise

    Args:
        n (pypsa.Network): the pypsa network object
    """
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        c.df["location"] = c.df.index.str.split(" ", expand=True).get_level_values(0)


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


if __name__ == "__main__":
    set_plot_style()
    plt.plot([1, 2, 3, 4])
    plt.show()
