import pypsa
import matplotlib.pyplot as plt
import json
from os import PathLike
import os.path
import pandas as pd


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
