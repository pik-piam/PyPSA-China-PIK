"""
Helper/utility functions for plotting, including legacy functions yet to be removed
"""

import os.path
import re
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa


def validate_hex_colors(tech_colors: dict[str, str], fill_color = "#999999") -> dict[str, str]:
    """Validate and standardize hex color codes in technology color mappings.
    
    Ensures all color codes in the technology colors dictionary are valid hexadecimal
    color codes. Invalid or malformed colors are replaced with a default gray color.
    
    Args:
        tech_colors (Dict[str, str]): Dictionary mapping technology names to color codes. Expected
            format is {'tech_name': '#RRGGBB'} or {'tech_name': '#RGB'}.
        fill_color (str, optional): Default color to use for invalid entries. Defaults to '#999999'.
            
    Returns:
        dict[str,str] with validated hex color codes. All valid colors are converted
        to lowercase, while invalid colors are replaced with '#999999' (gray).
        
    Example:
        >>> colors = {'solar': '#FFD700', 'wind': 'invalid', 'coal': '#8B4513'}
        >>> validated = validate_hex_colors(colors)
        >>> print(validated)
        {'solar': '#ffd700', 'wind': '#999999', 'coal': '#8b4513'}
        
    Note:
        Accepts both 3-digit (#RGB) and 6-digit (#RRGGBB) hex color formats.
        All valid colors are standardized to lowercase.
    """
    hex_color_pattern = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
    validated_colors = {}

    for tech, color in tech_colors.items():
        if not isinstance(color, str) or not hex_color_pattern.match(color):
            validated_colors[tech] = fill_color
        else:
            validated_colors[tech] = color.lower()

    return validated_colors


def find_weeks_of_interest(
    n: pypsa.Network, summer_start="2060-04-01", summer_end="2060-10-06"
) -> tuple:
    """Find the most expensive price times and return index ranges of ±3.5 days around them.

    Args:
        n (pypsa.Network): The network object.
        summer_start (str, optional): start of the summer period. Defaults to "2060-04-01".
        summer_end (str, optional): end of the summer period. Defaults to "2060-10-06".

    Returns:
        tuple: Index ranges of ±3.5 days around the winter_max and summer_max.
    """
    prices_w = (
        -1
        * n.statistics.revenue(comps="Load", bus_carrier="AC", aggregate_time=False)
        .T.resample("W")
        .sum()
        / n.statistics.withdrawal(comps="Load", bus_carrier="AC", aggregate_time=False)
        .T.resample("W")
        .sum()
    )

    summer = prices_w.query("snapshot > @summer_start and snapshot < @summer_end")
    summer_peak = summer.idxmax().iloc[0]
    summer_peak_w = n.snapshots[
        (n.snapshots >= summer_peak - pd.Timedelta(days=3.5))
        & (n.snapshots <= summer_peak + pd.Timedelta(days=3.5))
    ]
    winter_peak = prices_w.loc[~prices_w.index.isin(summer.index)].idxmax().iloc[0]
    winter_peak_w = n.snapshots[
        (n.snapshots >= winter_peak - pd.Timedelta(days=3.5))
        & (n.snapshots <= winter_peak + pd.Timedelta(days=3.5))
    ]

    return winter_peak_w, summer_peak_w


def label_stacked_bars(ax: object, nbars: int, fontsize=8, small_values=350):
    """Add value labels to stacked bar charts.

    Args:
        ax (object): The matplotlib Axes object containing the stacked bar chart.
        nbars (int): The number of bars in the stacked chart.
        fontsize (int, optional): Font size for the labels. Defaults to 8.
        small_values (int, optional): Threshold for small values. Small values
            adjacent to one another are not printed to avoid overlap. Defaults to 350.
    """

    # reorganize patches by bar
    stacked = [ax.patches[i::nbars] for i in range(len(ax.patches) // nbars)]
    # loop over bars and patches, so we can avoid overlapping labels
    for stacked_bar in stacked:
        prev = 0
        cutoff = 100
        for bar in stacked_bar:
            value = round(bar.get_height())
            yoffset = -1.8 * fontsize * np.sign(value)
            if abs(value) < cutoff:
                continue
            if 0 < value < small_values:
                yoffset -= 20
            elif -1 * small_values < value < 0 and (prev < -1 * small_values or prev > 0):
                yoffset += 50
            ax.text(
                # Put the text in the middle of each bar. get_x returns the start
                # so we add half the width to get to the middle.
                bar.get_x() + bar.get_width() / 2,
                # Vertically, add the height of the bar to the start of the bar,
                # along with the offset.
                bar.get_height() / 2 + bar.get_y() + yoffset,
                # This is actual value we'll show.
                value,
                # Center the labels and style them a bit.
                ha="center",
                color="w",
                weight="bold",
                size=fontsize,
            )

            prev = value


def make_nice_tech_colors(tech_colors: dict, nice_names: dict) -> dict:
    """Add the nice names to the tech_colors dict keys

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


def get_solver_tolerance(config: dict, tol_name="BarConvTol") -> float:
    """Get the solver tolerance from the config

    Args:
        config (dict): the config
        tol_name (str): the name of the tolerance option. Defaults to "BarConvTol"

    Returns:
        float: the value
    """
    solver_opts = config["solving"]["solver"]["options"]
    return config["solving"]["solver_options"][solver_opts][tol_name]


def find_numerical_zeros(n, config, tolerance_name="BarConvTol") -> list:
    """
    Identify numerical zeros in the network's optimization results.

    This function checks for numerical zeros in the network's optimization results,
    such as link capacities or weighted prices, based on a specified solver tolerance.

    Args:
        n (pypsa.Network): The PyPSA network object containing optimization results.
        config (dict): Configuration dictionary containing solver options.
        tolerance_name (str): The name of the solver tolerance option to use.
                Defaults to "BarConvTol".

    Returns:
        list: A list of items (e.g., links or buses) where numerical zeros are detected.
    """

    tol = get_solver_tolerance(config, tolerance_name)
    threshold = n.objective * float(tol)
    costs = pd.concat([n.statistics.expanded_capex(), n.statistics.opex()], axis=1)
    return costs.fillna(0).sum(axis=1).loc[costs.sum(axis=1) < threshold].index


def rename_techs(label: list) -> list:
    """From pypsa-Eur

    Args:
        label (str): a list of labels

    Returns:
        list: renamed labels
    """
    prefix_to_remove = [
        "central ",
        "decentral ",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "H2": "H2",
        "coal cc": "CC",
    }

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
        NAN_COLOR = config["plotting"].get("nan_color", "lightgrey")
        n.carriers["color"].fillna(NAN_COLOR, inplace=True)


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


def determine_plottable(n: pypsa.Network):
    """Determine whether links should be plotted

    Args:
        n (pypsa.Network): the pypsa network object
    """
    for c in n.iterate_components(n.branch_components):
        c.df["plottable"] = c.df.bus0.map(n.buses.location != "") & c.df.bus1.map(
            n.buses.location != ""
        )

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


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Args:
        data (np.ndarray): The data to plot.
        row_labels (list): The labels for the rows.
        col_labels (list): The labels for the columns.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        cbar_kw (dict, optional): Arguments to pass to colorbar. Defaults to {}.
        cbarlabel (str, optional): The label for the colorbar. Defaults to "".
        **kwargs: Additional arguments for imshow.

    Returns:
        im: The image.
        cbar: The colorbar.
    """
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(data, aspect="auto", interpolation="none", **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    return im, cbar


def annotate_heatmap(
    im, data=None, valfmt="{x:.1f}", textcolors=("black", "white"), threshold=None, **textkw
):
    """
    Annotate a heatmap.

    Args:
        im: The AxesImage to annotate.
        data (np.ndarray, optional): Data to annotate. Defaults to im.get_array().
        valfmt (str, optional): Format for values. Defaults to "{x:.1f}".
        textcolors (tuple, optional): Colors for values below/above threshold. Defaults to ("black", "white").
        threshold (float, optional): Value in data units according to which the colors are applied. Defaults to half the max.
        **textkw: Additional arguments for text.

    Returns:
        list: List of text annotations.
    """
    if data is None:
        data = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt.format(x=data[i, j]), **kw)
            texts.append(text)
    return texts

def setup_plot_export_hook(plot_accessor_class, export_dir="plot_exports", verbose=True):
    """Setup a monkey patch to auto-export data to CSV whenever pandas plots are created.
    
    Args:
        plot_accessor_class: The PlotAccessor class to patch (e.g., pandas.plotting.PlotAccessor).
        export_dir (str, optional): Directory where CSV exports will be saved. 
            Defaults to "plot_exports".
        verbose (bool, optional): Whether to print export messages. Defaults to True.
    
    Returns:
        callable: Function to remove the patch and restore original behavior.
    
    Example:
        >>> from pandas.plotting import PlotAccessor
        >>> remove_hook = setup_plot_export_hook(PlotAccessor)
        >>> # ... do plotting ...
        >>> remove_hook()  # Restore original behavior
    """
    import os
    import time
    import pandas as pd
    
    # Create export directory
    os.makedirs(export_dir, exist_ok=True)
    
    # Store original __call__ if not already stored
    if not hasattr(plot_accessor_class, '_original_call'):
        plot_accessor_class._original_call = plot_accessor_class.__call__
    
    def patched_plot_call(self, *args, **kwargs):
        """Patched __call__ method for PlotAccessor to export data before plotting."""
        # Create timestamped filename
        if 'fname' in kwargs:
            fname = kwargs.pop('fname')
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(export_dir, f"plot_export_{ts}.csv")
        
        # Export the data
        if isinstance(self._parent, pd.Series):
            self._parent.to_frame().to_csv(fname, index=True)
        else:
            self._parent.to_csv(fname, index=True)
        
        if verbose:
            print(f"[pandas-plot-hook] Exported plotted data to {fname}")
        
        # Call the original __call__ method
        return self._original_call(*args, **kwargs)
    
    # Apply the patch
    plot_accessor_class.__call__ = patched_plot_call
    
    # Return function to remove the patch
    def remove_hook():
        """Remove the plot export hook and restore original behavior."""
        if hasattr(plot_accessor_class, '_original_call'):
            plot_accessor_class.__call__ = plot_accessor_class._original_call
            delattr(plot_accessor_class, '_original_call')
            if verbose:
                print("[pandas-plot-hook] Hook removed, original behavior restored.")
    
    return remove_hook


if __name__ == "__main__":
    set_plot_style()
    plt.plot([1, 2, 3, 4])
    plt.show()
