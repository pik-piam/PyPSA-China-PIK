import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
import geopandas as gpd
import xarray as xr

from _plot_utilities import set_plot_style
from readers_geospatial import read_province_shapes
from _helpers import (
    configure_logging,
    mock_snakemake,
    set_plot_test_backend,
)

logger = logging.getLogger(__name__)


def plot_average_distances(distances: xr.DataArray, ax: plt.Axes = None)-> tuple[plt.Figure, plt.Axes]:
    """Plot the average distances to the node (region com/repr point) for each vre class
    Args:
        distances (xr.DataArray): the average distances for each class to the node 
        ax (plt.Axes, optional): the axes to plot on. Defaults to None.
    Returns:
        tuple[plt.Figure, plt.Axes]: the figure and axes
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    distances.to_series().plot.hist(
        bins=30,
        figsize=(10, 6),
        title="Frequency Distribution of Average Distance",
        xlabel="Average Distance (km)",
        ylabel="Frequency",
        # color="skyblue",
        edgecolor="black",
    )
    fig.tight_layout()
    return fig, ax


def plot_resource_class_bins(
    resource_classes: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    technology: str,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Map of VRE grades (by grade/ bin number) for each node
    Args:
        resource_classes (gpd.GeoDataFrame): the resource classes
        regions (gpd.GeoDataFrame): the regions/node regions
        technology (str): the technology name
        ax (plt.Axes, optional): the axes to plot on. Defaults to None.
    Returns:
        tuple[plt.Figure, plt.Axes]: the figure and axes
    """

    if not ax:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    nbins = resource_classes.bin.nunique()

    cmap = plt.cm.get_cmap("magma_r", nbins)  # Inverted discrete colormap with `nbins` colors
    cmap.set_bad(color="lightgrey")  # Set color for missing values
    norm = mcolors.BoundaryNorm(boundaries=range(nbins + 1), ncolors=nbins)

    # Plot the class regions
    resource_classes.reset_index().plot("bin", cmap=cmap, legend=True, ax=ax, norm=norm)
    ax.set_title(f"Resource Classes for {technology.capitalize()} simple")

    # Add administrative region borders
    regions.boundary.plot(ax=ax, color="black", linewidth=0.5, linestyle="--")

    # coords.plot(ax=ax, color="black", markersize=1)
    fig.tight_layout()

    return fig, ax


def plot_resource_class_cfs(
    resource_classes: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    technology: str,
    ax: plt.Axes = None,
):
    """Map of VRE capacity factors for each node and vre grade
    Args:
        resource_classes (gpd.GeoDataFrame): the resource classes
        regions (gpd.GeoDataFrame): the regions/node regions
        technology (str): the technology name
        ax (plt.Axes, optional): the axes to plot on. Defaults to None.
    Returns:
        tuple[plt.Figure, plt.Axes]: the figure and axes
    """

    if not ax:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    # Plot the class regions
    resource_classes.plot("cf", cmap="magma_r", legend=True, ax=ax)
    ax.set_title(f"Resource Classes for {technology.capitalize()} simple")

    # Add administrative region borders
    regions.boundary.plot(ax=ax, color="black", linewidth=0.5, linestyle="--")

    # coords.plot(ax=ax, color="black", markersize=1)
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_renewable_classes",
            technology="onwind",
            rc_params="n3_min_cf_delta0.05",
        )
    set_plot_test_backend(snakemake.config)
    configure_logging(snakemake, logger=logger)

    regions = read_province_shapes(snakemake.input.province_shape)
    tech = snakemake.wildcards.technology
    resource_classes = gpd.read_file(snakemake.input.renewable_classes)

    fig, ax = plot_resource_class_bins(resource_classes, regions, tech)
    fig.savefig(
        snakemake.output.renewable_grades_bins,
        bbox_inches="tight",
    )

    fig, ax = plot_resource_class_cfs(resource_classes, regions, tech)
    fig.savefig(
        snakemake.output.renewable_grades_cf,
        bbox_inches="tight",
    )

    distances = xr.open_dataset(snakemake.input.average_distance)
    fig, ax = plot_average_distances(distances)
    distances.close()
    fig.savefig(
        snakemake.output.average_distances,
        bbox_inches="tight",
    )
