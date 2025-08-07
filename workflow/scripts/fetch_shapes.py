"""
Data fetch operation for region/province/country shapes
"""

import geopandas as gpd
import cartopy.io.shapereader as shpreader
import requests
import logging
import numpy as np
from os import PathLike
from pandas import DataFrame

from constants import (
    CRS,
    COUNTRY_ISO,
    COUNTRY_NAME,
    PROV_NAMES,
    EEZ_PREFIX,
    OFFSHORE_WIND_NODES,
)
from _helpers import mock_snakemake, configure_logging

NATURAL_EARTH_RESOLUTION = "10m"

logger = logging.getLogger(__name__)


def fetch_natural_earth_shape(
    dataset_name: str, filter_key: str, filter_value="China", region_key=None
) -> gpd.GeoDataFrame:
    """fetch region or country shape from natural earth dataset and filter

    Args:
        dataset_name (str): the name of the natural earth dataset to fetch
        filter_key (str): key to filter the records by
        filter_value (str|list, optional): filter pass value. Defaults to "China".

    Example:
        china country: build_natural_earth_shape("admin_0_countries", "ADMIN", "China")
        china provinces: build_natural_earth_shape("admin_1_states_provinces", "iso_a2", "CN", region_key = "name_en")

    Returns:
        gpd.GeoDataFrame: the filtered records
    """
    shpfilename = shpreader.natural_earth(
        resolution=NATURAL_EARTH_RESOLUTION, category="cultural", name=dataset_name
    )
    reader = shpreader.Reader(shpfilename)
    records = list(reader.records())
    if not region_key:
        region_key = filter_key
    if isinstance(filter_value, list):
        gdf = gpd.GeoDataFrame(
            [
                {"region": c.attributes[region_key], "geometry": c.geometry}
                for c in records
                if c.attributes[filter_key] in filter_value
            ]
        )
    else:
        gdf = gpd.GeoDataFrame(
            [
                {"region": c.attributes[region_key], "geometry": c.geometry}
                for c in records
                if c.attributes[filter_key] == filter_value
            ]
        )
    gdf.set_crs(epsg=CRS, inplace=True)
    return gdf


def fetch_country_shape(outp_path: PathLike):
    """fetch the country shape from natural earth and save it to the outpath

    Args:
        outp_path (PathLike): the path to save the country shape (geojson)
    """

    country_shape = fetch_natural_earth_shape("admin_0_countries", "ADMIN", COUNTRY_NAME)
    country_shape.set_index("region", inplace=True)
    country_shape.to_file(outp_path, driver="GeoJSON")


def fetch_province_shapes() -> gpd.GeoDataFrame:
    """fetch the province shapes from natural earth and save it to the outpath

    Returns:
        gpd.GeoDataFrame: the province shapes
    """

    province_shapes = fetch_natural_earth_shape(
        "admin_1_states_provinces", "iso_a2", COUNTRY_ISO, region_key="name_en"
    )
    province_shapes.rename(columns={"region": "province"}, inplace=True)
    province_shapes.province = province_shapes.province.str.replace(" ", "")
    province_shapes.sort_values("province", inplace=True)
    logger.debug("province shapes:\n", province_shapes)

    filtered = province_shapes[province_shapes["province"].isin(PROV_NAMES)]
    if (filtered["province"].unique() != sorted(PROV_NAMES)).all():
        logger.warning(
            f"Missing provinces: {set(PROV_NAMES) - set(province_shapes['province'].unique())}"
        )
    filtered.set_index("province", inplace=True)

    return filtered.sort_index()


def fetch_maritime_eez(zone_name: str) -> gpd.GeoDataFrame:
    """fetch maritime data for a country from Maritime Gazette API#
    (Royal marine institute of Flanders data base)

    Args:
        zone_name (str): the country's zone name, e.g "Chinese" for china

    Raises:
        requests.HTTPError: if the request fails
    Returns:
        dict: the maritime data
    """

    def find_record_id(zone_name: str) -> int:
        # get Maritime Gazette record ID for the country
        # the eez ID is 70: see https://www.marineregions.org/gazetteer.php?p=webservices&type=rest#/
        url = f"https://www.marineregions.org/rest/getGazetteerRecordsByName.json/{zone_name}/?like=true&fuzzy=false&typeID=70&offset=0&count=100"
        response = requests.get(url)
        if response.status_code != 200:
            raise requests.HTTPError(
                f"Failed to retrieve Maritime Gazette ID. Status code: {response.status_code}"
            )
        record_data = response.json()
        logger.debug(record_data)
        return [
            data
            for data in record_data
            if (data["status"] == "standard")
            and (data["preferredGazetteerName"].lower().find(zone_name.lower()) != -1)
        ][0]["MRGID"]

    mgrid = find_record_id(zone_name)
    logger.debug(f"Found Maritime Gazette ID for {zone_name}: {mgrid}")
    #  URL of the WFS service
    url = "https://geo.vliz.be/geoserver/wfs"
    # WFS request parameters + record ID filter
    params = dict(
        service="WFS",
        version="1.1.0",
        request="GetFeature",
        typeName="MarineRegions:eez",
        outputFormat="json",
        filter=f"<Filter><PropertyIsEqualTo><PropertyName>mrgid_eez</PropertyName><Literal>{mgrid}</Literal></PropertyIsEqualTo></Filter>",
    )

    # Fetch data from WFS using requests
    response_eez = requests.get(url, params=params)

    # Check for successful request
    if response_eez.status_code == 200:
        data = response_eez.json()
    else:
        logger.error(f"Error: {response_eez.status_code}")
        raise requests.HTTPError(
            f"Failed to retrieve Maritime Gazette data. Status code: {response_eez.status_code}"
        )
    if data["totalFeatures"] != 1:
        raise ValueError(f"Expected 1 feature, got {data['totalFeatures']}\n: {data}")
    crs = data["crs"]["properties"]["name"].split("EPSG::")[-1]
    eez = gpd.GeoDataFrame.from_features(data["features"])
    return eez.set_crs(epsg=crs)


def cut_smaller_from_larger(
    row: gpd.GeoSeries, gdf: gpd.GeoDataFrame, overlaps: DataFrame
) -> gpd.GeoSeries:
    """automatically assign overlapping area to the smaller region

    Example:
        areas_gdf.apply(cut_smaller_from_larger, args=(areas_gdf, overlaps), axis=1)

    Args:
        row (gpd.GeoSeries): the row from pandas apply
        gdf (gpd.GeoDataFrame): the geodataframe on which the operation is performed
        overlaps (DataFrame): the boolean overlap table

    Raises:
        ValueError: in case areas are exactly equal

    Returns:
        gpd.GeoSeries: the row with overlaps removed or not
    """
    ovrlap_idx = np.where(overlaps.loc[row.name].values == True)[0].tolist()
    for idx in ovrlap_idx:
        geom = gdf.iloc[idx].geometry
        if row.geometry.area > geom.area:
            row.geometry = row.geometry.difference(geom)
        elif row.geometry.area == geom.area:
            raise ValueError(f"Equal area overlap between {row.name} and {idx} - unhandled")
    return row


def has_overlap(gdf: gpd.GeoDataFrame) -> DataFrame:
    """Check for spatial overlaps across rows

    Args:
        gdf (gpd.GeoDataFrame): the geodataframe to check

    Returns:
        DataFrame: Index x Index boolean dataframe
    """
    return gdf.apply(
        lambda row: gdf[gdf.index != row.name].geometry.apply(
            lambda geom: row.geometry.intersects(geom)
        ),
        axis=1,
    )


def remove_overlaps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """remove inter row overlaps from a GeoDataFrame, cutting out the smaller region from the larger one

    Args:
        gdf (gpd.GeoDataFrame): the geodataframe to be treated

    Returns:
        gpd.GeoDataFrame: the treated geodataframe
    """
    overlaps = has_overlap(gdf)
    return gdf.apply(cut_smaller_from_larger, args=(gdf, overlaps), axis=1)


def eez_by_region(
    eez: gpd.GeoDataFrame,
    province_shapes: gpd.GeoDataFrame,
    prov_key="region",
    simplify_tol=0.5,
) -> gpd.GeoDataFrame:
    """break up the eez by admin1 regions based on voronoi polygons of the centroids

    Args:
        eez (gpd.GeoDataFrame): _description_
        province_shapes (gpd.GeoDataFrame): _description_
        prov_key (str, optional): name of the provinces col in province_shapes. Defaults to "region".
        simplify_tol (float, optional): tolerance for simplifying the voronoi polygons. Defaults to 0.5.

    Returns:
        gpd.GeoDataFrame: _description_
    """
    # generate voronoi cells (more than one per province & can overlap)
    voronois_simple = gpd.GeoDataFrame(
        geometry=province_shapes.simplify(tolerance=simplify_tol).voronoi_polygons(),
        crs=province_shapes.crs,
    )
    # assign region
    prov_voronoi = (
        voronois_simple.sjoin(province_shapes, predicate="intersects")
        .groupby(prov_key)
        .apply(lambda x: x.union_all("unary"))
    )
    prov_voronoi = gpd.GeoDataFrame(
        geometry=prov_voronoi.values,
        crs=province_shapes.crs,
        data={prov_key: prov_voronoi.index},
    )

    # remove overlaps
    gdf_ = remove_overlaps(prov_voronoi.set_index(prov_key))

    eez_prov = (
        gdf_.reset_index()
        .overlay(eez, how="intersection")[[prov_key, "geometry"]]
        .groupby(prov_key)
        .apply(lambda x: x.union_all("unary"))
    )
    eez_prov = gpd.GeoDataFrame(
        geometry=eez_prov.values,
        crs=province_shapes.crs,
        data={prov_key: eez_prov.index},
    )

    return eez_prov[eez_prov[prov_key].isin(OFFSHORE_WIND_NODES)].set_index(prov_key)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake("fetch_region_shapes")
    configure_logging(snakemake, logger=logger)

    logger.info(f"Fetching country shape {COUNTRY_NAME} from cartopy")
    fetch_country_shape(snakemake.output.country_shape)
    logger.info(f"Country shape saved to {snakemake.output.country_shape}")

    logger.info(f"Fetching province shapes for {COUNTRY_ISO} from cartopy")
    # TODO it would be better to filter by set regions after making the voronoi polygons
    regions = fetch_province_shapes()
    regions.to_file(snakemake.output.province_shapes, driver="GeoJSON")
    regions.to_file(snakemake.output.prov_shpfile)
    logger.info(f"Province shapes saved to {snakemake.output.province_shapes}")

    logger.info(f"Fetching maritime zones for EEZ prefix {EEZ_PREFIX}")
    eez_country = fetch_maritime_eez(EEZ_PREFIX)
    logger.info("Breaking by reion")
    tol = snakemake.config["fetch_regions"]["simplify_tol"]
    eez_by_region(eez_country, regions, prov_key="province", simplify_tol=tol).to_file(
        snakemake.output.offshore_shapes, driver="GeoJSON"
    )

    logger.info("Regions succesfully built")
