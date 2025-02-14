"""
Data fetch operation for region/province/country shapes
"""

import geopandas as gpd
import cartopy.io.shapereader as shpreader
import requests
import logging
from os import PathLike

from constants import CRS, COUNTRY_ISO, COUNTRY_NAME, PROV_NAMES, EEZ_PREFIX
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


def eez_by_region(
    eez: gpd.GeoDataFrame, province_shapes: gpd.GeoDataFrame, prov_key="region"
) -> gpd.GeoDataFrame:
    """break up the eez by admin1 regions based on voronoi polygons of the centroids

    Args:
        eez (gpd.GeoDataFrame): _description_
        province_shapes (gpd.GeoDataFrame): _description_
        prov_key (str, optional): name of the provinces col in province_shapes. Defaults to "region".

    Returns:
        gpd.GeoDataFrame: _description_
    """

    voronoi_cells = gpd.GeoDataFrame(
        geometry=province_shapes.centroid.voronoi_polygons(), crs=province_shapes.crs
    )
    prov_centroids = province_shapes.copy()
    prov_centroids.geometry = prov_centroids.centroid
    # need to assign cells to province
    voronoi_cells = voronoi_cells.sjoin(prov_centroids, predicate="contains").reset_index()
    if "index_right" in voronoi_cells.columns:
        voronoi_cells.drop(columns=["index_right"], inplace=True)
    logger.debug(f"Voronoi cells: {voronoi_cells}")
    # check the below with ez.overlay(voronoi_cells, how="intersection").boundary.plot()
    eez_by_region = voronoi_cells.overlay(eez, how="intersection")[[prov_key, "geometry"]]
    return eez_by_region[eez_by_region[prov_key].isin(PROV_NAMES)].set_index(prov_key)


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
    eez_by_region(eez_country, regions, prov_key="province").to_file(
        snakemake.output.offshore_shapes, driver="GeoJSON"
    )

    logger.info("Regions succesfully built")
