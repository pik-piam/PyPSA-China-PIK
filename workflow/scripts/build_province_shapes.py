import cartopy.io.shapereader as shpreader
import geopandas as gpd
import os.path
import logging

from _helpers import configure_logging, mock_snakemake
from constants import PROV_NAMES, CRS

# TODO integrate constants with repo/config files/snakefile
NATURAL_EARTH_RESOLUTION = "10m"  # 1:10m scale
# first administration level
NATURAL_EARTH_DATA_SET = "admin_1_states_provinces"

# TODO fix this messy path
DEFAULT_SHAPE_OUTPATH = "resources/data/province_shapes/CHN_adm1.shp"

logger = logging.getLogger(__name__)


def fetch_natural_earth_records(country_iso2_code="CN") -> object:
    """fetch the province/state level (1st admin level) from the NATURAL_EARTH data store and make a file

    Args:
        country_iso2_code (str, optional): the country code (iso_a2) for which
          provincial records will be extracted. None will not filter (untestetd) Defaults to 'CN'
    Returns:
        Records: the natural earth records
    """

    shpfilename = shpreader.natural_earth(
        resolution=NATURAL_EARTH_RESOLUTION, category="cultural", name=NATURAL_EARTH_DATA_SET
    )
    reader = shpreader.Reader(shpfilename)
    logger.info("Succesfully downloaded natural earth shapefiles")
    provinces_states = reader.records()

    def filter_country_code(records: object, target_iso_a2_code="CN") -> list:
        """filter provincial/state (admin level 1) records for one country

        Args:
            records (shpreader.Reader.records): the records object from cartopy shpreader for natural earth dataset
            target_iso_a2_code (str, optional): the country code (iso_a2) for which provincial records will be extracted. Defaults to 'CN'.

        Returns:
            list: records list
        """
        results = []
        for rec in records:
            if rec.attributes["iso_a2"] == target_iso_a2_code:
                results.append(rec)

        return results

    # TODO test with none
    if country_iso2_code is not None:
        provinces_states = filter_country_code(
            provinces_states, target_iso_a2_code=country_iso2_code
        )

    return provinces_states


def records_to_data_frame(records: object) -> gpd.GeoDataFrame:
    """dump irrelevant info and make records into a GeoDataFrame that matches the PROV_NAMES

    Args:
        records (object): the cartopy shpread records from natural earth

    Returns:
        gpd.GeoDataFrame: the cleaned up & sorted data in a format that can be saved
    """

    records[0].attributes["name"]
    d = {"province": [r.attributes["name_en"] for r in records]}
    geo = [r.geometry for r in records]
    gdf = gpd.GeoDataFrame(d, geometry=geo)
    gdf.sort_values(by="province", inplace=True)
    # remove white spaces
    gdf["province"] = gdf.province.str.replace(" ", "")

    filtered = gdf[gdf.province.isin(PROV_NAMES)]

    if not filtered.province.to_list() == sorted(PROV_NAMES):
        raise ValueError(
            "Built cut-out does not have the right provinces - do your province lists have white spaces?"
        )

    return filtered


def save_province_data(
    provinces_gdf: gpd.GeoDataFrame,
    crs: int = CRS,
    output_file: os.PathLike = DEFAULT_SHAPE_OUTPATH,
):
    """save to file

    Args:
        provinces_gdf (GeoDataFrame): the cleaned up province records
        crs (int, optional): the crs in epsg format. Defaults to CRS.
        output_file (os.pathlike): the output path. defaults to DEFAULT_SHAPE_OUTPATH
    """
    provinces_gdf.set_crs(epsg=crs, inplace=True)  # WGS84
    provinces_gdf.to_file(os.path.abspath(output_file))


if __name__ == "__main__":
    if not "snakemake" in globals():
        snakemake = mock_snakemake("build_province_shapes")
    configure_logging(snakemake, logger=logger)
    records = fetch_natural_earth_records(country_iso2_code="CN")
    provinces_gdf = records_to_data_frame(records)
    save_province_data(provinces_gdf, CRS, DEFAULT_SHAPE_OUTPATH)

    logger.info("Province shapes successfully built")
