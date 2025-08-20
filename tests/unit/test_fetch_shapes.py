import geopandas as gpd
import pytest
from constants import PROV_NAMES
from fetch_shapes import (
    eez_by_region,
    fetch_country_shape,
    fetch_maritime_eez,
    fetch_natural_earth_shape,
    fetch_province_shapes,
)
from pandas import DataFrame
from shapely.geometry import Polygon


@pytest.fixture
def mock_country_shape():
    """Fixture for mock country shape GeoDataFrame"""
    data = {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]}
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def mock_province_shapes():
    """Fixture for mock province shapes GeoDataFrame"""
    data = {
        "province": ["Shanghai", "Jiangsu"],  # must be in prov names
        "geometry": [
            Polygon([(0, 0), (0.6, 0), (0.6, 0.5), (0, 0.5), (0, 0)]),
            Polygon([(0.6, 0), (1, 0), (1, 0.5), (0.6, 0.5), (0.6, 0)]),
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def mock_eez():
    """Fixture for mock EEZ GeoDataFrame"""
    data = {"geometry": [Polygon([(0, 0.5), (1, 0.5), (1, 1), (0, 1), (0, 0.5)])]}
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def calc_overlap_area(gdf: gpd.GeoDataFrame) -> DataFrame:
    """Calculate the overlap area between rows

    Args:
        gdf (gpd.GeoDataFrame): the geodataframe for which to calculate overlap areas

    Returns:
        DataFrame: Index x Index float overlap areas dataframe
    """
    return gdf.apply(
        lambda row: gdf[gdf.index != row.name].geometry.apply(
            lambda geom: (
                row.geometry.intersection(geom).area if row.geometry.intersects(geom) else 0
            )
        ),
        axis=1,
    )


def test_fetch_natural_earth_shape():
    result = fetch_natural_earth_shape("admin_0_countries", "ADMIN", "China")
    assert not result.empty
    assert "geometry" in result.columns
    assert result.crs.to_string() == "EPSG:4326"


def test_fetch_country_shape(tmp_path):
    outp_path = tmp_path / "country_shape.geojson"
    fetch_country_shape(outp_path)
    assert outp_path.exists()


def test_fetch_province_shapes():
    result = fetch_province_shapes()

    assert not result.empty
    assert all(prov in result.index for prov in PROV_NAMES)
    assert "geometry" in result.columns
    assert result.crs.to_string() == "EPSG:4326"


def test_fetch_maritime_eez():
    result = fetch_maritime_eez("Chinese")
    assert not result.empty
    assert "geometry" in result.columns
    assert result.crs.to_string() == "EPSG:4326"


def test_eez_by_region(mock_eez, mock_province_shapes):
    result = eez_by_region(mock_eez, mock_province_shapes, prov_key="province", simplify_tol=0)
    assert not result.empty
    assert "geometry" in result.columns
    assert result.crs == mock_eez.crs
    # Expecting two regions after splitting, since have a square divided in 3, with bottom half land
    assert len(result) == 2
    # bottom half must be of different area but given shapes, will be area preserving
    assert result.area.sum() == mock_eez.area.sum()
    assert "Shanghai" in result.index
    overlap_areas = calc_overlap_area(result)
    assert overlap_areas.max().max() <= 0.01
