import pytest
import geopandas as gpd
from shapely.geometry import Polygon

from fetch_shapes import (
    fetch_natural_earth_shape,
    fetch_country_shape,
    fetch_province_shapes,
    fetch_maritime_eez,
    eez_by_region,
)
from constants import PROV_NAMES


@pytest.fixture
def mock_country_shape():
    """Fixture for mock country shape GeoDataFrame"""
    data = {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]}
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def mock_province_shapes():
    """Fixture for mock province shapes GeoDataFrame"""
    data = {
        "province": ["Shanghai", "Anhui"],  # must be in prov names
        "geometry": [
            Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5), (0, 0)]),
            Polygon([(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 0)]),
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def mock_eez():
    """Fixture for mock EEZ GeoDataFrame"""
    data = {"geometry": [Polygon([(0, 0.5), (1, 0.5), (1, 1), (0, 1), (0, 0.5)])]}
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


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
    result = eez_by_region(mock_eez, mock_province_shapes, prov_key="province")
    assert not result.empty
    assert "geometry" in result.columns
    assert result.crs == mock_eez.crs
    assert len(result) == 2  # Expecting two regions after splitting
    assert result.area.iloc[0] == result.area.iloc[1]  # of same area
    assert "Shanghai" in result.index
