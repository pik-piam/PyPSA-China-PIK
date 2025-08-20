# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: CC0-1.0
"""
Maths calculations used in the PyPSA-China workflow."""

from functools import partial
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import pyproj
from pyproj import transform
from scipy import interpolate


# TODO make function
# polynomial centroid for plotting
def get_poly_center(poly):
    """Get the centroid of a polygon."""
    return (poly.centroid.xy[0][0], poly.centroid.xy[1][0])


def cartesian(s1: pd.Series, s2: pd.Series) -> pd.DataFrame:
    """
    Compute the Cartesian product of two pandas Series.

    Args:
        s1 (pd.Series): first series
        s2 (pd.Series): second series
    Returns:
        pd.DataFrame: A DataFrame representing the Cartesian product of s1 and s2.

    Examples:
        >>> s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
        >>> s2 = pd.Series([4, 5, 6], index=["d", "e", "f"])
        >>> cartesian(s1, s2)
        d  e   f
        a  4  5   6
        b  8 10  12
        c 12 15  18
    """
    return pd.DataFrame(np.outer(s1, s2), index=s1.index, columns=s2.index)


def haversine(p1, p2) -> float:
    """Calculate the great circle distance in km between two points on
    the earth (specified in decimal degrees)

    Args:
        p1 (shapely.Point): location 1 in decimal deg
        p2 (shapely.Point): location 2 in decimal deg

    Returns:
        float: great circle distance in [km]
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


# This function follows http://toblerity.org/shapely/manual.html
def area_from_lon_lat_poly(geometry):
    """For shapely geometry in lon-lat coordinates,
    returns area in km^2.
    """

    project = partial(
        pyproj.transform,
        pyproj.Proj(init="epsg:4326"),
        pyproj.Proj(proj="aea"),  # Source: Lon-Lat
    )  # Target: Albers Equal Area Conical https://en.wikipedia.org/wiki/Albers_projection
    # TODO fix
    new_geometry = transform(project, geometry)

    # default area is in m^2
    return new_geometry.area / 1e6


# TODO fix this/ DELETE
def HVAC_cost_curve(distance):
    """Calculate the cost of HVAC lines based on distance.

    Args:
        distance (float): distance in km
    Returns:
        float: cost in currency
    """
    d = np.array([608, 656, 730, 780, 903, 920, 1300])
    c = 1000 / 7.5 * np.array([5.5, 4.71, 5.5, 5.57, 5.5, 5.5, 5.51])

    c_func = interpolate.interp1d(d, c, fill_value="extrapolate")
    c_results = c_func(distance)

    return c_results
