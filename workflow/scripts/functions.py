# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: CC0-1.0
""" 
Maths calculations used in the PyPSA-China workflow."""
import numpy as np
from scipy import interpolate
import pyproj

from math import radians, cos, sin, asin, sqrt
from functools import partial

# TODO make function
# polynomial centroid for plotting
get_poly_center = lambda poly: (poly.centroid.xy[0][0], poly.centroid.xy[1][0])


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
    returns area in km^2."""

    project = partial(
        pyproj.transform, pyproj.Proj(init="epsg:4326"), pyproj.Proj(proj="aea")  # Source: Lon-Lat
    )  # Target: Albers Equal Area Conical https://en.wikipedia.org/wiki/Albers_projection
    # TODO fix
    new_geometry = transform(project, geometry)

    # default area is in m^2
    return new_geometry.area / 1e6


# TODO fix this/ DELETE
def HVAC_cost_curve(distance):
    """ Calculate the cost of HVAC lines based on distance.
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
