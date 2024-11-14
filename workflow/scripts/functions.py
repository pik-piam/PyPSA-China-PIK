# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: CC0-1.0

import numpy as np
from scipy import interpolate

from math import radians, cos, sin, asin, sqrt


offwind_nodes = np.array(
    [
        "Fujian",
        "Guangdong",
        "Guangxi",
        "Hainan",
        "Hebei",
        "Jiangsu",
        "Liaoning",
        "Shandong",
        "Shanghai",
        "Tianjin",
        "Zhejiang",
    ],
    dtype=str,
)

# polynomial centroid for plotting
get_poly_center = lambda poly: (poly.centroid.xy[0][0], poly.centroid.xy[1][0])


def haversine(p1, p2):
    """Calculate the great circle distance in km between two points on
    the earth (specified in decimal degrees)
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


# TODO fix this
def HVAC_cost_curve(distance):

    d = np.array([608, 656, 730, 780, 903, 920, 1300])
    c = 1000 / 7.5 * np.array([5.5, 4.71, 5.5, 5.57, 5.5, 5.5, 5.51])

    c_func = interpolate.interp1d(d, c, fill_value="extrapolate")
    c_results = c_func(distance)

    return c_results
