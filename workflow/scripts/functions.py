# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: CC0-1.0

import numpy as np
from scipy import interpolate

offwind_nodes = np.array(['Fujian', 'Guangdong', 'Guangxi', 'Hainan', 'Hebei', 'Jiangsu',
       'Liaoning', 'Shandong', 'Shanghai', 'Tianjin', 'Zhejiang'],
      dtype=str)

# polynomial centroid for plotting
get_poly_center = lambda poly: (poly.centroid.xy[0][0],poly.centroid.xy[1][0])


def HVAC_cost_curve(distance):

  d = np.array([608, 656, 730, 780, 903, 920, 1300])
  c = 1000 / 7.5 * np.array([5.5, 4.71, 5.5, 5.57, 5.5, 5.5, 5.51])

  c_func = interpolate.interp1d(d, c, fill_value='extrapolate')
  c_results = c_func(distance)

  return c_results
