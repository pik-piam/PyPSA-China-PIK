# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: CC0-1.0

import numpy as np
from scipy import interpolate

pro_names = np.array(['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong',
       'Guangxi', 'Guizhou', 'Hainan', 'Hebei', 'Heilongjiang', 'Henan',
       'Hubei', 'Hunan','InnerMongolia', 'Jiangsu', 'Jiangxi', 'Jilin', 'Liaoning',
       'Ningxia', 'Qinghai', 'Shaanxi', 'Shandong',
       'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Tibet', 'Xinjiang',
       'Yunnan', 'Zhejiang'],
      dtype=str)

offwind_nodes = np.array(['Fujian', 'Guangdong', 'Guangxi', 'Hainan', 'Hebei', 'Jiangsu',
       'Liaoning', 'Shandong', 'Shanghai', 'Tianjin', 'Zhejiang'],
      dtype=str)

def HVAC_cost_curve(distance):

  d = np.array([608, 656, 730, 780, 903, 920, 1300])
  c = 1000 / 7.5 * np.array([5.5, 4.71, 5.5, 5.57, 5.5, 5.5, 5.51])

  c_func = interpolate.interp1d(d, c, fill_value='extrapolate')
  c_results = c_func(distance)

  return c_results
