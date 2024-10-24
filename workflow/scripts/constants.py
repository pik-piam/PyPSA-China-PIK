import numpy as np

CRS = 4326  # WGS84
YEARBOOK_DATA2POP = 1e4
POP_YEAR = 2020
FREQ = "1H"

PROV_NAMES = [
    "Anhui",
    "Beijing",
    "Chongqing",
    "Fujian",
    "Gansu",
    "Guangdong",
    "Guangxi",
    "Guizhou",
    "Hainan",
    "Hebei",
    "Heilongjiang",
    "Henan",
    "Hubei",
    "Hunan",
    "InnerMongolia",
    "Jiangsu",
    "Jiangxi",
    "Jilin",
    "Liaoning",
    "Ningxia",
    "Qinghai",
    "Shaanxi",
    "Shandong",
    "Shanghai",
    "Shanxi",
    "Sichuan",
    "Tianjin",
    "Tibet",
    "Xinjiang",
    "Yunnan",
    "Zhejiang",
]

OFFSHORE_WIND_NODES = np.array(
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
