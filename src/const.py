from datetime import datetime

import matplotlib as mpl
import numpy as np

# Constant
DATA_SOURCE = "CWA_RWRF"
VAR_SUFFIX = "WE01H0202500"

# Path
BLACKLIST_PATH = "./assets/blacklist_complete_rwrf.txt"
CHECKPOINT_DIR = "./checkpoints/"
LAND_SEA_MASK_PATH = "./assets/constant_masks/land_sea_mask_2km.npy"
TOPOGRAPHY_MASK_PATH = "./assets/constant_masks/topography_mask_2km.npy"
COUNTY_SHP_PATH = "./assets/town_shp/COUNTY_MOI_1090820.shp"
STANDARDIZATION_PATH = "./assets/standardization.json"
DATA_PATH = "/work/dong1128/rwrf_data/"
FIGURE_PATH = "./gallery/"
DATA_CONFIG_PATH = "./config/data/rwrf_dense.yaml"

# Radar color bar
DBZ_LV = np.arange(0, 66, 1)
DBZ_COLOR = np.concatenate(
    [
        np.array([[255, 255, 255]]),  # 0
        np.array(
            [np.linspace(0, 0, 14), np.linspace(255, 0, 14), np.linspace(255, 255, 14)]
        ).T,  # 1~14
        np.array(
            [np.linspace(0, 0, 11), np.linspace(255, 150, 11), np.linspace(0, 0, 11)]
        ).T,  # 15~25
        np.array(
            [np.linspace(51, 204, 4), np.linspace(171, 234, 4), np.linspace(0, 0, 4)]
        ).T,  # 26~29
        np.array(
            [np.linspace(255, 255, 5), np.linspace(255, 211, 5), np.linspace(0, 0, 5)]
        ).T,  # 30~34
        np.array(
            [np.linspace(255, 255, 6), np.linspace(200, 120, 6), np.linspace(0, 0, 6)]
        ).T,  # 35~40
        np.array(
            [np.linspace(255, 255, 5), np.linspace(96, 0, 5), np.linspace(0, 0, 5)]
        ).T,  # 41~45
        np.array(
            [np.linspace(244, 150, 10), np.linspace(0, 0, 10), np.linspace(0, 0, 10)]
        ).T,  # 46~55
        np.array(
            [np.linspace(171, 255, 5), np.linspace(0, 0, 5), np.linspace(51, 255, 5)]
        ).T,  # 56~60
        np.array(
            [np.linspace(234, 150, 5), np.linspace(0, 0, 5), np.linspace(255, 255, 5)]
        ).T,  # 61~65
    ]
)
DBZ_COLOR = mpl.colors.ListedColormap(DBZ_COLOR / 255)
DBZ_NORM = mpl.colors.BoundaryNorm(DBZ_LV, DBZ_COLOR.N)

# Rain rate color bar
RR_LV = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
RR_COLOR = mpl.colors.ListedColormap(
    [
        "#FFFFFF",
        "#9CFCFF",
        "#03C8FF",
        "#059BFF",
        "#0363FF",
        "#059902",
        "#39FF03",
        "#FFFB03",
        "#FFC800",
        "#FF9500",
        "#FF0000",
        "#CC0000",
        "#990000",
        "#960099",
        "#C900CC",
        "#FB00FF",
        "#FDC9FF",
    ]
)
RR_NORM = mpl.colors.BoundaryNorm(RR_LV, RR_COLOR.N)

# Wind speed color bar and intervals
WSP_LV = [
    0,
    4,
    6,
    8,
    10,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    34,
    36,
    38,
    40,
    43,
    46,
    49,
    52,
    55,
    58,
    61,
    64,
    67,
    70,
    73,
    76,
    79,
    82,
    85,
]
WSP_COLOR = [
    "#ffffff",
    "#80ffff",
    "#6fedf1",
    "#5fdde4",
    "#50cdd5",
    "#40bbc7",
    "#2facba",
    "#1f9bac",
    "#108c9f",
    "#007a92",
    "#00b432",
    "#33c341",
    "#67d251",
    "#99e060",
    "#cbf06f",
    "#ffff80",
    "#ffdd52",
    "#ffdc52",
    "#ffa63e",
    "#ff6d29",
    "#ff3713",
    "#ff0000",
    "#d70000",
    "#af0000",
    "#870000",
    "#5f0000",
    "#aa00ff",
    "#b722fe",
    "#c446ff",
    "#d46aff",
    "#e38dff",
    "#f1b1ff",
    "#ffd3ff",
    "#ffc6ea",
    "#ffb6d5",
    "#ffa6c1",
    "#ff97ac",
    "#ff8798",
    "#fe7884",
    "#ff696e",
    "#ff595a",
    "#e74954",
    "#cc3a4c",
    "#b22846",
    "#9a1941",
]

# Temperature color bar and intervals
TEMP_LV = [
    -15,
    -14,
    -13,
    -12,
    -11,
    -10,
    -9,
    -8,
    -7,
    -6,
    -5,
    -4,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
]
TEMP_COLOR = [
    "#a8acdf",
    "#9092d4",
    "#777acc",
    "#5f63c3",
    "#4949b6",
    "#4655c3",
    "#435aca",
    "#3b6ddf",
    "#3979ef",
    "#3386f5",
    "#2d99fe",
    "#22affe",
    "#1bc2ff",
    "#0ee6fe",
    "#07fbff",
    "#6ee699",
    "#65e08d",
    "#4fd06f",
    "#45c65f",
    "#34bd4b",
    "#28b338",
    "#16a71f",
    "#16a111",
    "#43b121",
    "#66c034",
    "#78c63c",
    "#9ad54d",
    "#c5e763",
    "#e1f26f",
    "#fef87b",
    "#fdeb76",
    "#fad66a",
    "#f9c662",
    "#f8b558",
    "#f6a24e",
    "#ef9043",
    "#e4692c",
    "#e15f27",
    "#cc3513",
    "#c8250a",
    "#c8250a",
]

# Evaluation cases
EVAL_CASES = {
    "one_day": [
        datetime(2021, 6, 4),  # ATS
        datetime(2022, 6, 24),  # ATS, observe graupel in Taipei
        datetime(2022, 8, 25),  # ATS
        datetime(2021, 8, 7),  # South-western flow + Tropical Depression
        datetime(2021, 8, 8),  # South-western flow
    ],
    "three_days": [
        # == harsh northward turning == #
        # datetime(2022, 9, 3), # TC HINNAMNOR
        datetime(2022, 9, 12),  # TC MUIFA
        # datetime(2021, 7, 23), # TC IN-FA
        # == north-eastern wind accompanied == #
        datetime(2022, 10, 16),  # TC NESAT
        # datetime(2022, 10, 31),  # TC NALGAE
        # == pass by northern Taiwan == #
        datetime(2020, 8, 3),  # TC HAGUPI
    ],
    "five_days": [
        # == pass by eastern Taiwan == #
        datetime(2023, 7, 26),  # TC DOKSURI
        # == landing == #
        # datetime(2023, 9, 3),  # TC HAIKUI
        datetime(2024, 7, 24),  # TC GAEMI
        # datetime(2024, 10, 31),  # TC Kong-rey
    ],
    "seven_days": [
        # == landing == #
        datetime(2024, 10, 3),  # TC Krathon
    ],
}
