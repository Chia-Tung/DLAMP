from datetime import datetime

import matplotlib as mpl
import numpy as np

# Constant
VAR_SUFFIX = "WE01H0202500"

# Path
BLACKLIST_PATH = "./assets/blacklist.txt"
CHECKPOINT_DIR = "./checkpoints/"
COUNTY_SHP_PATH = "./assets/town_shp/COUNTY_MOI_1090820.shp"
DATA_PATH = "/work/dong1128/rwrf/"
FIGURE_PATH = "./gallery/"

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
        # == harsh northward turning ==#
        # datetime(2022, 9, 3), # TC HINNAMNOR
        datetime(2022, 9, 12),  # TC MUIFA
        # datetime(2021, 7, 23), # TC IN-FA
        # == north-eastern wind accompanied ==#
        datetime(2022, 10, 16),  # TC NESAT
        # datetime(2022, 10, 31),  # TC NALGAE
        # == pass by northern Taiwan ==#
        datetime(2020, 8, 3),  # TC HAGUPIT
    ],
}
