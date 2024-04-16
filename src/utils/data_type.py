from enum import StrEnum

from src.const import VAR_SUFFIX


class DataType(StrEnum):
    GeoHeight = "000"
    P = "010"
    T = "100"
    U = "200"
    V = "210"
    Lat = "LAT"
    Lon = "LON"
    Radar = "MOS"

    @classmethod
    def gen_dir_name(cls, var: str, lv: str):
        """
        A function to generate a directory name based on the given variables and level.

        Parameters:
            var (str): The variable to be included in the directory name.
            lv (str): The level to be included in the directory name.

        Returns:
            str: The generated directory name based on the input variables and level.
        """
        if var in ["Lat", "Lon", "Radar"]:
            lv = "NoRule"
        return f"{Level[lv]}{cls[var]}{VAR_SUFFIX}"


class Level(StrEnum):
    Hpa200 = "200"
    Hpa300 = "300"
    Hpa500 = "500"
    Hpa700 = "700"
    Hpa850 = "850"
    Hpa925 = "925"
    Hpa1000 = "H00"
    LowestModelLevel = "B00"
    Meter2 = "B02"
    Meter10 = "B10"
    Meter100 = "H10"
    Surface = "S00"
    SeaSurface = "W00"
    NoRule = "X00"
