from __future__ import annotations

from enum import StrEnum


class DataType(StrEnum):
    GeoHeight = "000"
    P = "010"
    T = "100"
    U = "200"
    V = "210"
    Lat = "LAT"
    Lon = "LON"
    Radar = "MOS"


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

    def is_surface(self) -> bool:
        return self in [self.Surface, self.SeaSurface, self.NoRule]
