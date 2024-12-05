from __future__ import annotations

from enum import StrEnum


class DataType(StrEnum):
    Z = "000"
    P = "010"
    T = "100"
    U = "200"
    V = "210"
    W = "220"
    Qv = "300"
    Qr = "310"
    Qs = "320"
    Qg = "330"
    Qc = "340"
    Qi = "350"
    RH = "ReH"
    SST = "SST"
    PSFC = "SFP"
    PW = "PW"
    Lat = "LAT"
    Lon = "LON"
    Radar = "MOS"


class Level(StrEnum):
    Hpa100 = "100"
    Hpa150 = "150"
    Hpa200 = "200"
    Hpa250 = "250"
    Hpa300 = "300"
    Hpa350 = "350"
    Hpa400 = "400"
    Hpa450 = "450"
    Hpa500 = "500"
    Hpa550 = "550"
    Hpa600 = "600"
    Hpa650 = "650"
    Hpa700 = "700"
    Hpa750 = "750"
    Hpa800 = "800"
    Hpa850 = "850"
    Hpa900 = "900"
    Hpa925 = "925"
    Hpa950 = "950"
    Hpa975 = "975"
    Hpa1000 = "H00"
    LowestModelLevel = "B00"
    Meter2 = "B02"
    Meter10 = "B10"
    Meter100 = "H10"
    Surface = "S00"
    SeaSurface = "W00"
    NoRule = "X00"

    def is_surface(self) -> bool:
        return self in [
            self.LowestModelLevel,
            self.Meter2,
            self.Meter10,
            self.Meter100,
            self.Surface,
            self.SeaSurface,
            self.NoRule,
        ]
