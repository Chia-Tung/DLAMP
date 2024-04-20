from pydantic.dataclasses import dataclass

from src.const import VAR_SUFFIX
from src.utils.data_type import DataType, Level


@dataclass
class DataCompose:
    var_name: DataType
    level: Level

    def __post_init__(self):
        """
        This method is called automatically after an instance of the class is created.

        It sets the `level` attribute to `Level.NoRule` if the `var_name` attribute is
        either `DataType.Radar`, `DataType.Lat`, or `DataType.Lon`.

        Parameters:
            self (DataCompose): The instance of the class.

        Returns:
            None
        """
        if self.var_name in [DataType.Radar, DataType.Lat, DataType.Lon]:
            self.level = Level.NoRule
        self.sub_dir_name = f"{self.level}{self.var_name}{VAR_SUFFIX}"
        self.is_radar = self.var_name == DataType.Radar
