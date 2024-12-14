from __future__ import annotations

from enum import Enum
from typing import Callable

from pydantic.dataclasses import dataclass

from ..const import VAR_SUFFIX
from .data_type import DataType, Level


@dataclass
class DataCompose:
    var_name: DataType
    level: Level

    def __post_init__(self):
        """
        This method is called automatically after an instance of the class is created.

        It sets the `level` attribute to `Level.NoRule` if the `var_name` attribute is
        either `DataType.Radar`, `DataType.Lat`, or `DataType.Lon`.

        Args:
            self (DataCompose): The instance of the class.

        Returns:
            None
        """
        if self.var_name in [DataType.Radar, DataType.Lat, DataType.Lon]:
            self.level = Level.NoRule
        if self.var_name in [DataType.Td, DataType.RH]:
            self.level = Level.Meter2

        self.basename = f"{self.level.code}{self.var_name.code}{VAR_SUFFIX}"
        self.combined_key = self.get_combined_key()
        self.is_radar = self.var_name == DataType.Radar

    def __str__(self) -> str:
        return f"{self.var_name.value}@{self.level.value}"

    @staticmethod
    def retrive_var_level_from_string(sentence: str) -> tuple[DataType, Level]:
        var_str = sentence.split("@")[0]
        level_str = sentence.split("@")[1]
        return DataType(var_str), Level(level_str)

    def get_combined_key(self):
        """
        Combine the NetCDF key of the variable and level into a single string.
        """
        if self.level not in [Level.Meter2, Level.Meter10, Level.Meter100]:
            return self.var_name.nc_key

        if self.var_name in [DataType.Td, DataType.RH]:
            return f"{self.var_name.nc_key}{self.level.nc_key}"

        if self.var_name in [DataType.U, DataType.V]:
            prefix = self.var_name.nc_key.split("_")[0]
            return f"{prefix}{self.level.nc_key}"

        if self.var_name in [DataType.T, DataType.Qv]:
            prefix = self.var_name.name[0]
            return f"{prefix}{self.level.nc_key}"

    @classmethod
    def from_config(cls, config: dict[str, list[str]]) -> list[DataCompose]:
        """
        A method to create a list of DataCompose objects based on the provided config.

        Args:
            config (dict[str, str]): A dictionary containing the configuration.
                The dictoinary should have the following structure:
                    {
                        "GeoHeight": ["Hpa200", "Hpa500", "Hpa700", "Hpa850", "Hpa925"],
                        "T": ["Hpa200", "Hpa500", "Hpa700", "Hpa850", "Hpa925"],
                        ...
                    }
        Returns:
            list[DataCompose]: A list of DataCompose objects.
        """
        data_list = []
        for var, lvs in config.items():
            for lv in lvs:
                data_list.append(cls(DataType[var], Level[lv]))
        return data_list

    def get_all_hook(fn: Callable):
        def wrapper(
            data_list: list[DataCompose],
            only_upper: bool = False,
            only_surface: bool = False,
            to_str: bool = False,
        ) -> list[Enum] | list[str]:
            """
            A wrapper function that takes a list of `DataCompose` objects, along with optional
            boolean flags `only_upper` and `only_surface`, and returns a list of `StrEnum` or `str` values.

            Parameters:
                data_list (list[DataCompose]): A list of `DataCompose` objects.
                only_upper (bool, optional): If True, only the levels that are not surface levels will be
                    included in the returned list. Defaults to False.
                only_surface (bool, optional): If True, only the surface levels will be included in the returned
                    list. Defaults to False.
                to_str (bool, optional): If True, the returned list will contain `str` values instead of `StrEnum`
                    objects. Defaults to False.

            Raises:
                ValueError: If both `only_upper` and `only_surface` are True.

            Returns:
                list[StrEnum] | list[str]: A list of `StrEnum` or `str` values, depending on the value of `to_str`.
            """
            if only_upper and only_surface:
                raise ValueError("only_upper and only_surface cannot both be True")

            ret = []
            for data_compose in data_list:
                lv = data_compose.level
                if lv.is_surface() and only_surface:
                    fn(ret, data_compose)
                elif not lv.is_surface() and only_upper:
                    fn(ret, data_compose)
                elif not only_surface and not only_upper:
                    fn(ret, data_compose)
            return [x.name for x in ret] if to_str else ret

        return wrapper

    @staticmethod
    @get_all_hook
    def get_all_levels(result: list[Level], data_compose: DataCompose) -> None:
        """
        Get a list of `Level` objects or a list of Level `str` from a given list of `DataCompose` objects.

        Can choose to include only surface levels or only upper levels.
        """
        if data_compose.level not in result:
            result.append(data_compose.level)

    @staticmethod
    @get_all_hook
    def get_all_vars(result: list[DataType], data_compose: DataCompose) -> None:
        """
        Get a list of `DataType` objects or a list of DataType `str` from a given list of `DataCompose` objects.

        Can choose to include only surface levels or only upper levels.
        """
        if data_compose.var_name not in result:
            result.append(data_compose.var_name)
