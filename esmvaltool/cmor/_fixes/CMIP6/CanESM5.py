# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for CanESM5"""
import iris
import numpy as np

from ..fix import Fix


class allvars(Fix):
    """Fixes common to all vars"""

    def fix_metadata(self, cubes):
        """
        Fixes latitude and latitude.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """

        for cube in cubes:
            latitude = cube.coord('latitude')
            latitude.var_name = 'lat'

            longitude = cube.coord('longitude')
            longitude.var_name = 'lon'
        return cubes





