# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for MIROC6"""
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
        print ('Running fixes for MIROC6')
        for cube in cubes:
            latitude = cube.coord('latitude')
            latitude.var_name = 'lat'

            longitude = cube.coord('longitude')
            longitude.var_name = 'lon'
        return cubes





