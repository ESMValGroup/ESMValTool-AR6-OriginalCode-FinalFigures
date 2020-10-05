# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for CESM2-WACCM."""
import numpy as np
import iris

from ..fix import Fix

class tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubelist):
        """
        Fix missing scalar dimension.

        Parameters
        ----------
        cubelist: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        height2m = iris.coords.AuxCoord(
            '2.d',
            standard_name='height',
            long_name='height',
            var_name='height',
            units='m',
            bounds=None)
        for cube in cubelist:
            cube.add_aux_coord(height2m)
        return cubelist