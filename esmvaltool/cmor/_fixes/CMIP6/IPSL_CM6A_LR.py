# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for IPSL-CM6A-LR"""
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
            latitude = cube.coord(standard_name='latitude')
            latitude.var_name = 'lat'

            longitude = cube.coord(standard_name='longitude')
            longitude.var_name = 'lon'
        print ('**************** IPSL_CM6A_LR.py ******************')
        print (cubes)
        print (len(cubes))
        print(type (cubes))
        print ('*********************************')
        print (cubes[0:1])
        print (cubes[0].name)
        if "cell_area" in str(cubes[0].name):
             return cubes[1:2]
        else:
             return cubes[0:1]
# Removes first cube containing cell area if there are two cubes. 
#        if len(cubes) != 1:
#
#            return cubes[0:1]
#
#        else:
#
#            return cubes

#        return cubes[0:1]



