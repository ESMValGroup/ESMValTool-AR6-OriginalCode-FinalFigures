import geopandas
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

shed = geopandas.read_file('/home/acrnemr/esmvaltool_aux_data/IPCC-WGI-reference-regions-v4.shp')

selection = shed [53:57]

for n in range(53,57):
    x,y = selection.geometry[n].exterior.coords.xy
    plt.plot(x,y, c='b',transform=ccrs.PlateCarree())

coordinates = [(21, -36), (110, -36), (148, -45), (148, -65), (21, -65), (21, -36)]
poly = Polygon(coordinates)
selection.at[57, 'geometry'] = poly
selection.loc[57, 'Name'] = 'South Ind Ocean'
selection.loc[57, 'Type'] = 'Ocean'
selection.loc[57, 'Continent'] = 'INDIAN'
selection.loc[57, 'Acronym'] = 'SSI'

plt.plot(*poly.exterior.coords.xy, c='r', linewidth=3, transform=ccrs.PlateCarree())

selection.to_file('/home/acrnemr/esmvaltool_aux_data/indian_ocean.shp')

selection = shed [50:53]

for n in range(50,53):
    x,y = selection.geometry[n].exterior.coords.xy
    plt.plot(x,y, c='g',transform=ccrs.PlateCarree())

coordinates = [(-56, -56), (8, -36), (21, -36), (21, -65), (8, -65), (-5,-68), (-56,-75), (-56, -56)]
poly = Polygon(coordinates)
selection.at[53, 'geometry'] = poly
selection.loc[53, 'Name'] = 'South Atlantic'
selection.loc[53, 'Type'] = 'Ocean'
selection.loc[53, 'Continent'] = 'ATLANTIC'
selection.loc[53, 'Acronym'] = 'SSA'

plt.plot(*poly.exterior.coords.xy, c='r', linewidth = 3, transform=ccrs.PlateCarree())

selection.to_file('/home/acrnemr/esmvaltool_aux_data/atlantic.shp')

selection = shed [47:50]

for n in range(47,50):
    for pol in list(selection.geometry[n]):
        plt.plot(*pol.exterior.xy, c= 'magenta', transform=ccrs.PlateCarree())

coords_west_pac =[(-77, -56),(-77,-70), (-180, -75), (-180, -56), (-77, -56)]
polyg_west_pac = Polygon(coords_west_pac)

plt.plot(*polyg_west_pac.exterior.coords.xy, c='r', linewidth = 3, transform=ccrs.PlateCarree())

coords_east_pac = [(180, -50), (150, -50), (150, -65), (180, -65), (180, -50)]
polyg_est_pac = Polygon(coords_east_pac)

plt.plot(*polyg_est_pac.exterior.coords.xy, c='r', linewidth = 3, transform=ccrs.PlateCarree())

south_pac_mult_pol = MultiPolygon([polyg_west_pac, polyg_est_pac])

selection.loc[50, 'geometry'] = None
selection.at[50, 'geometry'] = south_pac_mult_pol
selection.loc[50, 'Name'] = 'South Tip of Pacific'
selection.loc[50, 'Type'] = 'Ocean'
selection.loc[50, 'Continent'] = 'PACIFIC'
selection.loc[50, 'Acronym'] = 'SSP'

selection.to_file('/home/acrnemr/esmvaltool_aux_data/pacific.shp')

plt.show()