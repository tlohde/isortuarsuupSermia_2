'''
for querying ITS_LIVE v2. velocity cubes
and computing annual average composite fields
'''

import argparse
from dask.distributed import Client, LocalCluster
import geopandas as gpd
import os
import utils

# print(f'current dir is: {os.getcwd()}')
parser = argparse.ArgumentParser()
parser.add_argument("--v_area")
parser.add_argument("--velocity",
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--z_area")
parser.add_argument("--elevation",
                    action=argparse.BooleanOptionalAction)

args = parser.parse_args()
v_area = args.v_area
velocity = args.velocity
elevation = args.elevation
z_area = args.z_area

def read_aoi(file):
    aoi_gdf = gpd.read_file(file)
    assert(len(aoi_gdf)==1), 'more than one geometry in input .geojson'
    assert(aoi_gdf.crs.to_epsg() == 4326), 'input .geojson not in epsg:4326'
    return aoi_gdf.geometry[0]


if elevation:
    aoi = read_aoi(z_area)
    print('getting elevation data')
    
    # set dask cluster running
    if __name__ == '__main__':
        print('name is main, apparently')
        cluster = LocalCluster()
        client = cluster.get_client()

        e = utils.Elevation(aoi)

        client.shutdown()
        client.close()
        
### the velocity chunk of code below has been run. and it works fine
### leave it alone ###
if velocity:
    aoi = read_aoi(z_area)
    v = utils.Velocity(aoi)

    # set dask cluster running
    if __name__ == "__main__":
        cluster = LocalCluster()
        client = cluster.get_client()
        print(f'dask dashboard link: {client.dashboard_link}')
        print('starting export')
        v.export()
        client.shutdown()
        cluster.close()

    print('all done')
### / leave it alone \ ###
