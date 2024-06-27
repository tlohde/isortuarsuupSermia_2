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
parser.add_argument("--file")
parser.add_argument("--velocity",
                    action=argparse.BooleanOptionalAction)

parser.add_argument("--elevation",
                    action=argparse.BooleanOptionalAction)

args = parser.parse_args()
file = args.file
velocity = args.velocity
elevation = args.elevation

aoi_gdf = gpd.read_file(file)
assert(len(aoi_gdf)==1), 'more than one geometry in input .geojson'
assert(aoi_gdf.crs.to_epsg() == 4326), 'input .geojson not in epsg:4326'

aoi = aoi_gdf.geometry[0]
aoi_3413 = aoi_gdf.to_crs(3413).geometry[0]

if elevation:
    print('getting elevation data')
    if __name__ == '__main__':
        print('name is main, apparently')
        with LocalCluster() as cluster, Client(cluster) as client:
            print(client.dashboard_link)
            e = utils.Elevation(aoi)

print('done, i think')
        
### the velocity chunk of code below has been run. and it works fine
### leave it alone ###
if velocity:
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