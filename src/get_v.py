'''
for querying ITS_LIVE v2. velocity cubes
and computing annual average composite fields
'''

import argparse
from dask.distributed import Client, LocalCluster
import geopandas as gpd
import os
import utils

print(f'current dir is: {os.getcwd()}')
parser = argparse.ArgumentParser()
parser.add_argument("--file")
args = parser.parse_args()
file = args.file


aoi_gdf = gpd.read_file(file)
assert(len(aoi_gdf)==1), 'more than one geometry in input .geojson'
assert(aoi_gdf.crs.to_epsg() == 4326), 'input .geojson not in epsg:4326'

aoi = aoi_gdf.geometry[0]


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

# ds = xr.open_dataset('../data/velocity_output.zarr')

print('all done')
