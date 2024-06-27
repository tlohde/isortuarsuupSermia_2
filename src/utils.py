'''
class for generating median velocity composite stacks
from itslive cubes
'''
import dask.delayed
from dask.distributed import Lock
import dask
import geopandas as gpd
from glob import glob
import itslive
import numpy as np
import os
import pandas as pd
import pyproj
import shapely
from shapely import wkt
import rioxarray as rio
from scipy.stats import median_abs_deviation
import subprocess
from tqdm import tqdm
import xarray as xr


def geo_reproject(geo,
                  src_crs: int=3413,
                  target_crs: int=4326):
    """
    reproject shapely point (geo) from src_crs to target_crs
    avoids having to create geopandas series to handle crs transformations
    """
    assert isinstance(geo,
                      (shapely.geometry.polygon.Polygon,
                       shapely.geometry.linestring.LineString,
                       shapely.geometry.point.Point)
                      ), 'geo must be shapely geometry'

    transformer = pyproj.Transformer.from_crs(
        src_crs,
        target_crs,
        always_xy=True
    )
    if isinstance(geo, shapely.geometry.point.Point):
        _x, _y = geo.coords.xy
        return shapely.Point(*transformer.transform(_x, _y))
    elif isinstance(geo, shapely.geometry.polygon.Polygon):
        _x, _y = geo.exterior.coords.xy
        return shapely.Polygon(zip(*transformer.transform(_x, _y)))
    elif isinstance(geo, shapely.geometry.linestring.LineString):
        _x, _y = geo.coords.xy
        return shapely.Linestring(zip(*transformer.transform(_x, _y)))


class Elevation():
    '''
    class for handling arcticDEM strips;
    co-registering DEMs; creating annual composites
    and rates of SEC
    '''
    def __init__(self,
                 geo: shapely.geometry.polygon.Polygon,
                 months: list=[6,7,8,9,10],
                 **kwargs):
        
        self.geo = geo
        self.geo_projected = geo_reproject(geo, 4326, 3413)
        self.months = months
        self.get_catalog()
        self.get_dems()        
        self.export_dems()
    
    def get_catalog(self):
        '''
        opens arctic DEM catalog,
        - filters by months of acquisiton date (acqdate1)
        - formats download urls 
        - intersects with self.geometry
        - resultant geodataframe (self.catalog)
            contains rows where each arcticDEM is one of interest
        
        if the arcticDEM catalog can't be found
        it downloads it by calling `arcticDEMCatalog.sh`
        
        '''
        if len(glob('../**/*.parquet', recursive=True)) == 0:
            # need to download it
            print('downloading and unzipping arcticDEM catalog')
            subprocess.call("./arcticDEMCatalog.sh")
        
        _filepath = glob('../**/*.parquet', recursive=True)[0]
        catalog = gpd.read_parquet(_filepath)
        # make timestamps, datetimes, and sort (so downloads in date order)
        catalog['acqdate1'] = (catalog['acqdate1']
                               .dt.tz_localize(None)
                               .astype('datetime64[ns]')
                               )
        catalog['acqdate2'] = (catalog['acqdate2']
                               .dt.tz_localize(None)
                               .astype('datetime64[ns]')
                               )

        catalog.sort_values(by='acqdate1', inplace=True)

        # only select dems from 'self.months'
        catalog = catalog.loc[catalog['acqdate1'].dt.month.isin(self.months)]

        # fix download urls
        text_to_replace = 'polargeospatialcenter.github.io/stac-browser/#/external/'
        catalog['downloadurl'] = (catalog['s3url']
                                .apply(lambda x: (
                                    x.replace(text_to_replace, "")
                                    .replace('.json', '_dem.tif')))
                                )
        catalog['bitmaskurl'] = (catalog['downloadurl']
                                .apply(lambda x: x.replace('_dem', '_bitmask')))

        catalog = catalog.to_crs(4326)
        self.catalog = catalog.loc[catalog.intersects(self.geo)]
        print(f'filtered catalog has {len(self.catalog)} dems')
    
    @dask.delayed
    def get_dem(self, row):
        '''
        lazy function
        where row is a single row from `self.catalog`
        lazily open DEM COG and bitmask COG
        apply bitmask and pad to self.geo bounds
        return dask delayed object for writing to .tiff
        '''
        with rio.open_rasterio(row.downloadurl, chunks='auto') as dem,\
            rio.open_rasterio(row.bitmaskurl, chunks='auto') as bm:
                
                _fill_value = dem.attrs['_FillValue']
                _dem_crs = dem.rio.crs.to_epsg()

                dem_clip = dem.rio.clip_box(*self.geo_projected.bounds)
                bm_clip = bm.rio.clip_box(*self.geo_projected.bounds)
                
                # apply bit mask
                _masked = (xr.where(
                    (dem_clip == _fill_value)
                    | (bm_clip[:, :, :] > 0),
                    np.nan,
                    dem_clip)
                            .rename('z')
                            .squeeze()
                            .rio.write_crs(_dem_crs)   
                            )
                
                _padded = _masked.rio.pad_box(*self.geo_projected.bounds,
                                              constant_values=np.nan)
                
                # _padded = _padded.expand_dims(
                #     dim={'acqdate1': [row.acqdate1.to_numpy()]}
                #     )
                # self.dems.append(_padded)
                
                _fname = f'../data/arcticDEM/padded_{row.dem_id}.tiff'
                _delayed_write = _padded.rio.to_raster(_fname,
                                                       compute=True,
                                                       lock=Lock()
                                                       )
                
                return _delayed_write
    

    def get_dems(self):
        '''
        get lazy delayed objects for each row in self.catalog
        '''
        self.rasters = []
        for row in tqdm(self.catalog.itertuples()):
            self.rasters.append(self.get_dem(row))
    

    def export_dems(self):
        '''
        compute / export all the dems
        '''
        dask.compute(*self.rasters)
        
        
class Velocity():
    '''
    class for handling itslive velocity cubes
    and generating median composite fields
    '''
    def __init__(self,
                 geo: shapely.geometry.polygon.Polygon,
                 **kwargs):
        
        
        self.geo = geo
        
        # default for ~ annual 11-13 month ddt
        self.ddt_range = kwargs.get('ddt_range', ('335d', '395d'))

        # use itslive package for identifying urls to zarr cubes
        self.coords = list(zip(*self.geo.exterior.coords.xy))
        self.cubes = itslive.velocity_cubes.find_by_polygon(self.coords)
        
        self.get_cubes()
        self.filter_by_ddt()
        
        ## apply compositing functions ##
        
        # lazily compute median along velocity per year
        self.median = self.apply_func_along_groupby_time(
            func=np.nanmedian,
            prefix='median_'
        )

        # lazily compute median abs deviation in v per year
        def mad(x):
            return median_abs_deviation(x, axis=-1, nan_policy='omit')
        self.mad = self.apply_func_along_groupby_time(
            func=mad,
            prefix='mad_'
        )

        # lazily count number of pixels in v in per year
        def count_valid(x):
            return (~np.isnan(x)).sum(axis=-1)            
        self.valid_count = self.apply_func_along_groupby_time(
            func=count_valid,
            prefix='n_valid_'
        )

        ## ## ## ## ## ## ## ## ## ## ##
        
        # merge/stack median, mad and valid count into single
        # lazy array dataset, ready for export/compute
        self.merge_add_metadata()
    
    # load and and clip cubes
    def get_cubes(self):
        '''
        for each itslive cube that intersects with self.geo
        open the url, assign projected crs and then clip to self.geo,
        which is in lat/lon, and append to list
        
        note: the list of xr.Datasets cannot be merged at this point
        because neighbouring itslive cubes have different values along
        the time dimension.
        '''
        self.dss = []
        for cube in self.cubes:
            _ds = (xr.open_dataset(cube['properties']['zarr_url'],
                                   engine='zarr',
                                   chunks='auto')
                   .rio.write_crs(3413)
                   .rio.clip_box(*self.geo.bounds, crs=4326)
                  )
            _ds.rio.write_transform(
                _ds.rio.transform(recalc=True),
                inplace=True
            )
            self.dss.append(_ds)
    
    def filter_by_ddt(self):
        '''
        filters itslive cubes along time dimension by date_dt
        still lazy.
        filtered cubes stored as list in `self.filtered`        
        '''
        lower, upper = [pd.Timedelta(dt) for dt in self.ddt_range]
        
        def ddt_filter(ds, lower, upper):
            idx = (
                (ds['date_dt'] >= lower) & (ds['date_dt'] < upper))
            return ds.sel(mid_date=idx)

        self.filtered = [ddt_filter(ds, lower, upper) for ds in self.dss]

    def apply_func_along_groupby_time(self,
                                      func,  ## TODO need to be able to pass kwargs to this!! for nan policy
                                      prefix,
                                      variables=['v', 'vx', 'vy'],
                                      core_dim=['mid_date'],
                                      grouper='mid_date.year'):
        '''
        wrapper for `xr.apply_ufunc()` for applying functions along
        time (`mid_date`) dimension whilst utilising dask for 
        parallelization
        
        this wrapper assumes that the supplied function `func` applies
        a reduction along the time dimension so that each xr.dataset in
        list of self.filtered (or self.dss) can be merged after the func()
        has been applied
        
        inputs:
            func: function to apply
            prefix: str. to add to to variable names after having func applied
            variables: list. list of variables in itslive cube to apply func to
            core_dim: dimension along which to apply
            grouper: for groupby apply functionality
        
        '''
        _outputs = []
        for ds in self.filtered:
            _output = xr.apply_ufunc(func,
                                     ds[variables].groupby(grouper),
                                     input_core_dims=[core_dim],
                                     vectorize=True,
                                     dask='parallelized',
                                     dask_gufunc_kwargs={
                                         'allow_rechunk': True
                                     }
                                    )
            _outputs.append(_output)
        _output = xr.merge(_outputs)
        
        # append prefix to variable names
        new_name_dict = dict(
            zip(
                list(_output.data_vars.keys()),
                [str(prefix + name) for name in list(_output.data_vars.keys())]
            )
        )
        return _output.rename(new_name_dict)

    def merge_add_metadata(self):
        _output = xr.merge([self.median, self.mad, self.valid_count])
        _output.attrs['description'] = '''
        velocity cube: dimensions: (year, y, x).
        varaibles: median_*, mad_* and n_valid_* for v, vx and vy.
        method: (1) the itslive velocity cube was first filtered on date_dt
        (see below attributed)
        (2) velocity fields were grouped by the year of the mid_date field
        (i.e., velocity field generated from two images taken in November 2015, 
        and October 2016, would have its the year of its midate in 2016.)
        for each mid_date_year group the median velocity (exc. nans) is calculated
        (median_*) for the resultant velocity (v), as well as the
        individual vx and vy components.
        (3) uncertainty in the velocity is described using the median absolute 
        deviation (mad_*) within each mid_date_year group, again for both v, vx, and vy
        (4) the number of not nan measurements that each median_ and mad_ value is
        is computed from is held in the variable n_valid_*
        '''
        _output.attrs['ddt_range'] = str(self.ddt_range)
        _output.attrs['aoi'] = wkt.dumps(self.geo)
        _output.attrs['date_processed'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        _output.attrs['processed_by'] = 'tlohde'
        
        # rechunking allows `to_zarr()` export to work
        _output = _output.chunk(
            dict(zip(
                _output['median_v'].dims,
                _output['median_v'].shape
            )))
        self.output = _output.rio.write_crs(3413)
        self.output.rio.write_grid_mapping('spatial_ref',
                                           inplace=True)
        print('ready to export')
    
    def export(self, path='../data/velocity_output'):
        '''
        exports to .zarr because that is quick and dask friendly
        doesn't require holding the whole thing in memory as can write
        out chunks as they are computed
        
        read in the output zarr and then immediately export as .nc
        '''
        self.output.to_zarr(f'{path}.zarr')
        with xr.open_dataset(f'{path}.zarr') as _ds:
            _ds.to_netcdf(f'{path}.nc')
