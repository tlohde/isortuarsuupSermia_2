'''
class for generating median velocity composite stacks
from itslive cubes
'''
import itslive
import numpy as np
import os
import pandas as pd
import shapely
from shapely import wkt
import rioxarray as rio
from scipy.stats import median_abs_deviation
import xarray as xr

class Velocity():
    '''
    class for handling itslive velocity cubes
    and generating median composite fields
    '''
    def __init__(self,
                 geo: shapely.geometry.polygon.Polygon,
                 **kwargs):
        
        self.geo = geo
        self.ddt_range = kwargs.get('ddt_range', ('335d', '395d'))

        self.coords = list(zip(*self.geo.exterior.coords.xy))
        self.cubes = itslive.velocity_cubes.find_by_polygon(self.coords)
        self.get_cubes()
        self.filter_by_ddt()

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

        # merge/stack median, mad and valid count into single
        # lazy array dataset, ready for export/compute
        self.merge_add_metadata()
            
    def get_cubes(self):
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
        self.output.to_zarr(f'{path}.zarr')
        with xr.open_dataset(f'{path}.zarr') as _ds:
            _ds.to_netcdf(f'{path}.nc')
            os.remove(f'{path}.zarr')