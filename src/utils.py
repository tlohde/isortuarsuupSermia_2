'''
class for generating median velocity composite stacks
from itslive cubes
'''
import dask.delayed
import dask.delayed
import dask.delayed
import dask.delayed
import dask.delayed
from dask.distributed import Lock
import dask
import geopandas as gpd
from glob import glob
import itslive
from itertools import product
import numpy as np
import os
import pandas as pd
import planetary_computer as pc
import pyproj
import pystac_client
from pystac.extensions.eo import EOExtension as eo
import shapely
from shapely import wkt
import rioxarray as rio
from scipy.stats import median_abs_deviation
import stackstac
import subprocess
from tqdm import tqdm
import xarray as xr
import xdem


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
        
        # looks to see if padded rasters have already been downloaded
        self.downloaded_rasters = glob('../data/arcticDEM/padded*.tiff')
        
        self.get_missing_dems()
        
        if self.missing_dems is False:
            print('already downloaded and padded all the DEMs')
        elif len(self.downloaded_rasters) == 0:
            self.missing_dems = self.catalog
            self.get_dems()
            self.export_dems()
        else:
            print(f'need to download {len(self.missing_dems)} DEMs')
            self.get_dems()
            self.export_dems()

        self.get_counts_and_reference()
        self.get_masks()
        # self.register()
        
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


    def get_missing_dems(self):
        '''
        checks directory for downloaded (& padded DEMs)
        compares that list with those in the catalog
        and identifies those that still need to be downloaded
        TODO add some logic/option here to allow for cases where
        you don't want to download the missing ones
        '''
        def id_from_path(path):
            _fname = os.path.basename(path)
            return _fname.split('padded_')[-1].split('.tiff')[0]
        
        _dem_ids = [id_from_path(p) for p in self.downloaded_rasters]
        _downloaded = [dem in _dem_ids for dem in self.catalog.dem_id.tolist()]
        if all(_downloaded):
            self.missing_dems = False
        else:
            _missing = [dem for dem, dld in zip(_dem_ids, _downloaded) if not dld]
            self.missing_dems = self.catalog.loc[self.catalog.dem_id.isin(_missing)]

    
    @dask.delayed
    def get_dem(self, row):
        '''
        lazy function
        where row is a single row from `self.catalog` (or `self.missing_dems`)
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
                    _fill_value,
                    dem_clip)
                            .rename('z')
                            .squeeze()
                            .rio.write_crs(_dem_crs)   
                            )
                
                _padded = _masked.rio.pad_box(*self.geo_projected.bounds,
                                              constant_values=_fill_value)
                _padded.rio.write_nodata(_fill_value, inplace=True)
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
        # for row in tqdm(self.catalog.itertuples()):
        for row in tqdm(self.missing_dems.itertuples()):
            self.rasters.append(self.get_dem(row))
    

    def export_dems(self):
        '''
        compute / export all the dems
        '''
        dask.compute(*self.rasters)
        
    ############# coregistration routines #############
    
    @dask.delayed
    def get_lazy_count(filepath):
        '''
        lazily count valid pixels in downloaded dem
        '''
        with rio.open_rasterio(filepath, chunks='auto') as dem:
            _fill_value = dem.attrs['_FillValue']
            _total = (dem != _fill_value).sum().compute()
            # _total = (~dem.isnull()).sum().compute()
            return _total.data.item()

    def get_counts_and_reference(self):
        '''
        compute count of valid pixels
        dem with greatest number --> reference DEM
        get list of dems to register (self.to_reg)
        '''
        _lazy_counts = []
        for f in self.downloaded_rasters:
            count = Elevation.get_lazy_count(f)
            _lazy_counts.append(count)
        
        _result = dask.compute(*_lazy_counts)
        self.count_result = dict(zip(self.downloaded_rasters, _result))
        
        _max = max(_result)
        for k, v in self.count_result.items():
            if v == _max:
                self.ref = k
        
        self.to_reg = [x for x in self.downloaded_rasters if x != self.ref]
        self.pairs = list(product([self.ref], self.to_reg))

    def date_from_path(self, path):
        '''
        get date from padded dem file path
        '''
        _fname = os.path.basename(path)
        _id = _fname.split('padded_')[-1].split('.tiff')[0]
        return self.catalog.loc[self.catalog.dem_id == _id, 'acqdate1'].item()
    
    @dask.delayed(nout=2)
    def make_mask(self, path):
        '''
        make stable terrain mask for given saved/padded dem
        '''
        _catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
            )
        
        _date = self.date_from_path(path)
        d1 = (_date - pd.Timedelta('14d')).strftime('%Y-%m-%d')
        d2 = (_date + pd.Timedelta('14d')).strftime('%Y-%m-%d')
        _search_period = f'{d1}/{d2}'
        _search = _catalog.search(collections=['sentinel-2-l2a',
                                               'landsat-c2-l2'],
                                  intersects=self.geo,
                                  datetime=_search_period)
        _items = _search.item_collection()
        assert len(_items) > 0, 'did not find any images'
        
        least_cloudy_item = min(_items, key=lambda item: eo.ext(item).cloud_cover)
        
        _asset_dict = {'l':['green','nir08'],
                       'S':['B03', 'B08']}

        _assets = _asset_dict[least_cloudy_item.properties['platform'][0]]
        
        img = (stackstac.stack(
            least_cloudy_item, epsg=3413,assets=_assets
            ).squeeze()
            .rio.clip_box(*self.geo.bounds, crs=4326)
            )
        
        # can use [] indexing here because the order
        # of assets in _asset dict is consistent 
        ndwi = ((img[0,:,:] - img[1,:,:]) /
                (img[0,:,:] + img[1,:,:]))
        
        
        with rio.open_rasterio(path, chunks='auto') as _ds:
            return (least_cloudy_item.id, xr.where(ndwi < 0, 1, 0).rio.reproject_match(_ds))

    def get_masks(self):
        '''
        make stable terrain mask from reference dem
        and & it with stable terrain for every other dem
        to make list combined masks.
        '''
        self.ref_img_id, self.ref_mask = self.make_mask(self.ref)
        self.combined_to_reg_masks = []
        self.combined_to_reg_img_ids = []
        for f in self.to_reg:
            _id, _mask = self.make_mask(f)
            _combined_mask = ((self.ref_mask & _mask) == 1).data
            self.combined_to_reg_masks.append(_combined_mask)
            self.combined_to_reg_img_ids.append(_id)
            
        _img_ids = list(product([self.ref_img_id],
                                self.combined_to_reg_img_ids))

        self.pairs_and_mask_id = list(zip(self.pairs,
                                       self.combined_to_reg_masks,
                                       _img_ids))
        
    # @dask.delayed
    def lazy_register(self, pair_and_mask):
        '''
        coregister dems
        convert to xarray
        calculate statistics
        add metadata
        export
        pair_and_mask = 
        (  
            (ref_path, _to_reg_path),
            stable_terrain_mask,
            (_ref_img_id, _to_reg_img_id)
        )
        '''
        
        def id_from_path(path):
            _fname = os.path.basename(path)
            return _fname.split('padded_')[-1].split('.tiff')[0]
        
        _pair, _mask, _id = pair_and_mask
        _ref_path, _to_reg_path = _pair
        _ref_img_id, _to_reg_img_id = _id

        _mask = _mask.compute()
        _ref_img_id = _ref_img_id.compute()
        _to_reg_img_id = _to_reg_img_id.compute()
        
        print(
            f'_ref_path: {_ref_path}\n',
            f'_to_reg_path: {_to_reg_path}\n',
            f'_ref_img_id: {_ref_img_id}\n',
            f'_to_reg_img_id: {_to_reg_img_id}\n',
            f'_mask: {_mask}'
            )
        
        _ref = xdem.DEM(_ref_path)
        _to_reg = xdem.DEM(_to_reg_path)
        
        _pipeline = xdem.coreg.NuthKaab() + xdem.coreg.Tilt()
        _pipeline.fit(
            reference_dem=_ref,
            dem_to_be_aligned=_to_reg,
            inlier_mask=_mask
        )
        
        _regd = _pipeline.apply(_to_reg)
        
        stable_diff_before = (_ref - _to_reg)[_mask]
        stable_diff_after = (_ref - _regd)[_mask]
        
        before_median = np.ma.median(stable_diff_before)
        after_median = np.ma.median(stable_diff_after)
        
        before_nmad = xdem.spatialstats.nmad(stable_diff_before)
        after_nmad = xdem.spatialstats.nmad(stable_diff_after)

        output = _regd.to_xarray()
        
        output.attrs['to_register'] = id_from_path(_to_reg_path)
        output.attrs['to_register_date'] = (
            self.date_from_path(_to_reg_path).strftime('%Y-%m-%d')
        )
        output.attrs['to_reg_mask'] = _to_reg_img_id
        
        output.attrs['reference'] = id_from_path(_ref_path)
        output.attrs['reference_date'] = (
            self.date_from_path(_ref_path).strftime('%Y-%m-%d')
        )
        output.attrs['ref_mask'] = _ref_img_id
        
        output.attrs['before_nmad'] = before_nmad
        output.attrs['after_nmad'] = after_nmad
        output.attrs['before_median'] = before_median
        output.attrs['after_median'] = after_median
        
        return output
        
    def register(self):
        '''
        apply coregistration and export
        '''
        self.registered_dems = []
        # lazily coregister and export
        for i, p_and_m in enumerate(self.pairs_and_mask_id):
            print(f'working on {i}/{len(self.pairs_and_mask_id)}')
            try:
                _registered = self.lazy_register(p_and_m)
                
                # self.registered_dems.append(_registered)
                _fname = os.path.basename(p_and_m[0][1])
                _fname = _fname.replace('padded', 'coregd')
    
                # check if it has already been computed & exported
                if os.path.exists(f'../data/arcticDEM/coregd/{_fname}'):
                    continue
                else:
                    _delayed_write = _registered.rio.to_raster(
                        f'../data/arcticDEM/coregd/{_fname}',
                        compute=True,
                        lock=Lock()
                        )
                    self.registered_dems.append(_delayed_write)

            except Exception as e:
                print(e)
                print(f'skipping: {p_and_m[0][1]}')
            
    
    def expot_registered_dems(self):
        '''
        compute / export all the dems
        '''
        dask.compute(*self.registered_dems)
        

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
