#!/n/home06/jeast/.conda/envs/imi_env/bin/python

# # Make regridding weights for satellite data
#     James East
#     2024-02-12

# ## Conservative regridding example here:
# https://discourse.pangeo.io/t/conservative-region-aggregation-with-xarray-geopandas-and-sparse/2715

# ## Area-weighted avg GC output to GC grid cells

import xarray as xr
import itertools
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import multiprocessing
import glob
import re
import os
import sys
from shapely.geometry import Polygon
import datetime
import geopandas as gpd
from glob import glob
import os


# #### 1. functions from imi 

def read_blended(filename):
    """
    Read Blended TROPOMI+GOSAT data and save important variables to dictionary.
    Arguments
        filename [str]  : Blended TROPOMI+GOSAT netcdf data file to read
    Returns
        dat      [dict] : Dictionary of important variables from Blended TROPOMI+GOSAT:
                            - CH4
                            - Latitude
                            - Longitude
                            - Time (utc time reshaped for orbit)
                            - Averaging kernel
                            - SWIR albedo
                            - NIR albedo
                            - Blended albedo
                            - CH4 prior profile
                            - Dry air subcolumns
                            - Latitude bounds
                            - Longitude bounds
                            - Surface classification
                            - Chi-Square for SWIR
                            - Vertical pressure profile
    """
    assert "BLND" in filename, f"BLND not in filename {filename}, but a blended function is being used"

    try:
        # Initialize dictionary for Blended TROPOMI+GOSAT data
        dat = {}

        # Extract data from netCDF file to our dictionary
        with xr.open_dataset(filename) as blended_data:

            dat["methane"] = blended_data["methane_mixing_ratio_blended"].values[:]
            dat["longitude"] = blended_data["longitude"].values[:]
            dat["latitude"] = blended_data["latitude"].values[:]
            dat["column_AK"] = blended_data["column_averaging_kernel"].values[:, ::-1]
            dat["swir_albedo"] = blended_data["surface_albedo_SWIR"][:]
            dat["nir_albedo"] = blended_data["surface_albedo_NIR"].values[:]
            dat["blended_albedo"] = 2.4 * dat["nir_albedo"] - 1.13 * dat["swir_albedo"]
            dat["methane_profile_apriori"] = blended_data["methane_profile_apriori"].values[:, ::-1]
            dat["dry_air_subcolumns"] = blended_data["dry_air_subcolumns"].values[:, ::-1]
            dat["longitude_bounds"] = blended_data["longitude_bounds"].values[:]
            dat["latitude_bounds"] = blended_data["latitude_bounds"].values[:]
            dat["surface_classification"] = (blended_data["surface_classification"].values[:].astype("uint8") & 0x03).astype(int)
            dat["chi_square_SWIR"] = blended_data["chi_square_SWIR"].values[:]

            # Remove "Z" from time so that numpy doesn't throw a warning
            utc_str = blended_data["time_utc"].values[:]
            dat["time"] = np.array([d.replace("Z","") for d in utc_str]).astype("datetime64[ns]")

            # Need to calculate the pressure for the 13 TROPOMI levels (12 layer edges)
            pressure_interval = (blended_data["pressure_interval"].values[:] / 100)  # Pa -> hPa
            surface_pressure = (blended_data["surface_pressure"].values[:] / 100)    # Pa -> hPa
            n = len(dat["methane"])
            pressures = np.full([n, 12 + 1], np.nan, dtype=np.float32)
            for i in range(12 + 1):
                pressures[:, i] = surface_pressure - i * pressure_interval
            dat["pressures"] = pressures

        # Add an axis here to mimic the (scanline, groundpixel) format of operational TROPOMI data
        # This is so the blended data will be compatible with the TROPOMI operators
        for key in dat.keys():
            dat[key] = np.expand_dims(dat[key], axis=0)

    except Exception as e:
        print(f"Error opening {filename}: {e}")
        return None

    return dat


def filter_blended(blended_data, xlim, ylim, startdate, enddate):
    """
    Description:
        Filter out any data that does not meet the following
        criteria: We only consider data within lat/lon/time bounds,
        that don't cross the antimeridian, and we filter out all
        coastal pixels (surface classification 3) and inland water
        pixels with a poor fit (surface classifcation 2, 
        SWIR chi-2 > 20000) (recommendation from Balasus et al. 2023).
        Also, we filter out water pixels and south of 60S.
    Returns:
        numpy array with satellite indices for filtered tropomi data.
    """
    return np.where(
        (blended_data["longitude"] > xlim[0])
        & (blended_data["longitude"] < xlim[1])
        & (blended_data["latitude"] > ylim[0])
        & (blended_data["latitude"] < ylim[1])
        & (blended_data["time"] >= startdate)
        & (blended_data["time"] <= enddate)
        & (blended_data["longitude_bounds"].ptp(axis=-1) < 100) #JDE
        & ~((blended_data["surface_classification"] == 3) | ((blended_data["surface_classification"] == 2) & (blended_data["chi_square_SWIR"][:] > 20000)))
        & (blended_data["surface_classification"] != 1)
        & (blended_data["latitude"] > -60)
    )[0]


def get_gc_lat_lon(gc_cache, start_date):
    """
    get dictionary of lat/lon values for gc gridcells

    Arguments
        gc_cache    [str]   : path to gc data
        start_date  [str]   : start date of the inversion

    Returns
        output      [dict]  : dictionary with the following fields:
                                - lat : list of GC latitudes
                                - lon : list of GC longitudes
    """
    gc_ll = {}
    date = pd.to_datetime(start_date).strftime("%Y%m%d_%H")
    file_species = f"GEOSChem.SpeciesConc.{date}00z.nc4"
    filename = f"{gc_cache}/{file_species}"
    with xr.open_dataset(filename) as gc_file_ds:
        gc_data = gc_file_ds[['lon','lat']]
    gc_ll["lon"] = gc_data["lon"].values
    gc_ll["lat"] = gc_data["lat"].values

    gc_file_ds.close()
    return gc_ll


def get_gridcell_list(lons, lats):
    """
    Create a 2d array of dictionaries, with each dictionary representing a GC gridcell.
    Dictionaries also initialize the fields necessary to store for tropomi data
    (eg. methane, time, p_sat, etc.)

    Arguments
        lons     [float[]]      : list of gc longitudes for region of interest
        lats     [float[]]      : list of gc latitudes for region of interest

    Returns
        gridcells [dict[][]]     : 2D array of dicts representing a gridcell
    """
    # create array of dictionaries to represent gridcells
    gridcells = []
    for i in range(len(lons)):
        for j in range(len(lats)):
            gridcells.append(
                {
                    "lat": lats[j],
                    "lon": lons[i],
                    "iGC": i,
                    "jGC": j,
                    "methane": [],
                    "p_sat": [],
                    "dry_air_subcolumns": [],
                    "apriori": [],
                    "avkern": [],
                    "time": [],
                    "overlap_area": [],
                    "lat_sat": [],
                    "lon_sat": [],
                    "observation_count": 0,
                    "observation_weights": [],
                }
            )
    gridcells = np.array(gridcells).reshape(len(lons), len(lats))
    return gridcells


def nearest_loc(query_location, reference_grid, tolerance=0.5):
    """Find the index of the nearest grid location to a query location, with some tolerance."""

    distances = np.abs(reference_grid - query_location)
    ind = distances.argmin()
    if distances[ind] >= tolerance:
        return np.nan
    else:
        return ind


def process_list_of_netcdf_files(file):

    # Loop through all of the files and only keep the observations
    # that are north of 60°S, are not over water, and are not the
    # problematic coastal pixels. Also make sure that we only use
    # observations within the month we are processing.
    #df = pd.DataFrame()
    #for idx,file in enumerate(subset_of_files):
    with Dataset(file) as ds:

        sc = (ds["surface_classification"][:] & 0x03).astype(int)
        start_date = pd.to_datetime(file.split('_')[-6][0:6], format='%Y%m')#pd.to_datetime(month, format="%Y%m")
        end_date = start_date + pd.DateOffset(months=1)
        f = "%Y-%m-%dT%H:%M:%S.%fZ"

        valid = (
            (ds["latitude"][:] > -60) &
             ~((sc == 3) | ((sc == 2) & (ds["chi_square_SWIR"][:] > 20000))) &
            (sc != 1) &
            (pd.to_datetime(ds["time_utc"][:], format=f) >= start_date) &
            (pd.to_datetime(ds["time_utc"][:], format=f) < end_date)
        )

        df = pd.DataFrame({
            "latitude": ds["latitude"][:],
            "longitude": ds["longitude"][:],
            "xch4": ds["methane_mixing_ratio_blended"][:],
            "surface_pressure": ds["surface_pressure"][:],
            "pressure_interval": ds["pressure_interval"][:],
            "time_utc": pd.to_datetime(ds["time_utc"][:], format=f),
            "latitude_bounds": list(ds["latitude_bounds"][:]),
            "longitude_bounds": list(ds["longitude_bounds"][:]),
            "averaging_kernel": list(ds["column_averaging_kernel"][:]),
            "methane_profile_apriori": list(ds["methane_profile_apriori"][:]),
            "dry_air_subcolumns": list(ds["dry_air_subcolumns"][:]),
            "valid": valid,
            "t_step": pd.to_datetime(ds["time_utc"][:], format=f).floor('h')
        })

        #df = pd.concat([df, tmp_df], ignore_index=True)

    return df


def make_blank_dataset(lat, lon, k_shape, base_run):
    nan_data = np.full(
        (lat.shape[0],lon.shape[0]),
        np.nan, dtype=np.float32
    )
    var_dict = {
       'geoschem_methane': (
           ('lat','lon'), nan_data.copy(),
           {
               'units':'ppb',
               'long_name':'gc_xch4_sat_operator_applied'
           }
       ),
    }

    if base_run:
        nan_data_k = np.full(
            (lat.shape[0], lon.shape[0], k_shape),
            np.nan, dtype=np.float32
        )
        var_dict_full =  {
            'tropomi_methane': (
                 ('lat','lon'), nan_data.copy(),
                 {
                     'units':'ppb',
                     'long_name':'tropomi_xch4'
                 }
            ),
            'lat_sat': (
                ('lat','lon'), nan_data.copy(),
                {
                    'standard_name': 'latitude',
                    'long_name': 'satellite_avg_Latitude',
                    'units': 'degrees_north'
                }
            ),
            'lon_sat': (
                ('lat','lon'), nan_data.copy(),
                {
                    'standard_name': 'longitude',
                    'long_name': 'satellite_avg_Longitude',
                    'units': 'degrees_east'
                }
            ),
            'observation_count': (
                ('lat','lon'), nan_data.copy(),
                {
                    'units':'count',
                    'long_name':'number_of_obs_in_gridcell'
                }
            ),
            'iGC': (
                ('lat','lon'), nan_data.copy(),
                {
                    'units':'none',
                    'long_name':'gridcell_index_i'
                }
            ),
            'jGC': (
                ('lat','lon'), nan_data.copy(),
                {
                    'units':'none',
                    'long_name':'gridcell_index_j'
                }
            ),
            'K': (
                ('lat','lon','element'), nan_data_k.copy(),
                {
                    'units':'none',
                    'long_name':'jacobian_K'
                }
            )
        }

    if base_run:
        var_dict_out = {**var_dict, **var_dict_full}
    else:
        var_dict_out = var_dict

    dsout = xr.Dataset(
        data_vars = var_dict_out,
        coords = {
            'lat': (
                ('lat'), lat,
                {
                    'standard_name': 'latitude',
                    'long_name': 'Latitude',
                    'units': 'degrees_north'
                }
            ),
            'lon': (
                ('lon'), lon,
                {
                    'standard_name': 'longitude',
                    'long_name': 'Longitude',
                    'units': 'degrees_east'
                }
            ),
        }
    )
    return dsout




def calc_weights(sat_ll, gc_ll, sat_ind):
    '''
    save weights to regrid tropomi to gc
    '''
    
    n_obs = len(sat_ind) #JDE
    
    gc_lats = gc_ll["lat"]
    gc_lons = gc_ll["lon"]
    
    dlon = np.median(np.diff(gc_ll["lon"])) # GEOS-Chem lon resolution
    dlat = np.median(np.diff(gc_ll["lat"])) # GEOS-Chem lon resolution
    gridcell_dicts = get_gridcell_list(gc_lons, gc_lats)
    
    # create sat geometry
    # reshape data so geodataframe is happy
    ll_pairs = np.concatenate((
        sat_ll["longitude_bounds"][sat_ind][:,:,None],
        sat_ll["latitude_bounds"][sat_ind][:,:,None]
    ),axis=-1)
    # create data from from lat/lon pairs
    df_ll = pd.DataFrame(
        [tuple(ll_pairs[i]) for i in range(ll_pairs.shape[0])]
    )
    # create sat obs geometry
    gdf = gpd.GeoDataFrame(
        data = df_ll,
        geometry = df_ll.apply(lambda x: Polygon(np.stack(x.values)), axis=1),
        crs="EPSG:4326"
    )
    # column for index number
    gdf['sat_idx'] = sat_ind
    # bounds to filter GC gridcells
    sat_bounds = (
        ll_pairs[:,:,0].min(), # lon min
        ll_pairs[:,:,0].max(), # lon max
        ll_pairs[:,:,1].min(), # lat min
        ll_pairs[:,:,1].max()  # lat max
    )
    
    # functions for GC grid cell
    def corners_lon(inlon):
        arr = np.array([
            inlon - dlon / 2,
            inlon + dlon / 2,
            inlon + dlon / 2,
            inlon - dlon / 2,
        ])
        return arr

    def corners_lat(inlat):
        arr = np.array([
            inlat - dlat / 2,
            inlat - dlat / 2,
            inlat + dlat / 2,
            inlat + dlat / 2,
        ])
        return arr
    
    # Crreate GC geometry
    gc_lon_corners = (
        np.stack(
            pd.DataFrame(gc_ll['lon'])
            .apply(corners_lon, axis=1)
            .values
        ).squeeze()
    )
    gc_lat_corners = (
        np.stack(
            pd.DataFrame(gc_ll['lat'])
            .apply(corners_lat, axis=1)
            .values
        ).squeeze()
    )
    
    # limit to -90/90
    gc_lat_corners = np.fmin(90,np.fmax(-90, gc_lat_corners))
    
    # limit to -180
    # cuts off data in 178.75 to 180
    gc_lon_corners = np.fmax(-180,gc_lon_corners)
    
    # now we add back 178.75 to 180, need to rejoin later
    gc_lon_corners = np.vstack([gc_lon_corners, [178.75, 180, 180, 178.75]])
    
    lat_list = []
    lon_list = []
    lat_lon_prod = itertools.product(
        range(gc_lon_corners.shape[0]),
        range(gc_lat_corners.shape[0])
    )
    for idx,i in enumerate(lat_lon_prod):
        lon_list.append(i[0])
        lat_list.append(i[1])  

    # lat lon pairs
    ll_gc_pairs = np.concatenate((
        gc_lon_corners[lon_list,:][:,:,None],
        gc_lat_corners[lat_list,:][:,:,None]
    ), axis=-1)
    # clip GC grid so grid cells far from
    # sat obs are not included
    drop_gc_grid = (
        (ll_gc_pairs[:,:,0].max(-1) < sat_bounds[0]) |
        (ll_gc_pairs[:,:,0].min(-1) > sat_bounds[1]) |
        (ll_gc_pairs[:,:,1].max(-1) < sat_bounds[2]) |
        (ll_gc_pairs[:,:,1].min(-1) > sat_bounds[3])
    )
    ll_gc_pairs_clip = ll_gc_pairs[~drop_gc_grid]
    # create dataframe from from lat/lon pairs
    dfgc = pd.DataFrame(
        [tuple(ll_gc_pairs_clip[i]) for i in range(ll_gc_pairs_clip.shape[0])]
    )
    gc_ind = np.arange(ll_gc_pairs.shape[0])
    # create GC grid geometry
    gdf_gc = gpd.GeoDataFrame(
        data = dfgc,
        geometry = dfgc.apply(
            lambda x: Polygon(np.stack(x.values)),
            axis=1
        ),
        crs="EPSG:4326"
    )
    gdf_gc['gc_idx'] = gc_ind[~drop_gc_grid]
    # equal area projection (!!!)
    proj_str = (
        '+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 '
        '+datum=WGS84 +units=m +no_defs'
    )
    
    # project sat geometry and GC geometry
    # to equal area projections
    gdf_tr_ea = gdf.to_crs(proj_str)
    gdf_gc_ea = gdf_gc.to_crs(proj_str)
    
    # overlay geometries to get overlaps
    overlay = gdf_tr_ea.overlay(gdf_gc_ea)
    
    # now adjust so antimeridian obs go to correct GC cell
    # this is pretty rough...
    # basically, we appended extra longitudes
    # on the end, so now we just  "move" them
    # back to the correct longitude grid cells
    max_idx = gc_ll['lon'].shape[0] * gc_ll['lat'].shape[0]
    am_grids = overlay[(overlay.gc_idx >= max_idx)].index
    overlay.loc[am_grids, 'gc_idx'] = overlay.loc[am_grids, 'gc_idx'] - max_idx
    
    # area weights for each sat ob location
    # weight = area_obs_i / sum(area_all_obs_in_GC_gridcell)
    grid_cell_fraction = (
        overlay.geometry.area
        .groupby(overlay.gc_idx)
        .transform(lambda x: x / x.sum())
    )

    # fraction of obs in each gridcell
    obs_fraction = (
        overlay.geometry.area
        .groupby(overlay.sat_idx)
        .transform(lambda x: x / x.sum())
    )
    
    
    # setup weights in 1-D
    multi_index = overlay.set_index(['gc_idx', 'sat_idx']).index
    df_weights = pd.DataFrame(
        {
            "weights": grid_cell_fraction.values,
            "fractions": obs_fraction.values
        },
        index=multi_index
    )
    ds_weights = xr.Dataset(df_weights)
    
    # unstack
    # 1-D --> 2-D for matrix multiplication
    weights = ds_weights.unstack(fill_value = 0.)
    
    return weights
    
    


def make_regridding_weight(
    sat_file,
    base_run = True,
    k_shape = 3753, # num of state vector elements
    gc_path = (
        #'/n/home06/jeast/proj/globalinv/test_concat/'
        'prod/output/imi_20180601/jacobian_runs/'
        'imi_20180601_000000/OutputDir'
    ),
    weight_path = (
        #'/n/home06/jeast/proj/'
        #'globalinv/regrid_weights/2x2.5'
        '/n/holylfs05/LABS/jacob_lab/Users/jeast/'
        'proj/globalinv/prod/weights/2x2.5'
    )
):
    '''
    Make weight file for regridding
    '''

    # get time and valid flag for sat obs
    df = process_list_of_netcdf_files(sat_file)
    df = df[['t_step','valid']]
    
    # get all satellite data from file
    # including location data
    tropomi_dat = read_blended(sat_file)
    
    # for each GC timestep in the sat file,
    # get the corresponding GC output file
    # each sat file has 1 or 2 GC output timesteps
    dsout_list = []
    skip_tidx = []
    for itidx,tidx in enumerate(df.t_step.unique()):
        
        #corresponding GC output file
        fpat = 'output_%Y%m%dT%H.nc'
        gc_file = (
            pd.to_datetime(tidx)
            .strftime(f'{gc_path}/{fpat}')
        )
        
        # open the GC file
        if os.path.isfile(gc_file):
            with xr.open_dataset(gc_file) as gcf:
                gc_xch4 = gcf['gc_xch4']
        else:
            import warnings
            msg = f'Skipping time step...file {gc_file} does not exist'
            warnings.warn(msg)
            skip_tidx.append(itidx)
            continue
            
        # identify valid sat pixels
        # and subset for sat pixels
        # in this GC time step
        keep_vals = (
            df['valid'] &
            (df['t_step'] == tidx)
        ).values

        # subset the sat location data
        # based on valid pixels.
        # tropomi_dat is a dict and 
        # values are arrays of different
        # shapes, so need to subset each one
        # separately based on its shape
        tropomi_dat_out = dict()
        for k,v in tropomi_dat.items():
            if len(v.shape) < 3:
                keep_shaped = np.broadcast_to(
                    keep_vals, v.shape
                )
                tropomi_dat_out[k] = np.atleast_1d(
                    v[keep_shaped].squeeze()
                )
            elif len(v.shape) == 3:
                keep_shaped = np.broadcast_to(
                    keep_vals[None,:,None],
                    v.shape
                )
                tropomi_dat_out[k] = np.atleast_2d(
                    v[keep_shaped]
                    .reshape(
                        1, keep_vals.sum(), v.shape[2]
                    ).squeeze()
                )
            else:
                e = 'Unexpected TROPOMI data shape'
                raise Exception(e)
                
        # Now, tropomi data we subsetted exactly
        # matches GC output locations 
        # replace TROPOMI methane values with
        # GC "virtual" methane values
        tropomi_dat_out['gc_methane'] = tropomi_dat_out['methane'].copy()
        tropomi_dat_out['gc_methane'] = gc_xch4.values * 1e9 # to ppb
        
        #####
        # grid data to GC grid
        ####
        lat = np.arange(-90, 91, 2, dtype=np.float32)
        # poles are 0.5 degree smaller
        lat[0] = lat[0] + 0.5
        lat[0] = lat[0] - 0.5
        lon = np.arange(-180, 178, 2.5, dtype=np.float32)
        gc_ll = {'lon':lon, 'lat':lat}
        
        # filter data to match IMI filter
        # (redundant but included for consistency)
        dt_date = pd.to_datetime(tidx).floor('1d')
        sat_ind = filter_blended(
            tropomi_dat_out,
            [-180,180], [-90,90],
            dt_date,
            dt_date + pd.Timedelta('1d')
        )
        
        if sat_ind.size == 0:
            msg = (
                f'Skipping tstep {tidx} in file {sat_file}. '
                'No valid obs.'
            )
            import warnings
            warnings.warn(msg)
            skip_tidx.append(itidx)
            continue
        
        # check if weight file exists, create it if not
        fpat_weight = 'weight_%Y%m%dT%H.nc'
        weight_file = (
            pd.to_datetime(tidx)
            .strftime(f'{weight_path}/{fpat_weight}')
        )

        # if weight file does not exist
        if not os.path.isfile(weight_file):
            # create weights file

            weights = calc_weights(tropomi_dat_out, gc_ll, sat_ind)
            
            # #corresponding GC output file
            comment = (
                'weights to regrid file '
                f'{gc_file}'
            )
            
            weights.attrs['note'] = comment
            
            weights.to_netcdf(
                weight_file, encoding = {v: {'zlib':True, 'complevel':1} for v in weights.data_vars}
            )
    
        else:
            # open existing file 
            with xr.open_dataset(weight_file) as wds:
                weights = wds.copy(deep=True)
        
       # Regrid !

        # lat/lon and i/j indices of GC gridcells 
        # so that we can put the data onto the grid
        ll_prod = itertools.product(gc_ll['lon'], gc_ll['lat'])
        ij_prod = itertools.product(np.arange(gc_ll['lon'].size), np.arange(gc_ll['lat'].size))
        gc_idx = (
            pd.DataFrame(np.array([[i,j] for i,j in ll_prod]))
            .rename({0:'lon',1:'lat'}, axis=1)
        )
        igc_idx = (
            pd.DataFrame(np.array([[i,j] for i,j in ij_prod]))
            .rename({0:'iGC',1:'jGC'}, axis=1)
        )
        df_gc_idx = igc_idx.merge(
            gc_idx, how='outer', left_index=True, right_index=True
        )


        ii = df_gc_idx.loc[weights.gc_idx.values]['iGC'] # lons
        jj = df_gc_idx.loc[weights.gc_idx.values]['jGC'] # lats
        lon_clip = df_gc_idx.loc[weights.gc_idx.values]['lon'] # lons
        lat_clip = df_gc_idx.loc[weights.gc_idx.values]['lat'] # lats

        # make blank dataframe with vars we want to keep
        dsout = make_blank_dataset(gc_ll['lat'], gc_ll['lon'], k_shape, base_run)
        
        # fill dataset with regridded data
        dsout['geoschem_methane'].values[jj, ii] = np.dot(weights.weights.values, tropomi_dat_out['gc_methane'][sat_ind])
        if base_run:
            dsout['tropomi_methane'].values[jj, ii] = np.dot(weights.weights.values, tropomi_dat_out['methane'][sat_ind])
            dsout['observation_count'].values[jj, ii] = weights.fractions.values.sum(1)
            dsout['iGC'].values[jj, ii] = ii.values
            dsout['jGC'].values[jj, ii] = jj.values
            dsout['lon_sat'].values[jj, ii] = lon_clip.values
            dsout['lat_sat'].values[jj, ii] = lat_clip.values


        dsout_list.append(dsout)
    
    if len(dsout_list) == 0:
        return None

    dsout_combine = xr.concat(dsout_list, dim='time')
    keep_tidx = [i for i,_ in enumerate(df.t_step.unique()) if i not in skip_tidx]
    tcoords = df.t_step.unique()[keep_tidx]
    dsout_combine = (
        dsout_combine.assign_coords(time=tcoords)
    )
    
    return dsout_combine
    

if __name__ == '__main__':
    print('starting')
    from multiprocessing import get_context
    import sys
    import pandas as pd
    import os
    

    # call script like this (example):
    #
    # python -u ProcessGlobalJacobianRuns.py 201806 imi_20180601_000000 /n/holylfs06/SCRATCH/jacob_lab/jeast/proj/globalinv/
    #
    #     or, e.g.:
    #
    # python -u ProcessGlobalJacobianRuns.py 201807 imi_20180601_001375 /n/holylfs06/SCRATCH/jacob_lab/jeast/proj/globalinv/
    #


    # user inputs to script
    # month of data to process, format "YYYYMM"
    month = pd.to_datetime(sys.argv[1], format='%Y%m')

    # imi case to process, format "imi_YYYYMMDD_NNNNNN"
    imi_case = sys.argv[2] 
    imi_dir = '_'.join(imi_case.split('_')[:-1]) # drop sv suffix
    sv_case = imi_case.split('_')[-1]

    # input files dir
    input_dir = sys.argv[3]

    # whether it is base run or not
    if sv_case == '000000':
        base_run_input = True
    else:
        base_run_input = False

    # glob sat files
    month_str = month.strftime('S5P_BLND_L2__CH4____%Y%m??T*.nc')
    sat_files_path = (
        f'/n/holylfs05/LABS/jacob_lab/imi/ch4/blended/{month_str}'
    )
    blnd_files = sorted(glob(sat_files_path))

    # GC files path
    gc_path = (
        #'/n/holyscratch01/jacob_lab/jeast/proj/globalinv/'
        f'{input_dir}/'
        f'prod/output/{imi_dir}/jacobian_runs/'
        f'{imi_case}/OutputDir'
    )

    # number state vector elements, always same
    n_sv = 3753

    # arguments for function calls
    args_in = [(blnd_f, base_run_input, n_sv, gc_path) for blnd_f in blnd_files] 

    print('processing')
    # following https://pythonspeed.com/articles/python-multiprocessing/
    with get_context('spawn').Pool() as p:
        result = p.starmap(make_regridding_weight, args_in)

    print('merging')
    result_clean = list(filter(None, result))
    ds_merged = xr.concat(result_clean, 'time')
    print('writing to disk')

    output_path = lambda x,y: (
        '/n/holylfs05/LABS/jacob_lab/Users/jeast/proj/globalinv/'
        f'prod/output/{x}/inversion/data_converted_nc/out_{y}.nc'
    )    
    os.makedirs('/'.join(output_path(imi_dir, imi_case).split('/')[:-1]), exist_ok=True)

    ds_merged.to_netcdf(
        output_path(imi_dir, imi_case),
        encoding = {v: {'zlib':True, 'complevel':1} for v in ds_merged.data_vars},
        unlimited_dims='time'
    ) 
    print('done')

