#!/n/home06/jeast/.conda/envs/jpy01/bin/python

import xarray as xr
import numpy as np

# make a global 2x25 state vector file
# with only 3 elements, located in the
# Permian Basin

lat = np.arange(-90, 91, 2, dtype=np.float32)
lon = np.arange(-180, 178, 2.5, dtype=np.float32)
data = np.full((91,144),0,dtype=np.float64)

dsout = xr.Dataset(
    data_vars = {'StateVector': (('lat','lon'),data, {'units':'none'})},
    coords = {
        'lat': (('lat'), lat, {'standard_name': 'latitude', 'long_name': 'Latitude', 'units': 'degrees_north'}),
        'lon': (('lon'), lon, {'standard_name': 'longitude', 'long_name': 'Longitude', 'units': 'degrees_east'}),
    }
)

dsout['StateVector'].loc[32, np.arange(-105,-99,2.5)] = np.arange(1,4)

dsout.to_netcdf(
    'StateVector_testing.nc',
    encoding = {v: {'complevel':9, 'zlib':True} for v in dsout.data_vars}
)
