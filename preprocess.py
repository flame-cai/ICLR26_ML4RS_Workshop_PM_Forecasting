import os
import numpy as np
import multiprocessing
from tqdm import tqdm
from netCDF4 import Dataset
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from datetime import datetime, timezone

InDir = 'downloads'
OutDir = 'data'

xvar = {'forecast_period', 'forecast_reference_time', 'pressure_level', 'latitude', 'longitude', 'valid_time'}
LatLon = [
    (8, -453), (8, -276), (8, -99), (8, 78), (8, 255),
    (185, -453), (185, -276), (185, -99), (185, 78), (185, 255)
]

def process_zip(path):
    with TemporaryDirectory() as tmpdir:
        try:
            with ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
        except:
            print(path)
            return

        file = Dataset(os.path.join(tmpdir, 'data_sfc.nc'), 'r')
        nlat = len(file.variables['latitude'][:])
        nlon = len(file.variables['longitude'][:])
        variables = sorted(list(set(file.variables.keys()) - xvar))
        for i, t in enumerate(file.variables['valid_time'][:]):
            t = datetime.fromtimestamp(t[0], tz=timezone.utc).strftime('%Y-%m-%d_%H')
            for j, (lat, lon) in enumerate(LatLon):
                lon = lon % nlon
                l = lon + 256
                temp = {}
                for v in variables:
                    if l <= nlon:
                        temp[v] = file.variables[v][0, i, lat:lat+256, lon:l]
                    else:
                        temp[v] = np.concatenate((file.variables[v][0, i, lat:lat+256, lon:nlon], file.variables[v][0, i, lat:lat+256, :l - nlon]), axis=1)
                    assert temp[v].shape == (256, 256) 
                np.savez_compressed(os.path.join(OutDir, f'{t}_{j}.npz'), **temp)

year = int(input("Year: "))

paths = [os.path.join(InDir, p) for p in os.listdir(InDir) if p.startswith(f'{year}-') and p.endswith('.zip')]

with multiprocessing.Pool() as pool:
    list(tqdm(pool.imap_unordered(process_zip, paths), total=len(paths))) 
