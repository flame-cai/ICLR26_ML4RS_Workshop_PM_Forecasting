import os
import cdsapi
import pandas as pd
from pqdm.threads import pqdm

year = int(input("Year: "))
dates = pd.date_range(f'{year}-01-01', f'{year}-12-31').strftime('%Y-%m-%d').tolist()

c = cdsapi.Client()

def download(date):
    filepath = f'downloads/{date}.nc.zip'
    if os.path.exists(filepath):
        return
    c.retrieve(
        'cams-global-atmospheric-composition-forecasts', {
            'date': date, 'type': 'analysis', 'leadtime_hour': '0', 'format': 'netcdf_zip',
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'variable': [
                '2m_temperature',
                '2m_dewpoint_temperature',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'mean_sea_level_pressure',
                'land_sea_mask',
                'surface_geopotential',
                'particulate_matter_1um',
                'particulate_matter_2.5um',
                'particulate_matter_10um',
            ]
        }, filepath
    )

pqdm(dates, download, n_jobs=4)
