import requests
import time
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from pyproj import Proj, transform
import warnings
warnings.filterwarnings('ignore')

SECS = 20 # wait for 20 seconds to get 20 last positions
# API keys
# F54V5K-NF3K5F-M6QQUS-58AZ
# PJFAQD-M64K83-7UV42Z-58AA
ISS_URL = f'https://api.n2yo.com/rest/v1/satellite/positions/25544/0/0/0/{SECS}/&apiKey=F54V5K-NF3K5F-M6QQUS-58AZ'

source = ColumnDataSource({'latitudes': [], 'longitudes': []})

def get_positions(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch ISS position from {url}. \
        Status code: {response.status_code}")
    data = response.json()
    latitudes = [position['satlatitude'] for position in data['positions']]
    longitudes = [position['satlongitude'] for position in data['positions']]
    return latitudes, longitudes

def convert_to_web_mercator(latitudes, longitudes):
    in_proj = Proj(init='epsg:4326')  # WGS84 coordinate system
    out_proj = Proj(init='epsg:3857')  # Web Mercator projection
    web_mercator_longitudes, web_mercator_latitudes = transform(in_proj, out_proj, longitudes, latitudes)
    return web_mercator_latitudes, web_mercator_longitudes

def update_plot(latitude, longitude, source):
    source.stream({'latitudes': [latitude], 'longitudes': [longitude]})

def track_iss(interval=SECS + 1):
    p = figure(x_range=(-19971868.880408566, 19971868.880408566), 
               y_range=(-20037508.342789244, 20037508.342789244),
               x_axis_type='mercator', y_axis_type='mercator',
               x_axis_label='Longitude', y_axis_label='Latitude',
               width=1000, height=700)
    p.add_tile("CARTODBPOSITRON", retina=True)
    
    p.scatter(x='longitudes', y='latitudes', source=source, size=2, color='blue')

    def update():
        lats, longs = get_positions(ISS_URL)
        
        web_mercator_latitudes, web_mercator_longitudes = convert_to_web_mercator(lats, longs)
        for icoor in range(len(lats)):
            update_plot(web_mercator_latitudes[icoor], web_mercator_longitudes[icoor],
                        source)
            time.sleep(1)

    curdoc().add_root(p)
    curdoc().add_periodic_callback(update, interval * 1000)
    
track_iss()