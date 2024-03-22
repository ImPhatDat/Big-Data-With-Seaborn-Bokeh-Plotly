import requests
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from pyproj import Proj, transform

ISS_URL = 'https://api.n2yo.com/rest/v1/satellite/positions/25544/41.702/-76.014/0/1/&apiKey=PJFAQD-M64K83-7UV42Z-58AA'

source = ColumnDataSource({'latitudes': [], 'longitudes': []})

def get_iss_telemetry(url):
    """Return DataFrame of current ISS telemetry from URL."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch ISS position from {url}. \
        Status code: {response.status_code}")
    data = response.json()
    latitudes = [position['satlatitude'] for position in data['positions']]
    longitudes = [position['satlongitude'] for position in data['positions']]
    return latitudes, longitudes

def convert_to_web_mercator(latitudes, longitudes):
    """Convert latitude and longitude from degrees to Web Mercator projection units."""
    in_proj = Proj(init='epsg:4326')  # WGS84 coordinate system
    out_proj = Proj(init='epsg:3857')  # Web Mercator projection
    web_mercator_longitudes, web_mercator_latitudes = transform(in_proj, out_proj, longitudes, latitudes)
    return web_mercator_latitudes, web_mercator_longitudes

def update_plot(latitudes, longitudes, source):
    source.stream({'latitudes': [latitudes[-1]], 'longitudes': [longitudes[-1]]})

def track_iss(interval=2):
    latitudes = []
    longitudes = []
    
    # US coordinates range
    #(6340332.343706039, -13915064.36657361)
    #(2801774.86356037, -7451122.248866724)
    
    p = figure(x_range=(-14000000, -7500000), y_range=(2900000, 6300000),
               x_axis_type='mercator', y_axis_type='mercator',
               x_axis_label='Longitude', y_axis_label='Latitude')
    p.add_tile("CARTODBPOSITRON", retina=True)
    
    p.scatter(x='longitudes', y='latitudes', source=source, size=2, color='blue')

    def update():
        nonlocal latitudes, longitudes
        lats, longs = get_iss_telemetry(ISS_URL)
        
        latitudes += lats
        longitudes += longs
        
        web_mercator_latitudes, web_mercator_longitudes = convert_to_web_mercator(latitudes, longitudes)
        update_plot(web_mercator_latitudes, web_mercator_longitudes, source)

    curdoc().add_root(p)
    curdoc().add_periodic_callback(update, interval * 1000)
    
track_iss()