import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from pyproj import Proj, transform
import warnings
warnings.filterwarnings('ignore')

source = ColumnDataSource({'latitude': [], 'longitude': [], 'severity': [], 'time': []})

df_source = pd.read_csv('../data/US_Accidents_sub.csv')
df_source['Start_Time'] = pd.to_datetime(df_source['Start_Time'])

df_tmp = df_source.iloc[[0]]
tmp_date = df_source.iloc[0, 1]

MONTH_RANGE = 1

def get_us_accidents():
    global df_tmp, df_source, tmp_date
    # Group by month

    # Get the first group (until the month changes)
    if tmp_date.month != 12:
        new_items = df_source[(df_source['Start_Time'].dt.year == tmp_date.year) & 
                                            (df_source['Start_Time'].dt.month == tmp_date.month + MONTH_RANGE)]
    else:
        new_items = df_source[(df_source['Start_Time'].dt.year == tmp_date.year + 1) & 
                                            (df_source['Start_Time'].dt.month == 1)]
    n = len(new_items)
    if n == 0:
        return None
    print(f"New records ({tmp_date} to {new_items.iloc[-1, 1]}): {n}")
    df_tmp = pd.concat([df_tmp, new_items], axis=0)
    tmp_date = new_items.iloc[-1, 1]
    return new_items


def convert_to_web_mercator(latitudes, longitudes):
    """Convert latitude and longitude from degrees to Web Mercator projection units."""
    in_proj = Proj(init='epsg:4326')  # WGS84 coordinate system
    out_proj = Proj(init='epsg:3857')  # Web Mercator projection
    web_mercator_longitudes, web_mercator_latitudes = transform(in_proj, out_proj, longitudes, latitudes)
    return web_mercator_latitudes, web_mercator_longitudes

def update_plot(latitude, longitude, severity, time, source):
    source.stream({'latitude': [latitude], 
                   'longitude': [longitude],
                   'severity': [severity],
                   'time': [time]
                   })

def track_iss(interval=2):
    
    # US coordinates range
    #(6340332.343706039, -13915064.36657361)
    #(2801774.86356037, -7451122.248866724)
    
    p = figure(x_range=(-14000000, -7500000), y_range=(2900000, 6300000),
               x_axis_type='mercator', y_axis_type='mercator',
               x_axis_label='Longitude', y_axis_label='Latitude',
               width=1000, height=700,
               tooltips=[('Severity', '@severity'),
                         ('Time', '@time')])
    p.add_tile("CARTODBPOSITRON", retina=True)
    
    p.scatter(x='longitude', y='latitude', source=source, size=2, color='red')

    def update():
        new_items = get_us_accidents()
        if new_items is None:
            return

        lats = new_items['Start_Lat'].to_list()
        longs = new_items['Start_Lng'].to_list()
        
        severities = new_items['Severity'].to_list()
        times = new_items['Start_Time'].to_list()
        
        web_mercator_latitudes, web_mercator_longitudes = convert_to_web_mercator(lats, longs)
        for icoor in range(len(new_items)):
            update_plot(web_mercator_latitudes[icoor], web_mercator_longitudes[icoor], 
                        severities[icoor], times[icoor],
                        source)
    curdoc().add_root(p)
    curdoc().add_periodic_callback(update, interval * 1000)
    
track_iss()
