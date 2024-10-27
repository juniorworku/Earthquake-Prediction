import os
import numpy as np
import pandas as pd
import hopsworks
import requests
import io
from datetime import date, timedelta

def get_earthquakes_data(date=date.today()):
    params = {
        'format': 'csv',
        'starttime': date - timedelta(days=1),
        'endtime': date
    }

    try:
        r = requests.get('https://earthquake.usgs.gov/fdsnws/event/1/query', params=params)
        r.raise_for_status()
        earthquakes_df = pd.read_csv(io.StringIO(r.text))

    except requests.exceptions.RequestException as e:
        print(f"Error fetching earthquake data: {e}")
        return pd.DataFrame()  # Return empty DataFrame if there’s an error
    
    return earthquakes_df

def filter_earthquakes_data(df):
    if df.empty:
        print("No data to filter.")
        return df
    df.dropna(inplace=True)
    df = df[['id', 'time', 'latitude', 'longitude', 'depth', 'depthError', 'rms', 'status', 'type', 'mag']]
    df = df[(df.type == 'earthquake') & (df.mag >= 2.5)]  # Filter significant earthquakes
    df['reviewed'] = (df['status'] == 'reviewed').astype(float)
    df.drop(columns=['type', 'status'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={'depthError': 'deptherror'}, inplace=True)
    return df

def add_earthquakes_feature_group():
    project = hopsworks.login()
    fs = project.get_feature_store()

    earthquakes_df = filter_earthquakes_data(get_earthquakes_data())

    if earthquakes_df.empty:
        print("No data to insert into the feature group.")
        return
    
    earthquakes_fg = fs.get_feature_group(name='earthquakes', version=1)
    earthquakes_fg.insert(earthquakes_df)

if __name__ == "__main__":
    add_earthquakes_feature_group()