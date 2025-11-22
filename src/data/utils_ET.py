import cftime
from mpl_toolkits.basemap import Basemap
import rioxarray
import matplotlib.dates as mdates
from datetime import date, timedelta, datetime

import numpy as np
import xarray as xr
import pandas as pd
import cftime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
#import rioxarray
from shapely.geometry import mapping
from pyhdf.SD import SD, SDC
import fnmatch
import os

from tqdm import tqdm
import re
from pyproj import Transformer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Load required libraries (For download)
import sys
import urllib
import fnmatch
import lxml.html
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from http.cookiejar import CookieJar
import json
import os
import os.path
from datetime import datetime



# Databricks notebook source
# Define a function that lists all the items under the url
def url_lister(url):
    urls = []
    connection = urllib.request.urlopen(url)
    dom =  lxml.html.fromstring(connection.read())
    for link in dom.xpath('//a/@href'):
        urls.append(link)
    return urls

# COMMAND ----------

# Define a function that reads the metadata (ul and lr points x,y coordinates under sinusoidal crs) of MODIS PET hdf data. 
# Input: fattrs: data attributes; data: pyhdf read data
def get_x_y(fattrs, data):
    ga = fattrs['StructMetadata.0']
    gridmeta = ga[0]
    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                              (?P<upper_left_x>[+-]?\d+\.\d+)
                              ,
                              (?P<upper_left_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)
    match = ul_regex.search(gridmeta)
    x0 = float(match.group('upper_left_x')) 
    y0 = float(match.group('upper_left_y')) 

    lr_regex = re.compile(r'''LowerRightMtrs=\(
                              (?P<lower_right_x>[+-]?\d+\.\d+)
                              ,
                              (?P<lower_right_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)
    match = lr_regex.search(gridmeta)
    x1 = float(match.group('lower_right_x'))
    y1 = float(match.group('lower_right_y'))

    ny, nx = data.shape
    xinc = (x1 - x0) / nx
    yinc = (y1 - y0) / ny

    x = np.linspace(x0, x0 + xinc*nx, nx)
    y = np.linspace(y0, y0 + yinc*ny, ny)
    
    return x, y

# COMMAND ----------

# A function outputs if a input year is a leap year
def leap_year(year):
    if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0):
        trig=False
    else:
        trig=True
    return trig

 
# A function create a "eight-day seperation" list from start date and end date input
def create_eight_day_list(sdate, edate):
    day = sdate
    date_list = [day]
    
    while day < edate:
        if leap_year(day.year):
            if  day==date(day.year, 12, 26):
                day = sdate + timedelta(days=6)
                sdate=day
            else:
                day = sdate + timedelta(days=8)
                sdate=day
        else:
            if  day==date(day.year, 12, 27):
                day = sdate + timedelta(days=5)
                sdate=day
            else:
                day = sdate + timedelta(days=8)
                sdate=day

        date_list.append(day)
    return date_list
    
# Define a function that reads the metadata (ul and lr points x,y coordunates under sinusoidal crs) of MODIS PET hdf data and transform the x,y coordinates to lat and lon (not used anymore). It converts (x,y) coordinates under sinusoidal crs to (lat, lon) coordinates under EPSG:4326
# Input: fattrs: data attributes; data: pyhdf read data
def get_lat_lon(fattrs, data):
    ga = fattrs['StructMetadata.0']
    gridmeta = ga[0]
    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                              (?P<upper_left_x>[+-]?\d+\.\d+)
                              ,
                              (?P<upper_left_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)
    match = ul_regex.search(gridmeta)
    x0 = np.float(match.group('upper_left_x')) 
    y0 = np.float(match.group('upper_left_y')) 

    lr_regex = re.compile(r'''LowerRightMtrs=\(
                              (?P<lower_right_x>[+-]?\d+\.\d+)
                              ,
                              (?P<lower_right_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)
    match = lr_regex.search(gridmeta)
    x1 = np.float(match.group('lower_right_x'))
    y1 = np.float(match.group('lower_right_y'))

    ny, nx = data.shape
    xinc = (x1 - x0) / nx
    yinc = (y1 - y0) / ny

    x = np.linspace(x0, x0 + xinc*nx, nx)
    y = np.linspace(y0, y0 + yinc*ny, ny)
    xv, yv = np.meshgrid(x, y)
    
    # Reproject into epsg:4326
    sinu = '+proj=sinu +R=6371007.181 +nadgrids=@null +wktext'
    t = Transformer.from_crs(sinu, "epsg:4326", always_xy=True)
    lon, lat = t.transform(xv, yv)
    
    return lat, lon

# A function that: (1) Processing the data using formulas: real-data = (data - add_offset) * scale_factor ; invalid data = np.nan, and (2) Calculate the daily average (daily sum) from the original 8-day sum data
def valid_check(data, day, _FillValue=32767, valid_max=32700, valid_min=-32767, add_offset=0.0, scale_factor=0.1):
    
    invalid = np.logical_or(data> valid_max,
                            data < valid_min)
    invalid = np.logical_or(invalid, data == _FillValue)
    data[invalid] = np.nan
    data = (data - add_offset) * scale_factor 
    data_1 = np.ma.masked_array(data, np.isnan(data))
    
    if leap_year(day.year):
        if (day.month==12 and day.day==26):
            day_num=6
        else:
            day_num=8
    else:
        if (day.month==12 and day.day==27):
            day_num=5
        else:
            day_num=8
            
    daily = data_1.data / day_num
    return daily

# A function that outputs the index of the closest element in a list (lst) to a value (K) (not used anymore)
def closest(lst, K):
    idx = (np.abs(lst - K)).argmin()
    return idx

# COMMAND ----------

# A function that expands the original dataset to a 8-day daily dataset
def expand_eight_days(ds, date_list):
    
    start_date_f, end_date_f = date_list[0], date_list[-1]
    dataset_start = ds[0]
    data_tmp = []

    for m in tqdm(range(len(date_list))):

        if leap_year(date_list[m].year):
            if (date_list[m].month==12 and date_list[m].day==26):
                expand_num=6
            else:
                expand_num=8
        else:
            if (date_list[m].month==12 and date_list[m].day==27):
                expand_num=5
            else:
                expand_num=8

        ds_t = ds[m]
        for n in range(expand_num):
            data_tmp.append(ds_t)
            
    combined = xr.concat(data_tmp, dim='time')
    dataset_start = xr.concat([dataset_start, combined], "time")
    dataset_start = dataset_start.transpose('time', 'latitude', 'longitude')

    dataset_start = dataset_start[1:]

    day_expand = start_date_f
    day_start = start_date_f

    if leap_year(end_date_f.year):
        if (end_date_f.month==12 and end_date_f.day==26):
            expand_num=6
        else:
            expand_num=8
    else:
        if (end_date_f.month==12 and end_date_f.day==27):
            expand_num=5
        else:
            expand_num=8

    date_list_expand = [day_expand]

    while day_expand < (end_date_f + timedelta(days=expand_num-1)):
        day_expand = day_start + timedelta(days=1)
        day_start = day_expand
        date_list_expand.append(day_expand)    


    datetime64_expand = pd.to_datetime(date_list_expand)
    dataset_tile_expand = dataset_start.assign_coords(time = datetime64_expand)
    
    return dataset_tile_expand

# COMMAND ----------

# A function: 1) reads the data using pyhdf; 2) get x,y coordinates using "get_x_y"; 3) conduct valid_check function for averaging 8-day data into daily and replace the fillvalues;
# 4) reproject the data from sinu to epsg:4326 using package rioxarray.reproject.
# Input: file_list_h contains the found file path for the right date; date: the date specified for this file

def hdf_to_reprojected_xarray(file_list_h, date, PET):
    fattrs_start = SD(file_list_h, SDC.READ).attributes(full=1)

    if PET:
        layer = 'PET_500m'
    else:
        layer = 'ET_500m'

    col_start = SD(file_list_h, SDC.READ).select(layer).get().astype(np.double)
    x_cor, y_cor = get_x_y(fattrs_start,col_start)
    col_start = valid_check(col_start, date)

    dataset_na = xr.DataArray(col_start).rename({'dim_0':'y'}).rename({'dim_1':'x'})
    dataset_na = dataset_na.assign_coords(x = x_cor)
    dataset_na = dataset_na.assign_coords(y = y_cor)
    dataset_na = dataset_na.rio.write_crs("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    dataset_na = dataset_na.rio.reproject("EPSG:4326")
    fill_val = dataset_na.attrs['_FillValue']
    c = dataset_na.where(dataset_na != fill_val)
    c = c.rename({'y':'latitude'}).rename({'x':'longitude'})
    dataset_tile = c.expand_dims({'time':1})
    return dataset_tile

# COMMAND ----------

# Single map plot
def plotMap(ax, map_slice, date_object=None, member_id=None):
    """Create a map plot on the given axes, with min/max as text"""
    
    # map_slice is a dataarray on a time step
    img = ax.imshow(map_slice, origin='lower', aspect='auto',extent =[map_slice.longitude.min(),map_slice.longitude.max(),map_slice.latitude.min(),map_slice.latitude.max()],interpolation='nearest')

    minval = map_slice.min(dim = ['latitude', 'longitude'],skipna=True)
    maxval = map_slice.max(dim = ['latitude', 'longitude'],skipna=True)

    # Format values to have at least 4 digits of precision.
    ax.text(0.01, 0.03, "Min: %1.4f" % minval, transform=ax.transAxes, fontsize=17, color='r')
    ax.text(0.99, 0.03, "Max: %1.4f" % maxval, transform=ax.transAxes, fontsize=17, horizontalalignment='right',color='r')
    
    ax.set_xlabel('longitude (degree_east)')
    ax.set_ylabel('latitude (degree_north)')
    
    plt.colorbar(img, ax=ax)
    
    #If pass the date value, print the date to the title
    if date_object:
        #ax.set_title(date_object.values.astype(str)[:19], fontsize=12)
        ax.set_title(date_object.values.item()[:13], fontsize=12)
    if member_id:
        ax.set_ylabel(member_id, fontsize=12)
        
    return ax

# COMMAND ----------

# Plot the first, middel, and last time steps, calling the single map plot function
def plot_first_mid_last(ds, PET):
    """Plot the first, middle, and final time steps for several climate runs."""
    
    # Specify the figure size
    figWidth = 18
    figHeight = 6 
    # Plot three time steps (horizontally)
    numPlotColumns = 3
    fig, axs = plt.subplots(1, numPlotColumns, figsize=(figWidth, figHeight), constrained_layout=True)

    data_slice = ds
    
    # Compute the first, mid, and last time steps
    start_index, end_index = 0,len(ds.time)-1
    midDateIndex = np.floor(len(ds.time) / 2).astype(int)
    
    # Extract the data slice and plot the single figure
    startDate = ds.time[start_index]
    first_step = data_slice.sel(time=startDate) 
    ax = axs[0]
    plotMap(ax, first_step, startDate)
    

    midDate = ds.time[midDateIndex]
    mid_step = data_slice.sel(time=midDate)   
    ax = axs[1]
    plotMap(ax, mid_step, midDate)

    endDate = ds.time[end_index]
    last_step = data_slice.sel(time=endDate)            
    ax = axs[2]
    plotMap(ax, last_step, endDate)
    
    if PET:
        plt.suptitle(f'First, Middle, and Last Timesteps for Selected Runs (daily sum PET, mm/day)', fontsize=20)
    else:
        plt.suptitle(f'First, Middle, and Last Timesteps for Selected Runs (daily sum ET, mm/day)', fontsize=20)

    return fig

# COMMAND ----------

# Plot one tile at a given time step on a world map, calling the single map plot function
def plot_basemap(ds,time_step):
    """Plot the selected region with specified time step on a world basemap"""
    fig = plt.figure(figsize=(16, 12))
    ds[time_step].plot()
    plt.axis('off')
    m = Basemap(projection='cyl', resolution='l',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180)
    # Set ticks for lat and lon (with specified resolution)
    lat_ticks = np.arange(-90.,90.,30)
    lon_ticks = np.arange(-180.,180.,60)
    m.drawparallels(lat_ticks,labels=[1,0,0,0])
    m.drawmeridians(lon_ticks,labels=[0,0,0,1])
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='#46bcec')
    return fig

# COMMAND ----------

# # Plot two tiles at a given time step on a world map, calling the single map plot function
# def plot_basemap_multiple_tiles(ds1, ds2, time_step):
#     """Plot the selected region with specified time step on a world basemap"""
#     fig = plt.figure(figsize=(16, 12))
#     ds1[time_step].plot()
#     ds2[time_step].plot()
    
#     plt.axis('off')
#     m = Basemap(projection='cyl', resolution='l',
#                 llcrnrlat=-90, urcrnrlat=90,
#                 llcrnrlon=-180, urcrnrlon=180)
#     # Set ticks for lat and lon (with specified resolution)
#     lat_ticks = np.arange(-90.,90.,30)
#     lon_ticks = np.arange(-180.,180.,60)
#     m.drawparallels(lat_ticks,labels=[1,0,0,0])
#     m.drawmeridians(lon_ticks,labels=[0,0,0,1])
#     m.drawcoastlines()
#     m.drawcountries()
#     m.drawstates()
#     m.drawmapboundary(fill_color='#46bcec')
#     return fig

# COMMAND ----------

# Plot the regional statatistics by aggreating over time, calling the single map plot function
def plot_stat_maps(ds, PET):
    """Plot the mean, min, max, and standard deviation values for several climate runs, aggregated over time."""

    figWidth = 20 
    figHeight = 6
    # Plot four statistics (horizontally)
    numPlotColumns = 4
    
    fig, axs = plt.subplots(1, numPlotColumns, figsize=(figWidth, figHeight), constrained_layout=True)

    data_slice = ds
    index=0
    
    # Get regional statistics aggreating over time
    data_agg = data_slice.min(dim='time',skipna=True)
    plotMap(axs[0], data_agg)

    data_agg = data_slice.max(dim='time',skipna=True)
    plotMap(axs[1], data_agg)

    data_agg = data_slice.mean(dim='time',skipna=True)
    plotMap(axs[2], data_agg)

    data_agg = data_slice.std(dim='time',skipna=True)
    plotMap(axs[3], data_agg)

    if PET:
        axs[0].set_title(f'min(PET)', fontsize=15)
        axs[1].set_title(f'max(PET)', fontsize=15)
        axs[2].set_title(f'mean(PET)', fontsize=15)
        axs[3].set_title(f'std(PET)', fontsize=15)
    else:
        axs[0].set_title(f'min(ET)', fontsize=15)
        axs[1].set_title(f'max(ET)', fontsize=15)
        axs[2].set_title(f'mean(ET)', fontsize=15)
        axs[3].set_title(f'std(ET)', fontsize=15)
        

    if PET:
        plt.suptitle(f'Spatial Statistics for Selected Runs (daily sum PET, mm/day)', fontsize=20)
    else:
        plt.suptitle(f'Spatial Statistics for Selected Runs (daily sum ET, mm/day)', fontsize=20)

    return fig

# COMMAND ----------

# Plot the temporal statistics by aggreating over region, calling the single map plot function
def plot_timeseries(ds, PET):
    """Plot the mean, min, max, and standard deviation values for several climate runs, 
       aggregated over lat/lon dimensions."""

    figWidth = 15
    figHeight = 10
    linewidth = 3

    
    fig, axs = plt.subplots(1,1,figsize=(figWidth, figHeight))
    data_slice = ds

    # Get temproal statistics aggreating over region
    min_vals = data_slice.min(dim = ['latitude', 'longitude'],skipna=True)
    max_vals = data_slice.max(dim = ['latitude', 'longitude'],skipna=True)
    mean_vals = data_slice.mean(dim = ['latitude', 'longitude'],skipna=True)
    std_vals = data_slice.std(dim = ['latitude', 'longitude'],skipna=True)
    

    missing_indexes = np.isnan(min_vals)
    missing_times = ds.time[missing_indexes]

    
    gg_m = [w for w in missing_times.values]
    kk_m = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in gg_m]
    missing_times_m = [c.strftime("%Y-%m-%d")for c in kk_m]

    if len(missing_times_m)==0:
        print("There is no missing data")
    else:
        print(f"Missing data times include {missing_times_m}")

    gg =[w for w in ds.time.values]
    kk = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in gg]
    dt_list = [c.strftime("%Y-%m-%d")for c in kk]




    axs.plot(dt_list, max_vals, linewidth=linewidth, label='max', color='red')
    axs.plot(dt_list, mean_vals, linewidth=linewidth, label='mean', color='black')

    # Plot one standard deviation range
    axs.fill_between(dt_list, (mean_vals - std_vals), (mean_vals + std_vals), 
                                     color='grey', linewidth=0, label='std', alpha=0.5)
    axs.plot(dt_list, min_vals, linewidth=linewidth, label='min', color='blue')
    
    if len(ds.time) <= 8:
        axs.xaxis.set_major_locator(plt.MaxNLocator(len(ds.time)))
    else:
        axs.xaxis.set_major_locator(plt.MaxNLocator(8))


    ymin, ymax = axs.get_ylim()
    rug_y = ymin + 0.01*(ymax-ymin)
    axs.plot(missing_times_m, [rug_y]*len(missing_times), '|', color='m', label='missing')
    axs.legend(loc='upper right')


    plt.tight_layout(pad=10.2, w_pad=3.5, h_pad=3.5)

    if PET:
        plt.suptitle(f'Temporal Statistics for Selected Runs (daily sum PET, mm/day)', fontsize=20)
    else:
        plt.suptitle(f'Temporal Statistics for Selected Runs (daily sum ET, mm/day)', fontsize=20)

    return fig

# COMMAND ----------

# Plot the temporal statistics by aggreating over two regions, calling the single map plot function
# At each time step, the maximum value is the max values over two regions, the minimum values is the min values over two regions, the mean value is the area-weighted averaged value over two regions, the standard deviation value is the std over two regions when assuming they are not correlated.
# Not used anymore

# def plot_timeseries_multiple_tiles(ds1, ds2):
#     """Plot the mean, min, max, and standard deviation values for several climate runs, 
#        aggregated over lat/lon dimensions."""

#     figWidth = 15
#     figHeight = 10
#     linewidth = 3


#     fig, axs = plt.subplots(1,1,figsize=(figWidth, figHeight))
    
#     data_slice_1 = ds1
#     # Get temproal statistics aggreating over region
#     min_vals_1 = data_slice_1 .min(dim = ['latitude', 'longitude'],skipna=True)
#     max_vals_1 = data_slice_1 .max(dim = ['latitude', 'longitude'],skipna=True)
#     mean_vals_1 = data_slice_1 .mean(dim = ['latitude', 'longitude'],skipna=True)
#     std_vals_1 = data_slice_1 .std(dim = ['latitude', 'longitude'],skipna=True)
#     missing_indexes_1 = np.isnan(min_vals_1)
#     missing_times_1 = ds1.time[missing_indexes_1]

#     data_slice_2 = ds2
#     # Get temproal statistics aggreating over region
#     min_vals_2 = data_slice_2 .min(dim = ['latitude', 'longitude'],skipna=True)
#     max_vals_2 = data_slice_2 .max(dim = ['latitude', 'longitude'],skipna=True)
#     mean_vals_2 = data_slice_2 .mean(dim = ['latitude', 'longitude'],skipna=True)
#     std_vals_2 = data_slice_2 .std(dim = ['latitude', 'longitude'],skipna=True)
#     missing_indexes_2 = np.isnan(min_vals_2)
#     missing_times_2 = ds1.time[missing_indexes_2]

    
#     ds_1_area=np.count_nonzero(~np.isnan(ds1[0]))
#     ds_2_area=np.count_nonzero(~np.isnan(ds2[0]))
#     mean_combined = (mean_vals_1*ds_1_area + mean_vals_2*ds_2_area) / (ds_1_area + ds_2_area)
#     min_vals_combined = [min(*l) for l in zip(min_vals_1, min_vals_2)]
#     max_vals_combined = [max(*l) for l in zip(max_vals_1, max_vals_2)]
#     a = ds_1_area /(ds_1_area + ds_2_area)
#     b = ds_2_area /(ds_1_area + ds_2_area)
    
#     # Reasonably assume that the two areas in  different Hucs are independent
#     std_vals_combined = np.sqrt(a*a*std_vals_1 + b*b*std_vals_2)
    
    
#     axs.plot(ds1.time, max_vals_combined, linewidth=linewidth, label='max', color='red')
#     axs.plot(ds1.time, mean_combined, linewidth=linewidth, label='mean', color='black')
    

#     # Plot one standard deviation range
#     axs.fill_between(ds1.time, (mean_combined - std_vals_combined), (mean_combined + std_vals_combined), 
#                                      color='grey', linewidth=0, label='std', alpha=0.5)
#     axs.plot(ds1.time, min_vals_combined, linewidth=linewidth, label='min', color='blue')

#     ymin, ymax = axs.get_ylim()
#     rug_y = ymin + 0.01*(ymax-ymin)
#     axs.plot(missing_times_1, [rug_y]*len(missing_times_1), '|', color='m', label='missing 1')
#     axs.plot(missing_times_2, [rug_y]*len(missing_times_2), '|', color='m', label='missing 2')
#     axs.legend(loc='upper right')
    
#     # Set the y_lim for better comparision 
#     #axs.set_ylim(0, 1.3)
    
#     plt.tight_layout(pad=10.2, w_pad=3.5, h_pad=3.5)
#     plt.suptitle(f'Temporal Statistics for Selected Runs (daily sum PET, mm/day)', fontsize=20)

#     return fig

# COMMAND ----------

# This function gets the Modis tile number from a (lat, lon). 
# Source: https://gis.stackexchange.com/questions/265400/getting-tile-number-of-sinusoidal-modis-product-from-lat-long
import math
from pyproj import Proj
def lat_lon_to_modis(lat, lon):
    CELLS = 2400
    VERTICAL_TILES = 18
    HORIZONTAL_TILES = 36
    EARTH_RADIUS = 6371007.181
    EARTH_WIDTH = 2 * math.pi * EARTH_RADIUS

    TILE_WIDTH = EARTH_WIDTH / HORIZONTAL_TILES
    TILE_HEIGHT = TILE_WIDTH
    CELL_SIZE = TILE_WIDTH / CELLS
    MODIS_GRID = Proj(f'+proj=sinu +R={EARTH_RADIUS} +nadgrids=@null +wktext')

    
    x, y = MODIS_GRID(lon, lat)
    h = (EARTH_WIDTH * .5 + x) / TILE_WIDTH
    v = -(EARTH_WIDTH * .25 + y - (VERTICAL_TILES - 0) * TILE_HEIGHT) / TILE_HEIGHT
    return int(h), int(v)


def extract_dates_and_times(text_string):
    """
    Extracts the beginning date, beginning time, ending date, and ending time from a given text string.

    Parameters:
    - text_string (str): The text string from which to extract the dates and times.

    Returns:
    - Tuple[str]: A tuple containing the beginning date, beginning time, ending date, and ending time extracted from the text string.
                    Each element in the tuple is a string representing the corresponding date/time value.
                    If a particular date/time cannot be found in the text string, the corresponding element in the tuple will be an empty string.
    """
    beginning_date_pattern = re.compile(r'OBJECT\s*=\s*RANGEBEGINNINGDATE\s*.*?VALUE\s*=\s*"([^"]+)"', re.DOTALL)
    beginning_time_pattern = re.compile(r'OBJECT\s*=\s*RANGEBEGINNINGTIME\s*.*?VALUE\s*=\s*"([^"]+)"', re.DOTALL)
    ending_date_pattern = re.compile(r'OBJECT\s*=\s*RANGEENDINGDATE\s*.*?VALUE\s*=\s*"([^"]+)"', re.DOTALL)
    ending_time_pattern = re.compile(r'OBJECT\s*=\s*RANGEENDINGTIME\s*.*?VALUE\s*=\s*"([^"]+)"', re.DOTALL)

    beginning_date = beginning_date_pattern.search(text_string).group(1) if beginning_date_pattern.search(text_string) else ""
    beginning_time = beginning_time_pattern.search(text_string).group(1) if beginning_time_pattern.search(text_string) else ""
    ending_date = ending_date_pattern.search(text_string).group(1) if ending_date_pattern.search(text_string) else ""
    ending_time = ending_time_pattern.search(text_string).group(1) if ending_time_pattern.search(text_string) else ""

    return beginning_date, beginning_time, ending_date, ending_time

def extract_tile_numbers(filename):
    """
    Extracts horizontal (h) and vertical (v) tile numbers from a filename.

    Parameters:
    filename (str): The filename containing tile numbers in the format ".hXXvXX.".

    Returns:
    tuple: A tuple containing the horizontal and vertical tile numbers as integers.

    Raises:
    ValueError: If the tile numbers are not found in the filename.
    """
    # Define the regular expression pattern to match h and v numbers
    pattern = r'\.h(\d{2})v(\d{2})\.'
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    if match:
        # Extract horizontal and vertical tile numbers
        h_num = match.group(1)
        v_num = match.group(2)
        return int(h_num), int(v_num)
    else:
        raise ValueError("Tile numbers not found in the filename")

# A function that: (1) Processing the data using formulas: real-data = (data - add_offset) * scale_factor ; invalid data = np.nan, and (2) Calculate the daily average (daily sum) from the original 8-day sum data
def valid_check_day_input(data, days, _FillValue=32767, valid_max=32700, valid_min=-32767, add_offset=0.0, scale_factor=0.1):
    """
    Validates and scales input data for daily values calculation.

    Parameters:
    data (np.array): The input data array.
    days (int): The number of days for scaling the data.
    _FillValue (int, optional): The fill value to be treated as invalid. Default is 32767.
    valid_max (int, optional): The maximum valid value for the data. Default is 32700.
    valid_min (int, optional): The minimum valid value for the data. Default is -32767.
    add_offset (float, optional): The offset to be subtracted from the data. Default is 0.0.
    scale_factor (float, optional): The factor by which to scale the data. Default is 0.1.

    Returns:
    np.array: The processed daily values.
    """
    invalid = np.logical_or(data > valid_max, data < valid_min)
    invalid = np.logical_or(invalid, data == _FillValue)
    data[invalid] = np.nan
    data = (data - add_offset) * scale_factor 
    data_1 = np.ma.masked_array(data, np.isnan(data))
    
    daily = data_1.data / days
    return daily

def process_ET_per_huc(gdf_row, df_huc, ET_dataset_clipped):
    """
    Process Evapotranspiration (ET) data for a specific HUC12 region.

    Parameters:
    gdf_row (GeoDataFrame): A GeoDataFrame containing the specific HUC12 region.
    df_huc (DataFrame): A DataFrame to which the processed ET data will be added.
    ET_dataset_clipped (xarray.DataArray): An xarray DataArray containing the clipped ET dataset.

    Returns:
    DataFrame: The input DataFrame with added PET(mm/day) and huc12 columns.
    """
    geometry = gdf_row.geometry
    fill_val = ET_dataset_clipped.attrs['_FillValue']

    try:
        ET_by_huc = ET_dataset_clipped.rio.clip([geometry])
        ET_by_huc_clipped = ET_by_huc.where(ET_by_huc != fill_val)

        ET_avg_array_for_huc = ET_by_huc_clipped.mean(dim=['latitude', 'longitude'], skipna=True).values
        ET_med_array_for_huc = ET_by_huc_clipped.median(dim=['latitude', 'longitude'], skipna=True).values
        df_huc['PET_avg(mm/day)'] = ET_avg_array_for_huc
        df_huc['PET_median(mm/day)'] = ET_med_array_for_huc
        df_huc.insert(0, 'huc12',[gdf_row.huc12] * len(df_huc))
 
    except:
        df_huc['PET_avg(mm/day)'] = [np.nan] * len(df_huc)
        df_huc['PET_median(mm/day)'] = [np.nan] * len(df_huc)
        df_huc.insert(0, 'huc12', [gdf_row.huc12] * len(df_huc))
        
    median_days = df_huc['days'].median()
    if median_days >= 365:
        df_huc = df_huc[df_huc['days'] >= median_days].reset_index(drop=True)
    return df_huc

def process_hdf_files(files, local_dir_path):
    """
    Processes HDF files to extract metadata and calculate day differences.

    Parameters:
    files (list): List of filenames to process.
    local_dir_path (str): Directory path where the HDF files are located.
    extract_dates_and_times (function): Function to extract dates and times from HDF file attributes.

    Returns:
    DataFrame: A pandas DataFrame containing the extracted metadata and calculated day differences.
    """
    days_differences = []
    data = []

    for file in files:
        hdf_file = SD(local_dir_path + '/' + file, SDC.READ)
        attributes = hdf_file.attributes(full=1)
        beginning_date, beginning_time, ending_date, ending_time = extract_dates_and_times(attributes['CoreMetadata.0'][0])

        # Parse the dates
        begin_date = datetime.strptime(beginning_date, "%Y-%m-%d")
        end_date = datetime.strptime(ending_date, "%Y-%m-%d")

        # Calculate the difference in days
        date_difference = (end_date - begin_date).days
        days_differences.append(date_difference + 1)

        # Tile Numbers
        h_num, v_num = extract_tile_numbers(file)

        # Append the data to the list
        data.append({
            'filename': file,
            'beginning date': begin_date,
            'ending date': end_date,
            'days': date_difference + 1,
            'tile_vertical_num': v_num,
            'tile_horizontal_num': h_num
        })

    df = pd.DataFrame(data).sort_values(by=['beginning date', 'tile_vertical_num', 'tile_horizontal_num']).reset_index(drop=True)
    
    return df

def process_hdf_to_dataset(df, local_dir_path, dataset_name):
    """
    Processes HDF files into a reprojected xarray DataSet, organized by time and coordinates.

    Parameters:
    df (DataFrame): DataFrame containing file metadata and tile numbers.
    local_dir_path (str): Directory path where the HDF files are located.
    get_x_y (function): Function to extract X and Y coordinates from HDF file attributes and data.
    valid_check_day_input (function): Function to validate and scale data.

    Returns:
    xarray.Dataset: The reprojected dataset organized by time and coordinates.
    """

    # Initialize dictionaries and lists to store data and coordinates
    data_by_tile_dict = {}
    data_by_time = []
    X_coordinates = np.array([])
    Y_coordinates = np.array([])

    # Get unique tile numbers
    h_list = df['tile_horizontal_num'].unique().tolist()
    v_list = df['tile_vertical_num'].unique().tolist()

    # Determine minimum tile numbers to use for coordinate extraction
    min_v_tile_num = df['tile_vertical_num'].min()
    min_h_tile_num = df['tile_horizontal_num'].min()

    df_groups = df.groupby('beginning date')
    group_0_df = df_groups.get_group(list(df_groups.groups)[0])

    # For the first group of tiles (by date) get the coordinates of the merged tiles
    filtered_groups_df = group_0_df[(group_0_df['tile_horizontal_num'] == min_h_tile_num) | (group_0_df['tile_vertical_num'] == min_v_tile_num) ]
    for _, row in filtered_groups_df.iterrows():

        #Get the file name and open the file
        file_name = row['filename']
        hdf_file = SD(local_dir_path + '/' + file_name, SDC.READ)
        hdf_file_attributes = hdf_file .attributes(full=1)
        data = hdf_file.select(dataset_name).get().astype(np.double)

        if row['tile_horizontal_num'] == min_h_tile_num and row['tile_vertical_num'] == min_v_tile_num:
            X_coordinates, Y_coordinates = get_x_y(hdf_file_attributes, data)
        elif row['tile_vertical_num'] == min_v_tile_num:
            X_coords_by_tile, _ = get_x_y(hdf_file_attributes, data)
            X_coords_by_tile[0] += 0.0001
            X_coordinates = np.hstack([X_coordinates, X_coords_by_tile])
        elif row['tile_horizontal_num'] == min_h_tile_num:
            _, Y_coords_by_tile = get_x_y(hdf_file_attributes, data)
            Y_coords_by_tile[0] += 0.0001
            Y_coordinates = np.hstack([Y_coordinates, Y_coords_by_tile])
    #Get scale factor
    scale_factor_dict = {
    'ET_500m': 0.1,
    'LE_500m': 10000,
    'PET_500m': 0.1,
    'PLE_500m': 10000,
    'ET_QC_500m': 'N/A'
    }

    scale_factor = scale_factor_dict[dataset_name]

    #Create a single xarray of the data. For each date, merge the geospatial tiles and add coordinates    
    for beginning_date, df_group in df_groups:
        print(beginning_date)
        for _, row in df_group .iterrows():
            #Dict key for tracking data
            key = (row['tile_vertical_num'],row['tile_horizontal_num'])
            #Get the file name
            file_name = row['filename']
            #Open the file and data
            hdf_file = SD(local_dir_path + '/' + file_name, SDC.READ)
            data = hdf_file.select(dataset_name).get().astype(np.double)
            data_by_tile_dict[key] = valid_check_day_input(data,row['days'],scale_factor=scale_factor)

        data_blocks = [[data_by_tile_dict[(v, h)] for h in h_list] for v in v_list]
        data_merged = np.block(data_blocks)

        dataset_na = (
            xr.DataArray(data_merged, dims=["y", "x"])
            .assign_coords(x=X_coordinates, y=Y_coordinates)
            .rio.write_crs("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
            .rio.reproject("EPSG:4326")
            )
    
        fill_val = dataset_na.attrs['_FillValue']

        dataset_tile = (
            dataset_na.where(dataset_na != fill_val)
            .rename({'y': 'latitude', 'x': 'longitude'})
            .expand_dims({'time': 1})
            .sortby('latitude')
            .transpose('time', 'latitude', 'longitude')
            .drop('spatial_ref')
            )
        data_by_time.append(dataset_tile)

    if len(data_by_time) == 1:
        ET_dataset = data_by_time[0]
    else:
        ET_dataset = xr.concat(data_by_time, dim="time")

    unique_beginning_dates = sorted({begin_date for begin_date in df['beginning date']})
    unique_beginning_dates_formatted = [date(d.year, d.month, d.day) for d in unique_beginning_dates]    
    datetime64 = pd.to_datetime(unique_beginning_dates_formatted)

    ET_dataset_f = (
        ET_dataset
        .assign_coords(time=datetime64)
        )
    

    return ET_dataset_f

def process_hdf_to_dataset_new(df, local_dir_path, dataset_name, batch_size = 200, use_zarr = False, zarr_store_path="ET_dataset.zarr"):
    """
    Processes HDF files into a reprojected xarray DataSet, organized by time and coordinates.

    Parameters:
    df (DataFrame): DataFrame containing file metadata and tile numbers.
    local_dir_path (str): Directory path where the HDF files are located.
    get_x_y (function): Function to extract X and Y coordinates from HDF file attributes and data.
    valid_check_day_input (function): Function to validate and scale data.

    Returns:
    xarray.Dataset: The reprojected dataset organized by time and coordinates.
    """

    # Initialize dictionaries and lists to store data and coordinates
    data_by_tile_dict = {}
    data_by_time = []
    counter = 0

    X_coordinates = np.array([])
    Y_coordinates = np.array([])

    # Get unique tile numbers
    h_list = df['tile_horizontal_num'].unique().tolist()
    v_list = df['tile_vertical_num'].unique().tolist()

    # Determine minimum tile numbers to use for coordinate extraction
    min_v_tile_num = df['tile_vertical_num'].min()
    min_h_tile_num = df['tile_horizontal_num'].min()

    df_groups = df.groupby('beginning date')
    group_0_df = df_groups.get_group(list(df_groups.groups)[0])

    # For the first group of tiles (by date) get the coordinates of the merged tiles
    filtered_groups_df = group_0_df[(group_0_df['tile_horizontal_num'] == min_h_tile_num) | (group_0_df['tile_vertical_num'] == min_v_tile_num) ]
    for _, row in filtered_groups_df.iterrows():

        #Get the file name and open the file
        file_name = row['filename']
        hdf_file = SD(local_dir_path + '/' + file_name, SDC.READ)
        hdf_file_attributes = hdf_file .attributes(full=1)
        data = hdf_file.select(dataset_name).get().astype(np.double)

        if row['tile_horizontal_num'] == min_h_tile_num and row['tile_vertical_num'] == min_v_tile_num:
            X_coordinates, Y_coordinates = get_x_y(hdf_file_attributes, data)
        elif row['tile_vertical_num'] == min_v_tile_num:
            X_coords_by_tile, _ = get_x_y(hdf_file_attributes, data)
            X_coords_by_tile[0] += 0.0001
            X_coordinates = np.hstack([X_coordinates, X_coords_by_tile])
        elif row['tile_horizontal_num'] == min_h_tile_num:
            _, Y_coords_by_tile = get_x_y(hdf_file_attributes, data)
            Y_coords_by_tile[0] += 0.0001
            Y_coordinates = np.hstack([Y_coordinates, Y_coords_by_tile])
    #Get scale factor
    scale_factor_dict = {
    'ET_500m': 0.1,
    'LE_500m': 10000,
    'PET_500m': 0.1,
    'PLE_500m': 10000,
    'ET_QC_500m': 'N/A'
    }

    scale_factor = scale_factor_dict[dataset_name]

    #Create a single xarray of the data. For each date, merge the geospatial tiles and add coordinates 
    if use_zarr ==True:
        print("Using Zarr for incremental storage.")
        for beginning_date, df_group in df_groups:
            print(f"Processing {beginning_date}")
            for _, row in df_group.iterrows():
                key = (row['tile_vertical_num'], row['tile_horizontal_num'])
                file_name = row['filename']
                hdf_file = SD(local_dir_path + '/' + file_name, SDC.READ)
                data = hdf_file.select(dataset_name).get().astype(np.double)
                data_by_tile_dict[key] = valid_check_day_input(data, row['days'], scale_factor=scale_factor)

            data_blocks = [[data_by_tile_dict[(v, h)] for h in h_list] for v in v_list]
            data_merged = np.block(data_blocks)

            dataset_na = (
                xr.DataArray(data_merged, dims=["y", "x"])
                .assign_coords(x=X_coordinates, y=Y_coordinates)
                .rio.write_crs("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
                .rio.reproject("EPSG:4326")
                )

            fill_val = dataset_na.attrs['_FillValue']

            dataset_tile = (
                dataset_na.where(dataset_na != fill_val)
                .rename({'y': 'latitude', 'x': 'longitude'})
                .expand_dims(time=[pd.to_datetime(beginning_date)]) #.expand_dims({'time': 1})
                .sortby('latitude')
                .transpose('time', 'latitude', 'longitude')
                .drop('spatial_ref')
                )
            
            # Remove the '_FillValue' attribute from the DataArray
            if '_FillValue' in dataset_tile.attrs:
                del dataset_tile.attrs['_FillValue']

            data_by_time.append(dataset_tile)
            counter += 1  # Increment the counter

            # Check if it's time to write a batch to Zarr
            if counter % batch_size == 0:
                print(f"Writing batch of {batch_size} datasets to Zarr...")

                if len(data_by_time) == 1:
                    ET_dataset_agg = data_by_time[0].to_dataset(name=dataset_name)
                else:
                    ET_dataset_agg  = xr.concat(data_by_time, dim="time").to_dataset(name=dataset_name)

                try:
                    print('appending to zarr')
                    # Save the data incrementally to the Zarr store
                    ET_dataset_agg .to_zarr(zarr_store_path, mode='a', append_dim='time')
                except:
                    print('initial write to zarr')
                    ET_dataset_agg .to_zarr(zarr_store_path, mode='w')

                # Clear memory for next batch
                data_by_time = []

        # Write remaining data (if any) after the loop finishes
        if data_by_time:
            print(f"Writing remaining {len(data_by_time)} datasets to Zarr...")
            if len(data_by_time) == 1:
                ET_dataset_agg = data_by_time[0].to_dataset(name=dataset_name)
            else:
                ET_dataset_agg = xr.concat(data_by_time, dim="time").to_dataset(name=dataset_name)
            
            ET_dataset_agg.to_zarr(zarr_store_path, mode='a', append_dim='time')# consolidated=True

        # Consolidate metadata after all batches are written
        print("Consolidating Zarr metadata...")
        xr.Dataset().to_zarr(zarr_store_path, mode='a', consolidated=True)

        # Open the final Zarr dataset and assign time coordinates
        ET_dataset = xr.open_zarr(zarr_store_path, consolidated=True)

    else:
        for beginning_date, df_group in df_groups:
            print(f"Processing {beginning_date}")
            for _, row in df_group .iterrows():
                #Dict key for tracking data
                key = (row['tile_vertical_num'],row['tile_horizontal_num'])
                #Get the file name
                file_name = row['filename']
                #Open the file and data
                hdf_file = SD(local_dir_path + '/' + file_name, SDC.READ)
                data = hdf_file.select(dataset_name).get().astype(np.double)
                data_by_tile_dict[key] = valid_check_day_input(data,row['days'],scale_factor=scale_factor)

            data_blocks = [[data_by_tile_dict[(v, h)] for h in h_list] for v in v_list]
            data_merged = np.block(data_blocks)

            dataset_na = (
                xr.DataArray(data_merged, dims=["y", "x"])
                .assign_coords(x=X_coordinates, y=Y_coordinates)
                .rio.write_crs("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
                .rio.reproject("EPSG:4326")
                )
    
            fill_val = dataset_na.attrs['_FillValue']

            dataset_tile = (
                dataset_na.where(dataset_na != fill_val)
                .rename({'y': 'latitude', 'x': 'longitude'})
                .expand_dims(time=[pd.to_datetime(beginning_date)])
                .sortby('latitude')
                .transpose('time', 'latitude', 'longitude')
                .drop('spatial_ref')
                )
            data_by_time.append(dataset_tile)

        if len(data_by_time) == 1:
            ET_dataset = data_by_time[0].to_dataset(name=dataset_name)
        else:
            ET_dataset = xr.concat(data_by_time, dim="time").to_dataset(name=dataset_name)

    #unique_beginning_dates = sorted({begin_date for begin_date in df['beginning date']})
    #unique_beginning_dates_formatted = [date(d.year, d.month, d.day) for d in unique_beginning_dates]    
    #datetime64 = pd.to_datetime(unique_beginning_dates_formatted)

    #ET_dataset_f = (
    #    ET_dataset
    #    .assign_coords(time=datetime64)
    #    )
    
    return ET_dataset #ET_dataset_f   

def get_tile_numbers(min_lat, min_lon, max_lat, max_lon):
    """
    Determine the MODIS tile numbers for the bounding box points and check if a single tile can cover the area.

    Parameters:
    min_lat (float): Minimum latitude of the bounding box.
    min_lon (float): Minimum longitude of the bounding box.
    max_lat (float): Maximum latitude of the bounding box.
    max_lon (float): Maximum longitude of the bounding box.
    lat_lon_to_modis (function): Function to convert latitude and longitude to MODIS tile numbers.

    Returns:
    tuple: (h_list, v_list, single_tile) 
           h_list (list): List of horizontal tile numbers.
           v_list (list): List of vertical tile numbers.
           single_tile (bool): True if a single tile covers the area, False otherwise.
    """
    # Lower-left
    ll_h, ll_v = lat_lon_to_modis(min_lat, min_lon)
    # Upper-right
    ur_h, ur_v = lat_lon_to_modis(max_lat, max_lon)
    # Lower-right
    lr_h, lr_v = lat_lon_to_modis(min_lat, max_lon)
    # Upper-left
    ul_h, ul_v = lat_lon_to_modis(max_lat, min_lon)

    # Check if a single tile contains all the bounding box points
    single_tile = False
    if ll_h == ur_h == lr_h == ul_h and ll_v == ur_v == lr_v == ul_v:
        print('Found a single tile that contains the selected Hucs:')
        h_list = [ll_h]
        v_list = [ll_v]
        print(f'h={h_list}')
        print(f'v={v_list}')
        single_tile = True
    else:
        # List all tile numbers within the min and max tile number of bounding box points
        print('No single tile found: Will merge multiple tiles')
        h_list = [ll_h, ul_h, lr_h, ur_h]
        v_list = [ll_v, ul_v, lr_v, ur_v]
        h_list = list(range(min(h_list), max(h_list) + 1))
        v_list = list(range(min(v_list), max(v_list) + 1))
        print(f'h={h_list}')
        print(f'v={v_list}')

    return h_list, v_list, single_tile
