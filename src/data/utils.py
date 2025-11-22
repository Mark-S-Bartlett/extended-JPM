"""
This file contains a collection of Python functions designed to support the analysis and visualization.
"""
import plotly.express as px
from dataretrieval import nwis, codes
import requests, datetime, warnings, folium, baseflow, pickle
from plotly.offline import plot
import plotly.express as px
import contextily as ctx
from contextily.tile import _calculate_zoom  # internal helper
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, shape

import copy
import errno
from functools import reduce
from itertools import product
import json
import logging 
import math as m
import multiprocessing as multi
import os
import pathlib as pl
import re 
import requests
import shutil
import sys
import time
import warnings; warnings.filterwarnings('ignore')
import zipfile
from zipfile import ZipFile



from grass_session import Session
from grass.pygrass.modules import Module, ParallelModuleQueue
from grass_session import Session
import grass.script as gs
from grass.script import core as gcore
import grass.script.array as garray
import grass.script.setup as gsetup
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import gdal
import pyarrow.feather as feather
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
import rasterio

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def save_raster(raster:str, save_path_posix: 'pathlib.Path', nodata: int=0) -> 'gtif':
    '''
Saves a GIS raster to a TIFF file.

Parameters:
-----------
raster: str
    Name of the GIS raster layer
save_path_posix: PosixPath
    Path to the save location
nodata: int, optional
    Value for missing data in the raster

Returns:
--------
None
'''
    gs.run_command('r.out.gdal',
        input=raster, 
        output=save_path_posix,
        format='GTiff',
        nodata=nodata
    )

def run_parallel(mod: 'GRASS_Module', workers: int, jobs: list, **kwargs) -> list:
    '''
    Runs GRASS modules in parallel using specified workers and job names.

    Parameters:
    -----------
    mod: Module
        GRASS Module object
    workers: int
        Number of CPUs for parallel processing
    jobs: list
        List of job names
    **kwargs:
        Additional options for the module. Lambda functions can be used to include job names.

    Returns:
    --------
    list:
        List of finished Module objects
    '''
    queue = ParallelModuleQueue(nprocs=workers)

    for job in jobs:
        new_mod = deepcopy(mod)
        new_kwargs = {k: (v(job) if callable(v) else v) for k, v in kwargs.items()}
        queue.put(new_mod(**new_kwargs))

    queue.wait()
    return queue.get_finished_modules()

def process_usgs_data(
    state,
    siteStatus,
    siteType,
    parameterCd,
    siteOutput="basic",
    sites=None,
    outputDataTypeCd=None,
    seriesCatalogOutput="false",
):
    """
    Fetches and processes USGS site data for a given state, extracts longitudes and latitudes,
    and converts them to a GeoDataFrame with the specified target Coordinate Reference System (CRS).

    Args:
        state (str): The two-letter state code (e.g., "LA" for Louisiana).
        target_crs (str): The target CRS to which the GeoDataFrame should be reprojected.
            Default is "EPSG:4267" (NAD27).

    Returns:
        GeoDataFrame: GeoDataFrame constructed from valid sites, reprojected to the specified CRS.

    Note:
        If there's an issue fetching the data, the function prints an error message and returns None. The references for the parameters is found here
        https://waterservices.usgs.gov/rest/Site-Service.html#siteStatus
    """
    BASE_URL = "https://waterservices.usgs.gov/nwis/site/"
    PARAMS = {
        "format": "rdb",
        "stateCd": state,
        "siteStatus": siteStatus,
        "siteOutput": siteOutput,
        "siteType": siteType,
        "parameterCd": parameterCd,
        "outputDataTypeCd": outputDataTypeCd,
        "seriesCatalogOutput": seriesCatalogOutput,
        "sites": sites,
    }

    response = requests.get(BASE_URL, params=PARAMS)

    if response.status_code != 200:
        print("Error fetching data from USGS.")
        return

    lines = response.text.split("\n")

    data2 = [line for line in lines if not line.startswith("#") and line]
    data3 = [line.split("\t") for line in data2]
    df = pd.DataFrame(data3[2:], columns=data3[0])

    # Filter out any sites that don't have valid numerical latitude and longitude
    columns_to_check = ["dec_long_va", "dec_lat_va"]
    df = df.dropna(subset=columns_to_check)

    return df

def process_usgs_data_for_param_list(
    state,
    siteStatus,
    siteType,
    parameterCd,
    siteOutput="basic",
    sites=None,
    outputDataTypeCd=None,
    seriesCatalogOutput=False
):
    """
    Processes USGS data for a specified parameter list.

    Parameters:
    - state (str): The US state code (e.g., "CA" for California).
    - siteStatus (str): The site status code (e.g., "active", "inactive").
    - siteType (str): The site type code.
    - parameterCd (str): The parameter code for the data to be processed.
    - siteOutput (str, optional): Output format for site data (default is "basic").
    - sites (list or None, optional): List of site numbers. If provided, the function processes data in chunks defined by the 'increment' parameter.
    - outputDataTypeCd (str or None, optional): Output data type code (default is None).
    - seriesCatalogOutput (bool, optional): Whether to include series catalog output (default is False).
    - increment (int, optional): The number of site numbers to process in each chunk (default is 1000).

    Returns:
    - pandas.DataFrame: Merged DataFrame containing processed USGS data for the specified parameter list.

    Example:
    >>> df_result = process_usgs_data_for_param_list("CA", "active", "streamgage", "00060", siteOutput="basic", sites=[1, 2, 3], outputDataTypeCd="dv", seriesCatalogOutput=True, increment=500)
    >>> print(df_result.head())
         agency_cd  site_no parameter_cd   datetime  value_cd
    0          USGS        1        00060 2022-01-01       1.0
    1          USGS        1        00060 2022-01-02       2.0
    ...        ...      ...          ...        ...       ...
    """
    args = [state, siteStatus, siteType, parameterCd, sites]
    list_count = sum(isinstance(arg, list) for arg in args)

    
    if list_count == 1:
        list_arg, index = next((arg, i) for i, arg in enumerate(args) if isinstance(arg, list))

        if index == 0:
            increments = [list_arg[i : i + 1] for i in range(0, len(list_arg), 1)]
        elif index == 4:
            increments = [list_arg[i : i + 1000] for i in range(0, len(list_arg),1000)]
        else:
            raise ValueError("Invalid index; only input a list for one input.")
        dfs = []
        for increment_list in increments:
            df = process_usgs_data(
                state=",".join(map(str, increment_list)) if index == 0 else state,
                siteStatus=siteStatus,
                siteType=siteType,
                parameterCd=parameterCd,
                siteOutput=siteOutput,
                sites=",".join(map(str, increment_list)) if index == 4 else sites,
                outputDataTypeCd=outputDataTypeCd,
                seriesCatalogOutput=seriesCatalogOutput,
            )
            dfs.append(df)

        df_merged = pd.concat(dfs, ignore_index=True)
        return df_merged
    else:
        raise ValueError("Exactly one input should be a list for the 'sites' or 'state' parameter.")
        return None

def save_vector(vector:str, save_path_posix: 'pathlib.Path', output_type:str) -> 'geojson':
    '''
Saves a GIS vector layer to a GeoJSON file.

Parameters:
-----------
vector: str
    Name of the GIS vector layer
save_path_posix: PosixPath
    Path to the save location
output_type: str
    Type of vector object to save ('line' or 'boundary')

Returns:
--------
None
'''
    gs.run_command('v.out.ogr', 
        input=vector, 
        output=save_path_posix, 
        format='GeoJSON',
        output_type=output_type
    )

def filter_and_merge_gage_data(df, parm_cd, min_no_years, min_year_inactive_site):
    """
    Filter and merge water gauge data in a DataFrame. For a given parameter code, this function augments the instantaneous data with daily values, but only for those gages lacking instantaneous data. The combined dataset is filtered to stations with a minimum number of years

    Parameters:
    - df (DataFrame): The input DataFrame containing water gauge data.
    - min_no_years (float): The minimum sum of years of data for a site to be included.
    - min_year_inactive_site (int): The minimum year for inactive sites.

    Returns:
    - filtered_merged_df (DataFrame): The filtered and merged DataFrame.

    This function performs the following steps:
    1. Filters rows with 'parm_cd' equal to '00060'.
    2. Separates data into 'uv' and 'dv' categories based on 'data_type_cd'.
    3. Identifies sites not present in both 'uv' and 'dv' categories.
    4. Filters rows based on the identified keys.
    5. Calculates the number of years of data for each site.
    6. Filters rows with a sum of years of data greater than 'min_no_years'.
    7. Filters rows with 'end_date' in the present year or earlier and 'begin_date' in or after 'min_year_inactive_site'.

    Example Usage:
    df_filtered = filter_and_merge_gauge_data(df, 5, 2010)
    """
    # Filter rows with 'parm_cd' equal to '00060'
    df_gage_filtered_by_parmd_cd = df[df["parm_cd"] == parm_cd]

    # Separate data into 'uv' and 'dv' categories
    df_uv = df_gage_filtered_by_parmd_cd[
        df_gage_filtered_by_parmd_cd["data_type_cd"] == "uv"
    ]
    df_dv = df_gage_filtered_by_parmd_cd[
        df_gage_filtered_by_parmd_cd["data_type_cd"] == "dv"
    ]

    # Create lists of 'site_no' for 'uv' and 'dv'
    list_site_uv = df_uv["site_no"].tolist()
    list_site_dv = df_dv["site_no"].tolist()

    # Identify sites not present in both 'uv' and 'dv' categories
    not_in_list1 = [x for x in list_site_uv if x not in list_site_dv]
    not_in_list2 = [x for x in list_site_dv if x not in list_site_uv]

    # Keys to merge
    keys_to_merge = not_in_list2

    # Filter rows from 'df_dv' based on the list of keys
    filtered_df_dv = df_dv[df_dv["site_no"].isin(keys_to_merge)]

    # Concatenate 'df_uv' and 'filtered_df_dv' to add rows from 'df_dv' to 'df_uv'
    merged_df = pd.concat([df_uv, filtered_df_dv], ignore_index=True)

    # Convert date columns to datetime objects
    merged_df["begin_date"] = pd.to_datetime(merged_df["begin_date"])
    merged_df["end_date"] = pd.to_datetime(merged_df["end_date"])

    # Calculate the number of years between 'begin_date' and 'end_date'
    merged_df["years_of_data"] = (
        merged_df["end_date"] - merged_df["begin_date"]
    ).dt.days / 365.25

    # Calculate the sum of years of data per site
    merged_df["sum_of_years"] = merged_df.groupby("site_no")["years_of_data"].transform(
        "sum"
    )

    # Filter rows with a sum of years of data greater than 'min_no_years'
    merged_sum_filter_df = merged_df[merged_df["sum_of_years"] > min_no_years]

    # Get the present year
    present_year = datetime.datetime.now().year

    # Create a boolean mask for rows with 'end_date' in the present year
    mask_present_year = merged_sum_filter_df["end_date"].dt.year == present_year

    # Create a boolean mask for rows with 'end_date' in a past year and 'begin_date' in or after 'min_year_inactive_site'
    mask_past_year = (merged_sum_filter_df["end_date"].dt.year < present_year) & (
        merged_sum_filter_df["begin_date"].dt.year >= min_year_inactive_site
    )

    # Combine the masks using logical OR to get the final mask
    final_mask = mask_present_year | mask_past_year

    # Apply the final mask to the DataFrame
    filtered_merged_df = merged_sum_filter_df[final_mask]

    return filtered_merged_df

def dem_vert_unit_check(dem_source: 'pathlib.Path', raster_name: str, vertical_unit: str):
    '''
    Reviews and converts vertical units in DEM metadata between feet and meters.

    Parameters:
    -----------
    dem_source: pathlib.Path
        Full path to the DEM source file
    raster_name: str
        Name of the raster layer in the GRASS GIS session
    vertical_unit: str
        Desired DEM units ('meters' or 'feet')

    Returns:
    --------
    None
    '''
    vert_value = 0.3048 if vertical_unit == 'feet' else 1.0
    meta = gs.parse_command('g.proj', georef= dem_source, flags='g')
    vert_value_m = float(meta['meters'])
    vert_conversion = (vert_value_m/vert_value)
    if vert_conversion != float(1):
        logger.info('converting vertical units to '+vertical_unit+'. Raw DEM * '+str(vert_conversion))
        gs.run_command('g.region', raster=raster_name)
        gs.mapcalc("new_rast = {0} * {1}".format(raster_name, vert_conversion))
        gs.run_command('g.rename', raster= "new_rast,{}".format(raster_name))
    else:
        pass
    return None

def retrieve_gage_time_series(clipped_gages, parameterCd):
    """
    Retrieves time series data for gage sites based on the information in the clipped_gages DataFrame.

    Parameters:
    - clipped_gages (pd.DataFrame): DataFrame containing information about gage sites, including columns
      'site_no', 'begin_date', 'end_date', and 'data_type_cd'.

    Returns:
    - pandas.DataFrame: Merged DataFrame containing processed time series data for the specified gage sites.

    Example:
    >>> clipped_gages = pd.DataFrame({
    ...     'site_no': [1, 2, 3],
    ...     'begin_date': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01']),
    ...     'end_date': pd.to_datetime(['2022-01-31', '2022-02-28', '2022-03-31']),
    ...     'data_type_cd': ['uv', 'uv', 'uv']
    ... })

    >>> result_df = retrieve_gage_time_series1(clipped_gages)
    >>> print(result_df.head())
         1_Qf  2_Qf  3_Qf
    ...

    Note:
    The function uses NWIS (National Water Information System) data to retrieve time series information
    for specified gage sites. The resulting DataFrame contains processed data columns labeled based on
    the 'site_no' values with appended '_Qf'. Missing values (denoted as -999999.00) are replaced with NaN.
    The 'data_type_cd' is utilized to determine the service ('uv' or 'iv') when retrieving the data.

    The function relies on external services and network connectivity to fetch data; hence, internet access
    and service availability may affect its functionality.
    """

    aggregated = clipped_gages.groupby('site_no').agg(
        oldest_begin_date=('begin_date', 'min'),
        newest_end_date=('end_date', 'max')
        ).reset_index()
    
    gages_no_duplicates = pd.merge(clipped_gages, aggregated, on='site_no').drop_duplicates(subset=['site_no'], keep='first')


    sites, START_DATEs, END_DATEs,cds = (
        gages_no_duplicates['site_no'].tolist(),
        gages_no_duplicates['oldest_begin_date'].dt.strftime('%Y-%m-%d'),
        gages_no_duplicates['newest_end_date'].dt.strftime('%Y-%m-%d'),
        gages_no_duplicates['data_type_cd']
    )

    result_df = pd.concat([
    (df := nwis.get_record(
        sites=site,
        service=cd.replace('uv', 'iv'),
        start=start,
        end=end,
        parameterCd=parameterCd
    )
     .replace(-999999.00, np.nan)
     .groupby(by=pd.Grouper(freq='D'))
     .mean(numeric_only=True)).rename(
        columns=lambda x: x.replace('00060', site) + '_Qf') 
    
        for i, (site, cd, start, end) in enumerate(zip(sites, cds, START_DATEs, END_DATEs))
        ], axis=1)

    return result_df

def filter_columns_by_strings(df, strings_to_exclude):
    """
    Filters columns in a DataFrame based on a list of strings to exclude.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.
    - strings_to_exclude (list): A list of strings. Columns containing any of these strings will be excluded.

    Returns:
    - pandas.DataFrame: A DataFrame containing only the columns that do not contain any string from the
      strings_to_exclude list.

    Example:
    >>> import pandas as pd
    >>> data = {
    ...     'col1_exclude_this': [1, 2, 3],
    ...     'col2': [4, 5, 6],
    ...     'col3_and_this_too': [7, 8, 9]
    ... }
    >>> df = pd.DataFrame(data)
    >>> excluded_strings = ['exclude_this', 'and_this_too']
    >>> filtered_df = filter_columns_by_strings(df, excluded_strings)
    >>> print(filtered_df)
       col2
    0     4
    1     5
    2     6

    Note:
    This function filters the columns of the input DataFrame based on the presence of strings provided
    in the strings_to_exclude list. It retains columns that do not contain any of these strings.
    """
    filtered_columns = [col for col in df.columns if not any(string in col for string in strings_to_exclude)]
    return df[filtered_columns]

def process_baseflow(df_gage_time_series, clipped_gages):
    """
    Process baseflow data from gage time series without control.

    Parameters:
    - df_gage_time_series (pandas.DataFrame): DataFrame containing time series data from gage sites
      without control. Columns likely represent different sites and time steps.
    - clipped_gages (pandas.DataFrame): DataFrame containing clipped gage information, including 'site_no' and
      'drain_area_va'.

    Returns:
    - tuple: A tuple containing:
        - pandas.DataFrame: DataFrame containing KGEs (Kling-Gupta Efficiency) for each site based on baseflow
          separation calculations.
        - dict: A dictionary containing baseflow DataFrames indexed by site number.
        - list: A list of site numbers.
        - numpy.ndarray: An array containing drain area values for each site.

    Note:
    This function processes time series data from gage sites without control to calculate baseflow using the baseflow
    separation method. It iterates over the columns of the input DataFrame to perform the baseflow separation
    for each site, generating KGEs for each site and collecting baseflow dataframes indexed by site number.

    The specifics of the baseflow separation method and its implementation using the 'baseflow' module are assumed
    to be defined within that module or elsewhere in the codebase.
    """

    KGE_data = []
    baseflow_data = {}

    n_end_list_nan = 10

    for i, arr in enumerate(df_gage_time_series.columns):
        site_no = arr.split('_')[0]
        area = clipped_gages.loc[clipped_gages['site_no']==site_no]['drain_area_va'].values.astype(float)[0]

        try:
            Q1, date = df_gage_time_series.iloc[:, i].dropna(), df_gage_time_series.iloc[:, i].dropna().index
            min_value = Q1.min()
            if min_value<0:
                Q1+= abs(min_value)
            b, KGEs = baseflow.separation(Q1, date, area=area)  
            KGE_data.append(KGEs)
            baseflow_data[arr] = pd.DataFrame(b, index=date)
            if min_value<0:
                baseflow_data[arr]=baseflow_data[arr].sub(abs(min_value))
            start_date = baseflow_data[arr].index.min()
            end_date = baseflow_data[arr].index.max()
            date_range = pd.date_range(start_date, end_date)
            baseflow_data[arr] = baseflow_data[arr].reindex(date_range)
        except Exception as e:
            print(f"An error occurred for site_no {site_no}: {e}")
            try:
                print("retrying with last n days set to nan")
                Q1, date = df_gage_time_series.iloc[:, i].dropna(), df_gage_time_series.iloc[:, i].dropna().index
                Q1.iloc[-n_end_list_nan:] = np.nan
                min_value = Q1.min()
                #This is for areas where flow is tidal or how a tiday signal and could be negative (in reverse), i.e.,flowing from the ocean inland.
                if min_value<0:
                    Q1+= abs(min_value)
                b, KGEs = baseflow.separation(Q1, date, area=area)  
                KGE_data.append(KGEs)
                baseflow_data[arr] = pd.DataFrame(b, index=date)
                if min_value<0:
                    baseflow_data[arr]=baseflow_data[arr].sub(abs(min_value))
                start_date = baseflow_data[arr].index.min()
                end_date = baseflow_data[arr].index.max()
                date_range = pd.date_range(start_date, end_date)
                baseflow_data[arr] = baseflow_data[arr].reindex(date_range)
                print("Successful retry")
            except Exception as e:
                print(f"An error occurred for site_no {site_no}: {e}")

        #Q1, date = df_gage_time_series.iloc[:, i].dropna(), df_gage_time_series.iloc[:, i].dropna().index
        #b, KGEs = baseflow.separation(Q1, date, area=area)
        #KGE_data.append(KGEs)
        #baseflow_data[arr] = pd.DataFrame(b, index=date)
        #start_date = baseflow_data[arr].index.min()
        #end_date = baseflow_data[arr].index.max()
        #date_range = pd.date_range(start_date, end_date)
        #baseflow_data[arr] = baseflow_data[arr].reindex(date_range)

    KGEs_df = pd.DataFrame({list(baseflow_data.keys())[i]: arr for i, arr in enumerate(KGE_data)}, index=b.dtype.names)
    KGEs_df['median'] = KGEs_df.median(axis=1)
    return KGEs_df, baseflow_data


def filter_data_by_intersection(df1, df2):
    """
    Filter two DataFrames based on their intersecting index.

    Parameters:
    - df1 (pandas.DataFrame): First DataFrame.
    - df2 (pandas.DataFrame): Second DataFrame.

    Returns:
    - pandas.DataFrame: Filtered DataFrame 1 based on the intersecting index with DataFrame 2.
    - pandas.DataFrame: Filtered DataFrame 2 based on the intersecting index with DataFrame 1.
    """
    intersecting_index = df1.index.intersection(df2.index)
    df1_filtered = df1.loc[intersecting_index]
    return df1_filtered

def plot_pdf_histogram(data, label, percentile=95, num_bins=20, subplots = None ):
    """
    Generate a probability density function (PDF) histogram based on the specified data.

    Parameters:
    label (str): The label for the x-axis.
    data (array-like): The input data array.
    percentile (float, optional): The percentile value for filtering the data. Defaults to 95.
    num_bins (int, optional): The number of bins for the histogram. Defaults to 20.
    """
    percentile_limit = np.percentile(data, percentile)
    data_percentile = data[data <= percentile_limit]

    if subplots == None:
        plt.hist(data_percentile, bins=num_bins, density=True, alpha=0.7, edgecolor='black')
        plt.title('PDF Histogram')
        plt.xlabel(label)
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.show()
    else:
        subplots.hist(data_percentile, bins=num_bins, density=True, alpha=0.7, edgecolor='black')
        subplots.set_title('PDF Histogram')
        subplots.set_xlabel(label)
        subplots.set_ylabel('Probability Density')
        subplots.grid(True)

def process_and_plot_data(index, method, df_gage_time_series, clipped_gages, baseflow_data, df_KGEs, percentile = 100):
    """
    Process data from multiple sources, generate visualizations, and perform analysis.

    Parameters:
    - index (int, optional): Index indicating the site to be processed. Defaults to 2.
    - method (str, optional): Method for flow calculation. Defaults to 'Local'.
    - df_gage_time_series (pd.DataFrame): DataFrame containing gage time series data.
    - clipped_gages (pd.DataFrame): DataFrame with clipped gages information.
    - baseflow_data (dict): Dictionary of baseflow data.
    - percentile (int, optional): Percentile value for analysis. Defaults to 100.

    Returns:
    - fig (plotly.graph_objs.Figure): Plotly figure showing line graphs of processed data.
    - axes (tuple): Tuple containing Matplotlib axes for generated histograms.

    Details:
    This function processes and visualizes data for a specific site, combining gage time series 
    data with baseflow information. It generates line plots and histograms to illustrate the 
    relationship between runoff and flow at the specified site. The `index` parameter determines 
    the site to be processed, while `method` defines the baseflow calculation method.

    The processed data and visualizations provide insights into the hydrological characteristics 
    of the site, including runoff, flow, and their distribution across different percentiles.
    """

    # Perform data processing and visualization
    site_list = list(baseflow_data.keys())
    site_no = site_list[index].split('_')[0]
    area = clipped_gages.loc[clipped_gages['site_no']==site_no]['drain_area_va'].values.astype(float)[0]
    elev = clipped_gages[clipped_gages['site_no'] == site_no]['alt_va'].values[0]
    station_name = clipped_gages[clipped_gages['site_no'] == site_no]['station_nm'].values[0]
    print(station_name + ' at ' + str(elev) + ' ft. with a drainage area of ' + str(area) + ' sq mi.')
    
    df_gage_time_series_w_baseflow = df_gage_time_series[site_list]

    df_gages_filtered = filter_data_by_intersection(df_gage_time_series_w_baseflow.iloc[:, index], baseflow_data[site_list[index]])
    print(df_gages_filtered.name)

    methods_to_exclude = ["Local", "Fixed", "Slide"]
    if area < 10:
        print('Best performing method ',(df_KGEs.iloc[:,index].drop(methods_to_exclude, axis=0)).idxmax())   
    else:
        print('Best performing method ',df_KGEs.iloc[:, index].idxmax())  
    
    fig = px.line(df_gages_filtered, x=df_gages_filtered.index, y=df_gages_filtered)
    fig.add_scatter(x=baseflow_data[site_list[index]].index, y=baseflow_data[site_list[index]][method], mode='lines')
    fig.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  
    plt.subplots_adjust(wspace=0.25)
    
    data = ((df_gages_filtered.values - baseflow_data[site_list[index]][method].values) / area) * (60*60*24*12**3) / (5280*5280*12*12)
    data_cm = ((df_gages_filtered.values - baseflow_data[site_list[index]][method].values) / area) * (60*60*24*12**3) / (5280*5280*12*12)*2.54
   
    non_zero_non_nan_data = data[(data != 0) & (~np.isnan(data))]
    non_zero_non_nan_data_cm = data_cm[(data_cm != 0) & (~np.isnan(data_cm))]
    print(type( non_zero_non_nan_data))
    print('Variance in^2')
    print(np.var(non_zero_non_nan_data))
    print('Variance cm^2')
    print(np.var(non_zero_non_nan_data_cm))
    plot_pdf_histogram(non_zero_non_nan_data, 'Runoff (in.)', percentile=percentile, num_bins=20, subplots=ax1)
    
    data = baseflow_data[site_list[index]][method].values
    non_nan_data = data[(~np.isnan(data))]
    plot_pdf_histogram(non_nan_data, 'Flow (cfs)', percentile=percentile, num_bins=30, subplots=ax2)

    return fig, (ax1, ax2) 


def visualize_stations(
    gdf,
    gdf2=None,
    gdf_line_color="red",
    gdf_fill_color="none",
    gdf2_line_color="blue",
    gdf2_fill_color="none",
    title = 'Please give name',
    table_data=None,
    table_title="Site Nos.",
    table_bbox =[0.02, 0.02, 0.35, 0.6],
    output_pdf=None
    ):
    """
    Visualizes the USGS site data (reprojected to NAD27) on a map with a basemap background.

    This function takes GeoDataFrames containing USGS site data reprojected to NAD27 (EPSG:4267) coordinate reference system (CRS) and plots them on a map with a basemap background. Optionally, you can provide a second GeoDataFrame (`gdf2`) to plot additional data on the same map.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame containing the USGS site data reprojected to NAD27 (EPSG:4267) CRS.
        gdf2 (GeoDataFrame, optional): A second GeoDataFrame to plot additional data on the map (default is None).
        gdf_line_color (str, optional): The color of the line outlining the data points in the first GeoDataFrame (default is 'red').
        gdf_fill_color (str, optional): The fill color of the data points in the first GeoDataFrame (default is 'none').
        gdf2_line_color (str, optional): The color of the line outlining the data points in the second GeoDataFrame (default is 'blue').
        gdf2_fill_color (str, optional): The fill color of the data points in the second GeoDataFrame (default is 'none').

    Note:
        Before using this function, ensure that the GeoDataFrames `gdf` and `gdf2` (if provided) are properly defined and populated with the respective data.

    Example:
        To visualize USGS site data in Louisiana, you can call the function like this:
        visualize_usgs_data(gdf_nad27)

        To visualize USGS site data along with another GeoDataFrame and customize the line and fill colors, provide the parameters as follows:
        visualize_usgs_data(gdf_nad27, gdf2_other, gdf_line_color='red', gdf_fill_color='none', gdf2_line_color='blue', gdf2_fill_color='none')

    Returns:
        None: The function displays the map as a plot but does not return any values.
    """
    # Ensure that gdf is not empty
    if gdf is None or gdf.empty:
        print("GeoDataFrame is empty. Please check that gdf is defined and populated.")
        return

    # Plot the first GeoDataFrame
    ax = gdf.to_crs(epsg=3857).plot(
        figsize=(12, 12),
        markersize=20,
        color=gdf_fill_color,
        edgecolor=gdf_line_color,
        alpha=0.7,
    )

    # Optionally, plot the second GeoDataFrame if provided
    if gdf2 is not None and not gdf2.empty:
        gdf2.to_crs(epsg=3857).plot(
            ax=ax,
            markersize=20,
            color=gdf2_fill_color,
            edgecolor=gdf2_line_color,
            alpha=0.7,
        )

    # Get bounds in Web Mercato
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Ensure proper bbox format (west, south, east, north)
    bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    # Automatically choose zoom (max zoom that covers extent)
    # Unpack bbox into separate arguments (w, s, e, n)
    zoom = _calculate_zoom(*bbox)
    print(zoom)
    zoom = min(zoom, 14)  # cap at 14 so you don't overload memory

    # Add a basemap in the background
    ctx.add_basemap(ax,zoom='auto', source=ctx.providers.OpenStreetMap.HOT) #ctx.providers.OpenStreetMap.Mapnik

    # Set map title and axis labels
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_axis_off()

    # Add a table of USGS site numbers if provided
    if table_data:
        # Split the list into two columns
        col_count = 2
        rows = int(np.ceil(len(table_data) / col_count))  # Calculate the number of rows
        table_text = [  # Reshape the list into a 2D array for the table
            table_data[i : i + col_count] for i in range(0, len(table_data), col_count)
        ]
        # Add padding for shorter columns
        if len(table_text[-1]) < col_count:
            table_text[-1].extend([""] * (col_count - len(table_text[-1])))

        # Create the table
        table = ax.table(
            cellText=table_text,
            colLabels=[f"{table_title}", f"{table_title}"],
            loc="center",  # Position the table within the map
            colWidths=[0.25] * col_count,  # Adjust column widths
            bbox=table_bbox,  # Adjust table's position (x, y, width, height)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.auto_set_column_width([0])

        # Make the table background transparent
        for key, cell in table.get_celld().items():
            cell.set_facecolor("none")  # Set cell background to transparent
            #cell.set_text_props(color="white")  # Set text color to white
            #cell.set_edgecolor("white")  # Set cell edge color to white
            

    # Save the figure to a PDF if a path is provided
    if output_pdf:
        plt.savefig(output_pdf, format="pdf", bbox_inches="tight",dpi=600)
        print(f"Figure saved as {output_pdf}")

    plt.tight_layout()
    plt.show()