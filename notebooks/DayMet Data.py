# Databricks notebook source
# MAGIC %md
# MAGIC # Download DayMet Data and Calculate HUC12 Statistic
# MAGIC
# MAGIC This notebook downloads the DayMet Data and based on the downloaded data, it calculates rainfall statistics on a HUC 12 basis. It does the HUC 12 analysis for all HUC 12s in given HUC 8 watersheds.
# MAGIC
# MAGIC - For the Florida HUCs (St. Johns River), run with ['03070205','03080103','03080101','03080102']
# MAGIC For the Louisiana HUCs, run with ['12040201','12010005','12020003','12020006','12020007','03180002','03180003','03180005','03170009','03180004','08090100','08070202','08090301','08070205','08070300','08090203','08070203','08070204','08070100','08080201','08080206','08090202','08070201','08080204','08080103','08080202','08090201','08080101','08080203','08090302','08080102','08080205']
# MAGIC - Also, change the s3 information and project_name accordingly to your data storage setup (or modify to not use AWS s3 storage).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC
# MAGIC For using this notebook, update the huc08s, project, mount_name, and aws_bucket_name 

# COMMAND ----------

huc08s =  ['12040201','12010005','12020003','12020006','12020007','03180002','03180003','03180005','03170009','03180004','08090100','08070202','08090301','08070205','08070300','08090203','08070203','08070204','08070100','08080201','08080206','08090202','08070201','08080204','08080103','08080202','08090201','08080101','08080203','08090302','08080102','08080205']

#Jacksonville HUCs ['03070205','03080103','03080101','03080102']
#TWI-Transition-Zone HUCS ['12040201','12010005','12020003','12020006','12020007','03180002','03180003','03180005','03170009','03180004','08090100','08070202','08090301','08070205','08070300','08090203','08070203','08070204','08070100','08080201','08080206','08090202','08070201','08080204','08080103','08080202','08090201','08080101','08080203','08090302','08080102','08080205']

project="lwi-transition-zone" #"lwi-transition-zone" "jacksonville"

#S3 bucket information or DBFS
mount_name  = "lwi-transition-zone" #"lwi-transition-zone" "jacksonville-data"
aws_bucket_name = "lwi-transition-zone" #"lwi-transition-zone" "jacksonville-data"

#File Storage Paths and File Names
directory_path_for_daymet="data/hydrology/daymet"
local_dir_path = '/local_disk0/daymet'
s3_bucket_dir = f"/dbfs/mnt/{aws_bucket_name}/{directory_path_for_daymet}"

transfer_to_dbfs=False

# COMMAND ----------

# MAGIC %md
# MAGIC #### Importing Packages
# MAGIC Defining local data location on disk

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC import pydaymet as daymet
# MAGIC from pynhd import WaterData
# MAGIC import os, pyproj, multiprocessing, copy
# MAGIC import xarray as xr
# MAGIC import rioxarray as rio
# MAGIC from tqdm import tqdm
# MAGIC from scipy import optimize
# MAGIC import scipy.stats as stats
# MAGIC from shapely.ops import unary_union
# MAGIC from sklearn.metrics import mean_squared_error
# MAGIC import concurrent.futures, time, shutil
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC from scipy.optimize import fsolve, newton, brentq
# MAGIC from src.data.utils_s3 import *
# MAGIC from src.data.utils_files import *
# MAGIC from src.data.utils import *
# MAGIC from src.data.utils_geo import *
# MAGIC from src.data.utils_statistics import *
# MAGIC user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
# MAGIC
# MAGIC os.environ["HYRIVER_CACHE_NAME"]='/local_disk0/cache/aiohttp_cache.sqlite'
# MAGIC os.environ["HYRIVER_CACHE_NAME_HTTP"] = "/local_disk0/cache/http_cache.sqlite"
# MAGIC os.environ["HYRIVER_CACHE_EXPIRE"] = "0"
# MAGIC #os.environ["HYRIVER_CACHE_DISABLE"] = "false"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Mount s3 Bucket
# MAGIC Mounting the s3 bucket is important because it extends the local files system. Without the bucket, the downloaded files may exceed the storage capacity of the computer.
# MAGIC
# MAGIC Here we mount the bucket with an instance profile, so the selected cluster must have the appropriate instance profile selected, which in this case is 'ec2-s3-access-role'. For reference see: 
# MAGIC - https://docs.databricks.com/en/aws/iam/instance-profile-tutorial.html
# MAGIC - https://docs.databricks.com/en/dbfs/mounts.html
# MAGIC
# MAGIC Here, we also check for the directories where we will store files and create any missing directories.

# COMMAND ----------

mount_aws_bucket(aws_bucket_name, mount_name)
create_dir_list = [f"/dbfs/mnt/{aws_bucket_name}/"+directory_path_for_daymet,local_dir_path]
create_directories(create_dir_list)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Retreive HUC12 and HUC08 Basins Separately
# MAGIC

# COMMAND ----------

nhd_param = {'outFields':'huc8','outSR':4326,'where':list_to_sql(huc08s, "huc8"),'f':'geojson','returnGeometry':'true'}
all_huc8_vector_dict = esri_query(nhd_param, level = 4)
gdf_huc8 = create_geodataframe_from_features(all_huc8_vector_dict, crs="EPSG:4326").set_crs("EPSG:4326")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Retreive HUC12 Basins for the listed HUC8s

# COMMAND ----------

gdf_all_huc12 = gpd.GeoDataFrame()
for huc8 in huc08s:
    nhd_param = {'outFields':'huc12','outSR':4326,'where':f"huc12 LIKE '{huc8}%'",'f':'geojson','returnGeometry':'true'}
    all_huc12_vector_dict = esri_query(nhd_param, level = 6)
    gdf_huc12 = create_geodataframe_from_features(all_huc12_vector_dict, crs="EPSG:4326").set_crs("EPSG:4326")
    gdf_all_huc12 = gdf = pd.concat([gdf_all_huc12,gdf_huc12])
    

# COMMAND ----------


gdf_all_huc12.reset_index(inplace=True)
gdf_all_huc12.to_file('/dbfs/mnt/lwi-transition-zone/data/hydrology/dimensionless_features/pi2/jax_huc12_geometry.geojson')
gdf_all_huc12.plot()

%md
#### Combines all the HUC 8's into a single dataframe
Merged dataframe is then plotted


# COMMAND ----------


# Merge all geometries into one
merged_geometry = unary_union(gdf_huc8.geometry)

# Create a new GeoDataFrame with the merged geometry
gdf_merged = gpd.GeoDataFrame(geometry=[merged_geometry])

gdf_merged.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Checks for HUC 12's based on previous code

# COMMAND ----------

gdf_all_huc12

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Daymet data and Transfer to Storage
# MAGIC Trying to download all the data for all years takes too much time; https://github.com/hyriver/pydaymet/issues/47. 
# MAGIC
# MAGIC Consequently, we look through the years and save out each year of data as a netcdf.
# MAGIC
# MAGIC Not Used
# MAGIC
# MAGIC ```for yr in range(1980, 2024):
# MAGIC     if os.path.exists(f"{s3_bucket_dir}/{file_id}_{yr}.nc")!=1:
# MAGIC         if os.path.exists(f"/local_disk0/{file_id}_{yr}.nc")!=1:
# MAGIC             start_time = time.time() 
# MAGIC             daymet.get_bygeom(merged_geometry, yr, variables=var, region = 'na', snow=True).to_netcdf(
# MAGIC                 f"{local_dir_path}/{file_id}_{yr}.nc"
# MAGIC                 )
# MAGIC             print(f"-----{yr} year took {(time.time()-start_time)} seconds------")```

# COMMAND ----------


var = ["prcp"]
file_id = project
s3_bucket_dir = f"/dbfs/mnt/{aws_bucket_name}/{directory_path_for_daymet}"

#https://hyriver.readthedocs.io/en/latest/readme/hydrosignatures.html
#https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/

# Merge all geometries into one
merged_geometry = unary_union(gdf_huc8.geometry)

for yr in range(1980, 2024):
    for _, row in gdf_huc8.iterrows():
        huc8_in = row['huc8']
        if os.path.exists(f"{s3_bucket_dir}/{file_id}_huc8_{huc8_in}_{yr}.nc")!=1:
            if os.path.exists(f"/local_disk0/{file_id}_huc8_{huc8_in}_{yr}.nc")!=1:
                start_time = time.time() 
                daymet.get_bygeom(row.geometry, yr, variables=var, region = 'na', snow=True).to_netcdf(
                f"{local_dir_path}/{file_id}_huc8_{huc8_in}_{yr}.nc"
                )
                print(f"-----huc 8 {huc8_in} for {yr} year took {(time.time()-start_time)} seconds------")

def copy_instance(file_pat,file_path_dest): 

    # printing process id to SHOW that we're actually using MULTIPROCESSING 
    print("ID of main process: {}".format(os.getpid()))     

    shutil.copy(file_path, file_path_dest) 

#https://stackoverflow.com/questions/62854292/copying-files-from-directory-via-multiprocessing-and-shutil-python
for file_name in os.listdir(local_dir_path):
    if os.path.exists(f"/local_disk0/{file_name}")!=1:
        file_path = os.path.join(local_dir_path, file_name)
        if os.path.isfile(file_path):
            p1 = multiprocessing.Process(target=copy_instance, args=(file_path, os.path.join(s3_bucket_dir, file_name),  )) 
            p1.start()

for file_name in os.listdir(local_dir_path):
    file_path = os.path.join(local_dir_path, file_name)
    if os.path.isfile(file_path):
        shutil.copy(file_path, os.path.join(s3_bucket_dir, file_name)) 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Move Daymet Data to the Local Drive
# MAGIC
# MAGIC for file_name in os.listdir(s3_bucket_dir):
# MAGIC     if os.path.exists(f"/local_disk0/{file_name}")!=1:
# MAGIC         file_path = os.path.join(s3_bucket_dir, file_name)
# MAGIC         if os.path.isfile(file_path):
# MAGIC             shutil.copy(file_path, os.path.join(local_dir_path, file_name))

# COMMAND ----------

for file_name in os.listdir(s3_bucket_dir): 
    if os.path.exists(f"/local_disk0/{file_name}")!=1: 
        file_path = os.path.join(s3_bucket_dir, file_name) 
        if os.path.isfile(file_path): 
            shutil.copy(file_path, os.path.join(local_dir_path, file_name))

# COMMAND ----------

# MAGIC %md
# MAGIC import shutil
# MAGIC import multiprocessing 
# MAGIC import os
# MAGIC
# MAGIC def copy_instance(file_pat,file_path_dest): 
# MAGIC
# MAGIC     # printing process id to SHOW that we're actually using MULTIPROCESSING 
# MAGIC     print("ID of main process: {}".format(os.getpid()))     
# MAGIC
# MAGIC     shutil.copy(file_path, file_path_dest) 
# MAGIC
# MAGIC for file_name in os.listdir(s3_bucket_dir):
# MAGIC     if os.path.exists(f"/local_disk0/{file_name}")!=1:
# MAGIC         file_path = os.path.join(s3_bucket_dir, file_name)
# MAGIC         if os.path.isfile(file_path):
# MAGIC             p1 = multiprocessing.Process(target=copy_instance, args=(file_path,  os.path.join(local_dir_path, file_name), )) 
# MAGIC             p1.start() 
# MAGIC

# COMMAND ----------


import shutil 
import multiprocessing 
import os

def copy_instance(file_pat,file_path_dest):
    # printing process id to SHOW that we're actually using MULTIPROCESSING 
    print("ID of main process: {}".format(os.getpid()))     
    shutil.copy(file_path, file_path_dest) 
for file_name in os.listdir(s3_bucket_dir): 
    if os.path.exists(f"/local_disk0/{file_name}")!=1: 
        file_path = os.path.join(s3_bucket_dir, file_name) 
        if os.path.isfile(file_path): 
            p1 = multiprocessing.Process(target=copy_instance, args=(file_path, os.path.join(local_dir_path, file_name), )) 
            p1.start()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Open and convert to rioxarray

# COMMAND ----------

gdf_all_huc12.reset_index(inplace=True)
gdf_all_huc12

# COMMAND ----------

row_index=gdf_all_huc12.index[gdf_all_huc12.huc12=='080902030900'].tolist()
if row_index:
    row = gdf_all_huc12.iloc[row_index[0]]
    print(row)
else:
    print("Row not found.")

# COMMAND ----------

# MAGIC %md
# MAGIC Extracting HUC12 and HUC8 codes from dataset
# MAGIC Loaded netcdf based on HUC 8 is projected onto ESRI projection
# MAGIC Data is plotted for a timestep 

# COMMAND ----------

ii=39
huc12_in = gdf_all_huc12.iloc[ii].huc12
huc8_in = huc12_in[:-4]
file_id = project

ds_open = xr.open_mfdataset(f"{s3_bucket_dir}/{file_id}_huc8_{huc8_in}*.nc")
spatial_ref_string = ds_open.lambert_conformal_conic.attrs['spatial_ref']
crs = pyproj.CRS.from_string(spatial_ref_string)
ds_open  = ds_open.rio.write_crs(crs)
new_crs = "EPSG:4326"
ds = ds_open.rio.reproject(new_crs)

ds.isel(time=4999).prcp.plot(x="x", y="y")

geometry = gdf_all_huc12.geometry.iloc[ii]

ax = gdf_all_huc12.plot()

# Plot the first row geometry with a different style
gdf_all_huc12.iloc[[ii]].plot(ax=ax, color='red', edgecolor='black', linewidth=2)

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore the data

# COMMAND ----------

geometry = gdf_all_huc12.geometry.iloc[ii]
# Convert the dataset to the new CRS
clipped_ds = ds.rio.clip([geometry])
print(gdf_all_huc12.iloc[ii].huc12)

fig, ax = plt.subplots(figsize=(10, 6))
clipped_ds.isel(time=4999).prcp.plot(x="x", y="y", ax=ax)
gdf_all_huc12.iloc[[ii]].plot(ax=ax,  edgecolor='black', linewidth=2, facecolor='none')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore the Data Statistics
# MAGIC

# COMMAND ----------

aggregate_to_events=True

if aggregate_to_events == False:
    series = clipped_ds.prcp.mean(dim = ['y', 'x'], skipna=True).to_pandas()
else:
    #Aggregate to storm events with logic from WRR paper
    df_precip = clipped_ds.prcp.mean(dim = ['y', 'x'], skipna=True).to_dataframe(name='rainfall (mm/day)').drop(columns='lambert_conformal_conic').tz_localize('UTC').drop(columns='spatial_ref')
    df_precip.reset_index(inplace=True)
    combine_rainfall_events(df_precip)
    series = df_precip['event_rainfall (mm/day)']

num_zeros = (series == 0).sum()

total_days = len(series)
rainfall_frequency = (total_days-num_zeros)/total_days
print(rainfall_frequency)

# Extract non-zero values
non_zero_values = np.sort(series[series != 0].values)
average_non_zero = np.mean(non_zero_values)

# Create histogram
plt.hist(non_zero_values, bins=20, density=True, alpha=0.6, color='g', edgecolor='black',label='Data (2002 - 2022)')
plt.yscale('log') 

guess = np.array([0.9, 1/(average_non_zero*1), 1/(average_non_zero)])
bnds = ((0.1,0.99),(1/(average_non_zero*50), 1/(average_non_zero)), (1/(average_non_zero), 1/(average_non_zero/1.25)))

## best function used for global fitting
minimizer_kwargs = {"method": 'L-BFGS-B',"bounds": bnds,"args": (non_zero_values,)} #'L-BFGS-B'  SLSQP
results = optimize.basinhopping(MLE, guess, minimizer_kwargs=minimizer_kwargs, niter=2000)


# Calculate MLE prediction
w1, k1, k2 = results.x
print(w1)
print(1/k1)
print(1/k2)

pdf_hyperexpon = (1 - w1) * k1 * np.exp(-k1 * non_zero_values) + w1 * k2 * np.exp(-k2 * non_zero_values)
x = np.linspace(0, np.max(non_zero_values), 1000)
plt.plot(x, stats.expon.pdf(x, scale=average_non_zero), color='r', linestyle='--', label='Exponential Distribution')

#Plot the exponential distribution based on the average from the mixed exponential distribution
plt.plot(x, stats.expon.pdf(x, scale=w1*(1/k1)+(1-w1)*(1/k2)), color='k', linestyle='--', label='Exponential Distribution w/ Mixed Exp. Avg.')

# Plot the line for MLE prediction
plt.plot(non_zero_values, pdf_hyperexpon, color='b', linestyle='--', linewidth=2, label='Mixed Exponential Distribution')

plt.xlabel('Rainfall (mm)')
plt.ylabel('PDF of Rainfall, p(R)')
plt.title('Rainy days, total rainfall')
plt.legend()

plt.show()

# Generate Q-Q plot
stats.probplot(non_zero_values, dist=stats.expon, sparams=(0,  average_non_zero,), plot=plt)

plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values')
plt.title('Q-Q Plot of Exponential Distribution')

# Calculate empirical quantiles for the data points
quantiles = [(i+1)/(len(non_zero_values)+1) for i in range(len(non_zero_values))]
theoretical_values = inverse_cdf_mixed_exponential(quantiles, w1, k1, k2)

rmse = np.sqrt(mean_squared_error(non_zero_values, theoretical_values))
print("rmse is " +str(rmse))

# Plot Q-Q plot
plt.figure(figsize=(6, 6))
plt.plot(theoretical_values, non_zero_values , 'o',color='lightgray', alpha=0.8)
plt.plot([np.min(theoretical_values), np.max(theoretical_values)], [np.min(theoretical_values), np.max(theoretical_values)], linestyle='-', color='k')  # True 1:1 line
#plt.plot([np.min(theoretical_quantiles), np.max(theoretical_quantiles)], [np.min(non_zero_values), np.max(non_zero_values)], linestyle='--', color='red')  # 45-degree line
plt.xlabel('theoretical Quantiles, R')
plt.ylabel('data Quantiles, R')
plt.title('Q-Q Plot')
plt.grid(True)
plt.show()

# COMMAND ----------

month = 1
iterations = 2000
aggregate_to_events=True

if aggregate_to_events == False:
    series = clipped_ds.prcp.mean(dim = ['y', 'x'], skipna=True).sel(time=ds['time.month'] == month).to_pandas()
else:
    #Aggregate to storm events with logic from WRR paper
    df_precip = clipped_ds.prcp.mean(dim = ['y', 'x'], skipna=True).sel(time=ds['time.month'] == month).to_dataframe(name='rainfall (mm/day)').drop(columns='lambert_conformal_conic').tz_localize('UTC').drop(columns='spatial_ref')
    df_precip.reset_index(inplace=True)
    combine_rainfall_events(df_precip)
    series = df_precip['event_rainfall (mm/day)']


# Extract non-zero values
non_zero_values = np.sort(series[series != 0].values)
average_non_zero = np.mean(non_zero_values)

guess = np.array([0.9, 1/(average_non_zero*1), 1/(average_non_zero)])
bnds = ((0.3,0.99),(1/(average_non_zero*50), 1/(average_non_zero)), (1/(average_non_zero), 1/(average_non_zero/1.25)))

## best function used for global fitting
minimizer_kwargs = {"method": 'L-BFGS-B',"bounds": bnds,"args": (non_zero_values,)} #'L-BFGS-B'  SLSQP
results = optimize.basinhopping(MLE, guess, minimizer_kwargs=minimizer_kwargs, niter=iterations)

w1, k1, k2 = results.x
print(w1)
print(1/k1)
print(1/k2)

#w1 = array_monthly_weight_hyperexpon[month-1]
#k1 = array_monthly_avg_1_hyperexpon[month-1]plt.plot(x, stats.expon.pdf(x, scale=average_non_zero), color='r', linestyle='--', label='Exponential Distribution')
#k2 = array_monthly_avg_2_hyperexpon[month-1]

quantiles = [(i+1)/(len(non_zero_values)+1) for i in range(len(non_zero_values))]
theoretical_values = inverse_cdf_mixed_exponential(quantiles, w1, k1, k2)

# Plot Q-Q plot
plt.figure(figsize=(6, 6))
plt.plot(theoretical_values, non_zero_values , 'o',color='lightgray', alpha=0.8)
plt.plot([np.min(theoretical_values), np.max(theoretical_values)], [np.min(theoretical_values), np.max(theoretical_values)], linestyle='-', color='k')  # True 1:1 line
#plt.plot([np.min(theoretical_quantiles), np.max(theoretical_quantiles)], [np.min(non_zero_values), np.max(non_zero_values)], linestyle='--', color='red')  # 45-degree line
plt.xlabel('theoretical Quantiles, R')
plt.ylabel('data Quantiles, R')
plt.title('Q-Q Plot')
plt.grid(True)
plt.show()

rmse = np.sqrt(mean_squared_error(non_zero_values, theoretical_values))
print(rmse)

# COMMAND ----------


season = 1
iterations = 2000
aggregate_to_events=True
list_season_months  = [x +((season-1)*3) for x in range(1, 4)]

if aggregate_to_events == False:
    series = clipped_ds.prcp.mean(dim = ['y', 'x'], skipna=True).sel(time=ds['time.month'].isin(list_season_months)).to_pandas()
else:
    #Aggregate to storm events with logic from WRR paper
    df_precip = clipped_ds.prcp.mean(dim = ['y', 'x'], skipna=True).sel(time=ds['time.month'].isin(list_season_months)).to_dataframe(name='rainfall (mm/day)').drop(columns='lambert_conformal_conic').tz_localize('UTC').drop(columns='spatial_ref')
    df_precip.reset_index(inplace=True)
    combine_rainfall_events(df_precip)
    series = df_precip['event_rainfall (mm/day)']

#df_precip_seasonal_data = clipped_ds.prcp.mean(dim = ['y', 'x'], skipna=True).sel(time=ds['time.month'].isin(list_season_months)).to_pandas()
#series = df_precip_seasonal_data

# Extract non-zero values
non_zero_values = np.sort(series[series != 0].values)
average_non_zero = np.mean(non_zero_values)

guess = np.array([0.9, 1/(average_non_zero*1), 1/(average_non_zero)])
bnds = ((0.3,0.99),(1/(average_non_zero*50), 1/(average_non_zero)), (1/(average_non_zero), 1/(average_non_zero/1.25)))

## best function used for global fitting
minimizer_kwargs = {"method": 'L-BFGS-B',"bounds": bnds,"args": (non_zero_values,)} #'L-BFGS-B'  SLSQP
results = optimize.basinhopping(MLE, guess, minimizer_kwargs=minimizer_kwargs, niter=iterations)

w1, k1, k2 = results.x
print(w1)
print(1/k1)
print(1/k2)

quantiles = [(i+1)/(len(non_zero_values)+1) for i in range(len(non_zero_values))]
theoretical_values = inverse_cdf_mixed_exponential(quantiles, w1, k1, k2)

# Plot Q-Q plot
plt.figure(figsize=(6, 6))
plt.plot(theoretical_values, non_zero_values , 'o',color='lightgray', alpha=0.8)
plt.plot([np.min(theoretical_values), np.max(theoretical_values)], [np.min(theoretical_values), np.max(theoretical_values)], linestyle='-', color='k')  # True 1:1 line
#plt.plot([np.min(theoretical_quantiles), np.max(theoretical_quantiles)], [np.min(non_zero_values), np.max(non_zero_values)], linestyle='--', color='red')  # 45-degree line
plt.xlabel('theoretical Quantiles, R')
plt.ylabel('data Quantiles, R')
plt.title('Q-Q Plot')
plt.grid(True)
plt.show()

rmse = np.sqrt(mean_squared_error(non_zero_values, theoretical_values))
print(rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Iterate Through the HUC12s and Calculate the Rain Stats

# COMMAND ----------

# MAGIC %md
# MAGIC #### Concat HUC 8 to HUC 12 for rainfall event

# COMMAND ----------

aggregate_to_event=True

df_list = []
failed_huc12_codes = []

for huc8_in in huc08s:
    print(huc8_in)
    #Select the HUC!2s in the HUC8
    selected_huc12_gdf = gdf_all_huc12[gdf_all_huc12['huc12'].str.startswith(huc8_in)]
    try:
        selected_huc12_calc_gdf = selected_huc12_gdf[~selected_huc12_gdf['huc12'].isin(remove_list)]
    except: 
        selected_huc12_calc_gdf = selected_huc12_gdf
    #Open the associated huc8 daymet data
    ds_open = xr.open_mfdataset(f"{s3_bucket_dir}/{file_id}_huc8_{huc8_in}*.nc")
    spatial_ref_string = ds_open.lambert_conformal_conic.attrs['spatial_ref']
    crs = pyproj.CRS.from_string(spatial_ref_string)
    ds_open  = ds_open.rio.write_crs(crs, inplace=True)
    new_crs = "EPSG:4326"
    ds = ds_open.rio.reproject(new_crs)

    tasks = []
    for index, gdf_row in selected_huc12_calc_gdf.iterrows():
        try:
            clipped_data = ds.rio.clip([gdf_row.geometry])
            tasks.append((gdf_row,clipped_data,None, None,'huc12',aggregate_to_event))
        except Exception as e:
            failed_huc12_codes.append(gdf_row['huc12'])

    #tasks = [(gdf_row, ds.rio.clip([gdf_row.geometry])) for index, gdf_row in #selected_huc12_no_calc_gdf.iterrows()] #.iloc[0:2].iterrows()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        # Submit tasks 
        futures = [executor.submit(rain_stats_per_gdf_geometry, *args) for args in tqdm(tasks)]
    
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

        for future in concurrent.futures.as_completed(futures):
            result, site_id = future.result()
            if result is not None:  # Only append if result is not None
                df_list.append(result)
            else:
                failed_huc12_codes.append(site_id)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Dataframe of Results
# MAGIC Compile Dataframe of Results for the HUC 12 Rainfall Statistics

# COMMAND ----------

# Initialize rainfall_stats_df and hucs_with_no_statistics_stats_df
rainfall_stats_df = None
hucs_with_no_statistics_stats_df = None

if rainfall_stats_df is None:
    rainfall_stats_df = pd.concat(df_list, ignore_index=False)
    remove_list = rainfall_stats_df.index.tolist()
else:
    df_list.append(rainfall_stats_df)
    rainfall_stats_df = pd.concat(df_list, ignore_index=False)

if hucs_with_no_statistics_stats_df is None:
    hucs_with_no_statistics_stats_df = gdf_all_huc12[gdf_all_huc12['huc12'].isin(failed_huc12_codes)]
else:
    failed_huc12_codes.append(hucs_with_no_statistics_stats_df['huc12'].tolist)
    hucs_with_no_statistics_stats_df = gdf_all_huc12[gdf_all_huc12['huc12'].isin(failed_huc12_codes)]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### View the Results

# COMMAND ----------

rainfall_stats_df

# COMMAND ----------

hucs_with_no_statistics_stats_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save out Data as Pickle

# COMMAND ----------


# Specify the new directory path
create_dir_list = [local_dir_path+'/output']
create_directories(create_dir_list)

with open(local_dir_path+f'/output/{project}_rainfall_statistics.pickle', 'wb') as file:
    pickle.dump(rainfall_stats_df, file)
if hucs_with_no_statistics_stats_df.empty is not True:
    with open(local_dir_path+f'/output/{project}_hucs_with_no_statistics.pickle', 'wb') as file:
        pickle.dump(hucs_with_no_statistics_stats_df, file)


if transfer_to_dbfs == True:
    create_dir_list = ['/dbfs/FileStore/'+user_name+'/'+project_name+'/daymet']
    create_directories(create_dir_list) 
    dbutils.fs.cp('file:'+local_dir_path+'/output', 'dbfs:/FileStore/'+user_name+'/'+project_name+'/daymet', recurse=True)
else: 
    dbutils.fs.cp('file:'+local_dir_path+'/output', f"dbfs:/mnt/{aws_bucket_name}/{directory_path_for_daymet}", recurse=True ) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore Data

# COMMAND ----------

rainfall_stats_df.iloc[1]['alpha_1_monthly']

# COMMAND ----------

plt.plot(rainfall_stats_df.iloc[1]['alpha_1_monthly'])

# COMMAND ----------

plt.plot(rainfall_stats_df.iloc[1]['alpha_2_monthly'])

# COMMAND ----------

plt.plot(rainfall_stats_df.iloc[1]['weight_1_monthly'])

# COMMAND ----------

plt.plot(rainfall_stats_df.iloc[1]['lambda_monthly'])

# COMMAND ----------

plt.plot(rainfall_stats_df.iloc[1]['alpha_monthly'])



# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore
# MAGIC Explore the fit of the distribution for different seasons and months

# COMMAND ----------

# MAGIC %md
# MAGIC #Referenes
# MAGIC
# MAGIC https://github.com/ornldaac/daymet-python-opendap-xarray/blob/master/1_daymetv4_discovery_access_subsetting.ipynb
