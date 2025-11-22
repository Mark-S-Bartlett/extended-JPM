# Databricks notebook source
# MAGIC %md
# MAGIC # Continuous Data for Stochastic Simulation
# MAGIC ### Jacksonville

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variables and Inputs

# COMMAND ----------

huc08s = ['03070205','03080103','03080101','03080102']

project='jacksonville'

#S3 bucket information or DBFS
mount_name  = "jacksonville-data"
aws_bucket_name = "jacksonville-data"


transfer_to_dbfs = False

baseflow_separation_method = 'Local'

local_dir_path = "/local_disk0"
directory_path_data="data/hydrology/USGS_gage_parameters"

directory_for_baseflow_data=f"data/hydrology/baseflow"
local_dir_path_baseflow = f'{local_dir_path}/hydrology/baseflow'
s3_bucket_baseflow_dir = f"/dbfs/mnt/{aws_bucket_name}/{directory_for_baseflow_data}"

directory_path_for_daymet="data/hydrology/daymet"
local_dir_path_daymet = f'{local_dir_path}/hydrology/daymet'
s3_bucket_daymet_dir = f"/dbfs/mnt/{aws_bucket_name}/{directory_path_for_daymet}"

directory_path_for_modis="data/hydrology/ET_MODIS"
s3_bucket_modis_dir = f"/dbfs/mnt/{aws_bucket_name}/{directory_path_for_modis}"

transfer_to_dbfs=False

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Libraries

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC import concurrent.futures
# MAGIC import os, shutil, pickle
# MAGIC import sys
# MAGIC import xarray as xr
# MAGIC from pynhd import NLDI
# MAGIC import os, pyproj
# MAGIC import concurrent.futures
# MAGIC from tqdm import tqdm
# MAGIC import scipy.stats as stats
# MAGIC from src.data.utils_s3 import *
# MAGIC from src.data.utils_files import *
# MAGIC from src.data.utils_geo import *
# MAGIC from src.data.utils import *
# MAGIC from src.data.utils_statistics import *
# MAGIC
# MAGIC user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Move Data to Local Drive

# COMMAND ----------

create_dir_list = [local_dir_path_baseflow, local_dir_path_daymet]
create_directories(create_dir_list)
for file_name in os.listdir(s3_bucket_baseflow_dir):
    if os.path.exists(f"/local_disk0/{file_name}")!=1:
        file_path = os.path.join(s3_bucket_baseflow_dir, file_name)
        if os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(local_dir_path_baseflow, file_name))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load Data

# COMMAND ----------

directory = local_dir_path_baseflow
with open(directory+'/baseflow_time_series_dataframes.pickle', 'rb') as file:
    baseflow_data = pickle.load(file)
with open(directory+'/baseflow_KGEs_dataframes.pickle', 'rb') as file:
    df_KGEs = pickle.load(file)
with open(directory+'/usgs_gage_time_series_data.pickle', 'rb') as file:
    df_gage_time_series = pickle.load(file)
with open(directory+'/usgs_gage_data.pickle', 'rb') as file:
    clipped_gages = pickle.load(file)
with open(directory+'/baseflow_interpolated_time_series_dataframes.pickle', 'rb') as file:
    baseflow_data_interpolated = pickle.load(file)
with open(directory+'/baseflow_interpolated_KGEs_dataframes.pickle', 'rb') as file:
    df_KGEs_interpolated = pickle.load(file)
with open(directory+'/usgs_gage_interpolated_time_series_data.pickle', 'rb') as file:
    df_gage_time_series_interpolated = pickle.load(file)
    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download HUC 8 watershed boundaries

# COMMAND ----------

nhd_param = {'outFields':'huc8','outSR':4326,'where':list_to_sql(huc08s, "huc8"),'f':'geojson','returnGeometry':'true'}
all_huc8_vector_dict = esri_query(nhd_param, level = 4)
huc8_gdf = create_geodataframe_from_features(all_huc8_vector_dict, crs="EPSG:4326").set_crs("EPSG:4326")

# COMMAND ----------

visualize_stations(clipped_gages,huc8_gdf, gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Filter USGS Gages to HUC 10 or HUC 12 sizes
# MAGIC The USGS gages for the state area filtered to the smaller HUC 10 and HUC 12 sizes that resonable represent headwater watersheds without incoming flow. This is done to get a clear signal of the rainfall to runoff process.

# COMMAND ----------

clipped_gages_huc12_huc10 = clipped_gages[(pd.to_numeric(clipped_gages['drain_area_va'], errors='coerce').astype(float)< 298.0) & (pd.to_numeric(clipped_gages['drain_area_va'], errors='coerce').astype(float) > 2.0)]

visualize_stations(clipped_gages_huc12_huc10, huc8_gdf, gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select valid sites
# MAGIC Not all the HUC 10 to HUC 12 filtered watersheds have a clear signal because of a control structure or tidal influence. Though in some cases tidal influenced watersheds may still be used if the rainfall to runoff signal is apparent upon visual inspection.

# COMMAND ----------

valid_sites_indices = [0,1,2,3,4,5,7,8,9,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,28,31,32,33,37,38,39,40, 53, 54, 55, 56, 59, 60, 62, 63, 64, 65, 68,69, 70,71, 73,74, 75, 76, 77, 78, 79, 81, 83, 84]
clipped_gages_huc12_huc10_valid = clipped_gages_huc12_huc10.loc[valid_sites_indices]

# COMMAND ----------

index_0 = 11
#index = clipped_gages_huc12_huc10.index[index_0] #80
index = 18
print(index)
baseflow_separation_method = 'Chapman' #'Local' Eckhardt
process_and_plot_data(index, baseflow_separation_method, df_gage_time_series_interpolated, clipped_gages, baseflow_data_interpolated, df_KGEs_interpolated) #baseflow_separation_method
site_no = list(baseflow_data.keys())[index].split('_')[0]
print(site_no)
visualize_stations(clipped_gages[clipped_gages['site_no'] == site_no], huc8_gdf, gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retreive USGS Gage Geometry

# COMMAND ----------

# MAGIC %md
# MAGIC Set the USGS gage geometry to the active dataframe geometry

# COMMAND ----------

clipped_gages_huc12_huc10_valid['geometry2']=None
clipped_gages_huc12_huc10_valid.set_geometry('geometry2', inplace=False).set_crs('EPSG:4326')
clipped_gages_huc12_huc10_valid['geometry2'] = clipped_gages_huc12_huc10_valid['site_no'].apply(get_basin_geometry)

# COMMAND ----------

# MAGIC %md
# MAGIC For USGS gages that don't have a geometry, use the HUC 12 watershed geometry.

# COMMAND ----------

from shapely.geometry import Polygon

# Apply the function to rows where 'geometry2' is None
for idx, row in clipped_gages_huc12_huc10_valid[clipped_gages_huc12_huc10_valid['geometry2'].isnull()].iterrows():
    point_geometry = row['geometry']
    huc12_data = get_huc12_geometry_from_point(point_geometry)
    coords = huc12_data['features'][0]['geometry']['rings'][0]
    
    # Update the 'geometry2' column with the retrieved HUC12 geometry
    clipped_gages_huc12_huc10_valid.at[idx, 'geometry2'] = Polygon(coords)

# COMMAND ----------

visualize_stations(clipped_gages_huc12_huc10,clipped_gages_huc12_huc10_valid.set_geometry('geometry2').set_crs('EPSG:4326'), gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load PET Data

# COMMAND ----------

zarr_store_path=f'/dbfs/mnt/{aws_bucket_name}/data/hydrology/ET_MODIS/MODIS_8_Day_PET_500m_jacksonville_2000-01-01_to_2023-12-31.zarr'
PET_dataset = xr.open_zarr(zarr_store_path, consolidated=True)

# COMMAND ----------

PET_dataset

# COMMAND ----------

PET_dataset.isel(time=2).PET_500m.plot(vmin=0, vmax=6.5, cmap='viridis')

# COMMAND ----------

# MAGIC %md
# MAGIC Clip the PET data to the USGS gage watershed

# COMMAND ----------

site_no = '02246000'
clipped_gages_huc12_huc10[clipped_gages_huc12_huc10['site_no'] == site_no].index
geometry = clipped_gages_huc12_huc10_valid[clipped_gages_huc12_huc10_valid['site_no'] == site_no].geometry2.values[0]
# Convert the dataset to the new CRS
existing_crs = "EPSG:4326"
PET_dataset.rio.write_crs(existing_crs, inplace=True)
clipped_ds = PET_dataset.PET_500m.rio.clip([geometry])
#fill_val = clipped_ds.attrs['_FillValue']
PET_dataset_clipped = clipped_ds #clipped_ds.where(clipped_ds != fill_val)

# COMMAND ----------

# MAGIC %md
# MAGIC Expand the 8 day PET data to all days by defining a full time index and then infilling with linear interpolation

# COMMAND ----------

full_time_index = pd.date_range(start=PET_dataset_clipped.time.min().values, end=PET_dataset_clipped.time.max().values, freq='D')

PET_watershed = PET_dataset_clipped.mean(dim = ['latitude', 'longitude'], skipna=True).reindex(time=full_time_index).chunk({'time': -1}).interpolate_na(dim='time', method='linear')

#.chunk({'time': -1}).interpolate_na(dim='time', method='linear', allow_rechunk=True)
PET_watershed.isel(time=slice(0, None)).plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Clip Rainfall Data

# COMMAND ----------

file_id = project
huc8_in = clipped_gages_huc12_huc10_valid[clipped_gages_huc12_huc10_valid['site_no'] ==site_no].huc_cd.values[0]
print(f"{s3_bucket_daymet_dir}/{file_id}_huc8_{huc8_in}*.nc")
#Open the associated huc8 daymet data
ds_open = xr.open_mfdataset(f"{s3_bucket_daymet_dir}/{file_id}_huc8_{huc8_in}*.nc")
spatial_ref_string = ds_open.lambert_conformal_conic.attrs['spatial_ref']
crs = pyproj.CRS.from_string(spatial_ref_string)
ds_open  = ds_open.rio.write_crs(crs, inplace=True)
new_crs = "EPSG:4326"
ds_rainfall = ds_open.rio.reproject(new_crs)
ds_rainfall_clipped = ds_rainfall.rio.clip([geometry])
ds_rainfall_clipped

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Clip Rainfall Data

# COMMAND ----------

file_id = project
huc8_in = clipped_gages_huc12_huc10_valid[clipped_gages_huc12_huc10_valid['site_no'] ==site_no].huc_cd.values[0]
print(f"{s3_bucket_daymet_dir}/{file_id}_huc8_{huc8_in}*.nc")
#Open the associated huc8 daymet data
ds_open = xr.open_mfdataset(f"{s3_bucket_daymet_dir}/{file_id}_huc8_{huc8_in}*.nc")
spatial_ref_string = ds_open.lambert_conformal_conic.attrs['spatial_ref']
crs = pyproj.CRS.from_string(spatial_ref_string)
ds_open  = ds_open.rio.write_crs(crs, inplace=True)
new_crs = "EPSG:4326"
ds_rainfall = ds_open.rio.reproject(new_crs)
ds_rainfall_clipped = ds_rainfall.rio.clip([geometry])
ds_rainfall_clipped

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate the lambda multiplied by alpha
# MAGIC Calculation assumes ergodicity of the time series for a window of time. The window is a Gaussian filter at 45 days so that 30 days receives most of the weight of +/- 2σ
# MAGIC
# MAGIC Truncating at ±3 standard deviations means the total range covered by the window is 
# MAGIC 6σ (because 3σ to the left + 3σ to the right = 6σ).

# COMMAND ----------

import numpy as np
import xarray as xr

# 1. Create a Gaussian kernel (truncated at ±2 standard deviations)
window_size = 45  # Size of the window
sigma = window_size / 6  # Standard deviation controls the Gaussian spread

# Generate the Gaussian weights
x = np.linspace(-2 * sigma, 2 * sigma, window_size)
gaussian_weights = np.exp(-0.5 * (x / sigma) ** 2)

# Normalize the weights to ensure they sum to 1
gaussian_weights /= gaussian_weights.sum()

# Convert the weights into an xarray DataArray for proper alignment
weights_da = xr.DataArray(gaussian_weights, dims=["window"])

# 2. Apply the rolling window and Gaussian smoothing
rolling = (
    ds_rainfall_clipped['prcp']
    .mean(dim=['x', 'y'], skipna=True)  # Spatial mean
    .rolling(time=window_size, center=True)  # 45-day rolling window, where 45 is the windows size define above
    .construct('window')  # Create 'window' dimension
)


# Custom function to compute weighted sum ignoring NaNs
def weighted_nansum(arr, weights):
    valid = ~np.isnan(arr)  # Mask valid (non-NaN) values
    return np.nansum(arr[valid] * weights[valid]) / np.nansum(weights[valid])

# Use apply_ufunc for the weighted dot product
lambda_alpha_smoothed_rolling = xr.apply_ufunc(
    weighted_nansum, 
    rolling, 
    weights_da, 
    input_core_dims=[["window"], ["window"]],  # Align on 'window' dimension
    vectorize=True,  # Allow vectorized computation across dimensions
)

# 3. Plot a slice of the smoothed result
lambda_alpha_smoothed_rolling.isel(time=slice(0, 300)).plot(label="Smoothed", color='blue')

# COMMAND ----------

# MAGIC %md
# MAGIC Compare to the just a straight averaging without Gaussian weights

# COMMAND ----------

ds_rolling = ds_rainfall_clipped['prcp'].mean(dim=['x', 'y'], skipna=True).rolling(time=30, center=True).mean()
watershed_precip= ds_rainfall_clipped['prcp'].mean(dim=['x', 'y'], skipna=True)
ds_rolling.isel(time=slice(0,300)).plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate the Dryness Index

# COMMAND ----------

ds_DI = PET_watershed/lambda_alpha_smoothed_rolling
n1=1
ds_DI.isel(time=slice(365*n1, 364*(1+n1))).plot(label="Smoothed", color='blue')
#ds_DI.isel(time=slice(365, None)).plot(label="Smoothed", color='blue')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate alpha and lambda on a rolling basis
# MAGIC A Gaussian filter is used as in the previous step

# COMMAND ----------

import numpy as np
import xarray as xr

# 1. Create a Gaussian kernel (truncated at ±2 standard deviations)
window_size = 45  # Size of the window
sigma = window_size / 6  # Standard deviation controls the Gaussian spread

# Generate the Gaussian weights
x = np.linspace(-2 * sigma, 2 * sigma, window_size)
gaussian_weights = np.exp(-0.5 * (x / sigma) ** 2)

# Normalize the weights to ensure they sum to 1
gaussian_weights /= gaussian_weights.sum()

# Convert the weights into an xarray DataArray for proper alignment
weights_da = xr.DataArray(gaussian_weights, dims=["window"])

watershed_precip= ds_rainfall_clipped['prcp'].mean(dim=['x', 'y'], skipna=True)
watershed_precip = watershed_precip.where(watershed_precip != 0, np.nan)

# 2. Apply the rolling window and Gaussian smoothing
rolling = (
    watershed_precip  # Spatial mean
    .rolling(time=window_size, center=True) 
    .construct('window')  # Create 'window' dimension
)

# Custom function to compute weighted sum ignoring NaNs
def weighted_nansum(arr, weights):
    valid = ~np.isnan(arr)  # Mask valid (non-NaN) values
    return np.nansum(arr[valid] * weights[valid]) / np.nansum(weights[valid])


# Use apply_ufunc for the weighted dot product
alpha_smoothed_rolling = xr.apply_ufunc(
    weighted_nansum, 
    rolling, 
    weights_da, 
    input_core_dims=[["window"], ["window"]],  # Align on 'window' dimension
    vectorize=True,  # Allow vectorized computation across dimensions
)

# 3. Plot a slice of the smoothed result
alpha_smoothed_rolling.isel(time=slice(0, None)).plot(label="Smoothed", color='blue')

# COMMAND ----------

# MAGIC %md
# MAGIC Compare to a rolling window with equal weights and not Gaussian weights. Here, 30 days is used for the equal weights because that is roughly equivalent to the 45 days with the Gaussian weights.

# COMMAND ----------

window_size =30

# Create an array of equal weights
equal_weights = np.ones(window_size) / window_size

# Convert to xarray DataArray
weights_da = xr.DataArray(equal_weights, dims=["window"])


watershed_precip= ds_rainfall_clipped['prcp'].mean(dim=['x', 'y'], skipna=True)
watershed_precip = watershed_precip.where(watershed_precip != 0, np.nan)

# 2. Apply the rolling window and Gaussian smoothing
rolling = (
    watershed_precip  # Spatial mean
    .rolling(time=window_size, center=True)  # 15-day rolling window
    .construct('window')  # Create 'window' dimension
)

# Custom function to compute weighted sum ignoring NaNs
def weighted_nansum(arr, weights):
    valid = ~np.isnan(arr)  # Mask valid (non-NaN) values
    return np.nansum(arr[valid] * weights[valid]) / np.nansum(weights[valid])


# Use apply_ufunc for the weighted dot product
smoothed_rolling = xr.apply_ufunc(
    weighted_nansum, 
    rolling, 
    weights_da, 
    input_core_dims=[["window"], ["window"]],  # Align on 'window' dimension
    vectorize=True,  # Allow vectorized computation across dimensions
)

# 3. Plot a slice of the smoothed result
smoothed_rolling.isel(time=slice(0, 360)).plot(label="Smoothed", color='blue')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Calculate lambda

# COMMAND ----------

lambda_smoothed_rolling = lambda_alpha_smoothed_rolling/alpha_smoothed_rolling

# COMMAND ----------

n1=0
lambda_smoothed_rolling .isel(time=slice(365*n1, 364*(1+n1))).plot(label="Smoothed", color='blue')

# COMMAND ----------

# MAGIC %md
# MAGIC Compare to lambda calculated without the Gaussian weights. Again a 30 day window is used because that is roughly equivalent to the 45 day window for the Gaussian filter.

# COMMAND ----------

non_zero_indicator = (watershed_precip > 0).astype(int)

non_zero_count = non_zero_indicator.rolling(time=30, center=True).sum()/30

# 3. Plot the result (spatial average over x and y dimensions)
non_zero_count.isel(time=slice(0,365)).plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Data

# COMMAND ----------

# 1. Align the datasets based on date
smoothed_rolling_aligned, ds_DI_aligned = xr.align(alpha_lambda_smoothed_rolling, ds_DI, join="inner")

# 2. Combine the aligned datasets
ds_combined = xr.Dataset({
    "alpha": alpha_smoothed_rolling,
    "lambda": lambda_smoothed_rolling,
    "PET": PET_watershed,
    "DI": ds_DI_aligned
})
# 3. Convert to a pandas DataFrame
df_combined = ds_combined.to_dataframe().dropna()

# COMMAND ----------

df_combined.drop(columns=["spatial_ref", "lambert_conformal_conic"]).to_csv("time_varying_data.csv")
df_combined

# COMMAND ----------


