# Databricks notebook source
# MAGIC %md
# MAGIC # PET and ET data retreival from the MODIS data product
# MAGIC - Here, PET and ET data is retreived from the MODIS data product
# MAGIC - In the code, for the MODIS yearly data product use 'MOD16A3*', and for the 8-day proudct use the 'MOD16A2*
# MAGIC - For the Florida HUCs (St. Johns River), run with ['03070205','03080103','03080101','03080102']
# MAGIC For the Louisiana HUCs, run with ['12040201','12010005','12020003','12020006','12020007','03180002','03180003','03180005','03170009','03180004','08090100','08070202','08090301','08070205','08070300','08090203','08070203','08070204','08070100','08080201','08080206','08090202','08070201','08080204','08080103','08080202','08090201','08080101','08080203','08090302','08080102','08080205']
# MAGIC - Also, change the s3 information and project_name accordingly to your data storage setup (or modify to not use AWS s3 storage).
# MAGIC - For the 8--day data, make sure the 'use_zarr' flag is set to True. The 8-day dataset is quite large and zarr makes using it manageable.
# MAGIC

# COMMAND ----------

#huc08s = ['12040201','12010005','12020003','12020006','12020007','03180002','03180003','03180005','03170009','03180004','08090100','08070202','08090301','08070205','08070300','08090203','08070203','08070204','08070100','08080201','08080206','08090202','08070201','08080204','08080103','08080202','08090201','08080101','08080203','08090302','08080102','08080205']

#['03070205','03080103','03080101','03080102']
#['02020007', '02040104'] 
huc08s = ['12040201','12010005','12020003','12020006','12020007','03180002','03180003','03180005','03170009','03180004','08090100','08070202','08090301','08070205','08070300','08090203','08070203','08070204','08070100','08080201','08080206','08090202','08070201','08080204','08080103','08080202','08090201','08080101','08080203','08090302','08080102','08080205']

project= "lwi-transition-zone" #'lwi-transition-zone' "jacksonville"

#S3 bucket information or DBFS
mount_name  = "lwi-transition-zone" # "lwi-transition-zone" "jacksonville-data"
aws_bucket_name = "lwi-transition-zone" # "lwi-transition-zone" "jacksonville-data"

local_dir_path = '/local_disk0/MODIS/new'
directory_path_data="data/hydrology/ET_MODIS"
s3_bucket_dir = f"/dbfs/mnt/{aws_bucket_name}/{directory_path_data}"

transfer_to_dbfs=False
use_zarr = True

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Libraries

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC import earthaccess, pickle
# MAGIC from datetime import datetime
# MAGIC from shapely.ops import unary_union
# MAGIC import pandas as pd
# MAGIC from src.data.utils_geo import *
# MAGIC from src.data.utils_s3 import *
# MAGIC from src.data.utils_files import *
# MAGIC from src.data.utils_ET import *
# MAGIC user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Login into Earth Access

# COMMAND ----------

#https://github.com/nsidc/earthaccess/issues/297
earthaccess.login(strategy="interactive", persist=True)

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
create_dir_list = [f"/dbfs/mnt/{aws_bucket_name}/"+directory_path_data,local_dir_path]
create_directories(create_dir_list)

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieve HUC information
# MAGIC Here we retreive the HUC8 and HUC12 watersheds geometrics. The HUC8 geometries are used to define the area needed for the data download, while the downloaed and processed data HUC12s are used to define ET at the HUC12 scale.

# COMMAND ----------

nhd_param = {'outFields':'huc8','outSR':4326,'where':list_to_sql(huc08s, "huc8"),'f':'geojson','returnGeometry':'true'}
all_huc8_vector_dict = esri_query(nhd_param, level = 4)
gdf_huc8 = create_geodataframe_from_features(all_huc8_vector_dict, crs="EPSG:4326").set_crs("EPSG:4326")
min_lon, min_lat, max_lon, max_lat   = gdf_huc8.total_bounds

# Merge all geometries into one
merged_geometry = unary_union(gdf_huc8.geometry)

# Create a new GeoDataFrame with the merged geometry
gdf_merged = gpd.GeoDataFrame(geometry=[merged_geometry])
bounds = gdf_merged.iloc[0].geometry.bounds

print(bounds)

# COMMAND ----------

gdf_merged.plot()

# COMMAND ----------

gdf_all_huc12 = gpd.GeoDataFrame()
for huc8 in huc08s:
    nhd_param = {'outFields':'huc12','outSR':4326,'where':f"huc12 LIKE '{huc8}%'",'f':'geojson','returnGeometry':'true'}
    all_huc12_vector_dict = esri_query(nhd_param, level = 6)
    gdf_huc12 = create_geodataframe_from_features(all_huc12_vector_dict, crs="EPSG:4326").set_crs("EPSG:4326")
    gdf_all_huc12 = gdf = pd.concat([gdf_all_huc12,gdf_huc12])
    gdf_all_huc12.reset_index(drop=True,inplace=True)

gdf_all_huc12  

# COMMAND ----------

# MAGIC %md
# MAGIC # Search NASA for the MODIS ET Data
# MAGIC
# MAGIC Determine the number of tiles needed from the MODIS data. This can be done without using earth access, and while this part is redundant, it doesn't take long to run, so it is here for reference.

# COMMAND ----------

h_list, v_list, single_tile = get_tile_numbers(min_lat, min_lon, max_lat, max_lon)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Search Earth Access
# MAGIC Earthaccess is searched for the MODIS yearly data product 'MOD16A3*' for the yearly data
# MAGIC Earthaccess is searched for the MODIS 8-day data products 'MOD16A2*'.

# COMMAND ----------

Query = earthaccess.collection_query().keyword('MOD16A2*')  #'MOD16A3*' #'MOD16A2*'
print(f'Collections found: {Query.hits()}')

# COMMAND ----------

dir(Query)
collections = Query.fields(['ShortName']).get()
[product['short-name'] for product in [collection.summary() for collection in collections]]

# COMMAND ----------

short_name = 'MOD16A2GF'  #MOD16A3GF' #'MOD16A2GF'
records = Query.get_all()
for record in records:
    if record.summary()['short-name'] == short_name:
        print( record.summary()['concept-id'])
        print( record.summary()['version'])
        print(record['umm']['TemporalExtents'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Retreive the list of Results
# MAGIC Earthaccess resuts are retreived for MOD16A3GF up to the current date. Note that the MOD16A3GF collection starts in the year 2000.

# COMMAND ----------

start_yr='2000'
current_date = datetime.now().strftime('%Y-%m-%d')
results= earthaccess.search_data(
    short_name=short_name,
    cloud_hosted=False,
    bounding_box=bounds,
    temporal=(f"{start_yr}-01-01",current_date),
    count=-1
    )

# COMMAND ----------

df_results = pd.json_normalize(results)
df_results['BeginningDateTime'] = pd.to_datetime(df_results['umm.TemporalExtent.RangeDateTime.BeginningDateTime'])
df_results

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download the data
# MAGIC Based on the search results, the data is downloaded. If certain files fail to download, the process loops until the files download.

# COMMAND ----------

create_dir_list = [local_dir_path]
create_directories(create_dir_list)
selected_results=results
while len(selected_results)>=1:
    earthaccess.download(selected_results, local_dir_path)
    files = os.listdir(local_dir_path)
    file_names = [file.split('.hdf')[0] for file in files if 'hdf' in file and file.endswith('.hdf')]
    indices = df_results[~df_results['meta.native-id'].isin(file_names)].index.values.tolist()
    selected_results = [results[i] for i in indices]


# COMMAND ----------

# MAGIC %md
# MAGIC #### Examine the hdf File Structure
# MAGIC The attributes and datasets contained in the data are displayed for reference.

# COMMAND ----------

# List all files in the directory
files = os.listdir(local_dir_path)
files

hdf_file = SD(local_dir_path + '/' + files[1], SDC.READ)
attributes = hdf_file .attributes(full=1)
print(attributes.keys())
print("Available datasets:")
for dataset in hdf_file .datasets().keys():
    print(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC # Process the ET Data
# MAGIC
# MAGIC Create a dataframe containing metadata of the downloaded data.

# COMMAND ----------

df = process_hdf_files(files, local_dir_path)
df

# COMMAND ----------

# MAGIC %md
# MAGIC Process the files into on xarray
# MAGIC
# MAGIC Selet the data to process which is one of 'ET_500m'
# MAGIC     'LE_500m'
# MAGIC     'PET_500m'
# MAGIC     'PLE_500m'
# MAGIC     'ET_QC_500m'

# COMMAND ----------

modis_data = 'PET_500m' #'ET_500m' LE_500m' 'PET_500m'  'PLE_500m' 'ET_QC_500m'

# COMMAND ----------

min_date  = pd.to_datetime(df['beginning date']).min().strftime('%Y-%m-%d')
max_date =  pd.to_datetime(df['ending date']).max().strftime('%Y-%m-%d')

zarr_store_path=f'/dbfs/mnt/{aws_bucket_name}/data/hydrology/ET_MODIS/MODIS_8_Day_{modis_data}_{project}_{min_date}_to_{max_date}.zarr'

ET_dataset_f = process_hdf_to_dataset_new(df, local_dir_path, modis_data, use_zarr=use_zarr,  zarr_store_path=zarr_store_path)
ET_dataset_f

# COMMAND ----------

# MAGIC %md
# MAGIC #### Examine the overall Data
# MAGIC A few plots provide a quick visual check that the data processes appropriately.

# COMMAND ----------

geometry = merged_geometry
# Convert the dataset to the new CRS
existing_crs = "EPSG:4326"
ET_dataset_f.rio.write_crs(existing_crs, inplace=True)
clipped_ds = ET_dataset_f.PET_500m.rio.clip([geometry])
#fill_val = clipped_ds.attrs['_FillValue']
ET_dataset_clipped = clipped_ds #clipped_ds.where(clipped_ds != fill_val)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 8))
ET_dataset_clipped.isel(time=2).plot(ax=ax, vmin=0, vmax=6.5, cmap='viridis')
gdf_merged.iloc[[0]].plot(ax=ax,  edgecolor='black', linewidth=2, facecolor='none')

# COMMAND ----------


fig, ax = plt.subplots(figsize=(12, 8))
ET_dataset_clipped.isel(time=2).plot(ax=ax, vmin=0, vmax=6.5, cmap='viridis')
gdf_merged.iloc[[0]].plot(ax=ax,  edgecolor='black', linewidth=2, facecolor='none')

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 8))
ET_dataset_clipped.isel(time=0).plot(ax=ax, vmin=0, vmax=6.5, cmap='viridis')
gdf_merged.iloc[[0]].plot(ax=ax,  edgecolor='black', linewidth=2, facecolor='none')

# COMMAND ----------

ET_dataset_clipped.isel(time=slice(0,None)).mean(dim = ['latitude', 'longitude'], skipna=True).plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Data
# MAGIC Here, the overall xarray of the PET data is saved in netCDF form. The data is only saved out here if zarr was not used to incrementally store a large dataset.

# COMMAND ----------

min_date  = pd.to_datetime(df['beginning date']).min().strftime('%Y-%m-%d')
max_date =  pd.to_datetime(df['ending date']).max().strftime('%Y-%m-%d')

create_dir_list = [local_dir_path+'/output/']
create_directories(create_dir_list)

if not use_zarr:
    output_file_path = os.path.join(local_dir_path+'/output/', f"MODIS_{modis_data}_{project}_{min_date}_to_{max_date}.nc")
    ET_dataset_clipped.to_netcdf(output_file_path)


    if transfer_to_dbfs == True:
        create_dir_list = ['/dbfs/FileStore/'+user_name+'/'+project_name+'/ET_MODIS']
        create_directories(create_dir_list) 
        dbutils.fs.cp('file:'+local_dir_path+'/output', 'dbfs:/FileStore/'+user_name+'/'+project_name+'/ET_MODIS', recurse=True)
    else: 
        dbutils.fs.cp('file:'+local_dir_path+'/output', f"dbfs:/mnt/{aws_bucket_name}/{directory_path_data}", recurse=True ) 


# COMMAND ----------

ET_dataset_f.to_zarr(zarr_store_path, mode='a', append_dim='time', consolidated=True, safe_chunks=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate the PET for each HUC12
# MAGIC The script loops through the HUC12 geometries and calculates the average PET for each HUC 12 for each time step of the data. The results are saved in a dataframe.
# MAGIC
# MAGIC The first step shown an example (sanity check) of the PET at the HUC12 level for a selected HUC12.

# COMMAND ----------

ii=10
#gdf_row = gdf_all_huc12[gdf_all_huc12['huc12']=='080801030900'].iloc[[0]]
gdf_row = gdf_all_huc12.iloc[[ii]]

geometry = gdf_row.geometry.iloc[0]

try:
    ET_by_huc = ET_dataset_clipped.rio.clip([geometry])
    ET_by_huc_clipped = ET_by_huc.where(ET_by_huc != fill_val)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ET_by_huc_clipped.isel(time=5).plot(ax=ax,vmin=0, vmax=6.5, cmap='viridis')
    gdf_row.plot(ax=ax,  edgecolor='black', linewidth=2, facecolor='none')
except:
    print('no ET data b/c the watershed is ocean or covers a major river')
    gdf_row.plot(  edgecolor='black', linewidth=2, facecolor='none')

df_huc = df.drop(columns=[df.columns[0],df.columns[5],df.columns[4]]).drop_duplicates().reset_index(drop=True)
df_huc_w_ET_data = process_ET_per_huc(gdf_row.iloc[0], df_huc, ET_dataset_clipped)
df_huc_w_ET_data 

# COMMAND ----------

# MAGIC %md
# MAGIC The script loops through the HUC12s and saves the PET results in a dataframe.

# COMMAND ----------

df_list_ET = []
for ii, gdf_row in gdf_all_huc12.iterrows():
    print(ii)
    df_huc = df.drop(columns=[df.columns[0],df.columns[5],df.columns[4]]).drop_duplicates().reset_index(drop=True)
    df_huc_w_ET_data = process_ET_per_huc(gdf_row, df_huc, ET_dataset_clipped)
    df_list_ET.append(df_huc_w_ET_data)
df_ET_hucs12 = pd.concat(df_list_ET, ignore_index=False)
df_ET_hucs12

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save the Data
# MAGIC The dataframe of PET at the HUC12 scale is saved for future use.

# COMMAND ----------

min_date  = pd.to_datetime(df['beginning date']).min().strftime('%Y-%m-%d')
max_date =  pd.to_datetime(df['ending date']).max().strftime('%Y-%m-%d')

create_dir_list = [local_dir_path+'/output']
create_directories(create_dir_list)

with open(local_dir_path+f'/output/MODIS_{modis_data}_by_huc12_{project}_{min_date}_to_{max_date}.pickle', 'wb') as file:
    pickle.dump(df_ET_hucs12, file)

if transfer_to_dbfs == True:
    create_dir_list = ['/dbfs/FileStore/'+user_name+'/'+project_name+'/ET_MODIS']
    create_directories(create_dir_list) 
    dbutils.fs.cp('file:'+local_dir_path+'/output', 'dbfs:/FileStore/'+user_name+'/'+project_name+'/daymet', recurse=True)
else: 
    dbutils.fs.cp('file:'+local_dir_path+'/output', f"dbfs:/mnt/{aws_bucket_name}/{directory_path_data}", recurse=True ) 

# COMMAND ----------


