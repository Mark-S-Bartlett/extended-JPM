# Databricks notebook source
# MAGIC %md
# MAGIC #Flow, Runoff, and Baseflow Data Retreival
# MAGIC - Here, USGS gage stream and river flow data is retreived and processed into runoff and baseflow data.
# MAGIC - For the Florida HUCs (St. Johns River), run with ["FL"] and ['03070205','03080103','03080101','03080102']
# MAGIC - For the Louisiana HUCs, run with ["LA","MS","TX"] and  ['12040201','12010005','12020003','12020006','12020007','03180002','03180003','03180005','03170009','03180004','08090100','08070202','08090301','08070205','08070300','08090203','08070203','08070204','08070100','08080201','08080206','08090202','08070201','08080204','08080103','08080202','08090201','08080101','08080203','08090302','08080102','08080205']
# MAGIC - Also, change the s3 information and project_name accordingly to your data storage setup (or modify to not use AWS s3 storage).

# COMMAND ----------

# MAGIC %md
# MAGIC #### User Input -  Data Selection Criteria

# COMMAND ----------

states = ["FL"]#["LA","MS","TX"] ["FL"]#State postal abbreviation
siteStatus = "all" # siteStatus=[ all | active | inactive ]
siteType = "ST" #https://help.waterdata.usgs.gov/site_tp_cd
outputDataTypeCd = "all" #	outputDataTypeCd=[ all | { [iv | uv | rt], dv, pk, sv, gw, qw, id, aw, ad} ]
parameterCd = "00060" #https://help.waterdata.usgs.gov/codes-and-parameters/parameters
huc08s = ['03070205','03080103','03080101','03080102'] #Florida HUC8s
#huc08s = ['12040201','12010005','12020003','12020006','12020007','03180002','03180003','03180005','03170009','03180004','08090100','08070202','08090301','08070205','08070300','08090203','08070203','08070204','08070100','08080201','08080206','08090202','08070201','08080204','08080103','08080202','08090201','08080101','08080203','08090302','08080102','08080205'] #Louisiana HUC8s
min_no_years = 4.5 #Minimum number of years for a gage to be included in the analysis
baseflow_separation_method = 'Local' #baseflow separation method
max_area = 30000 # sq miles... to eliminate gages on large rivers like the Mississippi
min_area = 2 #sq miles... to eliminate gages on small areas that are not significant TODO add this in
min_year_inactive_site = 1965 #Only want to retreive inactive sites more recent than this year.

#S3 bucket information or DBFS
mount_name  = "jacksonville-data"  #"lwi-transition-zone" 
aws_bucket_name = "jacksonville-data" #"lwi-transition-zone" 

#Project
project_name = 'jacksonville'

#File Storage Paths and File Names
directory_path_for_data="data/hydrology/baseflow"
local_dir_path = '/local_disk0/hydrology'
s3_bucket_dir = f"/dbfs/mnt/{aws_bucket_name}/{directory_path_for_data}"

#Transfer to 
transfer_to_dbfs = False

# COMMAND ----------

# MAGIC %md
# MAGIC #### Library Imports
# MAGIC Also retreives the username for file storage

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC import src.data.noaa_datasets as noaa_datasets
# MAGIC from src.data.utils import *
# MAGIC from src.data.utils_geo import *
# MAGIC from src.data.utils_files import *
# MAGIC from src.data.utils_s3 import *
# MAGIC user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
# MAGIC

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
create_dir_list = [f"/dbfs/mnt/{aws_bucket_name}/"+directory_path_for_data,local_dir_path]
create_directories(create_dir_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ###USGS Data Retrieval
# MAGIC Here we initially, retreive data by a list of states, and download the expanded output. The retreived data is filtered by area and for stream station data. This filtering could be expanded to other data types

# COMMAND ----------

df = process_usgs_data_for_param_list(states, siteStatus, siteType, parameterCd,siteOutput="expanded")
if siteType in ['ST']:
    df_usgs_stream_data_filtered = df[(pd.to_numeric(df['drain_area_va'], errors='coerce').astype(float)> 0)  &  (pd.to_numeric(df['drain_area_va'], errors='coerce').astype(float)< max_area) & (pd.to_numeric(df['drain_area_va'], errors='coerce').astype(float) > min_area)]
else: 
    df_usgs_stream_data_filtered = df
df_usgs_stream_data_filtered

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Retreive Gage Record Counts
# MAGIC For the retrieved data, the active years of the gage are downloaded. In turn, the data is filtered to gages with a minimum number of years of data and inactive sites more recent than a given year.

# COMMAND ----------

df_usgs = process_usgs_data_for_param_list(None, siteStatus, siteType, parameterCd, sites = df_usgs_stream_data_filtered['site_no'].tolist(),outputDataTypeCd="dv,uv")
df_filtered_merged_no_area = filter_and_merge_gage_data(df_usgs,'00060', min_no_years, min_year_inactive_site)
df_filtered_merged = pd.merge(df_filtered_merged_no_area,df_usgs_stream_data_filtered[['site_no','drain_area_va']], on='site_no', how='left')
#pd.set_option('display.max_columns', None)
df_filtered_merged

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Retreive Gages Specific to the Study
# MAGIC For the retrieved and filtered gage data, just the gages in the study area are selected. The resulting gage locations are plotted.

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC gdf = gpd.GeoDataFrame(df_filtered_merged, geometry=gpd.points_from_xy(df_filtered_merged['dec_long_va'].astype(float),df_filtered_merged['dec_lat_va'].astype(float)), crs="EPSG:4326")
# MAGIC nhd_param = {'outFields':'huc8','outSR':4326,'where':list_to_sql(huc08s, "huc8"),'f':'geojson','returnGeometry':'true'}
# MAGIC all_huc8_vector_dict = esri_query(nhd_param, level = 4)
# MAGIC huc8_gdf = create_geodataframe_from_features(all_huc8_vector_dict, crs="EPSG:4326").set_crs("EPSG:4326")
# MAGIC clipped_gages = gpd.clip(gdf, huc8_gdf).reset_index(drop=True)
# MAGIC clipped_gages

# COMMAND ----------

visualize_stations(clipped_gages, huc8_gdf, gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

clipped_gages_huc12_huc10 = clipped_gages[(pd.to_numeric(clipped_gages['drain_area_va'], errors='coerce').astype(float)< 300.0) & (pd.to_numeric(clipped_gages['drain_area_va'], errors='coerce').astype(float) > 2.0)]

# COMMAND ----------

visualize_stations(clipped_gages_huc12_huc10, huc8_gdf, gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

# MAGIC %md
# MAGIC ### NOAA Rain Gage Data
# MAGIC Here, NOAA rain gages are selected by state. This data is retreived to quantify the rainfall/runoff relationship. It may make sense to remove this script and save in a separate notebook.

# COMMAND ----------

# load the stations dataset
stations_dataset = noaa_datasets.Stations()
# then load the inventory dataset
inventory_dataset = noaa_datasets.Inventory(stations_dataset)

# COMMAND ----------

stations_sel_by_state = inventory_dataset.df[ (inventory_dataset.df['State'].isin(states)) &(inventory_dataset.df['Element']=='PRCP') & (inventory_dataset.df['YearCount']>0)]
gdf_stations = gpd.GeoDataFrame(stations_sel_by_state, 
                                     geometry=gpd.points_from_xy(stations_sel_by_state['Longitude'], stations_sel_by_state['Latitude']),
                                     crs='EPSG:4326')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Retreive Gage Specific to the Study
# MAGIC For the retrieved and filtered gage data, just the gages in the study area are selected. The resulting gage locations are plotted.

# COMMAND ----------

huc8_gdf = huc8_gdf.to_crs(gdf_stations.crs)
clipped_noaa_gages = gpd.clip(gdf_stations, huc8_gdf).reset_index(drop=True)
clipped_noaa_gages

# COMMAND ----------

visualize_stations(clipped_noaa_gages.to_crs("EPSG:4267"), huc8_gdf,  gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate Baseflow
# MAGIC For the USGS gages in the study area, the time series of flow data is retreived for each gage. Certain time series related to reservoirs are excluded. The retrieved flow data is separated into baseflow and runoff and this separation is done for flow on a daily time scale and flow a variety of baseflow separation methods.

# COMMAND ----------

clipped_gages_b = clipped_gages.copy()
clipped_gages_b.set_index('site_no', inplace=True)
clipped_gages_b = clipped_gages_b [['dec_lat_va','dec_long_va','drain_area_va']]
clipped_gages_b

# COMMAND ----------

df_gage_time_series = retrieve_gage_time_series(clipped_gages, '00060')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Intepolate Data
# MAGIC Interpolation in log space to account for an exponential recession of flow between missing data points. Do this for up to 5 days.

# COMMAND ----------

log_df = np.log(df_gage_time_series)

# Perform the interpolation
log_interpolated = log_df.interpolate(
    method='linear',
    order=1,
    limit=5,
    limit_direction='forward',
    axis=0,
    limit_area='inside'
)

# Exponentiate the interpolated values to return to the original scale
df_gage_time_series_interpolated = np.exp(log_interpolated)

# COMMAND ----------

import plotly.graph_objs as go

# Select the column you want to compare (for example, the first column)
column_name = df_gage_time_series.columns[22]

# Create a mask where original data is NaN
mask = df_gage_time_series[column_name].isna()

# Apply the mask to the interpolated data
interpolated_only = df_gage_time_series_interpolated[column_name].where(mask)

# Create a trace for the interpolated data (masked)
trace_interpolated = go.Scatter(
    x=df_gage_time_series_interpolated.index,
    y=interpolated_only,
    mode='lines+markers',
    name='Interpolated (Only where NaN)',
    line=dict(color='red', dash='dash')
)

# Create a trace for the original data
trace_original = go.Scatter(
    x=df_gage_time_series.index,
    y=df_gage_time_series[column_name],
    mode='lines+markers',
    name='Original',
    line=dict(color='blue')
)

# Combine the traces
data = [trace_original, trace_interpolated]

# Define the layout
layout = go.Layout(
    title=f"Original vs Interpolated Data (Showing Interpolation Only for NaN Values) for {column_name}",
    xaxis=dict(title='Date'),
    yaxis=dict(title='Flow (Qf)'),
    hovermode='x unified'
)

# Create the figure
fig = go.Figure(data=data, layout=layout)

# Show the interactive plot
fig.show()

# COMMAND ----------

df_KGEs, baseflow_data = process_baseflow(df_gage_time_series, clipped_gages)
df_KGEs_interpolated, baseflow_data_intepolated = process_baseflow(df_gage_time_series_interpolated, clipped_gages)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examinination of the Results

# COMMAND ----------

index = 53 #80
baseflow_separation_method = 'Local' #'Local' Eckhardt
process_and_plot_data(index, baseflow_separation_method, df_gage_time_series_interpolated, clipped_gages, baseflow_data_intepolated, df_KGEs_interpolated)
site_no = list(baseflow_data.keys())[index].split('_')[0]
print(site_no)
visualize_stations(clipped_gages[clipped_gages['site_no'] == site_no], huc8_gdf, gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

df_KGEs.iloc[:,10]
#df_KGEs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export the Data
# MAGIC Data is exported to pickle files. This part needs to be updated to transfer the data to the designated location for the project.

# COMMAND ----------

# Specify the new directory path
directory = local_dir_path
with open(directory+'/baseflow_time_series_dataframes.pickle', 'wb') as file:
    pickle.dump(baseflow_data, file)
with open(directory+'/baseflow_KGEs_dataframes.pickle', 'wb') as file:
    pickle.dump(df_KGEs, file)
with open(directory+'/usgs_gage_time_series_data.pickle', 'wb') as file:
    pickle.dump(df_gage_time_series, file)
with open(directory+'/usgs_gage_data.pickle', 'wb') as file:
    pickle.dump(clipped_gages, file)
with open(directory+'/baseflow_interpolated_time_series_dataframes.pickle', 'wb') as file:
    pickle.dump(baseflow_data_intepolated, file)
with open(directory+'/baseflow_interpolated_KGEs_dataframes.pickle', 'wb') as file:
    pickle.dump(df_KGEs_interpolated, file)
with open(directory+'/usgs_gage_interpolated_time_series_data.pickle', 'wb') as file:
    pickle.dump(df_gage_time_series_interpolated, file)

# COMMAND ----------

create_dir_list = [f"/dbfs/mnt/{aws_bucket_name}/"+directory_path_for_data,local_dir_path]
if transfer_to_dbfs == True:
  dbutils.fs.cp('file:'+local_dir_path, 'dbfs:/FileStore/'+user_name+'/'+project_name+'/hydrology/baseflow', recurse=True)
else: 
  dbutils.fs.cp('file:'+local_dir_path, f"dbfs:/mnt/{aws_bucket_name}/"+directory_path_for_data, recurse=True ) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### References
# MAGIC Here is a list of useful coding references that were used in developing this notebook.
# MAGIC ### NOAA rainfall data
# MAGIC https://docs.opendata.aws/noaa-ghcn-pds/readme.html \
# MAGIC https://registry.opendata.aws/noaa-ghcn/
# MAGIC ### USGS data
# MAGIC https://help.waterdata.usgs.gov/site_tp_cd
# MAGIC
# MAGIC ### Code References
# MAGIC https://github.com/Gare-Uti/Gare-Uti-NOAA_US_Weather-Data_analysis_with_python/ \
# MAGIC https://github.com/analyticsnate/noaa-daily-weather/blob/master/Tutorial%20Demo.ipynb \
# MAGIC https://github.com/themonk911/covid-19-open-data/tree/1e8f7ea1986c0942b21e060cfd80d4881d3e9dac/src/pipelines/weather \
# MAGIC https://github.com/MarcosMJD/ghcn-d
