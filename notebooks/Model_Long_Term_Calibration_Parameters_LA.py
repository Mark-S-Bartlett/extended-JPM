# Databricks notebook source
# MAGIC %md
# MAGIC # Calculate USGS Gage Watershed Parameters
# MAGIC
# MAGIC
# MAGIC This notebook calculates the data necessary for the watershed model parameterization and calibration.  The notebook outputs four main datasets:
# MAGIC - **Hydrology variables**
# MAGIC - **Hydrology variables defined per year**
# MAGIC - **Rainfall statistics overall, seasonally, and per month**
# MAGIC - **Event rainfall**
# MAGIC - **Event runoff**
# MAGIC
# MAGIC The notebook relies on the following dataframes calculated from the 'Baseflow_Data' notebook
# MAGIC - **usgs_gage_data.pickle**
# MAGIC - **usgs_gage_interpolated_time_series_data.pickle**
# MAGIC - **baseflow_interpolated_time_series_dataframes.pickle**
# MAGIC - **baseflow_interpolated_KGEs_dataframes**
# MAGIC
# MAGIC The notebook also relies on the MODIS ET and PET data from the ET notebook including
# MAGIC
# MAGIC - **/MODIS_ET_500m_{project}_2000-02-18_to_2023-12-31.nc**
# MAGIC - **/MODIS_PET_500m_{project}_2000-02-18_to_2023-12-31.nc**
# MAGIC
# MAGIC where KGEs are the Kling–Gupta efficiencys. Note that the notebook also loads non-interpolated data. The interpolation is just gap filling over a certain number of days based on an exponential recession of flow.
# MAGIC
# MAGIC
# MAGIC ####**Hydrology variables** table
# MAGIC  indexed by USGS site which includes:
# MAGIC
# MAGIC Data about the watershed and observations
# MAGIC - **huc8** 
# MAGIC - **years_data**
# MAGIC - **area (mi²)**  
# MAGIC - **baseflow_sep**  Method used for the baseflow separation
# MAGIC
# MAGIC The water balance paritioniong:
# MAGIC - **<ET/R>**  The ensemble average evapotranspiration over rainfall
# MAGIC - **<Qb/Qf>** The basaeflow over total stream/river flow
# MAGIC - **&lt;Q &gt;(cm)**  Runoff on an event basis
# MAGIC - **&lt;R &gt;(cm)**  Rainfall on an event basis
# MAGIC
# MAGIC The variance in the water balance fluxes:
# MAGIC
# MAGIC - **sigmaR² (cm²)**  Rainfall ensemble variance
# MAGIC - **sigmaQ² (cm²)**  Runoff ensemble variance
# MAGIC
# MAGIC Climate parameters (calculated before rainfall is aggregated into a storm basis):
# MAGIC - **DI_mean**  Overall average Budyko dryness index
# MAGIC - **lambda_all**  Overall average frequency of rainfal
# MAGIC - **alpha_all**   Overall average amount of rainfall per storm event--units mm
# MAGIC - **weight_1_all**  Mixed exponential PDF parameter (fit to all storm events)
# MAGIC - **alpha_1_all**  Mixed exponential PDF parameter (fit to all storm events) --units mm
# MAGIC - **alpha_2_all**  Mixed exponential PDF parameter (fit to all storm events) --units mm
# MAGIC - **alpha_me_all**  Mixed exponential PDF parameter (fit to all storm events) --units mm
# MAGIC
# MAGIC
# MAGIC Data quality metrics
# MAGIC - **percent_rows_dropped**  Rows dropped where runoff was greater than rainfall---a clear hydrologic signal was not discerned.
# MAGIC - **&lt;Q &gt;(cm) diff**  Difference between the average runoff calculated from the water balance (on average) and the average runoff across the aggregated runoff data from individual events.
# MAGIC - **rmse_all**  Root mean squared error between the empirical quantiels of precipitation and the theoretical equantiles from the mixed exponential PDF.
# MAGIC
# MAGIC Runoff recession metrics
# MAGIC - **Rec_a**  The recession parameter from the equation -Log[dQ/dt]= b Log[Q]+a which is -dQ/dt==aQ^b [https://hess.copernicus.org/articles/24/1159/2020/]
# MAGIC - **Rec_b**  The recession parameter from the equation -Log[dQ/dt]= b Log[Q]+a which is -dQ/dt==aQ^b
# MAGIC - **Rec_r²** The r^2 value from the previous equations
# MAGIC - **Rec_event_avg_r²**  The average r^2 from fitting the recession equation to each event where the coefficient `a` is event specific and b is from the previous equations analyzed over all the data.
# MAGIC
# MAGIC
# MAGIC Data about the watershed and observations
# MAGIC - **huc8** 
# MAGIC - **years_data**
# MAGIC - **area (mi²)**  
# MAGIC - **baseflow_sep**  Method used for the baseflow separation
# MAGIC
# MAGIC #### **Hydrology variables defined per year** 
# MAGIC table indexed by HUC 12 watershed includes:
# MAGIC
# MAGIC Data about the watershed and observations
# MAGIC - **year** The year of the data
# MAGIC - **days** The number of days in the year with data
# MAGIC
# MAGIC Hydrologic flux totals over a year
# MAGIC - **flow(mm)** river and stream flow on a unit area basis
# MAGIC - **ET(mm)** Evapotranspirtation on a unit area basis
# MAGIC - **rainfall(mm)** Rainfall on a unit area basis
# MAGIC
# MAGIC Hydrologic flux yearly averages
# MAGIC - **ET/day(mm/day)** Based on the USGS gage water balance
# MAGIC - **MODIS_ET(mm/day)** From MODIS product
# MAGIC - **MODIS_PET(mm/day)** MODIS potential evapotranspiration
# MAGIC - **Budyko_ET(mm/day)** Evapotranspiration based on the Budyko curve
# MAGIC
# MAGIC Climate pareameters:
# MAGIC - **DI**   Budyko dryness index for the year
# MAGIC
# MAGIC The water balance paritioniong for the year:
# MAGIC - **<ET/R>**  The ensemble average evapotranspiration over rainfall
# MAGIC - **<Qb/Qf>** The basaeflow over total stream/river flow
# MAGIC
# MAGIC Note that the alpha and lambda value are found from filtering the event aggregated rainfall and runoff data.
# MAGIC
# MAGIC #### Rainfall statistics overall, seasonally, and per month
# MAGIC These are calculated before the rainfall is aggregated into a storm event basis
# MAGIC
# MAGIC Marked Poisson process of rainfall parameters  - Overall Data series
# MAGIC - **lambda_all** Frequency of rainfall
# MAGIC - **alpha_all** Average rainfall per storm event
# MAGIC - **weight_1_all** Weight parameter of mixed exponential PDF of rainfall
# MAGIC - **alpha_1_all** Average rainfaml parameter of mixed exponential PDF of rainfall
# MAGIC - **alpha_2_all** Average rainfaml parameter of mixed exponential PDF of rainfall
# MAGIC - **alpha_me_all** Average rainfaml calculated from the fited mixed exponential PDF
# MAGIC - **rmse_all** Root mean squared error for mixed exponential PDF fit to data based on empirical quantiles compared to theoretical quantiles
# MAGIC
# MAGIC Marked Poisson process of rainfall parameters  - Per Month
# MAGIC - **lambda_monthly** Frequency of rainfall
# MAGIC - **alpha_monthly** Average rainfall per storm event
# MAGIC - **weight_1_monthly** Weight parameter of mixed exponential PDF of rainfall
# MAGIC - **alpha_1_monthly** Average rainfall parameter of mixed exponential PDF of rainfall  alpha_me_all	
# MAGIC - **alpha_2_monthly** Average rainfall parameter of mixed exponential PDF of rainfall
# MAGIC - **alpha_me_monthly** Average rainfaml calculated from the fited mixed exponential PDF
# MAGIC - **rmse_monthly** Root mean squared error for mixed exponential PDF fit to data based on empirical quantiles compared to theoretical quantiles
# MAGIC
# MAGIC Marked Poisson process of rainfall parameters  - Per Meteorological Season
# MAGIC - **lambda_seasonal** Frequency of rainfall
# MAGIC - **alpha_seasonal** Average rainfall per storm event
# MAGIC - **weight_1_seasonal** Weight parameter of mixed exponential PDF of rainfall
# MAGIC - **alpha_1_seasonal** Average rainfall parameter of mixed exponential PDF of rainfall  alpha_me_all	
# MAGIC - **alpha_2_seasonal** Average rainfall parameter of mixed exponential PDF of rainfall
# MAGIC - **alpha_me_seasonal** Average rainfaml calculated from the fited mixed exponential PDF
# MAGIC - **rmse_seasonal** Root mean squared error for mixed exponential PDF fit to data based on empirical quantiles compared to theoretical quantiles
# MAGIC
# MAGIC #### Event Rainfall and Runoff
# MAGIC
# MAGIC Daily rainfall values were aggregated into storm events by combining two consecutive days of rainfall as one event. A third day's rainfall was included in the total if it was less than 25\% of the combined rainfall from the previous two days. Runoff aggregation began based on whether there was runoff the day before the rainfall event. If no runoff occurred the day before, the aggregation started on the same day as the rainfall event. However, if runoff was present before the rainfall event, the aggregation started the day after. Runoff was accumulated using the baseflow separation time series until zero runoff was detected after at least three consecutive days, or until the next rainfall event occurred. This calculation accounts for the runoff from the previous event based on the calculated recession coefficients. Runoff for each day is the aggreagate daily runoff minus the runoff from the previous event as calculated from the recession equation with coefficients `a` and `b`, where `b` is calculated from an overall examination of the data and `a` is calibrated and fitted to each specific storm event.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variables and Inputs

# COMMAND ----------

project='lwi-transition-zone'

#S3 bucket information or DBFS
mount_name  = "lwi-transition-zone"
aws_bucket_name = "lwi-transition-zone"

huc08s = ['12040201','12010005','12020003','12020006','12020007','03180002','03180003','03180005','03170009','03180004','08090100','08070202','08090301','08070205','08070300','08090203','08070203','08070204','08070100','08080201','08080206','08090202','08070201','08080204','08080103','08080202','08090201','08080101','08080203','08090302','08080102','08080205']

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
# MAGIC #### Import Libraries

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
# MAGIC import rioxarray as rxr
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
# MAGIC #### Move Baseflow Data to the Local Drive

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
# MAGIC ### Load Baseflow Data

# COMMAND ----------

directory = local_dir_path_baseflow

with open(directory+'/usgs_gage_data.pickle', 'rb') as file:
    clipped_gages = pickle.load(file)


#Non-interpolated series
with open(directory+'/baseflow_time_series_dataframes.pickle', 'rb') as file:
    baseflow_data = pickle.load(file)
with open(directory+'/baseflow_KGEs_dataframes.pickle', 'rb') as file:
    df_KGEs = pickle.load(file)
with open(directory+'/usgs_gage_time_series_data.pickle', 'rb') as file:
    df_gage_time_series = pickle.load(file)


#Interpolated series
with open(directory+'/baseflow_interpolated_time_series_dataframes.pickle', 'rb') as file:
    baseflow_data_interpolated = pickle.load(file)
with open(directory+'/baseflow_interpolated_KGEs_dataframes.pickle', 'rb') as file:
    df_KGEs_interpolated = pickle.load(file)
with open(directory+'/usgs_gage_interpolated_time_series_data.pickle', 'rb') as file:
    df_gage_time_series_interpolated = pickle.load(file)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download HUC 8 watershed boundaries
# MAGIC

# COMMAND ----------

nhd_param = {'outFields':'huc8','outSR':4326,'where':list_to_sql(huc08s, "huc8"),'f':'geojson','returnGeometry':'true'}
all_huc8_vector_dict = esri_query(nhd_param, level = 4)
huc8_gdf = create_geodataframe_from_features(all_huc8_vector_dict, crs="EPSG:4326").set_crs("EPSG:4326")

# COMMAND ----------

visualize_stations(clipped_gages,huc8_gdf, gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

clipped_gages_huc12_huc10 = clipped_gages[(pd.to_numeric(clipped_gages['drain_area_va'], errors='coerce').astype(float)< 298.0) & (pd.to_numeric(clipped_gages['drain_area_va'], errors='coerce').astype(float) > 2.0)]

# COMMAND ----------

visualize_stations(clipped_gages_huc12_huc10, huc8_gdf, gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

clipped_gages_huc12_huc10

# COMMAND ----------

# MAGIC %md
# MAGIC I manually went through and picked the sites with a reasonable baseflow and runoff signal visible from the hydrograph separation.

# COMMAND ----------

#Method = Local
valid_sites_indices = [5,9,12,17,18,19,21,23,25,27,28,29,35,36,38,39,40,43,45,46,47,58,59,63,68]
len(valid_sites_indices)
#Not 21
# Indices w/ Negative Flow
#10
# Indices w/ Control Structure: it's a lake and appears to be controlled
#45
# Incides w/limited data
#62 which has a maximum flow and not a daily average... also the same location as 63

# COMMAND ----------

clipped_gages_huc12_huc10[clipped_gages_huc12_huc10['site_no'] == '02487900'].index

# COMMAND ----------

df_KGEs_interpolated

# COMMAND ----------

index_0 = 11
#index = clipped_gages_huc12_huc10.index[index_0] #80
index = 63
print(index)
baseflow_separation_method = 'Chapman' #'Local' Eckhardt
process_and_plot_data(index, baseflow_separation_method, df_gage_time_series_interpolated, clipped_gages, baseflow_data_interpolated, df_KGEs_interpolated) #baseflow_separation_method
site_no = list(baseflow_data.keys())[index].split('_')[0]
print(site_no)
visualize_stations(clipped_gages[clipped_gages['site_no'] == site_no], huc8_gdf, gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gage Basin Geometries
# MAGIC For each USGS gage, the watershed outline (geometry) is retreived. In cases where the USGS basin geometry is not availlable, the USGS HUC 12 geometry is retreived. This HUC 12 is the basin encompassing the USGS gage point.

# COMMAND ----------

clipped_gages_huc12_huc10_valid = clipped_gages_huc12_huc10.loc[valid_sites_indices]
clipped_gages_huc12_huc10_valid['geometry2']=None
clipped_gages_huc12_huc10_valid.set_geometry('geometry2', inplace=False).set_crs('EPSG:4326')
clipped_gages_huc12_huc10_valid['geometry2'] = clipped_gages_huc12_huc10_valid['site_no'].apply(get_basin_geometry)

# Apply the function to rows where 'geometry2' is None
for idx, row in clipped_gages_huc12_huc10_valid[clipped_gages_huc12_huc10_valid['geometry2'].isnull()].iterrows():
    point_geometry = row['geometry']
    huc12_data = get_huc12_geometry_from_point(point_geometry)
    coords = huc12_data['features'][0]['geometry']['rings'][0]
    
    # Update the 'geometry2' column with the retrieved HUC12 geometry
    clipped_gages_huc12_huc10_valid.at[idx, 'geometry2'] = Polygon(coords)

# COMMAND ----------

table_data = clipped_gages_huc12_huc10_valid["site_no"].tolist()
visualize_stations(clipped_gages_huc12_huc10_valid,clipped_gages_huc12_huc10_valid.set_geometry('geometry2').set_crs('EPSG:4326'), gdf_line_color='black', gdf_fill_color='red', gdf2_line_color=  'blue' ,title = 'USGS Gage Sites in the Vicinity of Southern Louisiana',table_data=table_data,table_bbox =[0.01, 0.39, 0.15, 0.5],output_pdf='LouisianaSites.pdf')

# COMMAND ----------

clipped_gages_huc12_huc10_valid.to_file("clipped_gages_huc12_huc10_valid_Louisiana.geojson", driver="GeoJSON")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Precipitation Statistics

# COMMAND ----------

#Get Precip
file_id = project

df_list = []
failed_codes = []

aggregate_to_event=True

for huc8_in in clipped_gages_huc12_huc10_valid.huc_cd.unique():
    print(huc8_in)
    print(f"{s3_bucket_daymet_dir}/{file_id}_huc8_{huc8_in}*.nc")
    #Open the associated huc8 daymet data
    ds_open = xr.open_mfdataset(f"{s3_bucket_daymet_dir}/{file_id}_huc8_{huc8_in}*.nc")
    spatial_ref_string = ds_open.lambert_conformal_conic.attrs['spatial_ref']
    crs = pyproj.CRS.from_string(spatial_ref_string)
    ds_open  = ds_open.rio.write_crs(crs, inplace=True)
    new_crs = "EPSG:4326"
    ds = ds_open.rio.reproject(new_crs)

    tasks = []
    for index, gdf_row in clipped_gages_huc12_huc10_valid[clipped_gages_huc12_huc10_valid.huc_cd==huc8_in].set_geometry('geometry2').set_crs('EPSG:4326').iterrows():
        try:
            clipped_data = ds.rio.clip([gdf_row.geometry2])
            tasks.append((gdf_row, clipped_data, ,None, None, 'site_no',aggregate_to_event))
        except Exception as e:
            failed_codes.append(gdf_row['site_no'])

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
                failed_codes.append(site_id)

# COMMAND ----------

df_usgs_rainfall_stats = pd.concat(df_list, ignore_index=False)
df_usgs_rainfall_stats

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hydrology Parameters and Statistics

# COMMAND ----------

file_id = project
failed_codes = []

#Days of data required in a month, for a month's worth of data to be considered valid
threshold_days = 16

#Declare empty list
df_hydro_variables_list = []
dfs_hydro_year_list = []
df_rain_stats_list = []
df_rainfall_sorted = pd.DataFrame()
df_runoff_sorted = pd.DataFrame()
df_rainfall_sorted = pd.DataFrame()
failed_codes = []


#Get site list of baseflow data
site_list = list(baseflow_data_interpolated.keys())
#Get USGS site IDS
site_nos = [item.split('_')[0] for item in site_list]

# Get Gages with baseflow Data
df_gage_time_series_w_baseflow_interpolated = df_gage_time_series_interpolated[site_list]

#ET and PET data
ds_modis_et = xr.open_dataarray(f"{s3_bucket_modis_dir}/MODIS_ET_500m_{project}_2000-02-18_to_2023-12-31.nc")
ds_modis_pet = xr.open_dataarray(f"{s3_bucket_modis_dir}/MODIS_PET_500m_{project}_2000-02-18_to_2023-12-31.nc")

dfs_rainfal_runoff = []

for huc8_in in clipped_gages_huc12_huc10_valid.huc_cd.unique(): #[4:5]
    print(huc8_in)

    #Open the associated huc8 daymet data
    ds_open = xr.open_mfdataset(f"{s3_bucket_daymet_dir}/{file_id}_huc8_{huc8_in}*.nc")
    spatial_ref_string = ds_open.lambert_conformal_conic.attrs['spatial_ref']
    crs = pyproj.CRS.from_string(spatial_ref_string)
    ds_open  = ds_open.rio.write_crs(crs, inplace=True)
    new_crs = "EPSG:4326"
    ds = ds_open.rio.reproject(new_crs)

    for index, gdf_row in clipped_gages_huc12_huc10_valid[clipped_gages_huc12_huc10_valid.huc_cd==huc8_in].set_geometry('geometry2').set_crs('EPSG:4326').iterrows():
        print(gdf_row.site_no)
        usgs_site_no = gdf_row.site_no
        area = clipped_gages.loc[clipped_gages['site_no']==usgs_site_no]['drain_area_va'].values.astype(float)[0]
        index_site = site_nos.index(usgs_site_no)
        print(index)
        df_gages_filtered = filter_data_by_intersection(df_gage_time_series_w_baseflow_interpolated.iloc[:, index], baseflow_data_interpolated[site_list[index]])

        #Get gage and baseflow data, we exclude the hysep method if the area is too small since the empirical equation only allows for one day of runoff.... and that appears to short.
        #Then select the method with the best KGE score
        methods_to_exclude = ["Local", "Fixed", "Slide","UKIH"]
        if area < 10:
            baseflow_separation_method = (df_KGEs_interpolated.iloc[:,index].drop(methods_to_exclude, axis=0)).idxmax()
        else:
            baseflow_separation_method = (df_KGEs_interpolated.iloc[:, index].drop(["Fixed","Slide", "Local","UKIH"], axis=0)).idxmax() # 'Local' df_KGEs_interpolated.iloc[:, index].idxmax()

        df_baseflow = pd.DataFrame(baseflow_data_interpolated[site_list[index]][baseflow_separation_method])
        df_gages_filtered = pd.DataFrame(df_gages_filtered)

        try:
            ds_clipped = ds.rio.clip([gdf_row.geometry2])

            #Calculate hydro parameters
            df_hydro_one_row, df_grouped_year, df_gage_flow_and_baseflow_non_nan = process_hydro_data(
                df_gages_filtered, df_baseflow, ds_clipped, area, 
                baseflow_separation_method, threshold_days,
                ds_modis_et, ds_modis_pet, gdf_row
                )
            
            dfs_hydro_year_list.append(df_grouped_year)

            df_hydro_one_row['baseflow_sep'] = baseflow_separation_method
            df_hydro_one_row['area(mi^2)'] = area
            df_hydro_one_row['years_data'] = len(df_grouped_year)
            

            #Calculate evente based rainfall and runoff
            df_rainfall_runoff = calculate_rainfall_runoff_events(df_gage_flow_and_baseflow_non_nan, df_grouped_year,baseflow_separation_method)

            dfs_rainfal_runoff.append(df_rainfall_runoff)
            

            #Recession Calcs:
            #only rows where 'dQ/dt' is NaN should be dropped.
            df_clean = df_rainfall_runoff.dropna(subset=['dQ/dt'])
            df_clean = df_clean[df_clean[f'{usgs_site_no}_runoff'] > 0]

            # Apply log transformation to 'dQ/dt' and '02231268_runoff'
            df_clean['log_dQ/dt'] = np.log(df_clean['dQ/dt'])
            df_clean['log_runoff'] = np.log(df_clean[f'{usgs_site_no}_runoff'])

            # Perform linear regression on log-transformed data
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_clean['log_runoff'], df_clean['log_dQ/dt']
                )

            # Calculate coefficients 'a' and 'b'
            b = slope  # Slope is the exponent b
            a = np.exp(intercept)  # Intercept gives log(a), so we exponentiate to get a

            # Display the results
            print(f"a (coefficient): {a}")
            print(f"b (exponent): {b}")
            print(f"Recession fit, R-squared: {r_value**2}")

            df_hydro_one_row['Rec_a'] = a
            df_hydro_one_row['Rec_b'] = b
            df_hydro_one_row['Rec_r^2']=r_value**2

            #Update rainfall/runoff dataframe based on accounting for recession from previous events over consecutive events
            recession_param_b  = b
            df_rainfall_runoff = process_recession_events(df_rainfall_runoff , usgs_site_no, recession_param_b)
            df_rainfall_runoff = update_event_runoff_based_on_recession(df_rainfall_runoff, usgs_site_no)

            df_rainfall_runoff_non_zero = df_rainfall_runoff[df_rainfall_runoff[f'{usgs_site_no}_event_rainfall'] != 0]

            df_hydro_one_row['Rec_event_avg_r^2']=df_rainfall_runoff['r2'].mean()


            #Calculate rainfall statistics
            df_rainfall_stats, _ = rain_stats_per_gdf_geometry(gdf_row, ds_clipped, series_clipped=df_rainfall_runoff[f'{usgs_site_no}_event_rainfall'], site_id = 'site_no')
            df_hydro_one_row= df_hydro_one_row.join(df_rainfall_stats[['lambda_all','alpha_all','weight_1_all','alpha_1_all','alpha_2_all','alpha_me_all','rmse_all']])

            #Calculate true mean of runoff based on the water balance
            df_hydro_one_row['<Q>(cm)'] =  (df_hydro_one_row['alpha_all']/10)*(1-df_hydro_one_row['<ET/R>'])*(1-df_hydro_one_row['<Qb/Qf>'])

            #Adjust runoff time series so the mean equals the mean cacluated from the water balance (which is the true mean)
            #Q_adj1 = df_hydro_one_row['<Q>(cm)'].values[0]-np.mean(df_rainfall_runoff_non_zero[f'{usgs_site_no }_event_runoff']/10)
            #df_rainfall_runoff_non_zero.loc[:, f'{usgs_site_no}_event_runoff']=df_rainfall_runoff_non_zero[f'{usgs_site_no}_event_runoff']+Q_adj1
            #print(f"{usgs_site_no} site 1st Q adjust is  {Q_adj1}")
            
            #Drop rows where rainfall is greater than runoff; therefore the runoff/rainfall event was not sucessfully dissaggregated.
            # Number of rows before filtering
            rows_before = len(df_rainfall_runoff_non_zero)
            #Remove events where a signal was not cpatured because runoff>rainfall
            df_rainfall_runoff_non_zero = df_rainfall_runoff_non_zero[df_rainfall_runoff_non_zero[f'{usgs_site_no}_event_runoff'] <= df_rainfall_runoff_non_zero[f'{usgs_site_no}_event_rainfall']]
            # Number of rows after filtering
            rows_after = len(df_rainfall_runoff_non_zero)
            # Calculate the number of dropped rows
            rows_dropped = rows_before - rows_after

            #Adjust runoff again
            Q_avg_diff = df_hydro_one_row['<Q>(cm)'].values[0]-np.mean(df_rainfall_runoff_non_zero[f'{usgs_site_no }_event_runoff']/10)

            if Q_avg_diff < 0:
                Q_avg_diff_iter =  Q_avg_diff
                tolerance = 1e-6  # Define a small tolerance to stop the loop
                while Q_avg_diff_iter < 0:
                    # Adjust only values greater than the scaled difference
                    mask = df_rainfall_runoff_non_zero[f'{usgs_site_no}_event_runoff'] > abs(Q_avg_diff_iter * 10)
                    if not mask.any():
                        print("Warning: No values to adjust further.")
                        break

                    df_rainfall_runoff_non_zero.loc[mask, f'{usgs_site_no}_event_runoff'] += Q_avg_diff_iter * 10

                    # Recalculate the difference after adjustment
                    Q_avg_diff_iter = df_hydro_one_row['<Q>(cm)'].values[0] - np.mean(
                        df_rainfall_runoff_non_zero[f'{usgs_site_no}_event_runoff'] / 10
                    )

                    # Break the loop if the difference is within the tolerance
                    if abs(Q_avg_diff_iter) < tolerance:
                        break

            # If the difference is non-negative, apply a uniform adjustment
            else:
                df_rainfall_runoff_non_zero[f'{usgs_site_no}_event_runoff'] += Q_avg_diff * 10

            print(f"{usgs_site_no} site Q avg difference is  {Q_avg_diff}")

            #Divide by 10 to save in cm
            rainfall_series = df_rainfall_runoff_non_zero[f'{usgs_site_no }_event_rainfall']/10
            runoff_series = df_rainfall_runoff_non_zero[f'{usgs_site_no }_event_runoff']/10

            #Calculate variance and mean of rainfall_series
            sigmaR_squared = np.var(rainfall_series.values)
            R_mean = np.mean(rainfall_series.values)
            #Calculate variance and mean of runoff_series
            sigmaQ_squared = np.var(runoff_series.values)
            Q_mean = np.mean(runoff_series.values)
           

            df_hydro_one_row['percent_rows_dropped'] = 1.0*rows_dropped/rows_before
            df_hydro_one_row['sigmaQ^2(cm^2)'] = sigmaQ_squared
            df_hydro_one_row['<R>(cm)'] = R_mean
            df_hydro_one_row['sigmaR^2(cm^2)'] = sigmaR_squared
            df_hydro_one_row['<Q>(cm)_diff'] = Q_avg_diff

            # Save, concatenate data to objects
            df_rainfall_sorted = df_rainfall_sorted.join(rainfall_series, how='outer')
            df_runoff_sorted = df_runoff_sorted.join(runoff_series, how='outer')

            df_rain_stats_list.append(df_rainfall_stats)
            df_hydro_variables_list.append(df_hydro_one_row)

        except Exception as e:
            failed_codes.append(gdf_row['site_no'])
            print(e)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Rainfall Statistics

# COMMAND ----------

df_rain_stats = pd.concat(df_rain_stats_list)
df_rain_stats

# COMMAND ----------

# MAGIC %md
# MAGIC #### Overall Hydrology Model Parameters

# COMMAND ----------

df_hydro_variables = pd.concat(df_hydro_variables_list)
df_hydro_variables

# COMMAND ----------

# MAGIC %md
# MAGIC #### Hydrology specific to each year

# COMMAND ----------

dfs_hydro_year_list_w_id = []
for i, df in enumerate(dfs_hydro_year_list):
    df = df.reset_index()
    df[df_hydro_variables.index.name] = df_hydro_variables.index[i]
    df.set_index(df_hydro_variables.index.name, inplace=True)
    dfs_hydro_year_list_w_id.append(df)

df_hydro_variables_w_year = pd.concat(dfs_hydro_year_list_w_id)
df_hydro_variables_w_year

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save the Data

# COMMAND ----------

create_dir_list = [local_dir_path+'/output']
create_directories(create_dir_list)

df_hydro_variables.to_csv(local_dir_path+f'/output/USGS_gage_hydro_variables_{project}.csv', index=True)
df_rain_stats.to_csv(local_dir_path+f'/output/USGS_gage_rain_stats_{project}.csv', index=True)
df_runoff_sorted.to_csv(local_dir_path+f'/output/USGS_gage_event_runoff_{project}.csv', index=True)
df_rainfall_sorted.to_csv(local_dir_path+f'/output/USGS_gage_event_rainfall_{project}.csv', index=True)

# COMMAND ----------

create_dir_list = [local_dir_path+'/output']
create_directories(create_dir_list)

df_hydro_variables.to_csv(local_dir_path+f'/output/USGS_gage_hydro_variables_{project}.csv', index=True)
df_rain_stats.to_csv(local_dir_path+f'/output/USGS_gage_rain_stats_{project}.csv', index=True)
df_runoff_sorted.to_csv(local_dir_path+f'/output/USGS_gage_event_runoff_{project}.csv', index=True)
df_rainfall_sorted.to_csv(local_dir_path+f'/output/USGS_gage_event_rainfall_{project}.csv', index=True)
df_hydro_variables_w_year.to_csv(local_dir_path+f'/output/USGS_gage_hydro_variables_w_year_{project}.csv', index=True)

if transfer_to_dbfs == True:
    create_dir_list = ['/dbfs/FileStore/'+user_name+'/'+project+'/ET_MODIS']
    create_directories(create_dir_list) 
    dbutils.fs.cp('file:'+local_dir_path+'/output', 'dbfs:/FileStore/'+user_name+'/'+project+'/usgs_gage_parameters', recurse=True)
else: 
    dbutils.fs.cp('file:'+local_dir_path+'/output', f"dbfs:/mnt/{aws_bucket_name}/{directory_path_data}", recurse=True ) 

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploration

# COMMAND ----------

df_clean['log_runoff']
df_clean['log_runoff']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plotting the data

# COMMAND ----------

df_clean[f'{usgs_site_no}_runoff'].plot()

# COMMAND ----------

# Verify if there are any NaNs or -inf values after log transformation
print("Any NaNs in 'log_dQ/dt'? ->", df_clean['log_dQ/dt'].isna().any())
print("Any NaNs in 'log_runoff'? ->", df_clean['log_runoff'].isna().any())
print("Any -inf or inf values?")

# COMMAND ----------

index =23

usgs_site_no = dfs_rainfal_runoff[index ].columns[0].split('_')[0]
print(usgs_site_no)
df_input = dfs_rainfal_runoff[index ].copy()

# Drop rows where 'dQ/dt' is NaN or '02231268_runoff' is zero
df_clean = df_input.dropna(subset=['dQ/dt'])
df_clean = df_clean[df_clean[f'{usgs_site_no}_runoff'] > 0 ]

# Apply log transformation to 'dQ/dt' and '02231268_runoff'
df_clean['log_dQ/dt'] = np.log(df_clean['dQ/dt'])
df_clean['log_runoff'] = np.log(df_clean[f'{usgs_site_no}_runoff'])

# Perform linear regression on log-transformed data
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df_clean['log_runoff'], df_clean['log_dQ/dt']
)

# Calculate coefficients 'a' and 'b'
b = slope  # Slope is the exponent b
a = np.exp(intercept)  # Intercept gives log(a), so we exponentiate to get a

# Display the results
print(f"a (coefficient): {a}")
print(f"b (exponent): {b}")
print(f"R-squared: {r_value**2}")

# Plot the log-log regression
plt.figure(figsize=(8, 6))
plt.scatter(df_clean['log_runoff'], df_clean['log_dQ/dt'], c='blue', marker='o', alpha=0.7)
plt.plot(df_clean['log_runoff'], intercept + slope * df_clean['log_runoff'], color='red', linewidth=2)

# Add labels and title
plt.xlabel('log(02231268_runoff)', fontsize=14)
plt.ylabel('log(dQ/dt)', fontsize=14)
plt.title('Log-Log Plot of dQ/dt vs. Runoff', fontsize=16)
plt.grid(True)

# Display the plot
plt.show()

# COMMAND ----------



usgs_gage_no = '07377240'

series = df_rainfall_sorted[f'{usgs_site_no }_event_rainfall'].dropna()
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

guess = np.array([0.999, 1/(average_non_zero*10), 1/(average_non_zero/50)])
bnds = ((0.001,0.999),(1/(average_non_zero*20), 1/(average_non_zero)), (1/(average_non_zero), 1/(average_non_zero/100)))

## best function used for global fitting
minimizer_kwargs = {"method": 'L-BFGS-B',"bounds": bnds,"args": (non_zero_values,)} #'L-BFGS-B'  SLSQP
results = optimize.basinhopping(MLE, guess, minimizer_kwargs=minimizer_kwargs, niter=1000)


# Calculate MLE prediction
w1, k1, k2 = results.x
pdf_hyperexpon = w1 * k1 * np.exp(-k1 * non_zero_values) + (1-w1) * k2 * np.exp(-k2 * non_zero_values)
x = np.linspace(0, np.max(non_zero_values), 1000)
plt.plot(x, stats.expon.pdf(x, scale=average_non_zero), color='r', linestyle='--', label='Exponential Distribution')

#Plot the exponential distribution based on the average from the mixed exponential distribution
plt.plot(x, stats.expon.pdf(x, scale=w1*(1/k1)+(1-w1)*(1/k2)), color='k', linestyle='--', label='Exponential Distribution w/ Mixed Exp. Avg.')

# Plot the line for MLE prediction
plt.plot(non_zero_values, pdf_hyperexpon, color='b', linestyle='--', linewidth=2, label='Mixed Exponential Distribution')

plt.xlabel('Rainfall (cm)')
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


print(w1, 1/k1/10, 1/k2/10)

# COMMAND ----------

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

usgs_gage_no = '07377240'

series = df_runoff_sorted[f'{usgs_site_no }_event_runoff'].dropna()

plt.hist(series, bins=20, density=True, alpha=0.6, color='g', edgecolor='black',label='Data (2002 - 2022)')
plt.yscale('log') 

# Calculate the average of non_zero_non_nan_data_mm.values/10
average_non_zero  =  np.mean(series)

x = np.linspace(0, np.max(series), 1000)
plt.plot(x, stats.expon.pdf(x, scale=average_non_zero), color='r', linestyle='--', label='Exponential Distribution')

plt.xlabel('Rainfall (cm)')
plt.ylabel('PDF of Rainfall, p(R)')
plt.title('Rainy days, total rainfall')
plt.legend()

plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Assuming your DataFrame is named df_grouped_year
plt.figure(figsize=(10, 6))

# Scatter plot of MODIS_ET vs ET/day
plt.scatter(df_grouped_year['MODIS_ET(mm/day)'], df_grouped_year['ET/day(mm/day)'], color='blue', label='MODIS_ET vs ET/day')

# Scatter plot of MODIS_ET vs Budyko_ET
plt.scatter( df_grouped_year['MODIS_ET(mm/day)'],df_grouped_year['Budyko_ET(mm/day)'], color='green', label='MODIS_ET vs Budyko_ET')

# Scatter plot of MODIS_ET vs Budyko_ET
plt.scatter(df_grouped_year['ET/day(mm/day)'], df_grouped_year['Budyko_ET(mm/day)'], color='red', label='ET/day vs Budyko_ET')

# Add 1:1 line
min_val = min(df_grouped_year['MODIS_ET(mm/day)'].min(), df_grouped_year[['ET/day(mm/day)', 'Budyko_ET(mm/day)']].min().min())
max_val = max(df_grouped_year['MODIS_ET(mm/day)'].max(), df_grouped_year[['ET/day(mm/day)', 'Budyko_ET(mm/day)']].max().max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='1:1 Line')

# Add labels and title
plt.xlabel('MODIS_ET (mm/day)')
plt.ylabel('ET/day (mm/day)')
plt.title('MODIS_ET vs ET/day and Budyko_ET')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()


# COMMAND ----------


