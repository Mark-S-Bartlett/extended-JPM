# Databricks notebook source
# MAGIC %md
# MAGIC This series of cells will generate exceedances at gridcell centers

# COMMAND ----------

import pathlib as pl
import os
import os, sys, copy, shutil, subprocess,json
import importlib
import io
import pandas as pd
from pyproj import Transformer
import geopandas as gpd
import h5py
import numpy as np
import itertools
import math
from scipy.stats import norm


hdf_data_dir = pl.Path('/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/no/prcp/ras_output')
out_data_dir = pl.Path('/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/output_data')
out_data_dir2 = "/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/prod_no_aleatory/"

elev_file = hdf_data_dir / 'Amite_20200114.p01.tmp.hdf'
elev_file2 = hdf_data_dir / 'Amite_20200114.p01.tmp.hdf'

test_file = hdf_data_dir/'storm0002.hdf'
#bias is modelled-true
abs_bias=0.4223155
#relative bias is expectation of (modelled-true)/modelled
rel_bias= -0.07017379

recurrence_rate = 1.184297
prob_mass_file = pl.Path('/dbfs/FileStore/LWI_hydro/prob_masses/prob_masses.csv') 

prob_masses = pd.read_csv(prob_mass_file)
prob_masses.set_index('storm_id', inplace = True)

with h5py.File(elev_file2) as f:
    elevs = list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Minimum Elevation'])
    xy = list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Center Coordinate'])

geo_frame = pd.DataFrame(xy, columns = ['x','y'])
geo_frame['ras_id'] = range(1,len(elevs)+1)
geo_frame['elevs'] = elevs

ras_ids = range(1,len(elevs)+1)
storm_data_list = []
for i in prob_masses.index:
    print(f'loading storm {i}')
    this_file = hdf_data_dir/f'storm{str(i).zfill(4)}.hdf'
    with h5py.File(this_file) as f:
        new_df = pd.DataFrame({'WSE': list(f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['AmiteMaurepas']['Max Values']['Water Surface'])})
        new_df['ras_id'] = ras_ids
        new_df['prob'] = prob_masses.loc[i, 'prob']
        new_df['elevs'] = elevs
        new_df['storm_id']= i
        new_df['depth_raw'] = round((new_df['WSE'] - new_df['elevs']),2)
        new_df.drop(['WSE','elevs'], inplace = True, axis = 1)
        storm_data_list.append(new_df)

storm_data_frame = pd.concat(storm_data_list)


#the assistant tells me this will sum the probability by ras_id and depth, check to make sure this actually does that

#grouped_table = storm_data_frame.groupby(['ras_id', 'depth_raw']).agg(prob=('prob', 'sum')).reset_index() 

#Approach change MSB---b/c I wan to include the storm id in the table moving forward
grouped_table = storm_data_frame.groupby(['ras_id']).apply(
    lambda g: g.sort_values('depth_raw', ascending=True, na_position='first')
).reset_index(drop=True)


#for max vals: list(f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['AmiteMaurepas']['Max Values']['Water Surface']
#for coordinates: list(f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['AmiteMaurepas']['Max Values']['Cells Center Coordinate'])
'''
This was a quick check to ensure that the elevations were consistent between two files with geometry
with h5py.File(elev_file) as f:
    print(f.keys())
    #print(list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Center Coordinate']))
    print(list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Minimum Elevation'])[1:20])

with h5py.File(elev_file2) as f:
    print(f.keys())
    #print(list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Center Coordinate']))
    print(list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Minimum Elevation'])[1:20])


'''


# COMMAND ----------

#def calc_annualized_cdf_val(in_df, recurrence_rate, cum_prob_col):
#    out_col = (
#        (np.exp(-recurrence_rate) * 
#        ((in_df[cum_prob_col] * recurrence_rate)**0 / math.factorial(0) +
#        (in_df[cum_prob_col] * recurrence_rate)**1 / math.factorial(1) +
#        (in_df[cum_prob_col] * recurrence_rate)**2 / math.factorial(2) +
#        (in_df[cum_prob_col] * recurrence_rate)**3 / math.factorial(3) +
#        (in_df[cum_prob_col] * recurrence_rate)**4 / math.factorial(4) +
#        (in_df[cum_prob_col] * recurrence_rate)**5 / math.factorial(5) +
#        (in_df[cum_prob_col] * recurrence_rate)**6 / math.factorial(6) +
#        (in_df[cum_prob_col] * recurrence_rate)**7 / math.factorial(7) +
#        (in_df[cum_prob_col] * recurrence_rate)**8 / math.factorial(8) +
#        (in_df[cum_prob_col] * recurrence_rate)**9 / math.factorial(9) +
#        (in_df[cum_prob_col] * recurrence_rate)**10 / math.factorial(10)))
#        )
#    return out_col   CHANGED BY MSB as the series is convergent

def calc_annualized_cdf_val(in_df, recurrence_rate, cum_prob_col):
    return np.exp(-recurrence_rate * (1 - in_df[cum_prob_col]))

# COMMAND ----------

'''aleatory_abs = 0.2
aleatory_rel = 0.15
n_uncert_bins = 10

grouped_table['total_aleatory'] = (aleatory_abs**2 + (aleatory_rel* grouped_table['depth_raw'])**2)**0.5
#check this
uncert_table = pd.DataFrame(list(itertools.product(range(1,n_uncert_bins+1), ras_ids)), columns = ['uncert_bin', 'ras_id'])

storm_data_uncert = pd.merge(grouped_table, uncert_table, on = 'ras_id', how = 'left')
storm_data_uncert['z'] = norm.ppf((storm_data_uncert['uncert_bin']-1)/n_uncert_bins + 1/(2*n_uncert_bins))
storm_data_uncert['depth_raw'] = round(np.maximum(storm_data_uncert['depth_raw'] + storm_data_uncert['total_aleatory']*storm_data_uncert['z'],0),2)
storm_data_uncert['prob'] = storm_data_uncert['prob']/n_uncert_bins

storm_data_uncert = storm_data_uncert.groupby(['ras_id', 'depth_raw']).agg(prob=('prob', 'sum')).reset_index()
'''

storm_data_uncert = grouped_table
#MSB next line is redundant
#storm_data_uncert.sort_values(['ras_id', 'depth_raw'],inplace=True)

storm_data_uncert['cum_prob'] = storm_data_uncert.groupby('ras_id')['prob'].cumsum()

storm_data_uncert['annualized_cdf_val'] = calc_annualized_cdf_val(storm_data_uncert, recurrence_rate, 'cum_prob')

storm_data_uncert['return_period'] = 1/(1-storm_data_uncert['annualized_cdf_val'])




# COMMAND ----------

storm_data_uncert

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Filter the data for storm_id 498 and ras_id 77456
filtered_data = storm_data_uncert[ storm_data_uncert['ras_id'] == 77456]

# Plot depth_raw for the selected storm_id and ras_id
plt.figure(figsize=(8, 6))
plt.plot(filtered_data['depth_raw'], marker='o', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Depth (ft)')
plt.title('Depth at ras_id 77456 for storm_id 498')
plt.grid(True)
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Filter the data for storm_id 498 and ras_id 77456
filtered_data = storm_data_uncert[(storm_data_uncert['storm_id'] == 498) & (storm_data_uncert['ras_id'] == 77456)]

# Plot depth_raw for the selected storm_id and ras_id
plt.figure(figsize=(8, 6))
plt.plot(filtered_data['depth_raw'], marker='o', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Depth (ft)')
plt.title('Depth at ras_id 77456 for storm_id 498')
plt.grid(True)
plt.show()

# COMMAND ----------

def interp_group(g):
    x = g["annualized_cdf_val"].values
    y = g["depth_adj"].values

    # mask values outside range -> np.nan
    mask = (rp_cdf_vals >= x.min()) & (rp_cdf_vals <= x.max())
    depths = np.full_like(rp_cdf_vals, np.nan, dtype=float)

    depths[mask] = np.interp(rp_cdf_vals[mask], x, y)

    return pd.DataFrame({
        "ras_id": g["ras_id"].iloc[0],
        "return_period": return_periods,
        "depth_adj": depths
    })

# COMMAND ----------

storm_data_uncert['bias_abs'] = abs_bias
storm_data_uncert['bias_rel'] = rel_bias*storm_data_uncert['depth_raw']
storm_data_uncert['bias_bounded'] = np.minimum(np.abs(storm_data_uncert['bias_abs']),np.abs(storm_data_uncert['bias_rel']))*np.sign(storm_data_uncert['bias_abs'])
storm_data_uncert['depth_adj'] = np.maximum(storm_data_uncert['depth_raw'] - storm_data_uncert['bias_bounded'],0)

# COMMAND ----------

storm_data_uncert

# COMMAND ----------

storm_data_uncert.to_parquet(f'{out_data_dir2}/wse_dist_tc_no_uncert_surge_only_vectorized_bartlett_edits.parquet', engine="pyarrow", compression="snappy")

# COMMAND ----------

storm_data_uncert['bias_abs'] = abs_bias
storm_data_uncert['bias_rel'] = rel_bias*storm_data_uncert['depth_raw']
storm_data_uncert['bias_bounded'] = np.minimum(np.abs(storm_data_uncert['bias_abs']),np.abs(storm_data_uncert['bias_rel']))*np.sign(storm_data_uncert['bias_abs'])
storm_data_uncert['depth_adj'] = np.maximum(storm_data_uncert['depth_raw'] - storm_data_uncert['bias_bounded'],0)


return_periods = np.array([2, 5, 8, 10, 13, 15, 20, 25, 33, 42, 50, 75, 100, 125, 150,
                  200, 250, 300, 350, 400, 500, 1000, 2000])

aeps = 1/return_periods

rp_cdf_vals = 1-aeps 

rp_dataframe_list = []
#for rp in return_periods:
#    this_aep = 1/rp
#    this_cdf_val = 1-this_aep
#    this_rp_data = storm_data_uncert[storm_data_uncert['annualized_cdf_val'] <= this_cdf_val].groupby('ras_id')['depth_adj'].max().reset_index() #was depth_adj
#    this_rp_data['return_period'] = rp
#    rp_dataframe_list.append(this_rp_data)

return_period_table = (
    storm_data_uncert.groupby("ras_id", group_keys=False)
    .apply(interp_group, include_groups=False)[["ras_id","annualized_cdf_val", "depth_adj"]]
    .reset_index(drop=True)
    )

#return_period_table = pd.concat(rp_dataframe_list, axis = 0)
#return_period_table.sort_values(['ras_id', 'return_period'], inplace = True)

out_table = pd.merge(return_period_table, geo_frame, how = 'left', on = 'ras_id')
out_table.to_csv(out_data_dir/ 'exceedance_table_tropical_surge_only_no_aleatory.csv')
'''this is easy to do in a vectorized fashion in R, since R has the approx() function which supports piecewise linear
interpolation, returning a list of depths for an input list of return periods / cdf vals, which can be then applied
over ras ids with data.tables. In Python, we can vectorize over ras ids with Pandas but the closest relative to approx in python is deprecated/legacy and isn't clear whether it actually supports piecewise constant in the way we want here.
'''


# COMMAND ----------

# MAGIC %md
# MAGIC here we'll get that into polygon shapefiles

# COMMAND ----------

import geopandas as gpd
import pathlib as pl
import pandas as pd

local_data = pl.Path('/home/ngeldner/app')

if not local_data.exists():
    local_data.mkdir(parents = True)

in_geom_dir = pl.Path('/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/inputs')
out_data_dir = pl.Path('/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/output_data')

#this is admittedly gross, but for the sake of consistency with previous ad-hoc visualization efforts, what we need to do here is
#pull in a polygon shapefile, then do a spatial join with a point feature class with ras ids. That way we'll have
#polygons keyed to ras id, which we can then merge with the return period table, save out shapefiles for each return period,
#then convert the polygon data to rasters
in_geom = gpd.read_file(in_geom_dir/'ras_grid_pilot_3.shp')

in_table = pd.read_csv(out_data_dir/ 'exceedance_table_tropical_surge_only_no_aleatory.csv')
#in_table['WSE'] = in_table['depth_adj'] + in_table['elevs'] #was depth_adj

#Changed by MSB to make WSE nan if the depth is zero instead of just mirroring the elevation
in_table['WSE'] = np.where(in_table['depth_adj'] <= 0,
                           np.nan,
                           in_table['depth_adj'] + in_table['elevs'])

in_table = in_table.dropna(subset=['WSE'])

ex_table = in_table[in_table['return_period']==100]

ex_data = ex_table[['ras_id', 'x', 'y']]


base_pt_gdf = gpd.GeoDataFrame(ex_data, geometry=gpd.points_from_xy(ex_data['x'], ex_data['y']))

wkt_string = 'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-96.0],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["latitude_of_origin",23.0],UNIT["Foot_US",0.3048006096012192]]'

base_pt_gdf.set_crs(wkt_string, inplace=True)


point_poly_gdf = gpd.sjoin(base_pt_gdf,in_geom, how='right', predicate='within')
point_poly_gdf.dropna(inplace=True, subset=['ras_id'])
point_poly_gdf = point_poly_gdf[['ras_id', 'geometry']]
point_poly_gdf.set_index('ras_id', inplace=True)
point_poly_gdf.to_crs(epsg=6479, inplace=True)

return_periods = list(set(in_table['return_period']))

for this_return_period in return_periods:
    print(f'saving {this_return_period} year WSE')
    this_data = in_table[in_table['return_period']==this_return_period][['ras_id', 'WSE']]
    this_data.set_index('ras_id', inplace = True)

    out_gdf = point_poly_gdf.join(this_data, how = 'left')
    out_gdf.to_file(local_data / f'{str(this_return_period).zfill(4)}_year_WSE.shp')



# COMMAND ----------

this_data = in_table[in_table['return_period']==500][['ras_id', 'WSE']]
this_data.set_index('ras_id', inplace = True)

out_gdf = point_poly_gdf.join(this_data, how = 'left')

# COMMAND ----------

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Merge 'elevs' from geo_frame into storm_data_uncert by 'ras_id'
storm_data_uncert2 = pd.merge(storm_data_uncert, geo_frame[['ras_id', 'elevs']], on='ras_id', how='left')

# Compute centroids for each polygon to get x, y coordinates
out_gdf['centroid'] = out_gdf.geometry.centroid
out_gdf['centroid_x'] = out_gdf['centroid'].apply(lambda p: p.x)
out_gdf['centroid_y'] = out_gdf['centroid'].apply(lambda p: p.y)

# Find the ras_id closest to the median centroid x and y in the domain
mid_x = out_gdf['centroid_x'].median()-20000
mid_y = out_gdf['centroid_y'].median()-25000
out_gdf['dist_to_mid'] = ((out_gdf['centroid_x'] - mid_x)**2 + (out_gdf['centroid_y'] - mid_y)**2)**0.5
mid_ras_id = out_gdf['dist_to_mid'].idxmin()
selected_ras_id = mid_ras_id

# Select a second point 5 km north and 5 km west of the first point
target_x = out_gdf.loc[selected_ras_id, 'centroid_x'] - 5000
target_y = out_gdf.loc[selected_ras_id, 'centroid_y'] + 5000
out_gdf['dist_to_offset'] = ((out_gdf['centroid_x'] - target_x)**2 + (out_gdf['centroid_y'] - target_y)**2)**0.5
offset_ras_id = out_gdf['dist_to_offset'].idxmin()
selected_ras_id_2 = offset_ras_id

# Get the CDF data for both ras_ids from storm_data_uncert and calculate WSE = depth_raw + elevs
cdf_data_1 = storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id].sort_values('depth_adj').copy()
cdf_data_1['return_period'] = 1 / (1 - cdf_data_1['annualized_cdf_val'])
cdf_data_1['WSE'] = cdf_data_1['depth_adj'] + cdf_data_1['elevs']

in_table_1= in_table[in_table['ras_id'] == selected_ras_id]

cdf_data_2 = storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id_2].sort_values('depth_adj').copy()
cdf_data_2['return_period'] = 1 / (1 - cdf_data_2['annualized_cdf_val'])
cdf_data_2['WSE'] = cdf_data_2['depth_adj'] + cdf_data_2['elevs']

in_table_2= in_table[in_table['ras_id'] == selected_ras_id_2]

plt.figure(figsize=(8,6))
plt.plot(cdf_data_1['return_period'], cdf_data_1['WSE'], marker='o', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(cdf_data_2['return_period'], cdf_data_2['WSE'], marker='s', label=f'Point 2 (ras_id={selected_ras_id_2})')
plt.plot(in_table_1['return_period'], in_table_1['WSE'], marker='o', linestyle='--', color='red', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(in_table_2['return_period'], in_table_2['WSE'], marker='s', linestyle='--', color='blue', label=f'Point 2 (ras_id={selected_ras_id_2})')
plt.xscale('log')
plt.xlim(1, 5000)
plt.xlabel('Return Period (years)')
plt.ylabel('WSE (ft)')
plt.title('WSE vs Return Period at Two Points')
plt.grid(True)

# Interpolate WSE for 400 and 500 year events for both points
highlight_rps = [400, 500]
interp_wse_1 = np.interp(highlight_rps, cdf_data_1['return_period'], cdf_data_1['WSE'])
interp_wse_2 = np.interp(highlight_rps, cdf_data_2['return_period'], cdf_data_2['WSE'])
plt.scatter(highlight_rps, interp_wse_1, color='red', marker='*', s=150, label='400 & 500 Year (Point 1)')
plt.scatter(highlight_rps, interp_wse_2, color='blue', marker='*', s=150, label='400 & 500 Year (Point 2)')

plt.legend()
plt.show()

# Plot spatial location of the selected points
ax = out_gdf.plot(column='WSE', legend=True, cmap='viridis', figsize=(10,8))
out_gdf.loc[[selected_ras_id]].plot(ax=ax, color='red', markersize=100, edgecolor='black', label='Point 1')
out_gdf.loc[[selected_ras_id_2]].plot(ax=ax, color='blue', markersize=100, edgecolor='black', label='Point 2')
plt.title('Water Surface Elevation (WSE) with Selected Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(['Point 1', 'Point 2'])
plt.show()

# COMMAND ----------

selected_ras_id = 59698
in_table[in_table['ras_id'] == selected_ras_id]

# COMMAND ----------

selected_ras_id = 59698
display(storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id].sort_values('depth_raw').tail(20))

# COMMAND ----------

selected_ras_id = 58077 
in_table[in_table['ras_id'] == selected_ras_id]

# COMMAND ----------

selected_ras_id = 58077 
display(storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id].sort_values('depth_raw').tail(20))

# COMMAND ----------


import matplotlib.pyplot as plt

selected_ras_id = 50802 
selected_ras_id_2 = 48730

# Merge 'elevs' from geo_frame into storm_data_uncert by 'ras_id'
storm_data_uncert2 = pd.merge(storm_data_uncert, geo_frame[['ras_id', 'elevs']], on='ras_id', how='left')

# Get the CDF data for both ras_ids from storm_data_uncert and calculate WSE = depth_raw + elevs
cdf_data_1 = storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id].sort_values('depth_raw').copy()
cdf_data_1['return_period'] = 1 / (1 - cdf_data_1['cum_prob'])
cdf_data_1['WSE'] = cdf_data_1['depth_raw'] + cdf_data_1['elevs']

cdf_data_2 = storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id_2].sort_values('depth_raw').copy()
cdf_data_2['return_period'] = 1 / (1 - cdf_data_2['cum_prob'])
cdf_data_2['WSE'] = cdf_data_2['depth_raw'] + cdf_data_2['elevs']

plt.figure(figsize=(8,6))
plt.plot(cdf_data_1['return_period'], cdf_data_1['WSE'], marker='o', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(cdf_data_2['return_period'], cdf_data_2['WSE'], marker='s', label=f'Point 2 (ras_id={selected_ras_id_2})')
plt.xscale('log')
plt.xlim(1, 5000)
plt.xlabel('Return Period (years)')
plt.ylabel('WSE (ft)')
plt.title('WSE vs Return Period at Two Points')
plt.grid(True)

# COMMAND ----------

import os, sys, copy, shutil, subprocess,json
import numpy as np
# create GRASS GIS runtime environment
subprocess.check_output(["grass", "--config", "path"]).strip()
#Define GIS based directory and database (data) storage location

gisbase = subprocess.check_output(["grass", "--config", "path"]).strip().decode()
gisdb = "/home/jovyan/grassdata"
raster_dir_local = pl.Path("/home/jovyan/raster_outs")
if not raster_dir_local.exists():
    raster_dir_local.mkdir(parents = True)

local_data = pl.Path('/home/ngeldner/app')

if not local_data.exists():
    local_data.mkdir(parents = True)

"""Grass GIS settings and housekeeping
"""

# simply overwrite existing maps like we overwrite Python variable values
os.environ['GRASS_OVERWRITE'] = '1'
# enable map rendering to in Jupyter Notebook
os.environ['GRASS_FONT'] = 'sans'
# set display modules to render into a file (named map.png by default)
os.environ['GRASS_RENDER_IMMEDIATE'] = 'cairo'
os.environ['GRASS_RENDER_FILE_READ'] = 'TRUE'
os.environ['GRASS_LEGEND_FILE'] = 'legend.txt'
os.environ['GISBASE'] = gisbase

#Append python scripts to system path
grass_pydir = os.path.join(gisbase, "etc", "python")
sys.path.append(grass_pydir)

# do GRASS GIS imports
from grass.script import*
import grass.script as gs
import grass.jupyter as gj
import grass.script.setup as gsetup
import grass.script.array as garray
from grass.pygrass.modules import Module, ParallelModuleQueue
from grass_session import Session
from grass.script import core as gcore
#Python Bindings
from grass.script import*
import grass.script as gs
import grass.script.setup as gsetup
import grass.script.array as garray
from grass.pygrass.modules import Module, ParallelModuleQueue
#from grass.pygrass.modules.shortcuts import raster as r
#from grass.pygrass.modules.shortcuts import vector as v
from grass_session import Session
from grass.script import core as gcore

# We want functions to raise exceptions and see standard output of the modules in the notebook.
gs.set_raise_on_error(True)
gs.set_capture_stderr(True)


def clean_tmp_folder(Location: str, Grass_Data_Location: str)-> str:
    '''This function deletes the temporary files from the lat Grass GIS session.
    '''
    folder = pl.Path(Grass_Data_Location)/Location/'Permanent/.tmp/unknown'
    if (os.path.exists(folder)):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        return print('Deleted temporary files from last session')

grass_gis_projection = '6479'

#here we initialize grass gis
project = 'lwi_amite_results'
Location = project+'_'+grass_gis_projection
GrassGIS_Location = os.path.join(gisdb, Location)
if os.path.exists(GrassGIS_Location):
    print("Database Location Exists")
else:
    with Session(gisdb=gisdb, location=Location, create_opts="EPSG:"+grass_gis_projection):
        # run something in PERMANENT mapset:
        print(gcore.parse_command("g.gisenv", flags="s"))
        
clean_tmp_folder(Location, gisdb)
#Start new session
#rcfile = gsetup.init(gisbase,gisdb, GrassGIS_Location, 'PERMANENT')
session = gj.init(gisdb, Location, 'PERMANENT')
print (gs.gisenv())

gs.run_command('g.region', n = 809400, s = 541000, w = 3312000, e = 3674900, res = 100, overwrite = True)

return_periods = np.array([2, 5, 8, 10, 13, 15, 20, 25, 33, 42, 50, 75, 100, 125, 150,
                  200, 250, 300, 350, 400, 500, 1000, 2000])

for this_return_period in return_periods:
  print('importing shapefile')
  gs.run_command('v.import', input=local_data/ f'{str(this_return_period).zfill(4)}_year_WSE.shp',
                       output = f'surge_{str(this_return_period).zfill(4)}_year_WSE',
                          overwrite = True)
  print("converting to raster")
  gs.run_command('v.to.rast', input = f'surge_{str(this_return_period).zfill(4)}_year_WSE',
                       output = f'surge_{str(this_return_period).zfill(4)}_year_WSE_rast',
                       use = 'attr', attribute_column = 'WSE', overwrite = True)
  gs.run_command('r.out.gdal', input = f'surge_{str(this_return_period).zfill(4)}_year_WSE_rast',
                          output = raster_dir_local/ (f'EJPMOS_tropical_surge_only_{str(this_return_period).zfill(4)}_YR_WSE_no_aleatory_bartlett_edits.tif'), format = "GTiff",
                          overwrite = True)
  print(f'finished {this_return_period}')



# COMMAND ----------

import shutil

shutil.copytree(raster_dir_local,"/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/output_data" , dirs_exist_ok = True)

# COMMAND ----------


grouped_table2 = grouped_table.copy()
grouped_table2 = pd.merge(grouped_table2, geo_frame[['ras_id', 'elevs']], on='ras_id', how='left')
grouped_table2['WSE'] = np.where(grouped_table2['depth_adj'] <= 0,
                           np.nan,
                           grouped_table2['depth_adj'] + grouped_table2['elevs'])

grouped_table2 = grouped_table2.dropna(subset=['WSE'])

subset = grouped_table2[grouped_table2['storm_id'] == 267]
subset.set_index('ras_id', inplace = True)

out_gdf = point_poly_gdf.join(subset, how = 'left')
out_gdf.plot(column='WSE', legend=True, figsize=(10, 8))

subset = grouped_table2[grouped_table2['storm_id'] == 80]
subset.set_index('ras_id', inplace = True)

out_gdf = point_poly_gdf.join(subset, how = 'left')
out_gdf.plot(column='WSE', legend=True, figsize=(10, 8))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

subset_80 = grouped_table2[grouped_table2['storm_id'] == 80].set_index('ras_id')
subset_267 = grouped_table2[grouped_table2['storm_id'] == 267].set_index('ras_id')

wse_diff = subset_80[['WSE']].copy()
wse_diff['WSE_diff'] = subset_80['WSE'] - subset_267['WSE']

out_gdf_diff = point_poly_gdf.join(wse_diff[['WSE_diff']], how='left')

vmax = out_gdf_diff['WSE_diff'].abs().max()
fig, ax = plt.subplots(figsize=(10, 8))
out_gdf_diff.plot(
    column='WSE_diff',
    legend=True,
    cmap='seismic',
    vmin=-vmax,
    vmax=vmax,
    ax=ax
)
plt.title('WSE Difference (80 - 267)')
plt.show()
