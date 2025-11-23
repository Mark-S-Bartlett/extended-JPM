# Databricks notebook source
import pathlib as pl
import os
https://dbc-9d1717f8-d468.cloud.databricks.com/editor/notebooks/2617494881374018?o=5992598613794601$0import os, sys, copy, shutil, subprocess,json
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
import numpy as np
import matplotlib.pyplot as plt

hdf_data_dir = pl.Path('/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/riverine_aeps_const_mannings_final')
out_data_dir = pl.Path('/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/output_data')

elev_file = pl.Path('/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/no/prcp/ras_output') / 'Amite_20200114.p01.tmp.hdf'

#bias is modelled-true
abs_bias=0.4223155
#relative bias is expectation of (modelled-true)/modelled
rel_bias= -0.07017379



with h5py.File(elev_file) as f:
    elevs = list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Minimum Elevation'])
    xy = list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Center Coordinate'])
len(elevs)
geo_frame = pd.DataFrame(xy, columns = ['x','y'])
geo_frame['ras_id'] = range(1,len(elevs)+1)
geo_frame['elevs'] = elevs

ten_yr_runs = [21, 17, 14]
fifty_yr_runs = [22, 18, 15]
hundred_yr_runs = [23, 19, 6]
five_hundred_yr_runs = [24, 20, 16]

#new 500 plans are 24, 20, 16
#new 100 plans are 23, 19, 6
#50 plans are 22, 18, 15
#10 plans are 21, 17, 14

ras_ids = range(1,len(elevs)+1)


ten_yr_df = pd.DataFrame({'ras_id': ras_ids, 'elevs':elevs})
max_wse = elevs
for run_id in ten_yr_runs:
    this_file = hdf_data_dir / f'Amite_20200114.p{str(run_id).zfill(2)}.hdf'
    with h5py.File(this_file) as f:
        this_wse =  np.array(f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Summary Output']['2D Flow Areas']['AmiteMaurepas']['Maximum Water Surface'])[0,:]
        max_wse = np.maximum(this_wse,max_wse)
ten_yr_df['max_wse'] = max_wse
ten_yr_df['return_period'] = 10
sum(~np.isnan(this_wse))
max_wse.shape

fifty_yr_df = pd.DataFrame({'ras_id': ras_ids, 'elevs':elevs})
max_wse = elevs
for run_id in fifty_yr_runs:
    this_file = hdf_data_dir / f'Amite_20200114.p{str(run_id).zfill(2)}.hdf'
    with h5py.File(this_file) as f:
        this_wse =  np.array(f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Summary Output']['2D Flow Areas']['AmiteMaurepas']['Maximum Water Surface'])[0,:]
        max_wse = np.maximum(this_wse,max_wse)
fifty_yr_df['max_wse'] = max_wse
fifty_yr_df['return_period'] = 50

hundred_yr_df = pd.DataFrame({'ras_id': ras_ids, 'elevs':elevs})
max_wse = elevs
for run_id in hundred_yr_runs:
    this_file = hdf_data_dir / f'Amite_20200114.p{str(run_id).zfill(2)}.hdf'
    with h5py.File(this_file) as f:
        this_wse =  np.array(f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Summary Output']['2D Flow Areas']['AmiteMaurepas']['Maximum Water Surface'])[0,:]
        max_wse = np.maximum(this_wse,max_wse)
hundred_yr_df['max_wse'] = max_wse
hundred_yr_df['return_period'] = 100

five_hundred_yr_df = pd.DataFrame({'ras_id': ras_ids, 'elevs':elevs})
max_wse = elevs
for run_id in five_hundred_yr_runs:
    this_file = hdf_data_dir / f'Amite_20200114.p{str(run_id).zfill(2)}.hdf'
    with h5py.File(this_file) as f:
        this_wse =  np.array(f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Summary Output']['2D Flow Areas']['AmiteMaurepas']['Maximum Water Surface'])[0,:]
        max_wse = np.maximum(this_wse,max_wse)
five_hundred_yr_df['max_wse'] = max_wse
five_hundred_yr_df['return_period'] = 500



riverine_df = pd.concat([ten_yr_df,fifty_yr_df,hundred_yr_df,five_hundred_yr_df])

riverine_df['depth_raw'] = riverine_df['max_wse'] - riverine_df['elevs']
riverine_df['bias_abs'] = abs_bias
riverine_df['bias_rel'] = rel_bias*riverine_df['depth_raw']
riverine_df['bias_bounded'] = np.minimum(np.abs(riverine_df['bias_abs']),np.abs(riverine_df['bias_rel']))*np.sign(riverine_df['bias_abs'])
riverine_df['depth_adj'] = riverine_df['depth_raw'] - riverine_df['bias_bounded']
riverine_df['wse_adj'] = riverine_df['elevs'] + riverine_df['depth_adj']

out_table = pd.merge(riverine_df, geo_frame, how = 'left', on = 'ras_id')


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


ex_table = out_table[out_table['return_period']==100]

ex_data = ex_table[['ras_id', 'x', 'y']]


base_pt_gdf = gpd.GeoDataFrame(ex_data, geometry=gpd.points_from_xy(ex_data['x'], ex_data['y']))

wkt_string = 'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-96.0],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["latitude_of_origin",23.0],UNIT["Foot_US",0.3048006096012192]]'

base_pt_gdf.set_crs(wkt_string, inplace=True)


point_poly_gdf = gpd.sjoin(base_pt_gdf,in_geom, how='right', predicate='within')
point_poly_gdf.dropna(inplace=True, subset=['ras_id'])
point_poly_gdf = point_poly_gdf[['ras_id', 'geometry']]
point_poly_gdf.set_index('ras_id', inplace=True)
point_poly_gdf.to_crs(epsg=6479, inplace=True)


return_periods = list(set(out_table['return_period']))

for this_return_period in return_periods:
    print(f'saving {this_return_period} year WSE')
    this_data = out_table[out_table['return_period']==this_return_period][['ras_id', 'wse_adj']]
    this_data.set_index('ras_id', inplace = True)

    out_gdf = point_poly_gdf.join(this_data, how = 'left')
    out_gdf.to_file(local_data / f'{str(this_return_period).zfill(4)}_year_WSE.shp')

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
project = 'lwi_amite_results_riverine'
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

return_periods = np.array([10, 50, 100, 500])

for this_return_period in return_periods:
  print('importing shapefile')
  gs.run_command('v.import', input=local_data/ f'{str(this_return_period).zfill(4)}_year_WSE.shp',
                       output = f'riverine_{str(this_return_period).zfill(4)}_year_WSE',
                          overwrite = True)
  print("converting to raster")
  gs.run_command('v.to.rast', input = f'riverine_{str(this_return_period).zfill(4)}_year_WSE',
                       output = f'riverine_{str(this_return_period).zfill(4)}_year_WSE_rast',
                       use = 'attr', attribute_column = 'wse_adj', overwrite = True)
  gs.run_command('r.out.gdal', input = f'riverine_{str(this_return_period).zfill(4)}_year_WSE_rast',
                          output = raster_dir_local/ (f'EJPMOS_tropical_riverine_only_{str(this_return_period).zfill(4)}_YR_WSE_final.tif'), format = "GTiff",
                          overwrite = True)
  print(f'finished {this_return_period}')

# COMMAND ----------

import shutil

shutil.copytree(raster_dir_local,"/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/output_data" , dirs_exist_ok = True)
