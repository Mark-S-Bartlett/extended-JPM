# Databricks notebook source

import pandas as pd
import xarray as xr
import pathlib as pl
import h5py
import time
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt



ras_data_dir = "/dbfs/mnt/lwi-common/LWI_Production_forPurdue/LWI_Production_TCs_forPurdue/"
elev_data_file = "/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/no/prcp/ras_output/Amite_20200114.p01.tmp.hdf"
out_data_dir = "/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/prod_no_aleatory/"
non_TC_source_dir = '/dbfs/mnt/lwi-common/LWI_Production_forPurdue/LWI_Production_NonTCs_forPurdue'


antecedent_conds = [5,25,50,75,95]
events_per_ac = 100
recurrence_rate = 1.184297
non_trop_recurrence_rate = 44/18.0
ac_probs = [0.12888,0.274451,0.194052,0.294581,0.108036]
#bias is modelled-true
abs_bias=0.4223155
abs_sd= 2.020001
#relative bias is expectation of (modelled-true)/modelled
rel_bias= -0.07017379
#rel sd is sd of modelled-true)
rel_sd = 0.448964

prob_mass_file = pl.Path('/dbfs/FileStore/LWI_hydro/prob_masses/prob_masses.csv')
prob_mass_df = pd.read_csv(prob_mass_file)
storm_ids = prob_mass_df.storm_id.values

with h5py.File(elev_data_file) as f:
    elevs = np.array(list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Minimum Elevation']))
    xy = list(f['Geometry']['2D Flow Areas']['AmiteMaurepas']['Cells Center Coordinate'])

geo_frame = pd.DataFrame(xy, columns = ['x','y'])
geo_frame['ras_id'] = range(1,len(elevs)+1)
geo_frame['elevs'] = elevs


def calc_annualized_cdf_val(in_df, recurrence_rate, cum_prob_col):
    return np.exp(-recurrence_rate * (1 - in_df[cum_prob_col]))

# COMMAND ----------

# MAGIC %md
# MAGIC # Mount lwi-common s3 bucket
# MAGIC '''
# MAGIC note: we're accessing the bucket using databricks secrets, which is kind of a pain and we'll have to update the credentials every time they rotate.
# MAGIC When it rotates, we need to unmount and remount - apparently the credentials don't actually get checked when mounting, and once you mount with a set of credentials they stick around even if you unmount and remount without inputting credentials, but trying to access anything in the bucket will check the credentials and see that they're not valid if the credentials previously used to mount have been rotated out. Figuring this out was frustrating.
# MAGIC access_key = dbutils.secrets.get(scope = "ocd-lwi-s3-2", key = "access_key")
# MAGIC secret_key = dbutils.secrets.get(scope = "ocd-lwi-s3-2", key = "secret_key")
# MAGIC encoded_secret_key = secret_key.replace("/", "%2F")
# MAGIC aws_bucket_name = "lwi-common"
# MAGIC mount_name = "lwi-common"
# MAGIC dbutils.fs.mount(f"s3a://{access_key}:{encoded_secret_key}@{aws_bucket_name}", f"/mnt/{mount_name}")
# MAGIC display(dbutils.fs.ls(f"/mnt/{mount_name}"))
# MAGIC '''

# COMMAND ----------

dbutils.fs.unmount("/mnt/lwi-common")
access_key = dbutils.secrets.get(scope = "ocd-lwi-s3-mb", key = "access_key")
secret_key = dbutils.secrets.get(scope = "ocd-lwi-s3-mb", key = "secret_key")
encoded_secret_key = secret_key.replace("/", "%2F")
aws_bucket_name = "lwi-common"
mount_name = "lwi-common"
dbutils.fs.mount(f"s3a://{access_key}:{encoded_secret_key}@{aws_bucket_name}", f"/mnt/{mount_name}")
display(dbutils.fs.ls(f"/mnt/{mount_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC #Functions

# COMMAND ----------

import os
import time
import gc
import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import psutil
import shutil
import os
import py7zr

MAX_WORKERS = max(1, mp.cpu_count() - 1)
BATCH_STORMS = 5 #3

def _process_file(args):
    (storm_num, this_ac, this_ac_ind, event_count,
     ras_data_dir, storm_prob, ac_probs, events_per_ac, elevs) = args

    try:
        storm_dir = f'{ras_data_dir}storm_{str(storm_num).zfill(4)}/'
        this_event = event_count + this_ac_ind *  events_per_ac
        this_filename = (
            f'{storm_dir}anteceedant_{str(this_ac).zfill(2)}P/'
            f'event_{str(this_event).zfill(3)}/qaqc/'
            f'storm{str(storm_num).zfill(4)}_antecedant{str(this_ac).zfill(2)}P_event{str(this_event).zfill(3)}.hdf'
        )

        # Check file existence first to avoid expensive h5py errors
        if not os.path.exists(this_filename):
            return None

        with h5py.File(this_filename, 'r') as f:
            wse = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output'] \
                    ['Unsteady Time Series']['2D Flow Areas']['AmiteMaurepas'] \
                    ['Max Values']['Water Surface'][()]
            wse = np.asarray(wse, dtype=np.float32)

        depths = np.round(wse - elevs, 2).astype(np.float32)
        prob_val = np.float32(storm_prob * ac_probs[this_ac_ind] / events_per_ac)
        ras_ids = np.arange(1, depths.shape[0] + 1, dtype=np.int32)
        probs = np.full_like(depths, prob_val, dtype=np.float32)

        return depths, ras_ids, probs, storm_num

    except Exception as e:
        print(f"ERROR reading file for storm {storm_num}: {e}")
        return None


def process_all_storms_vectorized(
    storm_ids, ras_data_dir, prob_mass_df, antecedent_conds,
    ac_probs, events_per_ac, elevs, out_path=None,
    max_workers=MAX_WORKERS, batch_storms=BATCH_STORMS
):
    t0 = time.perf_counter()
    storm_prob_dict = dict(zip(prob_mass_df['storm_id'], prob_mass_df['prob']))

    batch_dfs = []
    accumulated_dfs = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, storm_num in enumerate(storm_ids, 1):
            storm_prob = storm_prob_dict[storm_num]
            args_list = [
                (storm_num, ac, ac_ind, event_count, ras_data_dir, storm_prob, ac_probs, events_per_ac, elevs)
                for ac_ind, ac in enumerate(antecedent_conds)
                for event_count in range(1, 101)
            ]

            # Run all files for this storm in parallel, gather results
            results = list(executor.map(_process_file, args_list))
            valid_results = [r for r in results if r is not None]

            if not valid_results:
                continue

            # Combine results for this storm into a DataFrame
            depths = np.concatenate([r[0] for r in valid_results])
            ras_ids = np.concatenate([r[1] for r in valid_results])
            probs = np.concatenate([r[2] for r in valid_results])

            df_storm = pd.DataFrame({'depth': depths, 'ras_id': ras_ids, 'prob': probs})
            df_storm = df_storm.groupby(['depth', 'ras_id'], as_index=False)['prob'].sum()
            df_storm['storm_id'] = storm_num

            batch_dfs.append(df_storm)

            # Periodically flush to reduce memory fragmentation
            if i % batch_storms == 0:
                accumulated_dfs.append(pd.concat(batch_dfs, ignore_index=True))
                batch_dfs.clear()
                gc.collect()

            rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            print(f"Storms processed: {i}/{len(storm_ids)}; Time elapsed: {time.perf_counter()-t0:.1f}s; RSS: {rss:.0f}MB")

    # Final flush
    if batch_dfs:
        accumulated_dfs.append(pd.concat(batch_dfs, ignore_index=True))
        batch_dfs.clear()
        gc.collect()

    final_df = pd.concat(accumulated_dfs, ignore_index=True)
    accumulated_dfs.clear()

    if out_path:
        final_df.to_csv(out_path, index=False)

    print(f"Final rows: {final_df.shape[0]}; Total time: {time.perf_counter()-t0:.1f}s")
    return final_df



# Global variables for worker processes (set by initializer)
elevs_global = None
ac_probs_global = None

def init_worker(elevs_shared, ac_probs_shared):
    """Initialize worker with shared data to avoid serialization overhead"""
    global elevs_global, ac_probs_global
    elevs_global = elevs_shared
    ac_probs_global = ac_probs_shared

def process_file_optimized(args):
    """Optimized file processing with reduced argument passing"""
    (storm_num, this_ac, this_ac_ind, event_count, ras_data_dir, storm_prob, events_per_ac) = args
    
    try:
        storm_dir = f'{ras_data_dir}storm_{str(storm_num).zfill(4)}/'
        this_event = event_count + this_ac_ind * events_per_ac
        this_filename = (
            f'{storm_dir}antecedant_{str(this_ac).zfill(2)}P/'
            f'event_{str(this_event).zfill(3)}/qaqc/'
            f'storm{str(storm_num).zfill(4)}_antecedant{str(this_ac).zfill(2)}P_event{str(this_event).zfill(3)}.hdf'
        )
        
        # Check file existence first to avoid expensive h5py errors
        if not os.path.exists(this_filename):
            return None
            
        with h5py.File(this_filename, 'r') as f:
            wse = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output'] \
                    ['Unsteady Time Series']['2D Flow Areas']['AmiteMaurepas'] \
                    ['Max Values']['Water Surface'][()]
            wse = np.asarray(wse, dtype=np.float32)
        
        # Use global variables instead of passed parameters
        depths = np.round(wse - elevs_global, 2).astype(np.float32)
        prob_val = np.float32(storm_prob * ac_probs_global[this_ac_ind] / events_per_ac)
        
        # Return DataFrame directly instead of separate arrays
        ras_ids = np.arange(1, depths.shape[0] + 1, dtype=np.int32)
        
        return pd.DataFrame({
            'depth': depths,
            'ras_id': ras_ids,
            'prob': prob_val,  # Single value, pandas will broadcast
            'storm_id': storm_num
        })
        
    except Exception as e:
        print(f"ERROR reading file for storm {storm_num}, AC {this_ac}, event {event_count}: {e}")
        return None

def estimate_memory_usage(storm_ids, antecedent_conds, events_per_ac, avg_locations=10000):
    """Estimate memory usage to optimize batch size"""
    total_events = len(storm_ids) * len(antecedent_conds) * events_per_ac
    # Estimate: 4 bytes * 4 columns * locations per event
    estimated_mb = (total_events * avg_locations * 4 * 4) / (1024**2)
    return estimated_mb

def process_all_storms_optimized(
    storm_ids, ras_data_dir, prob_mass_df, antecedent_conds,
    ac_probs, events_per_ac, elevs, out_path=None,
    max_workers=MAX_WORKERS, batch_storms=BATCH_STORMS
):
    """Optimized storm processing with better memory management"""
    t0 = time.perf_counter()
    storm_prob_dict = dict(zip(prob_mass_df['storm_id'], prob_mass_df['prob']))
    
    # Estimate memory and adjust batch size if needed
    estimated_mb = estimate_memory_usage(storm_ids, antecedent_conds, events_per_ac)
    available_mb = psutil.virtual_memory().available / (1024**2)
    
    if estimated_mb > available_mb * 0.5:  # Use max 50% of available memory
        batch_storms = max(1, int(batch_storms * available_mb * 0.5 / estimated_mb * len(storm_ids)))
        print(f"Adjusted batch size to {batch_storms} based on memory estimate")
    
    accumulated_dfs = []
    
    # Process storms in batches
    for batch_start in range(0, len(storm_ids), batch_storms):
        batch_end = min(batch_start + batch_storms, len(storm_ids))
        batch_storm_ids = storm_ids[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_storms + 1}: storms {batch_start+1}-{batch_end}")
        
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(elevs, ac_probs)
        ) as executor:
            
            all_args = []
            for storm_num in batch_storm_ids:
                storm_prob = storm_prob_dict[storm_num]
                for ac_ind, ac in enumerate(antecedent_conds):
                    for event_count in range(1, events_per_ac + 1):  # Fixed bug
                        all_args.append((storm_num, ac, ac_ind, event_count, 
                                       ras_data_dir, storm_prob, events_per_ac))
            
            # Process all files in parallel
            results = list(executor.map(process_file_optimized, all_args))
            
        # Filter valid results and combine
        valid_dfs = [df for df in results if df is not None]
        
        if valid_dfs:
            # Combine all DataFrames for this batch
            batch_df = pd.concat(valid_dfs, ignore_index=True)
            
            # Group by depth, ras_id, storm_id to sum probabilities
            batch_df = batch_df.groupby(['depth', 'ras_id', 'storm_id'], as_index=False)['prob'].sum()
            
            accumulated_dfs.append(batch_df)
            
            # Clear memory
            del batch_df, valid_dfs, results
            gc.collect()
        
        # Memory status
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        print(f"Batch {batch_start//batch_storms + 1} complete. "
              f"Time: {time.perf_counter()-t0:.1f}s, RSS: {rss:.0f}MB")
    
    # Combine all batches
    if accumulated_dfs:
        print("Combining all batches...")
        final_df = pd.concat(accumulated_dfs, ignore_index=True)
        
        # Final groupby in case same depth/ras_id appears across batches
        final_df = final_df.groupby(['depth', 'ras_id', 'storm_id'], as_index=False)['prob'].sum()
        
        # Clear intermediate data
        accumulated_dfs.clear()
        gc.collect()
        
        if out_path:
            final_df.to_csv(out_path, index=False)
        
        print(f"Final rows: {final_df.shape[0]}, Total time: {time.perf_counter()-t0:.1f}s")
        return final_df
    else:
        print("No valid results found")
        return pd.DataFrame()

# Additional utility functions
def monitor_memory_usage(func):
    """Decorator to monitor memory usage of functions"""
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss / (1024**2)
        print(f"{func.__name__} memory delta: {mem_after - mem_before:.1f}MB")
        return result
    return wrapper

def chunk_storm_processing(storm_ids, chunk_size=10):
    """Process storms in smaller chunks for very large datasets"""
    for i in range(0, len(storm_ids), chunk_size):
        yield storm_ids[i:i + chunk_size]

def calc_annualized_cdf_val(in_df, recurrence_rate, cum_prob_col):
    return np.exp(-recurrence_rate * (1 - in_df[cum_prob_col]))

def interpolate_per_ras_id(storm_data_uncert, non_trop_storm_data_uncert):
    """
    Interpolate annualized CDF values from non_trop_storm_data_uncert onto
    storm_data_uncert's depths, per ras_id, with:
      - proper index alignment (no .values),
      - sorted and de-duplicated reference x (depth_adj),
      - cap right side at 1.0 and left side at first ref value.
    """
    # Build a per-ras_id reference dict with clean, sorted, unique xp/fp
    ref_map = {}
    for rid, g in non_trop_storm_data_uncert.groupby('ras_id'):
        g = g[['depth_adj', 'annualized_cdf_val']].dropna().sort_values('depth_adj')
        if g.empty:
            continue
        # Deduplicate any repeated depths (keep first; adjust if you prefer last/mean)
        xp, idx = np.unique(g['depth_adj'].values, return_index=True)
        fp = g['annualized_cdf_val'].values[idx]
        ref_map[rid] = (xp, fp)

    def interp_group(group: pd.DataFrame) -> pd.Series:
        rid = group.name
        if rid not in ref_map:
            # no reference for this ras_id
            return pd.Series(np.nan, index=group.index)
        xp, fp = ref_map[rid]
        # left uses first ref value; right capped at 1.0
        left_val = fp[0]
        y = np.interp(group['depth_adj'].values, xp, fp, left=left_val, right=1.0)
        return pd.Series(y, index=group.index)

    out = storm_data_uncert.copy()
    out['non_trop_annualized_cdf_interp'] = (
        out.groupby('ras_id', group_keys=False).apply(interp_group)
    )
    return out    



    

# COMMAND ----------

# -------------------------
# Example usage (adapt to your variable names/paths)
# -------------------------

#if __name__ == "__main__":
#    # make sure these variables are defined in your environment:
#    # storm_ids, ras_data_dir, prob_mass_df, antecedent_conds, ac_probs, events_per_ac, elevs, out_data_dir
#    OUT_CSV = os.path.join(out_data_dir, "wse_dist_tc_no_uncert_vectorized.csv")
#    result_df = process_all_storms_vectorized(
#        storm_ids, ras_data_dir, prob_mass_df, antecedent_conds,
#        ac_probs, events_per_ac, elevs, out_path=OUT_CSV,
#        max_workers=MAX_WORKERS, batch_storms=BATCH_STORMS
#    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Verifying the number of RAS IDs

# COMMAND ----------

# MAGIC %md
# MAGIC Confirm that elevation count and wse count are equivalent:

# COMMAND ----------

storm_num=52
this_ac=5
event_count = 0
this_ac_ind=1

storm_dir = f'{ras_data_dir}storm_{str(storm_num).zfill(4)}/'
this_event = event_count + this_ac_ind *  events_per_ac
this_filename = (
            f'{storm_dir}anteceedant_{str(this_ac).zfill(2)}P/'
            f'event_{str(this_event).zfill(3)}/qaqc/'
            f'storm{str(storm_num).zfill(4)}_antecedant{str(this_ac).zfill(2)}P_event{str(this_event).zfill(3)}.hdf'
        )



f = h5py.File(this_filename, 'r')
   

# COMMAND ----------

wse = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output'] \
                    ['Unsteady Time Series']['2D Flow Areas']['AmiteMaurepas'] \
                    ['Max Values']['Water Surface'][()]
wse = np.asarray(wse, dtype=np.float32)
        
# Use global variables instead of passed parameters
depths = np.round(wse - elevs, 2).astype(np.float32)
ras_ids = np.arange(1, depths.shape[0] + 1, dtype=np.int32)

# COMMAND ----------

# MAGIC %md
# MAGIC Verify the equivalence of the number of ras IDs between the elevation value and wse values. The difference is becauseof the nan in the elevation values.

# COMMAND ----------

import numpy as np

# Count the number of NaN values in the wse array
nan_count = np.isnan(elevs).sum()
nan_count

# COMMAND ----------

len(unique_ras_ids)

# COMMAND ----------

len(elevs)

# COMMAND ----------

82240-80413

# COMMAND ----------

len(wse)

# COMMAND ----------

len(ras_ids)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Aggregate TC Data
# MAGIC
# MAGIC Here we aggregate TC data to get all the data (across all the storm run hdf outputs) into one dataframe that includes the raw depth, the ras_id (which ist he grid cell id), the probability weight, and the JPM storm ID. This output is not used further here, but is used in the creation of figures 10, 11 and 12, which require the JPM storm ID.

# COMMAND ----------

OUT_CSV = os.path.join(out_data_dir, "wse_dist_tc_no_uncert_vectorized_bartlett_edits.csv")
result_df = process_all_storms_vectorized(
        storm_ids, ras_data_dir, prob_mass_df, antecedent_conds,
        ac_probs, events_per_ac, elevs, out_path=OUT_CSV,
        max_workers=MAX_WORKERS, batch_storms=BATCH_STORMS
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Data
# MAGIC Save the data for use in Figures 10, 11, and 12.

# COMMAND ----------

storm_data_w_id = pd.read_csv(f'{out_data_dir}/wse_dist_tc_no_uncert_vectorized_bartlett_edits.csv')
storm_data_w_id.to_parquet(f'{out_data_dir}/wse_dist_tc_no_uncert_vectorized_bartlett_edits.parquet', engine="pyarrow", compression="snappy")

# COMMAND ----------

result_df2 = result_df.groupby(['depth', 'ras_id']).sum().reset_index()


# COMMAND ----------

result_df2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Original TC Aggregation Code
# MAGIC The resuting aggregation of data excludes the JPM storm ID. This is not to be used as it is slower and it is better to include the JPM storm ID for design storm analysis.

# COMMAND ----------




init_time = time.perf_counter()

full_df = pd.DataFrame({'depth':pd.Series([], dtype='float'),
                   'ras_id':pd.Series([], dtype='int'),
                   'prob':pd.Series([], dtype='float')})
for storm_num in range(17,18): #storm_ids:
    print(f"processing storm {storm_num}: time elapsed: {time.perf_counter()-init_time}")
    storm_dir = f'{ras_data_dir}storm_{str(storm_num).zfill(4)}/'
    storm_prob = prob_mass_df.loc[prob_mass_df.storm_id == storm_num, 'prob'].values[0]
    storm_df_list = []
    for this_ac_ind in range(len(antecedent_conds)):
        print(f"processing storm {storm_num} antecedent condition {this_ac_ind} of {len(antecedent_conds)}: time elapsed: {time.perf_counter()-init_time}")
        for event_count in range(1,101):
            this_event = event_count + this_ac_ind*100
            this_ac = antecedent_conds[this_ac_ind]
            this_filename = f'{storm_dir}anteceedant_{str(this_ac).zfill(2)}P/event_{str(this_event).zfill(3)}/qaqc/storm{str(storm_num).zfill(4)}_antecedant{str(this_ac).zfill(2)}P_event{str(this_event).zfill(3)}.hdf'
            with h5py.File(this_filename, 'r') as f:
                this_wse = np.array(list(f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['AmiteMaurepas']['Max Values']['Water Surface']))
            this_depths = this_wse - elevs
            this_event_prob = storm_prob*ac_probs[this_ac_ind]/events_per_ac

            this_depth_df = pd.DataFrame({'depth': np.round(this_depths,2),
                                           'ras_id': range(1,len(this_depths)+1),
                                           'prob': this_event_prob,
                                           'storm_id': storm_num
                                           })
            

            storm_df_list.append(this_depth_df)
    last_storm_df = pd.concat(storm_df_list)
    full_df = pd.concat([last_storm_df, full_df])
    #now we sum probability by storm_id and ras_id
    #full_df = full_df.groupby(['depth', 'ras_id']).sum().reset_index()
    summed_df = full_df.groupby(['depth', 'ras_id',], as_index=False).sum()

    # Step 2: Collect storm_ids associated with each (depth, ras_id)
    storm_ids_df = full_df.groupby(['depth', 'ras_id'])['storm_id'].unique().reset_index()

    # Step 3: Merge back
    final_df = summed_df.merge(storm_ids_df, on=['depth', 'ras_id'])


#full_df.to_csv(f'{out_data_dir}wse_dist_tc_no_uncert.csv', index=False)        





# COMMAND ----------




# COMMAND ----------

# MAGIC %md
# MAGIC # Aggregate Non-TC Data
# MAGIC
# MAGIC Load and aggregate non-TC data

# COMMAND ----------



shutil.os.makedirs('/local_disk0/LWI_Production_NonTCs_forPurdue', exist_ok=True)


non_TC_source_dir = '/dbfs/mnt/lwi-common/LWI_Production_forPurdue/LWI_Production_NonTCs_forPurdue'
destination_dir = '/local_disk0'

for item in os.listdir(non_TC_source_dir):
    s = os.path.join(non_TC_source_dir, item)
    d = os.path.join(destination_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

# Unzip the specified 7z file
seven_z_file = '/local_disk0/lwi_hecras_nontc_production_hdf_extraction_data_hdf_20230303.7z'
with py7zr.SevenZipFile(seven_z_file, mode='r') as archive:
    archive.extractall(path='/local_disk0/lwi_hecras_nontc_production_hdf_extraction_data_hdf_2023030')

# COMMAND ----------

rain_events = 44
lag_conditions = 5
surge_condtitions = 5

nontropstorm_dir = '/local_disk0/lwi_hecras_nontc_production_hdf_extraction_data_hdf_2023030'

#non_trop_storm_df = pd.DataFrame({'depth':pd.Series([], dtype='float'),
#                   'ras_id':pd.Series([], dtype='int'),
#                   'prob':pd.Series([], dtype='float')})

non_trop_storm_df_list = []
ras_ids = range(1,len(elevs)+1)

for non_trop_storm_num in range(1,rain_events+1):
    for surge_condition in range(1,surge_condtitions+1):
        for lag_condition in range(1,lag_conditions+1):
            filename = f'{nontropstorm_dir}/storm_{str(non_trop_storm_num).zfill(2)}/storm{str(non_trop_storm_num).zfill(2)}_surge{str(surge_condition).zfill(2)}_lag{str(lag_condition).zfill(2)}.hdf'
            with h5py.File(filename, 'r') as f:
                this_wse = np.array(list(f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['AmiteMaurepas']['Max Values']['Water Surface']))
            this_depths = this_wse - elevs
            this_depth_df = pd.DataFrame({'depth': np.round(this_depths,2), 'ras_id': ras_ids})
            this_event_prob = (1/rain_events) *(1/lag_conditions)*(1/surge_condtitions)
            this_depth_df['prob'] = this_event_prob
            non_trop_storm_df_list.append(this_depth_df)

nont_trop_storm_df = pd.concat(non_trop_storm_df_list)
nont_trop_storm_df.rename(columns={'depth':'depth_raw'}, inplace=True)

non_trop_grouped_table =nont_trop_storm_df.groupby(['ras_id', 'depth_raw']).agg(prob=('prob', 'sum')).reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate non-TC statistics

# COMMAND ----------

non_trop_storm_data_uncert = non_trop_grouped_table

#non_trop_storm_data_uncert['cum_prob'] = non_trop_storm_data_uncert.groupby('ras_id')['prob'].cumsum()

N = len(non_trop_storm_data_uncert)

# Step 1: Equal-weight cumulative (ECDF): i/N for i=1,...,N

non_trop_storm_data_uncert['ecdf'] = non_trop_storm_data_uncert.groupby('ras_id')['prob'].cumsum()


# Step 2: Convert ECDF to Gringorten CDF
non_trop_storm_data_uncert['cum_prob'] = (N / (N + 0.12)) * non_trop_storm_data_uncert['ecdf'] - (0.44 / (N + 0.12))

non_trop_storm_data_uncert['annualized_cdf_val'] = calc_annualized_cdf_val(non_trop_storm_data_uncert,non_trop_recurrence_rate, 'cum_prob')

# COMMAND ----------

non_trop_storm_data_uncert['bias_abs'] = abs_bias
non_trop_storm_data_uncert['bias_rel'] = rel_bias*non_trop_storm_data_uncert['depth_raw']
non_trop_storm_data_uncert['bias_bounded'] = np.minimum(np.abs(non_trop_storm_data_uncert['bias_abs']),np.abs(non_trop_storm_data_uncert['bias_rel']))*np.sign(non_trop_storm_data_uncert['bias_abs'])
non_trop_storm_data_uncert['depth_adj'] = np.maximum(non_trop_storm_data_uncert['depth_raw'] - non_trop_storm_data_uncert['bias_bounded'],0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graph Non-TC results for two ras grid cells

# COMMAND ----------



selected_ras_id = 50802 
selected_ras_id_2 = 48730

# Merge 'elevs' from geo_frame into storm_data_uncert by 'ras_id'
storm_data_uncert2 = pd.merge(non_trop_storm_data_uncert, geo_frame[['ras_id', 'elevs']], on='ras_id', how='left')

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

storm_data_uncert

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate TC Statistics

# COMMAND ----------


storm_data_uncert = pd.read_csv(f'{out_data_dir}/wse_dist_tc_no_uncert.csv')
storm_data_uncert.rename(columns={'depth':'depth_raw'}, inplace=True)
storm_data_uncert.sort_values(['ras_id', 'depth_raw'])

storm_data_uncert['cum_prob'] = storm_data_uncert.groupby('ras_id')['prob'].cumsum()

storm_data_uncert['annualized_cdf_val'] = calc_annualized_cdf_val(storm_data_uncert, recurrence_rate, 'cum_prob')


# COMMAND ----------

storm_data_uncert['bias_abs'] = abs_bias
storm_data_uncert['bias_rel'] = rel_bias*storm_data_uncert['depth_raw']
storm_data_uncert['bias_bounded'] = np.minimum(np.abs(storm_data_uncert['bias_abs']),np.abs(storm_data_uncert['bias_rel']))*np.sign(storm_data_uncert['bias_abs'])
storm_data_uncert['depth_adj'] = np.maximum(storm_data_uncert['depth_raw'] - storm_data_uncert['bias_bounded'],0)

# COMMAND ----------

storm_data_uncert

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate Overall Statistics
# MAGIC Interpolate annualized CDF values from the non-TC CDF (annualized) onto the TC CDF (annualized) probabilities, per ras_id
# MAGIC Note: multiplying the annualized CDFs to get a overall non-exceedance probability is equivalent to the paper where we first combine the PDFs (by weighting by the relative frequency of events) and then subsequently annualize the result.

# COMMAND ----------

storm_data_uncert = interpolate_per_ras_id(storm_data_uncert, non_trop_storm_data_uncert)

# COMMAND ----------

storm_data_uncert 

# COMMAND ----------

storm_data_uncert['combined_annualized_cdf_val'] = storm_data_uncert['annualized_cdf_val'] * storm_data_uncert['non_trop_annualized_cdf_interp']

# COMMAND ----------

storm_data_uncert

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the Results for two ras_ids (two model grid cells). The interpolation provides for a smoother return period flood depth curve.

# COMMAND ----------

selected_ras_id = 50802 
selected_ras_id_2 = 48730

# Merge 'elevs' from geo_frame into storm_data_uncert by 'ras_id'
storm_data_uncert2 = pd.merge(storm_data_uncert, geo_frame[['ras_id', 'elevs']], on='ras_id', how='left')

# Get the CDF data for both ras_ids from storm_data_uncert and calculate WSE = depth_raw + elevs
cdf_data_1 = storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id].sort_values('depth_raw').copy()
cdf_data_1['return_period_trop'] = 1 / (1 - cdf_data_1['annualized_cdf_val'])
cdf_data_1['return_non_trop'] = 1 / (1 - cdf_data_1['non_trop_annualized_cdf_interp'])
cdf_data_1['return_combined'] = 1 / (1 - cdf_data_1['combined_annualized_cdf_val'])
cdf_data_1['WSE'] = cdf_data_1['depth_adj'] + cdf_data_1['elevs']


cdf_data_2 = storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id_2].sort_values('depth_raw').copy()
cdf_data_2['return_period_trop'] = 1 / (1 - cdf_data_2['annualized_cdf_val'])
cdf_data_2['return_non_trop'] = 1 / (1 - cdf_data_2['non_trop_annualized_cdf_interp'])
cdf_data_2['return_combined'] = 1 / (1 - cdf_data_2['combined_annualized_cdf_val'])
cdf_data_2['WSE'] = cdf_data_2['depth_adj'] + cdf_data_2['elevs']

plt.figure(figsize=(8,6))
plt.plot(cdf_data_1['return_period_trop'], cdf_data_1['WSE'], marker='o', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(cdf_data_2['return_period_trop'], cdf_data_2['WSE'], marker='s', label=f'Point 2 (ras_id={selected_ras_id_2})')

plt.plot(cdf_data_1['return_non_trop'], cdf_data_1['WSE'], marker='o', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(cdf_data_2['return_non_trop'], cdf_data_2['WSE'], marker='s', label=f'Point 2 (ras_id={selected_ras_id_2})')
plt.plot(cdf_data_1['return_combined'], cdf_data_1['WSE'], marker='o', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(cdf_data_2['return_combined'], cdf_data_2['WSE'], marker='s', label=f'Point 2 (ras_id={selected_ras_id_2})')

plt.xscale('log')
plt.xlim(1, 5000)
plt.xlabel('Return Period (years)')
plt.ylabel('WSE (ft)')
plt.title('WSE vs Return Period at Two Points')
plt.grid(True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Export TC WSE Rasters 
# MAGIC Rasters are exported for selecte return periods

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

return_periods = np.array([2, 5, 8, 10, 13, 15, 20, 25, 33, 42, 50, 75, 100, 125, 150,
                  200, 250, 300, 350, 400, 500, 1000, 2000])

aeps = 1/return_periods

rp_cdf_vals = 1-aeps 

rp_dataframe_list = []
#for rp in return_periods:
#    this_aep = 1/rp
#    this_cdf_val = 1-this_aep
#    this_rp_data = storm_data_uncert[storm_data_uncert['annualized_cdf_val'] <= this_cdf_val].groupby('ras_id')['depth_adj'].max().reset_index()
#    this_rp_data['return_period'] = rp
#    rp_dataframe_list.append(this_rp_data)

#return_period_table = pd.concat(rp_dataframe_list, axis = 0)
#return_period_table.sort_values(['ras_id', 'return_period'], inplace = True)

return_period_table = (
    storm_data_uncert.groupby("ras_id", group_keys=False)
    .apply(interp_group)
    .reset_index(drop=True)
    )
#[["ras_id","annualized_cdf_val", "depth_adj"]]
out_data_dir = pl.Path(out_data_dir)
out_table = pd.merge(return_period_table, geo_frame, how = 'left', on = 'ras_id')
out_table.to_csv(out_data_dir/ 'exceedance_table_compound_tropical_no_aleatory.csv')
'''this is easy to do in a vectorized fashion in R, since R has the approx() function which supports piecewise linear
interpolation, returning a list of depths for an input list of return periods / cdf vals, which can be then applied
over ras ids with data.tables. In Python, we can vectorize over ras ids with Pandas but the closest relative to approx in python is deprecated/legacy and isn't clear whether it actually supports piecewise constant in the way we want here.'''

# COMMAND ----------

out_data_dir

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

in_table = pd.read_csv(out_data_dir/ 'exceedance_table_compound_tropical_no_aleatory.csv')
in_table['WSE'] = in_table['depth_adj'] + in_table['elevs']

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
                          output = raster_dir_local/ (f'EJPMOS_tropical_compound_{str(this_return_period).zfill(4)}_YR_WSE_no_aleatory_bartlett_edits.tif'), format = "GTiff",
                          overwrite = True)
  print(f'finished {this_return_period}')


# COMMAND ----------

# MAGIC %md
# MAGIC # Export Overall Compound WSE Rasters (that incorporate both TC and non-TC events) 
# MAGIC Rasters are exported for selecte return periods

# COMMAND ----------

def interp_group(g):
    x = g["combined_annualized_cdf_val"].values
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

return_periods = np.array([2, 5, 8, 10, 13, 15, 20, 25, 33, 42, 50, 75, 100, 125, 150,
                  200, 250, 300, 350, 400, 500, 1000, 2000])

aeps = 1/return_periods

rp_cdf_vals = 1-aeps 

rp_dataframe_list = []
#for rp in return_periods:
#    this_aep = 1/rp
#    this_cdf_val = 1-this_aep
#    this_rp_data = storm_data_uncert[storm_data_uncert['annualized_cdf_val'] <= this_cdf_val].groupby('ras_id')['depth_adj'].max().reset_index()
#    this_rp_data['return_period'] = rp
#    rp_dataframe_list.append(this_rp_data)

#return_period_table = pd.concat(rp_dataframe_list, axis = 0)
#return_period_table.sort_values(['ras_id', 'return_period'], inplace = True)

return_period_table = (
    storm_data_uncert.groupby("ras_id", group_keys=False)
    .apply(interp_group)
    .reset_index(drop=True)
    )
#[["ras_id","annualized_cdf_val", "depth_adj"]]
out_data_dir = pl.Path(out_data_dir)
out_table = pd.merge(return_period_table, geo_frame, how = 'left', on = 'ras_id')
out_table.to_csv(out_data_dir/ 'exceedance_table_compound_combined_no_aleatory.csv')
'''this is easy to do in a vectorized fashion in R, since R has the approx() function which supports piecewise linear
interpolation, returning a list of depths for an input list of return periods / cdf vals, which can be then applied
over ras ids with data.tables. In Python, we can vectorize over ras ids with Pandas but the closest relative to approx in python is deprecated/legacy and isn't clear whether it actually supports piecewise constant in the way we want here.
'''

# COMMAND ----------

negative_wse = in_table[in_table['WSE'] < 0]
negative_wse

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

in_table = pd.read_csv(out_data_dir/ 'exceedance_table_compound_combined_no_aleatory.csv')
in_table['WSE'] = in_table['depth_adj'] + in_table['elevs']

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
                          output = raster_dir_local/ (f'EJPMOS_combined_compound_{str(this_return_period).zfill(4)}_YR_WSE_no_aleatory_bartlett_edits.tif'), format = "GTiff",
                          overwrite = True)
  print(f'finished {this_return_period}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Move Data to s3 
# MAGIC Data is used for figures in the paper

# COMMAND ----------

import shutil

shutil.copytree(raster_dir_local,"/dbfs/mnt/lwi-transition-zone/data/pilot_reanalysis/output_data" , dirs_exist_ok = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### NOT USED
# MAGIC This code is for graphing the results, but it was not used in the analysis here, but could be useful for future work.

# COMMAND ----------

selected_ras_id = 50802 
selected_ras_id_2 = 48730

# Merge 'elevs' from geo_frame into storm_data_uncert by 'ras_id'
storm_data_uncert2 = pd.merge(storm_data_uncert, geo_frame[['ras_id', 'elevs']], on='ras_id', how='left')

# Get the CDF data for both ras_ids from storm_data_uncert and calculate WSE = depth_raw + elevs
cdf_data_1 = storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id].sort_values('depth_raw').copy()
cdf_data_1['return_period_trop'] = 1 / (1 - cdf_data_1['annualized_cdf_val'])
cdf_data_1['return_non_trop'] = 1 / (1 - cdf_data_1['non_trop_annualized_cdf_interp'])
cdf_data_1['return_combined'] = 1 / (1 - cdf_data_1['combined_annualized_cdf_val'])
cdf_data_1['WSE'] = cdf_data_1['depth_adj'] + cdf_data_1['elevs']


cdf_data_2 = storm_data_uncert2[storm_data_uncert2['ras_id'] == selected_ras_id_2].sort_values('depth_raw').copy()
cdf_data_2['return_period_trop'] = 1 / (1 - cdf_data_2['annualized_cdf_val'])
cdf_data_2['return_non_trop'] = 1 / (1 - cdf_data_2['non_trop_annualized_cdf_interp'])
cdf_data_2['return_combined'] = 1 / (1 - cdf_data_2['combined_annualized_cdf_val'])
cdf_data_2['WSE'] = cdf_data_2['depth_adj'] + cdf_data_2['elevs']

plt.figure(figsize=(8,6))
plt.plot(cdf_data_1['return_period_trop'], cdf_data_1['WSE'], marker='o', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(cdf_data_2['return_period_trop'], cdf_data_2['WSE'], marker='s', label=f'Point 2 (ras_id={selected_ras_id_2})')

plt.plot(cdf_data_1['return_non_trop'], cdf_data_1['WSE'], marker='o', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(cdf_data_2['return_non_trop'], cdf_data_2['WSE'], marker='s', label=f'Point 2 (ras_id={selected_ras_id_2})')
plt.plot(cdf_data_1['return_combined'], cdf_data_1['WSE'], marker='o', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(cdf_data_2['return_combined'], cdf_data_2['WSE'], marker='s', label=f'Point 2 (ras_id={selected_ras_id_2})')

in_table_1= in_table[in_table['ras_id'] == selected_ras_id]
in_table_2= in_table[in_table['ras_id'] == selected_ras_id_2]

plt.plot(in_table_1['return_period'], in_table_1['WSE'], marker='o', linestyle='--', color='red', label=f'Point 1 (ras_id={selected_ras_id})')
plt.plot(in_table_2['return_period'], in_table_2['WSE'], marker='s', linestyle='--', color='blue', label=f'Point 2 (ras_id={selected_ras_id_2})')

plt.xscale('log')
plt.xlim(1, 5000)
plt.xlabel('Return Period (years)')
plt.ylabel('WSE (ft)')
plt.title('WSE vs Return Period at Two Points')
plt.grid(True)

# COMMAND ----------


