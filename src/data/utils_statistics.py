import numpy as np
from scipy import optimize
from scipy.optimize import fsolve, newton, brentq, curve_fit, OptimizeWarning
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import time
import warnings

#https://stackoverflow.com/questions/57722563/how-to-fit-double-exponential-distribution-using-mle-in-python
def MLE(params, data):
    """
    Find the maximum likelihood estimate for the given parameters and data.

    Args:
        params (tuple): A tuple containing the parameters w1, k1, and k2.
        data (array-like): The data to fit the model to.

    Returns:
        float: The negative log-likelihood of the model given the data.
    """
    w1, k1, k2 = params
    yPred = (w1)*k1*np.exp(-k1*data) + (1-w1)*k2*np.exp(-k2*data)
    negLL = -np.sum(np.log(yPred))
    return negLL



def inverse_cdf_mixed_exponential(p, w1, k1, k2):
    """
    Calculate the inverse cumulative distribution function (CDF) 
    of a distribution consisting of the weight addition of two exponential distributions.
    
    Parameters:
        p (float or array-like): Probability value(s) between 0 and 1.
        a1 (float): Weight of the first exponential distribution.
        k1 (float): Rate parameter of the first exponential distribution.
        k2 (float): Rate parameter of the second exponential distribution.
        
    Returns:
        float or array-like: The value(s) corresponding to the given probability value(s).
    """
    def equation(x, p, w1, k1, k2):
        return (1 - (w1) * np.exp(-k1 * x) - (1-w1) * np.exp(-k2 * x)) - p

    x_values = np.zeros_like(p)
    
    for i, prob in enumerate(p):
        # Define the function for brentq
        func = lambda x: equation(x, prob, w1, k1, k2)
        # Set the interval for brentq
        interval = (-1, 1000)  # Adjust the interval as needed
        # Use brentq to find the root within the interval
        sol = brentq(func, *interval)
        x_values[i] = sol
        
    return x_values


def rain_stats_per_gdf_geometry(gdf_row, ds_clipped, series_clipped=None, dates_select_array = None, site_id = 'huc12',aggregate_to_event=False):
    """
    Compute rainfall statistics for a given geometry within a GeoDataFrame row.
    
    This function calculates various rainfall statistics, including yearly, monthly,
    and seasonal statistics. It supports both direct rainfall data aggregation and 
    event-based aggregation using logic from the WRR paper.

    Args:
        gdf_row (GeoDataFrame row): A single row from a GeoDataFrame containing the target geometry.
        ds_clipped (xarray.Dataset): Clipped dataset containing precipitation data.
        series_clipped (pandas.Series, optional): Precomputed time series for direct statistical analysis.
            If provided, monthly and seasonal statistics are not computed.
        dates_select_array (array-like, optional): Array of selected dates for filtering the dataset.
        site_id (str, optional): Identifier for the site (default: 'huc12').
        aggregate_to_events (bool, optional): If True, aggregates rainfall into storm events using logic
            from the WRR paper. Default is False.
    
    Returns:
        dict: A dictionary containing computed statistics, including:
            - 'rainfall_frequency_all': Frequency of non-zero rainfall occurrences.
            - 'average_non_zero_all': Mean of all non-zero rainfall values.
            - 'monthly_stats': Dictionary containing monthly statistics (if applicable),
              including rainfall frequency, event averages, hyperexponential parameters,
              and RMSE values.
            - 'seasonal_stats': Dictionary containing seasonal statistics (if applicable),
              following a similar structure to 'monthly_stats'.
    
    Notes:
        - If `series_clipped` is provided, the function directly analyzes it without computing
          monthly or seasonal statistics.
        - Event-based aggregation is performed if `aggregate_to_events` is True, using storm 
          event logic from the WRR paper.
        - The function applies maximum likelihood estimation (MLE) for fitting a hyperexponential
          distribution to the non-zero rainfall values.
    """

    iterations=1000;
    start_time = time.time()
    #ds_clipped = ds_clipped.rio.write_crs("EPSG:4326", inplace=True)
    #ds_clipped = ds.rio.clip([gdf_row.geometry])
    #print(gdf_row.huc12 +' is clipped')

    if series_clipped is not None:
        series = series_clipped
    else:
        if dates_select_array is not None:
            ds_clipped = ds_clipped.sel(time=dates_select_array)
        #Here aggregate events based on logic on WRR paper.
        if aggregate_to_event == False:
            series = ds_clipped.prcp.mean(dim = ['y', 'x'], skipna=True).to_pandas()
        else:
            #Aggregate to storm events with logic from WRR paper
            df_precip = ds_clipped.prcp.mean(dim = ['y', 'x'], skipna=True).to_dataframe(name='rainfall (mm/day)').drop(columns='lambert_conformal_conic').tz_localize('UTC').drop(columns='spatial_ref')
            df_precip.reset_index(inplace=True)
            combine_rainfall_events(df_precip)

            series = df_precip['event_rainfall (mm/day)']
      
    num_zeros = (series == 0).sum()

    total_days = len(series)
    rainfall_frequency_all = (total_days-num_zeros)/total_days
    print(rainfall_frequency_all)

    # Extract non-zero values
    non_zero_values = np.sort(series[series != 0].values)
    average_non_zero_all = np.mean(non_zero_values)
    
    try:
        guess = np.array([0.999, 1/(average_non_zero_all*10), 1/(average_non_zero_all/50)])
        bnds = ((0.001,0.999),(1/(average_non_zero_all*20), 1/(average_non_zero_all)), (1/(average_non_zero_all), 1/(average_non_zero_all/100)))

        ## best function used for global fitting
        minimizer_kwargs = {"method": 'L-BFGS-B',"bounds": bnds,"args": (non_zero_values,)} #'L-BFGS-B'  SLSQP
        results = optimize.basinhopping(MLE, guess, minimizer_kwargs=minimizer_kwargs, niter=iterations)


        # Calculate MLE prediction
        w1_all, k1_all, k2_all = results.x

        quantiles = [(i+1)/(len(non_zero_values)+1) for i in range(len(non_zero_values))]
        theoretical_values = inverse_cdf_mixed_exponential(quantiles, w1_all, k1_all, k2_all)
        print(f'processing huc {gdf_row[site_id]}')
        rmse_all = np.sqrt(mean_squared_error(non_zero_values, theoretical_values))

        array_monthly_rainfall_freq = np.array([])
        array_monthly_rainfall_event_avg = np.array([])
        array_monthly_weight_hyperexpon = np.array([])
        array_monthly_avg_1_hyperexpon = np.array([])
        array_monthly_avg_2_hyperexpon = np.array([])
        array_monthly_avg_overall_hyperexpon = np.array([])
        array_monthly_rmse = np.array([])

        array_seasonal_rainfall_freq = np.array([])
        array_seasonal_rainfall_event_avg = np.array([])
        array_seasonal_weight_hyperexpon = np.array([])
        array_seasonal_avg_1_hyperexpon = np.array([])
        array_seasonal_avg_2_hyperexpon = np.array([])
        array_seasonal_avg_overall_hyperexpon = np.array([])
        array_seasonal_rmse = np.array([])

        for month in range(1,13):
            #Here aggregate events based on logic on WRR paper.
            if aggregate_to_event == False:
                series = ds_clipped.prcp.mean(dim = ['y', 'x'], skipna=True).sel(time=ds_clipped['time.month'] == month).to_pandas()
            else:
                #Aggregate to storm events with logic from WRR paper
                df_precip = ds_clipped.prcp.mean(dim = ['y', 'x'], skipna=True).sel(time=ds_clipped['time.month'] == month).to_dataframe(name='rainfall (mm/day)').drop(columns='lambert_conformal_conic').tz_localize('UTC').drop(columns='spatial_ref')
                df_precip.reset_index(inplace=True)
                combine_rainfall_events(df_precip)

                series = df_precip['event_rainfall (mm/day)']
            
            num_zeros = (series == 0).sum()
            total_days = len(series)

            rainfall_frequency = (total_days-num_zeros)/total_days
            array_monthly_rainfall_freq = np.append(array_monthly_rainfall_freq, rainfall_frequency)

            # Extract non-zero values
            non_zero_values = np.sort(series[series != 0].values)
            average_non_zero = np.mean(non_zero_values)

            array_monthly_rainfall_event_avg = np.append(array_monthly_rainfall_event_avg, average_non_zero )

            guess = np.array([0.9, 1/(average_non_zero*1.1), 1/(average_non_zero/1.05)])
            bnds = ((0.7,0.99),(1/(average_non_zero*50), 1/(average_non_zero)), (1/(average_non_zero), 1/(average_non_zero/1.25)))

            ## best function used for global fitting
            minimizer_kwargs = {"method": 'L-BFGS-B',"bounds": bnds,"args": (non_zero_values,)} #'L-BFGS-B',  SLSQP, Powell
            results = optimize.basinhopping(MLE, guess, minimizer_kwargs=minimizer_kwargs, niter=iterations)

            w1, k1, k2 = results.x
            array_monthly_weight_hyperexpon = np.append(array_monthly_weight_hyperexpon, w1 )
            array_monthly_avg_1_hyperexpon = np.append(array_monthly_avg_1_hyperexpon, 1/k1 )
            array_monthly_avg_2_hyperexpon = np.append(array_monthly_avg_2_hyperexpon, 1/k2 )
            array_monthly_avg_overall_hyperexpon = np.append(array_monthly_avg_overall_hyperexpon, w1*(1/k2)+(1-w1)*(1/k1) )

            quantiles = [(i+1)/(len(non_zero_values)+1) for i in range(len(non_zero_values))]
            theoretical_values = inverse_cdf_mixed_exponential(quantiles, w1, k1, k2)

            rmse = np.sqrt(mean_squared_error(non_zero_values, theoretical_values))
            array_monthly_rmse = np.append(array_monthly_rmse,rmse)

            #print(month)

        for season in range(1,5):
            list_season_months  = [x +((season-1)*3) for x in range(1, 4)]

            if aggregate_to_event == False:
                series = ds_clipped.prcp.mean(dim = ['y', 'x'], skipna=True).sel(time=ds_clipped['time.month'].isin(list_season_months)).to_pandas()
            else:
                #Aggregate to storm events with logic from WRR paper
                df_precip = ds_clipped.prcp.mean(dim = ['y', 'x'], skipna=True).sel(time=ds_clipped['time.month'].isin(list_season_months)).to_dataframe(name='rainfall (mm/day)').drop(columns='lambert_conformal_conic').tz_localize('UTC').drop(columns='spatial_ref')
                df_precip.reset_index(inplace=True)
                combine_rainfall_events(df_precip)

                series = df_precip['event_rainfall (mm/day)']

            num_zeros = (series == 0).sum()
            total_days = len(series)

            rainfall_frequency = (total_days-num_zeros)/total_days
            array_seasonal_rainfall_freq = np.append(array_seasonal_rainfall_freq, rainfall_frequency)

            # Extract non-zero values
            non_zero_values = np.sort(series[series != 0].values)
            average_non_zero = np.mean(non_zero_values)

            array_seasonal_rainfall_event_avg = np.append(array_seasonal_rainfall_event_avg, average_non_zero )

            guess = np.array([0.9, 1/(average_non_zero*1.1), 1/(average_non_zero/1.05)])
            bnds = ((0.7,0.99),(1/(average_non_zero*50), 1/(average_non_zero)), (1/(average_non_zero), 1/(average_non_zero/1.25)))

            ## best function used for global fitting
            minimizer_kwargs = {"method": 'L-BFGS-B',"bounds": bnds,"args": (non_zero_values,)} #'L-BFGS-B',  SLSQP, Powell
            results = optimize.basinhopping(MLE, guess, minimizer_kwargs=minimizer_kwargs, niter=iterations)

            w1, k1, k2 = results.x
            array_seasonal_weight_hyperexpon = np.append(array_seasonal_weight_hyperexpon, w1 )
            array_seasonal_avg_1_hyperexpon = np.append(array_seasonal_avg_1_hyperexpon, 1/k1 )
            array_seasonal_avg_2_hyperexpon = np.append(array_seasonal_avg_2_hyperexpon, 1/k2 )
            array_seasonal_avg_overall_hyperexpon = np.append(array_seasonal_avg_overall_hyperexpon, w1*(1/k2)+(1-w1)*(1/k1) )

            quantiles = [(i+1)/(len(non_zero_values)+1) for i in range(len(non_zero_values))]
            theoretical_values = inverse_cdf_mixed_exponential(quantiles, w1, k1, k2)

            rmse = np.sqrt(mean_squared_error(non_zero_values, theoretical_values))
            array_seasonal_rmse = np.append(array_seasonal_rmse,rmse)

            #print(season)
    
        #Append data to dataframe:

        data = {
            'lambda_all': rainfall_frequency_all,
            'alpha_all': average_non_zero_all,
            'weight_1_all': w1_all,
            'alpha_1_all': 1/k1_all,
            'alpha_2_all': 1/k2_all,
            'alpha_me_all': w1_all*(1/k1_all)+(1-w1_all)*(1/k2_all),
            'rmse_all': rmse_all,
            'lambda_monthly': array_monthly_rainfall_freq,
            'alpha_monthly': array_monthly_rainfall_event_avg,
            'weight_1_monthly':array_monthly_weight_hyperexpon,
            'alpha_1_monthly': array_monthly_avg_1_hyperexpon,
            'alpha_2_monthly': array_monthly_avg_2_hyperexpon,
            'alpha_me_monthly': array_monthly_avg_overall_hyperexpon,
            'rmse_monthly': array_monthly_rmse,
            'lambda_seasonal': array_seasonal_rainfall_freq,
            'alpha_seasonal': array_seasonal_rainfall_event_avg,
            'weight_1_seasonal':array_seasonal_weight_hyperexpon,
            'alpha_1_seasonal': array_seasonal_avg_1_hyperexpon,
            'alpha_2_seasonal': array_seasonal_avg_2_hyperexpon,
            'alpha_me_seasonal': array_seasonal_avg_overall_hyperexpon,
            'rmse_seasonal': array_seasonal_rmse
            }
    
        df = pd.DataFrame([data], index=[gdf_row[site_id]])
        print(gdf_row[site_id] +' has been processed')
        print(f"-----huc 12 {gdf_row[site_id]} took {(time.time()-start_time)} seconds------")
        return df, gdf_row[site_id]
    except Exception as e:
        print(f"Exception occurred for HUC ID {gdf_row[site_id]}: {e}")
        return (None, gdf_row[site_id])
    

    

def process_hydro_data(
    df_gages_filtered, df_baseflow, ds_clipped, area, 
    baseflow_separation_method, threshold_days, 
    ds_modis_et, ds_modis_pet, gdf_row
    ):
    """
    Process hydrological data for a specific USGS site.

    This function processes rainfall, flow, baseflow, and evapotranspiration (ET) data,
    calculates key hydrological metrics, and aggregates data by year. It also joins
    MODIS ET and PET data to the aggregated dataset and calculates the dryness index (DI)
    and Budyko curve-related values.

    Args:
        df_gages_filtered (pd.DataFrame): Filtered gage data.
        df_baseflow (pd.DataFrame): Baseflow data based on a specified separation method.
        ds_clipped (xarray.Dataset): Clipped dataset for precipitation data.
        area (float): Area of the gage site in square meters.
        baseflow_separation_method (str): Column name for the baseflow separation method.
        threshold_days (int): Minimum number of days required per month for data to be valid.
        ds_modis_et (xarray.Dataset): MODIS ET dataset.
        ds_modis_pet (xarray.Dataset): MODIS PET dataset.
        gdf_row (GeoDataFrame row): GeoDataFrame row containing site and watershed information.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A single-row DataFrame with average DI, ET/R, and Qb/Qf metrics.
            - pd.DataFrame: Yearly aggregated DataFrame with hydrological metrics.
            - pd.DataFrame: Non-NaN filtered gage flow, baseflow, and rainfall DataFrame.
    """

    #Get site name
    site_name = df_gages_filtered.columns.values[0]
    usgs_site_no = gdf_row.site_no
    huc8_in = gdf_row.huc_cd

    # Find dates in common for precipitation and the gage data
    dates_of_gage = df_gages_filtered.index.to_list()
    dates_of_gage_list = [date.tz_convert(None) for date in dates_of_gage ]
    dates_of_gage_array =  np.array([np.datetime64(ts, 'ns') for ts in dates_of_gage_list])
    common_dates = np.intersect1d(dates_of_gage_array, ds_clipped.time.values)

    #Get Rainfall data #TO DO what if no data is selected?
    ds_rainfall = ds_clipped.sel(time=common_dates).mean(dim=['x', 'y'])
    df_rainfall = ds_rainfall.prcp.to_dataframe(name='rainfall (mm/day)').drop(columns='lambert_conformal_conic').tz_localize('UTC').drop(columns='spatial_ref')


    #Convert cfs to mm based on gage area
    df_gages_filtered[df_gages_filtered.columns.values[0]] = (df_gages_filtered[df_gages_filtered.columns.values[0]].values/area ) * (60*60*24*12**3) / (5280*5280*12*12)*25.4
    df_baseflow[baseflow_separation_method] = (df_baseflow[baseflow_separation_method].values/area ) * (60*60*24*12**3) / (5280*5280*12*12)*25.4

    #Join into one dataframe and drop nan
    df_gage_flow_and_baseflow = df_gages_filtered.join([df_baseflow, df_rainfall], how='left')
    df_gage_flow_and_baseflow_non_nan = df_gage_flow_and_baseflow.dropna()

    # Group by month and year, then aggregate
    monthly_summary = df_gage_flow_and_baseflow_non_nan.resample('M').agg({
        df_gage_flow_and_baseflow_non_nan.columns.values[0]: ['count', 'sum'],
        baseflow_separation_method: ['count', 'sum'],
        'rainfall (mm/day)': ['count', 'sum']
        })

    monthly_summary_test = monthly_summary.copy()
    monthly_summary_test.loc[:, 'year'] = monthly_summary_test.index.year
    #Make a boolean for months with a sufficient number of days
    monthly_summary_test.loc[:, 'month_valid']= monthly_summary_test[site_name]['count'].apply(lambda x: 1 if x > threshold_days else 0)


    #Group data by year and only retain full years
    df_grouped = monthly_summary_test.groupby('year').filter(lambda x: x['month_valid'].sum() == 12).drop(columns = 'month_valid')

    #Calculate data and remove multidimensional columns
    df_grouped['days'] = df_grouped[site_name]['count']
    df_grouped['flow(mm)'] = df_grouped[site_name]['sum']
    df_grouped['baseflow(mm)'] = df_grouped[baseflow_separation_method]['sum']
    df_grouped['rainfall(mm)'] = df_grouped['rainfall (mm/day)']['sum']
    df_grouped['ET(mm)'] = df_grouped['rainfall (mm/day)']['sum']-df_grouped[site_name]['sum']
    df_grouped = df_grouped.drop(columns = [site_name,baseflow_separation_method, 'rainfall (mm/day)'])


    #Mask to areas that are not zero or nan
    ds_et_masked = ds_modis_et.where((ds_modis_et != 0) & (ds_modis_et.notnull()))
    ds_pet_masked = ds_modis_pet.where((ds_modis_pet != 0) & (ds_modis_pet.notnull()))

    #Clip the data to the watershed
    df_et_modis = ds_et_masked.rio.clip([gdf_row.geometry2]).mean(dim = ['latitude', 'longitude'], skipna=True).to_dataframe(name='MODIS_ET(mm/day)').drop(columns='spatial_ref')
    df_pet_modis = ds_pet_masked.rio.clip([gdf_row.geometry2]).mean(dim = ['latitude', 'longitude'], skipna=True).to_dataframe(name='MODIS_PET(mm/day)').drop(columns='spatial_ref')

    #Drop the first year if the beginning date is not January... so we only consider full years of data
    df_pet_modis = df_pet_modis.drop(df_pet_modis.index[0]) if df_pet_modis.index[0].strftime('%m') != '01' else df_pet_modis
    df_et_modis = df_et_modis.drop(df_et_modis.index[0]) if df_et_modis.index[0].strftime('%m') != '01' else df_et_modis

    #Calculate the mean PET over all years
    pet_modis_mean = df_pet_modis['MODIS_PET(mm/day)'].mean()

    #Extract the year as a column and make that column the index
    df_et_modis.loc[:, 'year'] = df_et_modis.index.year
    df_et_modis.set_index('year', inplace=True)

    df_pet_modis.loc[:, 'year'] = df_pet_modis.index.year
    df_pet_modis.set_index('year', inplace=True)

    df_grouped_year = df_grouped.groupby('year').sum()
    df_grouped_year['ET/day(mm/day)'] = df_grouped_year['ET(mm)']/df_grouped_year['days']

    df_grouped_year.columns = df_grouped_year.columns.get_level_values(0)

    df_grouped_year = df_grouped_year.join([df_et_modis,df_pet_modis], how='left')
    df_grouped_year['MODIS_PET(mm/day)'].fillna(df_pet_modis['MODIS_PET(mm/day)'].mean(), inplace=True)
    df_grouped_year['DI'] = (df_grouped_year['MODIS_PET(mm/day)'].fillna(pet_modis_mean))*df_grouped_year['days']/df_grouped_year['rainfall(mm)']
    df_grouped_year['Budyko_ET(mm/day)'] = (1/df_grouped_year['days'])*df_grouped_year['rainfall(mm)']*(df_grouped_year['DI'] * (1 - np.exp(-df_grouped_year['DI'])) * np.tanh(1 / df_grouped_year['DI']))**0.5
    df_grouped_year['<ET/R>'] = df_grouped_year['ET(mm)']/df_grouped_year['rainfall(mm)']
    df_grouped_year['<Qb/Qf>'] = df_grouped_year['baseflow(mm)']/df_grouped_year['flow(mm)']

    mean_di = df_grouped_year['DI'].mean()
    mean_et_r = df_grouped_year['<ET/R>'].mean()
    mean_qb_qf = df_grouped_year['<Qb/Qf>'].mean()

    #   Create a DataFrame with one row
    df_one_row = pd.DataFrame({
        'huc8': [huc8_in],
        'DI_mean': [mean_di],
        '<ET/R>': [mean_et_r],
        '<Qb/Qf>': [mean_qb_qf]
        }, index=[usgs_site_no])

    df_one_row.index.name = 'usgs_site_no'
    df_one_row

    return df_one_row, df_grouped_year, df_gage_flow_and_baseflow_non_nan

def Q_model(t, alpha, C1, b):
    """
    Recession curve model for streamflow.

    Args:
        t (float or array-like): Time since the start of recession.
        alpha (float): Recession coefficient controlling the rate of decline.
        C1 (float): Logarithmic constant for initial discharge.
        b (float, optional): Shape parameter controlling non-linearity. 
                             

    Returns:
        float or ndarray: Modeled discharge values over time.

    Note:
        Uses an exponential form when `b` is close to 1 or greater than 1. 
        Otherwise, applies a non-linear recession model.
    """
    if np.isclose(b, 1, atol=0.04) or b > 1:
        return np.exp(C1) * np.exp(-alpha * t)
    else:
        term = np.clip(-alpha * t + C1, 1e-10, None)
        return ((1 - b) * term) ** (1 / (1 - b))


def calculate_rainfall_runoff_events(df_gage_flow_and_baseflow_non_nan, df_grouped_year,baseflow_separation_method):
    """
    Calculate and adjust rainfall and runoff events based on non-zero rainfall days and their subsequent runoff.

    Args:
        df_gage_flow_and_baseflow_non_nan (pd.DataFrame): DataFrame containing gage flow, baseflow, and non-NaN rainfall data.
        df_grouped_year (pd.DataFrame): DataFrame grouped by year with aggregated hydrological data.

    Returns:
        pd.DataFrame: DataFrame with calculated and adjusted rainfall and runoff events.
    """

    site_name = df_gage_flow_and_baseflow_non_nan.columns.values[0]
    usgs_site_no = site_name.split('_')[0]

    #Get final dates
    dates_final = df_gage_flow_and_baseflow_non_nan.index.to_list()
    dates_to_select_list  = [date.tz_convert(None) for date in dates_final ]
    dates_to_select_array =  np.array([np.datetime64(ts, 'ns') for ts in dates_to_select_list])

    #Filter to dates that are in the final grouping.
    dates = pd.DatetimeIndex(dates_to_select_array)
    mask = (dates.year).isin(df_grouped_year.index.values)
    filtered_dates_to_select_array = dates_to_select_array[mask]

    df_gage_non_nan_masked = df_gage_flow_and_baseflow_non_nan[mask]
    data_runoff = df_gage_non_nan_masked [site_name]-df_gage_non_nan_masked[baseflow_separation_method]
    data_rainfall = df_gage_non_nan_masked ['rainfall (mm/day)']

    df_rainfall_runoff = data_rainfall.to_frame(name=f'{usgs_site_no }_rainfall').join(data_runoff.to_frame(name=f'{usgs_site_no }_runoff'))
    df_rainfall_runoff.reset_index(inplace=True)

    # Initialize a new column to store the combined rainfall
    df_rainfall_runoff[f'{usgs_site_no }_event_rainfall'] = df_rainfall_runoff[f'{usgs_site_no }_rainfall']

    # Iterate over the DataFrame to calculate and adjust rainfall values
    i = 0
    while i < len(df_rainfall_runoff) - 1:
        # Only consider non-zero rainfall days
        if df_rainfall_runoff.loc[i, f'{usgs_site_no }_event_rainfall'] > 0:
            # Start with the first non-zero day
            sum_rainfall = df_rainfall_runoff.loc[i, f'{usgs_site_no }_event_rainfall']
            days_counted = 1  # Counter for consecutive days summed

            j = i + 1

            # Iterate to find subsequent non-zero days
            while j < len(df_rainfall_runoff) and df_rainfall_runoff.loc[j, f'{usgs_site_no }_event_rainfall'] > 0:
                if days_counted == 2:
                    # If this is the third consecutive day
                    if df_rainfall_runoff.loc[j, f'{usgs_site_no }_event_rainfall'] < 0.25 * sum_rainfall:
                        # If the third day's rainfall is less than 25% of the sum of the first two days
                        sum_rainfall += df_rainfall_runoff.loc[j, f'{usgs_site_no }_event_rainfall']
                        df_rainfall_runoff.loc[j, f'{usgs_site_no }_event_rainfall'] = 0  # Set third day to 0
                    break  # Stop after considering the third day
                else:
                    sum_rainfall += df_rainfall_runoff.loc[j, f'{usgs_site_no }_event_rainfall']
                    df_rainfall_runoff.loc[j, f'{usgs_site_no }_event_rainfall'] = 0  # Set subsequent days to 0
                    days_counted += 1
                    j += 1

            # Place the total sum in the first non-zero day
            df_rainfall_runoff.loc[i, f'{usgs_site_no }_event_rainfall'] = sum_rainfall

            # Skip to the next day after the last processed day
            i = j
        else:
            i += 1

    #Initialize a new column to store the combined runoff
    df_rainfall_runoff[f'{usgs_site_no }_event_runoff'] = 0

    # Iterate over the DataFrame to calculate and adjust runoff values
    i = 0
    while i < len(df_rainfall_runoff):
        # Only consider non-zero event rainfall days
        if df_rainfall_runoff.loc[i, f'{usgs_site_no }_event_rainfall'] > 0:
            # Start with the first non-zero event rainfall day
            sum_runoff = 0
            days_summed = 0  # Counter to track the number of days summed
        
            # Include runoff from the current day only if the previous day's runoff was zero
            # CHANGE add in condition that if 6 day runoff total is detected then add the current day
            if i == 0 or df_rainfall_runoff.loc[i-1, f'{usgs_site_no }_runoff'] == 0:
                sum_runoff += df_rainfall_runoff.loc[i, f'{usgs_site_no }_runoff']
                days_summed += 1

            j = i + 1

            # Iterate to find the next non-zero event rainfall day or zero runoff day
            while j < len(df_rainfall_runoff):
                if df_rainfall_runoff.loc[j, f'{usgs_site_no }_runoff'] == 0:
                    # Stop summing if a zero runoff day is encountered after at least three days summed
                    if days_summed >= 3:
                        break
                # CHANGE if statement to break if total 6 day is detected then break and add in 6 day total.
                else:
                    sum_runoff += df_rainfall_runoff.loc[j, f'{usgs_site_no }_runoff']
                    days_summed += 1
                
                # If rainfall is detected then add runof from the rainfall day and then stop. 
                if df_rainfall_runoff.loc[j, f'{usgs_site_no }_event_rainfall'] > 0:
                    break

                j += 1
        
            # Place the total sum in the current non-zero combined rainfall day
            df_rainfall_runoff.loc[i, f'{usgs_site_no }_event_runoff'] = sum_runoff

            # Skip to the next day after the last processed day
            i = j
        else:
            i += 1

    #Following code lines calculate recession charcteristics
    df_rainfall_runoff.set_index('index', inplace=True)
    df_rainfall_runoff.index.name = None

    # Step 1: Identify recession periods (no rainfall and declining runoff)
    no_rainfall = df_rainfall_runoff[f'{usgs_site_no }_event_rainfall'] == 0
    declining_runoff = -df_rainfall_runoff[f'{usgs_site_no }_runoff'].diff(-1)  < 0

    # Combine conditions to isolate recession periods
    recession_periods = no_rainfall & declining_runoff

    # Group consecutive recession periods together
    df_rainfall_runoff['recession_period'] = (recession_periods != recession_periods.shift(1)).cumsum() * recession_periods

    # Calculate the difference of runoff for each site
    df_rainfall_runoff['dQ/dt'] = df_rainfall_runoff[f'{usgs_site_no}_runoff'].diff(-1)

    # Set dQ/dt to NaN where recession_period is 0 (non-recession periods)
    df_rainfall_runoff['dQ/dt'] = df_rainfall_runoff.apply(lambda row: np.nan if row['recession_period'] == 0 else row['dQ/dt'], axis=1)

    #Add time in days
    df_rainfall_runoff['time'] = df_rainfall_runoff.groupby('recession_period').cumcount() + 1
    # Set 'time' to NaN where recession_period is 0
    df_rainfall_runoff['time'] = df_rainfall_runoff.apply(
        lambda row: np.nan if row['recession_period'] == 0 else row['time'], axis=1)

    return df_rainfall_runoff



def process_recession_events(df, usgs_site_no, recession_param_b):
    """
    Processes rainfall events and preceding recession periods, fitting a recession model 
    to calculate runoff over the next six days and storing results in new DataFrame columns.

    Args:
        df (pd.DataFrame): DataFrame containing event and runoff data.
        usgs_site_no (str): USGS site number used to identify relevant columns.
        Q_model (callable): Function to model the recession curve.

    Returns:
        pd.DataFrame: DataFrame with updated columns for 6-day recession and runoff calculations.
    """

    def Q_model(t, alpha, C1, b=recession_param_b):
        """
        Recession curve model for streamflow.

        Args:
            t (float or array-like): Time since the start of recession.
            alpha (float): Recession coefficient controlling the rate of decline.
            C1 (float): Logarithmic constant for initial discharge.
            b (float, optional): Shape parameter controlling non-linearity. 
                             

        Returns:
            float or ndarray: Modeled discharge values over time.

        Note:
            Uses an exponential form when `b` is close to 1 or greater than 1. 
            Otherwise, applies a non-linear recession model.
        """
        if np.isclose(b, 1, atol=0.04) or b > 1:
            return np.exp(C1) * np.exp(-alpha * t)
        else:
            term = np.clip(-alpha * t + C1, 1e-10, None)
            return ((1 - b) * term) ** (1 / (1 - b))


    df[f'{usgs_site_no}_recession_prev_event'] = 0
    df['6_day_recession'] = 0

    # Identify non-zero rainfall events with runoff
    event_indices = df[(df[f'{usgs_site_no}_event_rainfall'] > 0) & (df[f'{usgs_site_no}_runoff'] > 0)].index

    for idx in event_indices:
        previous_day = idx - pd.Timedelta(days=1)

        if previous_day in df.index:
            recession_id = df.loc[previous_day, 'recession_period']

            if recession_id > 0:
                recession_data = df[df['recession_period'] == recession_id][['time', f'{usgs_site_no}_runoff']].dropna()

                if len(recession_data) >= 2:
                    t = recession_data['time'].values
                    Q = recession_data[f'{usgs_site_no}_runoff'].values

                    try:
                        # Suppress the OptimizeWarning
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=OptimizeWarning)
                            params, _ = curve_fit(Q_model, t, Q, p0=[0.01, 100])
                        alpha, C1 = params

                        Q_pred = Q_model(t, alpha, C1)
                        r2 = r2_score(Q, np.nan_to_num(Q_pred, nan=0))

                        t_future = np.arange(1, 7) + t.max()
                        modeled_runoff = Q_model(t_future, alpha, C1)
                        modeled_runoff = np.nan_to_num(np.maximum(modeled_runoff, 0), nan=0)

                        # Find dataframe rows overlapping the next six days
                        next_six_days = df.loc[idx:idx + pd.Timedelta(days=5)].index

                        #Find indices of the days in the dataframe b/c all six days might not be in the dataframe
                        next_six_days_actual = pd.date_range(start=idx, periods=6, freq='D')
                        valid_indices = [i for i, d in enumerate(next_six_days_actual) if d in next_six_days]

                        # Check if the column exists and handle missing values
                        recession_runoff = (
                            df.loc[next_six_days, f'{usgs_site_no}_recession_prev_event']
                            ).fillna(0).values  # Fill any NaNs with 0

                        #recession runoff from previou event is limited by the runoff calculated at each time step.
                        recession_runoff_adjusted = np.minimum(df.loc[next_six_days, f'{usgs_site_no}_runoff'].values-recession_runoff, modeled_runoff[valid_indices])

                        # Retrieve existing values or treat missing values as 0
                        existing_values = df.loc[next_six_days, f'{usgs_site_no}_recession_prev_event'].fillna(0).values

                        # Add new values to the existing ones
                        df.loc[next_six_days, f'{usgs_site_no}_recession_prev_event'] =+ (
                            existing_values +recession_runoff_adjusted
                            )

                        total_modeled_runoff = recession_runoff_adjusted.sum()

                        df.loc[idx, '6_day_recession'] = total_modeled_runoff
                        df.loc[idx, 'r2'] = r2

                    except RuntimeError as e:
                        print(f"Curve fitting failed for event at index {idx}: {e}")

    return df


def update_event_runoff_based_on_recession(df, usgs_site_no):
    """
    Calculates and adjusts runoff values for event rainfall days, accounting for 
    multi-day runoff totals, recession events, and zero-runoff conditions.

    Args:
        df (pd.DataFrame): DataFrame containing event rainfall, runoff, and recession data.
        usgs_site_no (str): USGS site number used to identify relevant columns.

    Returns:
        pd.DataFrame: DataFrame with the adjusted event runoff column.
    """
    # Initialize the new column
    df[f'{usgs_site_no}_event_runoff_new'] = 0

    i = 0
    while i < len(df):
        # Only consider non-zero event rainfall days
        if df.loc[df.index[i], f'{usgs_site_no }_event_rainfall'] > 0:
            # Start with the first non-zero event rainfall day
            sum_runoff = 0
            days_summed = 0  # Counter to track the number of days summed
        
            # Include runoff from the current day only if the previous day's runoff was zero
            #  if 6 day runoff total is detected then add the current day
            if i == 0 or df.loc[df.index[i-1], f'{usgs_site_no }_runoff'] == 0 or df.loc[df.index[i-1], '6_day_recession']!=0:
                sum_runoff += (df.loc[df.index[i], f'{usgs_site_no }_runoff']-df.loc[df.index[i], f'{usgs_site_no }_recession_prev_event'])
                days_summed += 1

            j = i + 1

            # Iterate to find the next non-zero event rainfall day or zero runoff day
            while j < len(df):
                if df.loc[df.index[j], f'{usgs_site_no }_runoff'] == 0:
                    # Stop summing if a zero runoff day is encountered after at least three days summed
                    if days_summed >= 3:
                        break
                # if statement to break if total 6 day is detected then break and add in 6 day total.
                else:
                    if df.loc[df.index[j], '6_day_recession']>0:
                        sum_runoff += df.loc[df.index[j],'6_day_recession']
                        break
                    sum_runoff += (df.loc[df.index[j], f'{usgs_site_no }_runoff']-df.loc[df.index[j], f'{usgs_site_no }_recession_prev_event'])
                    days_summed += 1
                
                # If rainfall is detected then add runof from the rainfall day and then stop. 
                if df.loc[df.index[j], f'{usgs_site_no}_event_rainfall'] > 0:
                    break

                j += 1
        
            # Place the total sum in the current non-zero combined rainfall day
            df.loc[df.index[i], f'{usgs_site_no}_event_runoff_new'] = sum_runoff

            # Skip to the next day after the last processed day
            i = j
        else:
            i += 1

    # Replace the original event runoff column with the new one
    df[f'{usgs_site_no}_event_runoff'] = df[ f'{usgs_site_no}_event_runoff_new']
    # Drop the temporary column
    df.drop(columns=[ f'{usgs_site_no}_event_runoff_new'], inplace=True)

    return df


def combine_rainfall_events(df_rainfall_runoff, usgs_site_no=None):
    """Aggregate consecutive non-zero rainfall days into a single event total.
    
    This function modifies the input DataFrame by summing consecutive non-zero rainfall
    values into the first day's entry and setting subsequent contributing days to zero.
    If a third consecutive day's rainfall is less than 25% of the first two days' total,
    it is included in the sum and also set to zero.

    Args:
        df_rainfall_runoff (pd.DataFrame): DataFrame containing rainfall data.
        usgs_site_no (str): Site identifier used to reference column names.

    Returns:
        None: The function modifies df_rainfall_runoff in place.
    """
    if usgs_site_no is not None:
        event_col = f'{usgs_site_no}_event_rainfall (mm/day)'
        rainfall_col = f'{usgs_site_no}_rainfall (mm/day)'
    else:
        event_col = f'event_rainfall (mm/day)'
        rainfall_col = f'rainfall (mm/day)'
    
    df_rainfall_runoff[event_col] = df_rainfall_runoff[rainfall_col]
    i = 0
    
    while i < len(df_rainfall_runoff) - 1:
        # Only consider non-zero rainfall days
        if df_rainfall_runoff.loc[i, event_col] > 0:
            # Start with the first non-zero day
            sum_rainfall = df_rainfall_runoff.loc[i, event_col]
            days_counted, j = 1, i + 1
            
            # Iterate to find subsequent non-zero days
            while j < len(df_rainfall_runoff) and df_rainfall_runoff.loc[j, event_col] > 0:
                if days_counted == 2:
                    # If this is the third consecutive day, check if it is less than 25%
                    if df_rainfall_runoff.loc[j, event_col] < 0.25 * sum_rainfall:
                        # If the third day's rainfall is less than 25% of the sum of the first two days
                        sum_rainfall += df_rainfall_runoff.loc[j, event_col]
                        df_rainfall_runoff.loc[j, event_col] = 0
                    break # Stop after considering the third day
                else:
                    sum_rainfall += df_rainfall_runoff.loc[j, event_col]
                    df_rainfall_runoff.loc[j, event_col] = 0
                    days_counted += 1
                    j += 1
            
            # Place the total sum in the first non-zero day
            df_rainfall_runoff.loc[i, event_col] = sum_rainfall

            # Skip to the next day after the last processed day
            i = j
        else:
            i += 1

    return None

def rain_stats_monthly_per_gdf_geometry(gdf_row, ds_clipped, series_clipped=None, dates_select_array = None, site_id = 'huc12',aggregate_to_event=False):

    """
    Computes monthly rainfall statistics for a given geographic feature.

    This function processes precipitation data for a specific geometry, calculating:
    1. The weighted frequency of nonzero rainfall events per month.
    2. The weighted average rainfall per month.
    Weighting considers the current month (70%), the previous month (20%), and 
    two months prior (10%), dynamically adjusting for missing data.

    Args:
        gdf_row (GeoDataFrame row): A row from a GeoDataFrame containing site metadata.
        ds_clipped (xarray.Dataset): Clipped dataset containing precipitation data.
        series_clipped (pd.Series, optional): Precomputed rainfall series; skips dataset processing if provided.
        dates_select_array (array-like, optional): Array of selected dates for filtering `ds_clipped`.
        site_id (str, optional): Column name identifying the site in `gdf_row`. Default is 'huc12'.
        aggregate_to_event (bool, optional): If True, aggregates rainfall into storm events.

    Returns:
        tuple:
            - pd.DataFrame: Weighted rainfall event frequency per month.
            - pd.DataFrame: Weighted average rainfall per month.
            - str: Site identifier from `gdf_row[site_id]`.

    Raises:
        Exception: Logs errors encountered during processing.

    """
    
    base_weights = {0: 0.7, -1: 0.2, -2: 0.1}

    start_time = time.time()


    if series_clipped is not None:
        series = series_clipped
    else:
        if dates_select_array is not None:
            ds_clipped = ds_clipped.sel(time=dates_select_array)
        #Here aggregate events based on logic on WRR paper.
        if aggregate_to_event == False:
            series = ds_clipped.prcp.mean(dim = ['y', 'x'], skipna=True).to_pandas()
        else:
            #Aggregate to storm events with logic from WRR paper
            df_precip = ds_clipped.prcp.mean(dim = ['y', 'x'], skipna=True).to_dataframe(name='rainfall (mm/day)').drop(columns='lambert_conformal_conic').tz_localize('UTC').drop(columns='spatial_ref')
            df_precip.reset_index(inplace=True)
            combine_rainfall_events(df_precip)

    try:
         # Ensure 'time' is a datetime type
        df_precip['time'] = pd.to_datetime(df_precip['time'])

        # Extract Year-Month as period
        df_precip['year_month'] = df_precip['time'].dt.to_period('M')

        # Generate a full range of months in the dataset
        all_months = pd.period_range(df_precip['year_month'].min(), df_precip['year_month'].max(), freq='M')

        # Count non-zero values per year-month
        df_nonzero_counts = df_precip[df_precip['event_rainfall (mm/day)'] > 0].groupby('year_month').size()

        # Ensure all months exist, filling missing ones with 0
        df_nonzero_counts = df_nonzero_counts.reindex(all_months, fill_value=0)

        # Compute number of days in each month (1 value per group)
        days_in_month = df_nonzero_counts.index.to_timestamp().days_in_month

        # Normalize by dividing counts by days in the month
        df_normalized = df_nonzero_counts / days_in_month

        # Compute dynamically weighted frequency
        weighted_freq = {}
        for period in df_normalized.index:
            current = df_normalized.get(period, None)
            prev_1 = df_normalized.get(period - 1, None)  # Previous month
            prev_2 = df_normalized.get(period - 2, None)  # Two months ago
            
            # Construct weight distribution dynamically
            values = []
            weights = []

            if current is not None:
                values.append(current)
                weights.append(base_weights[0])
            if prev_1 is not None:
                values.append(prev_1)
                weights.append(base_weights[-1])
            if prev_2 is not None:
                values.append(prev_2)
                weights.append(base_weights[-2])

            # Normalize weights to sum to 1
            if weights:
                weights = [w / sum(weights) for w in weights]

            # Compute weighted sum
            weighted_freq[period] = sum(v * w for v, w in zip(values, weights))

        # Convert to a single-row DataFrame
        df_rainfall_freq_result = pd.DataFrame([weighted_freq.values()], 
                                           columns=[str(p) for p in weighted_freq.keys()], 
                                           index=[gdf_row[site_id]])

        # Convert to a single-row DataFrame
        #df_rainfall_freq_result = pd.DataFrame([df_normalized.values], columns=df_normalized.index.values.astype(str),index= [huc12n])

        # ---- (2) Compute Average Rainfall per Year-Month ----
        df_avg_rainfall = df_precip[df_precip['event_rainfall (mm/day)'] > 0].groupby('year_month')['event_rainfall (mm/day)'].mean()

        # Ensure all months exist, filling missing ones with 0
        df_avg_rainfall = df_avg_rainfall.reindex(all_months, fill_value=0)

        # Compute dynamically weighted average rainfall
        weighted_avg_rainfall = {}
        for period in df_avg_rainfall.index:
            current = df_avg_rainfall.get(period, None)
            prev_1 = df_avg_rainfall.get(period - 1, None)  # Previous month
            prev_2 = df_avg_rainfall.get(period - 2, None)  # Two months ago

            values = []
            weights = []

            if current is not None:
                values.append(current)
                weights.append(base_weights[0])
            if prev_1 is not None:
                values.append(prev_1)
                weights.append(base_weights[-1])
            if prev_2 is not None:
                values.append(prev_2)
                weights.append(base_weights[-2])

            if weights:
                weights = [w / sum(weights) for w in weights]

            weighted_avg_rainfall[period] = sum(v * w for v, w in zip(values, weights))

        # Convert to a single-row DataFrame
        df_avg_rainfall_result = pd.DataFrame([weighted_avg_rainfall.values()], 
                                          columns=[str(p) for p in weighted_avg_rainfall.keys()], 
                                          index=[gdf_row[site_id]])

        # Convert to a single-row DataFrame
        #df_avg_rainfall_result = pd.DataFrame([df_avg_rainfall.values], columns=df_avg_rainfall.index.values.astype(str),index= [huc12n])

        print(gdf_row[site_id] +' has been processed')
        print(f"-----huc 12 {gdf_row[site_id]} took {(time.time()-start_time)} seconds------")
        return df_rainfall_freq_result, df_avg_rainfall_result, gdf_row[site_id]

    except Exception as e:
        print(f"Exception occurred for HUC ID {gdf_row[site_id]}: {e}")
        return (None, None, gdf_row[site_id])
