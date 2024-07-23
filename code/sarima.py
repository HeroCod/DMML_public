import os
import pandas as pd
import numpy as np
import json
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from pmdarima.arima import ndiffs, nsdiffs
from datetime import timedelta
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../datasets/completes/')
plot_dir = os.path.join(current_dir, '../plots/sarima/')
params_dir = os.path.join(current_dir, '../plots/sarima/params/')
console_output_file = os.path.join(current_dir, '../plots/sarima/console_output.json')
h_to_predict = 24 * 4  # 4 days ahead

# Create necessary directories if they do not exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(params_dir):
    os.makedirs(params_dir)

# List all CSV files in the directory
datasets = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'filtered_dataset.csv']

# Function to convert PeriodIndex to string
def convert_period_index_to_str(data):
    if isinstance(data.index, pd.PeriodIndex):
        data.index = data.index.astype(str)
    return data

# Function to check stationarity
def check_stationarity(series):
    output = {}
    try:
        # Augmented Dickey-Fuller Test
        result_adf = adfuller(series.dropna())
        test_stat_adf, p_value_adf = result_adf[0], result_adf[1]

        # KPSS Test
        result_kpss = kpss(series.dropna(), regression='c')
        test_stat_kpss, p_value_kpss = result_kpss[0], result_kpss[1]

        output['ADF'] = {'Test Statistic': test_stat_adf, 'p-value': p_value_adf}
        output['KPSS'] = {'Test Statistic': test_stat_kpss, 'p-value': p_value_kpss}

        print(f'ADF Test: Test Statistic = {test_stat_adf}, p-value = {p_value_adf}')
        print(f'KPSS Test: Test Statistic = {test_stat_kpss}, p-value = {p_value_kpss}')

        # Determining the d parameter 
        if p_value_adf > 0.05 and p_value_kpss < 0.05:
            print('Series needs differencing.')
            output['Stationarity'] = 'Non-Stationary'
            d = 1
        else:
            print('Series is stationary.')
            output['Stationarity'] = 'Stationary'
            d = 0

        output['d'] = d
    except Exception as e:
        print(f'Error in stationarity check: {e}')
        output['error'] = str(e)
    
    return output

# Function to load parameters if present
def load_best_params(dataset, column):
    params_file = os.path.join(params_dir, f'{dataset}_{column}_params.json')
    if os.path.exists(params_file):
        with open(params_file, 'r') as file:
            params = json.load(file)
        return params
    return None

# Function to save parameters
def save_best_params(dataset, column, order, seasonal_order, aic, mae_value, mape_value):
    params = {
        'order': order,
        'seasonal_order': seasonal_order,
        'aic': aic,
        'mae': mae_value,
        'mape': mape_value
    }
    params_file = os.path.join(params_dir, f'{dataset}_{column}_params.json')
    with open(params_file, 'w') as file:
        json.dump(params, file, indent=4)

# Initialize a dictionary to store console output
console_output = {}

for dataset in datasets:
    print(f'Processing {dataset}')
    try:
        console_output[dataset] = {}
        
        columns_to_test = ['HOURLY_MINUTES_OF_WORK_TIME', 'HOURLY_MINUTES_OF_TRAVEL_TIME', 'HOURLY_MINUTES_OF_HOME_TRAVEL_TIME', 'TOTAL_HOURLY_MINUTES']
        
        for col in columns_to_test:
            print(f'Analyzing {col}')
            try:
                console_output[dataset][col] = {}
                
                # Check for saved parameters and skip if present
                params = load_best_params(dataset, col)
                if params:
                    print(f'Skipping {col} in {dataset} as parameters already exist.')
                    console_output[dataset][col]['Status'] = 'Skipped: Parameters Exist'
                    continue
                
                # Load only required columns
                df = pd.read_csv(os.path.join(data_dir, dataset), parse_dates=['TASK_CREATION_HOUR'], usecols=['TASK_CREATION_HOUR', col])
                df.set_index('TASK_CREATION_HOUR', inplace=True)
                df.index = pd.to_datetime(df.index).to_period('h')

                series = df[col]
                console_output[dataset][col]['Stationarity Test'] = check_stationarity(series)
                d = console_output[dataset][col]['Stationarity Test']['d']

                # Plot ACF and PACF 
                try:
                    acf_fig, ax = plt.subplots(2, 1)
                    plot_acf(series.dropna(), ax=ax[0])
                    plot_pacf(series.dropna(), ax=ax[1])
                    plt.tight_layout()
                    acf_pacf_path = os.path.join(plot_dir, f'{dataset}_{col}_ACF_PACF.png')
                    plt.savefig(acf_pacf_path)
                    plt.close()
                    
                    console_output[dataset][col]['ACF and PACF Plot'] = acf_pacf_path
                except Exception as e:
                    print(f'Error in plotting ACF and PACF: {e}')
                    console_output[dataset][col]['ACF and PACF Plot Error'] = str(e)

                # Splitting Data
                first_date = pd.Timestamp('2023-01-01 00:00:00')
                last_date = df.index.max().to_timestamp()
                train_end = last_date - timedelta(hours=h_to_predict)
                test_start = last_date - timedelta(hours=h_to_predict-1)
                
                train = df[first_date:train_end]
                test = df[test_start:last_date]

                # Model selection using auto_arima
                try:
                    print('Running auto_arima to find the best model...')
                    seasonal_period = 24  # Data shows daily cycle
                    auto_arima_model = auto_arima(
                        train[col].dropna(),
                        seasonal=True,
                        m=seasonal_period,
                        stepwise=True,
                        suppress_warnings=False,
                        error_action='ignore',
                        trace=True,
                        d=d,
                        method='nm',
                    )

                    best_order = auto_arima_model.order
                    best_seasonal_order = auto_arima_model.seasonal_order
                    best_aic = auto_arima_model.aic()

                    final_model = auto_arima_model
                    mae_value = mae(test[col], final_model.predict_in_sample(start=len(train), end=len(train) + len(test) - 1))

                    test_col_altered = test[col].copy()
                    test_col_altered[test_col_altered == 0] = 0.1

                    mape_value = mape(test_col_altered, final_model.predict_in_sample(start=len(train), end=len(train) + len(test) - 1))
                    save_best_params(dataset, col, best_order, best_seasonal_order, best_aic, mae_value, mape_value)

                    console_output[dataset][col]['Best Model Order'] = best_order
                    console_output[dataset][col]['Best Seasonal Order'] = best_seasonal_order
                    console_output[dataset][col]['Best AIC'] = best_aic
                    console_output[dataset][col]['MAE'] = mae_value
                    console_output[dataset][col]['MAPE'] = mape_value


                    model_summary_str = str(auto_arima_model.summary())
                    console_output[dataset][col]['Model Summary'] = model_summary_str
                    print(model_summary_str)
                except Exception as e:
                    print(f'Error in model selection: {e}')
                    console_output[dataset][col]['Model Selection Error'] = str(e)
                    continue

                # Forecasting
                print('Generating forecasts...')
                try:
                    steps_ahead = h_to_predict
                    forecast, forecast_conf_intervals = final_model.predict(n_periods=steps_ahead, return_conf_int=True)                    

                    # Creating the forecast index
                    forecast_index = pd.date_range(start=test_start, periods=steps_ahead, freq='h')

                    # Aligning the forecast with actual and previous values
                    forecast_series = pd.Series(forecast)
                    lower_series = pd.Series(forecast_conf_intervals[:, 0], index=forecast_index)
                    upper_series = pd.Series(forecast_conf_intervals[:, 1], index=forecast_index)

                    # Combine the previous and test series for plotting
                    original_series = pd.concat([train[col], test[col]])

                    original_series = convert_period_index_to_str(original_series)
                    forecast_series = convert_period_index_to_str(forecast_series)
                    lower_series = convert_period_index_to_str(lower_series)
                    upper_series = convert_period_index_to_str(upper_series)

                    fig = px.line(title=f'{col} - Forecast')
                    fig.add_scatter(name='Actual Values', x=original_series.index, y=original_series.values)
                    fig.add_scatter(name='Forecast', x=forecast_series.index, y=forecast_series.values)
                    fig.add_scatter(name='Lower CI', x=lower_series.index, y=lower_series.values, fill='tonexty', mode='lines', line=dict(width=0))
                    fig.add_scatter(name='Upper CI', x=upper_series.index, y=upper_series.values, fill='tonexty', mode='lines', line=dict(width=0))

                    forecast_html_path = os.path.join(plot_dir, f'{dataset}_{col}_forecast.html')
                    fig.write_html(forecast_html_path)

                    console_output[dataset][col]['Forecast Plot'] = forecast_html_path
                except Exception as e:
                    print(f'Error in forecasting: {e}')
                    console_output[dataset][col]['Forecasting Error'] = str(e)
            except Exception as e:
                print(f'Error in processing column {col}: {e}')
                console_output[dataset][col]['Column Processing Error'] = str(e)
    except Exception as e:
        print(f'Error in processing dataset {dataset}: {e}')
        console_output[dataset]['Dataset Processing Error'] = str(e)

# Save console output to a JSON file
with open(console_output_file, 'w') as json_file:
    json.dump(console_output, json_file, indent=4)

print('All tasks completed and console output saved!')

#
# Copyright Notice - DO NOT REMOVE OR ALTERATE
#
# The content within this repository, including but not limited to all source code files, documentation files, and any other files contained herein, is the intellectual property of Francesco Boldrini, or, when applicable, the respective owners of publicly available libraries or code.
#
# Unauthorized reproduction, distribution, or usage of any content from this repository is strictly prohibited unless express written consent has been granted by Francesco Boldrini or the respective content owners, where applicable. Such consent must be documented through a formal, signed contract.
#
# This code is also permitted for use solely in the context of the examination process at the University of Pisa. Post-examination, any copies of the code must be irretrievably destroyed.
#
# In the event that you have come into possession of any code from this repository without proper authorization, you are required to contact Francesco Boldrini at the address provided below, as the code may have been obtained through unlawful distribution.
#
# Copyright © 2024, Francesco Boldrini, All rights reserved.
#
# For any commercial inquiries, please contact:
#
# Francesco Boldrini
# Email: commercial@francesco-boldrini.com
#