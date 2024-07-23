import os
import pandas as pd
import json
import plotly.express as px
from prophet import Prophet
from datetime import timedelta
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import holidays  # Importing holidays library

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../datasets/completes/')
plot_dir = os.path.join(current_dir, '../plots/prophet/')
params = os.path.join(current_dir, '../plots/prophet/params/')
console_output_file = os.path.join(current_dir, '../plots/prophet/console_output.json')

# Create necessary directories if they do not exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(params):
    os.makedirs(params)

h_to_predict = 24 * 4  # 4 days ahead

# List all CSV files in the directory
datasets = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'filtered_dataset.csv']

# Helper functions to generate exogenous variables
def is_weekend(date):
    return int(date.weekday() >= 5)

def is_holiday(date, holiday_list):
    return int(date in holiday_list)

def get_season(date):
    m = date.month
    if m in [12, 1, 2]:
        return 0  # Winter
    elif m in [3, 4, 5]:
        return 1  # Spring
    elif m in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Autumn

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
                
                # Load only required columns
                df = pd.read_csv(os.path.join(data_dir, dataset), parse_dates=['TASK_CREATION_HOUR'], usecols=['TASK_CREATION_HOUR', col, 'IS_HOLIDAY', 'IS_WEEKEND', 'SEASON'])
                
                # Convert date info into appropriate format and set index
                df.set_index('TASK_CREATION_HOUR', inplace=True)
                df.reset_index(inplace=True)
                df.rename(columns={'TASK_CREATION_HOUR': 'ds', col: 'y'}, inplace=True)
                
                # Fetch start and end date from dataset
                start_date = df['ds'].min()
                end_date = df['ds'].max()

                # Generate holidays list for Switzerland within the relevant period
                ch_holidays = holidays.Switzerland(years=range(start_date.year, end_date.year + 1))

                # Apply holiday, weekend, and season functions to the dataframe
                df['IS_HOLIDAY'] = df['ds'].apply(lambda x: is_holiday(x, ch_holidays))
                df['IS_WEEKEND'] = df['ds'].apply(is_weekend)
                df['SEASON'] = df['ds'].apply(get_season)

                # Ensure correct data types for exogenous variables
                df['IS_HOLIDAY'] = df['IS_HOLIDAY'].astype('category')
                df['IS_WEEKEND'] = df['IS_WEEKEND'].astype('category')
                df['SEASON'] = df['SEASON'].astype('category')
                
                # Split data into train and test
                test_start = end_date - timedelta(hours=h_to_predict)
                train = df[df['ds'] <= test_start]
                test = df[(df['ds'] > test_start) & (df['ds'] <= end_date)]


                # Define and fit the Prophet model
                model = Prophet()

                # Add exogenous variables
                model.add_regressor('IS_HOLIDAY')
                model.add_regressor('IS_WEEKEND')
                model.add_regressor('SEASON')
                
                model.fit(train)


                # Generate forecast
                print('Generating forecasts...')
                try:
                    future = model.make_future_dataframe(periods=h_to_predict, freq='h')
                    future['IS_HOLIDAY'] = future['ds'].apply(lambda x: is_holiday(x, ch_holidays))
                    future['IS_WEEKEND'] = future['ds'].apply(is_weekend)
                    future['SEASON'] = future['ds'].apply(get_season)

                    forecast = model.predict(future)

                    # Calculate and save MAE and MAPE
                    test_forecast = forecast.set_index('ds').loc[test['ds']]['yhat']
                    actuals = test['y'].values
                    predicted = test_forecast.values

                    mae_value = mae(actuals, predicted)

                    actuals_altered = actuals.copy()
                    actuals_altered[actuals_altered == 0] = 0.1

                    mape_value = mape(actuals_altered, predicted)
                    
                    params_file = os.path.join(params, f'{dataset}_{col}_params.json')
                    with open(params_file, 'w') as file:
                        json.dump({'MAE': mae_value,'MAPE': mape_value}, file, indent=4)
                    
                    console_output[dataset][col]['MAE'] = mae_value
                    console_output[dataset][col]['Params File'] = params_file

                    # Interactive forecast plot using plotly
                    fig = go.Figure()

                    # Add actual observations
                    fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Train'))
                    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Test'))

                    # Add forecast
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

                    # Add uncertainty intervals
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill=None, line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', fill='tonexty', line=dict(width=0), name='Uncertainty Interval'))

                    fig.update_layout(title=f'{col} Forecast with Prophet', xaxis_title='Date', yaxis_title='Value')

                    forecast_path = os.path.join(plot_dir, f'{dataset}_{col}_forecast.html')
                    fig.write_html(forecast_path)
                    
                    console_output[dataset][col]['Forecast Plot'] = forecast_path

                    # Diagnostic plots
                    components_fig = model.plot_components(forecast)
                    diagnostics_path = os.path.join(plot_dir, f'{dataset}_{col}_diagnostics.png')
                    components_fig.savefig(diagnostics_path)
                    
                    console_output[dataset][col]['Diagnostics Plot'] = diagnostics_path
                    
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