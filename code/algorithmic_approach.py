import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from datetime import timedelta
import json

# Define directories
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../datasets/completes/')
plot_dir = os.path.join(current_dir, '../plots/algorithmic_approach/')
params_dir = os.path.join(plot_dir, 'params')
console_output_file = os.path.join(current_dir, '../plots/algorithmic_approach/console_output.json')
forecasted_data_file = os.path.join(current_dir, '../plots/algorithmic_approach/forecasted_data.json')
h_to_predict = 24 * 4  # 4 days ahead

# Create necessary directories if they do not exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(params_dir):
    os.makedirs(params_dir)

# List all CSV files in the directory
datasets = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'filtered_dataset.csv']

# Initialize a dictionary to store console output and forecasted data
console_output = {}
forecasted_data = {}

for dataset in datasets:
    print(f'Processing {dataset}')
    try:
        console_output[dataset] = {}
        forecasted_data[dataset] = {}

        columns_to_test = ['HOURLY_MINUTES_OF_WORK_TIME', 'HOURLY_MINUTES_OF_TRAVEL_TIME', 'HOURLY_MINUTES_OF_HOME_TRAVEL_TIME', 'TOTAL_HOURLY_MINUTES']

        for col in columns_to_test:
            print(f'Analyzing {col}')
            try:
                console_output[dataset][col] = {}
                forecasted_data[dataset][col] = {}

                # Load the dataset and the required column
                df = pd.read_csv(os.path.join(data_dir, dataset), parse_dates=['TASK_CREATION_HOUR'], usecols=['TASK_CREATION_HOUR', col])
                df.set_index('TASK_CREATION_HOUR', inplace=True)
                df.index = pd.to_datetime(df.index).to_period('h')

                # Splitting Data
                first_date = pd.Timestamp('2023-01-01 00:00:00')
                last_date = df.index.max().to_timestamp() # align last date to midnight
                last_date = pd.Timestamp(last_date.year, last_date.month, last_date.day, 00, 00, 00)
                train_end = last_date - timedelta(hours=h_to_predict)
                test_start = last_date - timedelta(hours=h_to_predict-1)

                train = df[first_date:train_end]
                test = df[test_start:last_date]

                def average_past_three_months(df, hours_per_day):
                    avg = []
                    for i in range(hours_per_day):
                        hourly_indices = df.index[df.index.hour == i]
                        if len(hourly_indices) > 0:
                            past_values = df.loc[hourly_indices].tail(3*24).values.flatten()
                            avg.append(np.nanmean(past_values))
                        else:
                            avg.append(np.nan)
                    return avg

                predictions = []
                hours_per_day = 24
                daily_prediction = average_past_three_months(train, hours_per_day)
                num_days_to_predict = h_to_predict // hours_per_day

                predictions = daily_prediction * num_days_to_predict

                # Calculate the MAE and MAPE
                actual_values = test[col].values[:h_to_predict]
                error_mae = mae(actual_values, predictions)

                actual_values_altered = actual_values.copy()
                actual_values_altered[actual_values_altered == 0] = 0.1

                error_mape = mape(actual_values_altered, predictions)
                console_output[dataset][col]['MAE'] = error_mae
                console_output[dataset][col]['MAPE'] = error_mape

                # Save MAE and MAPE to JSON files in the params subfolder
                params_file = os.path.join(params_dir, f'{dataset}_{col}_params.json')
                with open(params_file, 'w') as param_file:
                    json.dump({'MAE': error_mae, 'MAPE': error_mape}, param_file, indent=4)

                # Creating the forecast index
                forecast_index = pd.date_range(start=test_start, periods=h_to_predict, freq='h')

                forecasted_data[dataset][col]['Metrics'] = {
                    'MAE': error_mae,
                    'MAPE': error_mape
                }
                forecasted_data[dataset][col]['Forecast'] = predictions
                forecasted_data[dataset][col]['Actual'] = actual_values.tolist()

                train_series = train[col]
                test_series = test[col]
                forecast_series = pd.Series(predictions, index=forecast_index)

                fig = px.line(title=f'{col} - Average-based Forecast')
                fig.add_scatter(name='Train Data', x=train_series.index.to_timestamp(), y=train_series.values)
                fig.add_scatter(name='Test Data', x=test_series.index.to_timestamp(), y=test_series.values)
                fig.add_scatter(name='Forecast', x=forecast_series.index, y=forecast_series.values)

                forecast_html_path = os.path.join(plot_dir, f'{dataset}_{col}_forecast.html')
                fig.write_html(forecast_html_path)

                console_output[dataset][col]['Forecast Plot'] = forecast_html_path

            except Exception as e:
                print(f'Error in processing column {col}: {e}')
                console_output[dataset][col]['Column Processing Error'] = str(e)
    except Exception as e:
        print(f'Error in processing dataset {dataset}: {e}')
        console_output[dataset]['Dataset Processing Error'] = str(e)

# Save console output and forecasted data to JSON files
with open(console_output_file, 'w') as json_file:
    json.dump(console_output, json_file, indent=4)

with open(forecasted_data_file, 'w') as forecast_file:
    json.dump(forecasted_data, forecast_file, indent=4)

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