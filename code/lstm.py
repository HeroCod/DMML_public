import os
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error as mape
import tensorflow as tf
from tensorflow import keras
from keras import models as km
from keras import layers as kl
from keras import callbacks as kc
import json

# Define paths
current_dir = os.path.dirname(os.path.realpath(__file__))
train_data_dir = os.path.join(current_dir, '../datasets/training/')
validation_data_dir = os.path.join(current_dir, '../datasets/validation/')
test_data_dir = os.path.join(current_dir, '../datasets/test/')
plot_dir = os.path.join(current_dir, '../plots/lstm/')
params_dir = os.path.join(plot_dir, 'params/')
models_dir = os.path.join(plot_dir, 'models/')
console_output_file = os.path.join(plot_dir, 'console_output.json')

# Create necessary directories if they do not exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(params_dir):
    os.makedirs(params_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# List all CSV files in the dataset directories
datasets = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]

columns_to_test = ['HOURLY_MINUTES_OF_WORK_TIME',
                   'HOURLY_MINUTES_OF_TRAVEL_TIME',
                   'HOURLY_MINUTES_OF_HOME_TRAVEL_TIME',
                   'TOTAL_HOURLY_MINUTES']

def load_data(file_path, columns):
    df = pd.read_csv(file_path, parse_dates=['TASK_CREATION_HOUR'], usecols=['TASK_CREATION_HOUR'] + columns)
    df.set_index('TASK_CREATION_HOUR', inplace=True)
    
    # Round off seconds and microseconds to consolidate near-identical timestamps
    df.index = df.index.round('h')
    return df

def create_lag_features(data, target_column, lags=[1, 24, 48, 72, 96]):
    lagged_data = data.copy()
    for lag in lags:
        lagged_data[f"Lag_{lag}"] = lagged_data[target_column].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

# Initialize a dictionary to store console output
console_output = {}

# Function to save the resume status
def save_resume_status(status_file, row_idx, best_params_idx):
    status = {'row_idx': row_idx, 'best_params_idx': best_params_idx}
    with open(status_file, 'w') as f:
        json.dump(status, f)

# Function to load the resume status
def load_resume_status(status_file):
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    return {'row_idx': 0, 'best_params_idx': 0}

for dataset in datasets:
    print(f'Processing {dataset}')
    
    try:
        console_output[dataset] = {}
        
        for col in columns_to_test:
            print(f'Analyzing {col}')
            try:
                console_output[dataset][col] = {}
                
                # Load train, validation, and test data
                train_df = load_data(os.path.join(train_data_dir, dataset), [col])
                validation_df = load_data(os.path.join(validation_data_dir, dataset), [col])
                test_df = load_data(os.path.join(test_data_dir, dataset), [col])
                
                # Combine dates 
                full_time_range = pd.date_range(start=min(train_df.index.min(), validation_df.index.min(), test_df.index.min()),
                                                end=max(train_df.index.max(), validation_df.index.max(), test_df.index.max()), freq='h')

                data = pd.concat([train_df, validation_df, test_df])
                lagged_data = create_lag_features(data, col)

                X = lagged_data.drop(columns=[col])
                y = lagged_data[[col]]

                X_train = X.loc[train_df.index.intersection(lagged_data.index)]
                y_train = y.loc[train_df.index.intersection(lagged_data.index)]
                X_valid = X.loc[validation_df.index.intersection(lagged_data.index)]
                y_valid = y.loc[validation_df.index.intersection(lagged_data.index)]
                X_test = X.loc[test_df.index.intersection(lagged_data.index)]
                y_test = y.loc[test_df.index.intersection(lagged_data.index)]

                # Reshape the data for LSTM input
                X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_valid = X_valid.values.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
                X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
                
                # Check if optimal model parameters already exist
                params_file = os.path.join(params_dir, f'{dataset}_{col}_params.json')
                model_file = os.path.join(models_dir, f'{dataset}_{col}_best_model.keras')
                status_file = os.path.join(params_dir, f'{dataset}_{col}_status.json')
                
                if os.path.exists(params_file) and os.path.exists(model_file):
                    print(f'Found saved model for {dataset} - {col}. Using existing model for evaluation and prediction...')
                    best_model = km.load_model(model_file)
                    with open(params_file, 'r') as file:
                        best_model_params = json.load(file)
                    best_model_details = best_model_params  # Initialize the variable for later usage
                else:
                    # Parameter grid for model training
                    param_grid = [
                        {'num_units': [50], 'dropout_rate': 0.1, 'dense_units': [10], 'batch_size': 32, 'epochs': 200},
                        {'num_units': [50, 50], 'dropout_rate': 0.1, 'dense_units': [10, 10], 'batch_size': 32, 'epochs': 250},
                        {'num_units': [100], 'dropout_rate': 0.1, 'dense_units': [10, 10], 'batch_size': 64, 'epochs': 300},
                        {'num_units': [100, 50], 'dropout_rate': 0.1, 'dense_units': [20, 10], 'batch_size': 64, 'epochs': 350},
                        {'num_units': [100, 100, 50], 'dropout_rate': 0.1, 'dense_units': [20, 20, 10], 'batch_size': 64, 'epochs': 400},
                        {'num_units': [150], 'dropout_rate': 0.1, 'dense_units': [20, 20], 'batch_size': 128, 'epochs': 500},
                        {'num_units': [150, 100], 'dropout_rate': 0.1, 'dense_units': [30, 20, 10], 'batch_size': 128, 'epochs': 600},
                        {'num_units': [150, 150, 100], 'dropout_rate': 0.1, 'dense_units': [40, 30, 20, 10], 'batch_size': 128, 'epochs': 700},
                        {'num_units': [50], 'dropout_rate': 0.2, 'dense_units': [10], 'batch_size': 32, 'epochs': 200},
                        {'num_units': [50, 50], 'dropout_rate': 0.2, 'dense_units': [10, 10], 'batch_size': 32, 'epochs': 250},
                        {'num_units': [100], 'dropout_rate': 0.2, 'dense_units': [10, 10], 'batch_size': 64, 'epochs': 300},
                        {'num_units': [100, 50], 'dropout_rate': 0.2, 'dense_units': [20, 10], 'batch_size': 64, 'epochs': 350},
                        {'num_units': [100, 100, 50], 'dropout_rate': 0.2, 'dense_units': [20, 20, 10], 'batch_size': 64, 'epochs': 400},
                        {'num_units': [150], 'dropout_rate': 0.2, 'dense_units': [20, 20], 'batch_size': 128, 'epochs': 500},
                        {'num_units': [150, 100], 'dropout_rate': 0.2, 'dense_units': [30, 20, 10], 'batch_size': 128, 'epochs': 600},
                        {'num_units': [150, 150, 100], 'dropout_rate': 0.2, 'dense_units': [40, 30, 20, 10], 'batch_size': 128, 'epochs': 700},
                        {'num_units': [50], 'dropout_rate': 0.3, 'dense_units': [10], 'batch_size': 32, 'epochs': 200},
                        {'num_units': [50, 50], 'dropout_rate': 0.3, 'dense_units': [10, 10], 'batch_size': 32, 'epochs': 250},
                        {'num_units': [100], 'dropout_rate': 0.3, 'dense_units': [10, 10], 'batch_size': 64, 'epochs': 300},
                        {'num_units': [100, 50], 'dropout_rate': 0.3, 'dense_units': [20, 10], 'batch_size': 64, 'epochs': 350},
                        {'num_units': [100, 100, 50], 'dropout_rate': 0.3, 'dense_units': [20, 20, 10], 'batch_size': 64, 'epochs': 400},
                        {'num_units': [150], 'dropout_rate': 0.3, 'dense_units': [20, 20], 'batch_size': 128, 'epochs': 500},
                        {'num_units': [150, 100], 'dropout_rate': 0.3, 'dense_units': [30, 20, 10], 'batch_size': 128, 'epochs': 600},
                        {'num_units': [150, 150, 100], 'dropout_rate': 0.3, 'dense_units': [40, 30, 20, 10], 'batch_size': 128, 'epochs': 700},
                        {'num_units': [50], 'dropout_rate': 0.4, 'dense_units': [10], 'batch_size': 32, 'epochs': 200},
                        {'num_units': [50, 50], 'dropout_rate': 0.4, 'dense_units': [10, 10], 'batch_size': 32, 'epochs': 250},
                        {'num_units': [100], 'dropout_rate': 0.4, 'dense_units': [10, 10], 'batch_size': 64, 'epochs': 300},
                        {'num_units': [100, 50], 'dropout_rate': 0.4, 'dense_units': [20, 10], 'batch_size': 64, 'epochs': 350},
                        {'num_units': [100, 100, 50], 'dropout_rate': 0.4, 'dense_units': [20, 20, 10], 'batch_size': 64, 'epochs': 400},
                        {'num_units': [150], 'dropout_rate': 0.4, 'dense_units': [20, 20], 'batch_size': 128, 'epochs': 500},
                        {'num_units': [150, 100], 'dropout_rate': 0.4, 'dense_units': [30, 20, 10], 'batch_size': 128, 'epochs': 600},
                        {'num_units': [150, 150, 100], 'dropout_rate': 0.4, 'dense_units': [40, 30, 20, 10], 'batch_size': 128, 'epochs': 700},
                        {'num_units': [50], 'dropout_rate': 0.5, 'dense_units': [10], 'batch_size': 32, 'epochs': 200},
                        {'num_units': [50, 50], 'dropout_rate': 0.5, 'dense_units': [10, 10], 'batch_size': 32, 'epochs': 250},
                        {'num_units': [100], 'dropout_rate': 0.5, 'dense_units': [10, 10], 'batch_size': 64, 'epochs': 300},
                        {'num_units': [100, 50], 'dropout_rate': 0.5, 'dense_units': [20, 10], 'batch_size': 64, 'epochs': 350},
                        {'num_units': [100, 100, 50], 'dropout_rate': 0.5, 'dense_units': [20, 20, 10], 'batch_size': 64, 'epochs': 400},
                        {'num_units': [150], 'dropout_rate': 0.5, 'dense_units': [20, 20], 'batch_size': 128, 'epochs': 500},
                        {'num_units': [150, 100], 'dropout_rate': 0.5, 'dense_units': [30, 20, 10], 'batch_size': 128, 'epochs': 600},
                        {'num_units': [150, 150, 100], 'dropout_rate': 0.5, 'dense_units': [40, 30, 20, 10], 'batch_size': 128, 'epochs': 700},
                    ]
                    
                    best_val_loss = float('inf')
                    best_model = None
                    best_model_params = {}
                    
                    # Load the resume status
                    status = load_resume_status(status_file)
                    row_idx = status['row_idx']
                    best_params_idx = status['best_params_idx']
                    
                    while row_idx < len(param_grid):
                        params = param_grid[row_idx]
                        print(f"Training with params: {params}")
                        
                        model = km.Sequential()
                        for i, units in enumerate(params['num_units']):
                            if i == 0:
                                model.add(kl.LSTM(units, return_sequences=len(params['num_units']) > 1, input_shape=(X_train.shape[1], X_train.shape[2])))
                            else:
                                model.add(kl.LSTM(units, return_sequences=(i < len(params['num_units']) - 1)))
                            model.add(kl.Dropout(params['dropout_rate']))
                        for units in params['dense_units']:
                            model.add(kl.Dense(units, activation='relu'))
                        model.add(kl.Dense(1))
                        
                        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
                        early_stop = kc.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                        
                        history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'],
                                            validation_data=(X_valid, y_valid), callbacks=[early_stop], verbose=1)
                        
                        val_loss = min(history.history['val_loss'])
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model = model
                            best_model_params = params
                            best_params_idx = row_idx

                        # Save intermediate status
                        save_resume_status(status_file, row_idx, best_params_idx)
                        
                        row_idx += 1
                    
                    console_output[dataset][col]['Best Model Params'] = best_model_params
                    
                    # Save the best model
                    best_model.save(model_file)
                    
                    # Save model parameters and performance metrics
                    best_model_details = {
                        'params': best_model_params,
                        'val_loss': best_val_loss
                    }
                    with open(params_file, 'w') as json_file:
                        json.dump(best_model_details, json_file, indent=4)
                
                # Evaluate on test data
                test_loss = best_model.evaluate(X_test, y_test, verbose=1)
                console_output[dataset][col]['Test Loss'] = test_loss
                
                y_pred = best_model.predict(X_test)
                
                # Performance metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                mae_val = mean_absolute_error(y_test, y_pred)

                # Replace zeroes in an altered copy of y_test
                y_test_altered = y_test.copy()
                y_test_altered[y_test_altered == 0] = 0.1

                mape_val = mape(y_test_altered, y_pred)

                console_output[dataset][col]['Performance'] = {'RMSE': rmse, 'R²': r2, 'MAE': mae_val, 'MAPE': mape_val}
                
                # Calculate prediction intervals
                pred_stds = np.std([y_test.values - y_pred])
                lower_bound = y_pred - 1.96 * pred_stds
                upper_bound = y_pred + 1.96 * pred_stds
                
                # Update model parameters JSON with new metrics
                best_model_details.update({
                    'test_loss': test_loss,
                    'metrics': {
                        'RMSE': rmse,
                        'R^2': r2,
                        'MAPE': mape_val,
                        'MAE': mae_val
                    }
                })
                with open(params_file, 'w') as json_file:
                    json.dump(best_model_details, json_file, indent=4)
                
                # Create plot
                fig = px.line(
                    x=y_test.index, 
                    y=[y_test.values.flatten(), y_pred.flatten(), lower_bound.flatten(), upper_bound.flatten()], 
                    labels={'x': 'Time', 'value': col},
                    title=f'Actual vs Predicted {col} with Uncertainty Bands'
                )
                
                fig.update_traces(mode='lines+markers', marker=dict(size=4))
                fig.data[0].name = 'Actual'
                fig.data[1].name = 'Predicted'
                
                # Add uncertainty bands
                fig.add_scatter(x=y_test.index, y=upper_bound.flatten(), mode='lines', line=dict(width=0), showlegend=False)
                fig.add_scatter(x=y_test.index, y=lower_bound.flatten(), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=True, name='95% Prediction Interval')
                
                fig.update_layout(showlegend=True)
                fig.write_html(os.path.join(plot_dir, f'{dataset}_{col}_actual_vs_predicted_with_uncertainty.html'))
                
                # Delete the status file after completing the training
                if os.path.exists(status_file):
                    os.remove(status_file)
                
            except Exception as e:
                console_output[dataset][col]['Error'] = str(e)
                print(f'Error with {col} in {dataset}: {e}')
    except Exception as e:
        console_output[dataset]['Error'] = str(e)
        print(f'Error with {dataset}: {e}')

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