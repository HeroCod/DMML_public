import os
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import plotly.graph_objects as go
import warnings
import matplotlib.pyplot as plt

# Function to import datasets
def import_datasets(data_dir, exclude_file_name):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != exclude_file_name]
    datasets = {}
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        try:
            datasets[file] = pd.read_csv(file_path)
            print(f"Imported {file}")
        except Exception as e:
            print(f"Error importing {file}: {e}")
    
    return datasets

# Function to handle zeros and subzero values
def handle_zeros_and_subzeros(df, column):
    if any(df[column] < 0):
        print(f"ERROR: Column '{column}' contains subzero values. Check data processing for errors!")

    df[column] = df[column].apply(lambda x: 0.01 if x <= 0 else x)

# Function to decompose series with handling missing values
def decompose_series(df, column, model='additive'):
    df_copy = df.copy()  # Make a copy of the dataframe to avoid modifying the original
    if 'TASK_CREATION_HOUR' not in df_copy.columns:
        raise KeyError("Column 'TASK_CREATION_HOUR' not found in the dataset.")
    
    df_copy['TASK_CREATION_HOUR'] = pd.to_datetime(df_copy['TASK_CREATION_HOUR'])
    df_copy.set_index('TASK_CREATION_HOUR', inplace=True)

    # Resample to hourly frequency
    df_copy = df_copy.resample('h').mean()
    
    # Fill missing values
    df_copy[column].fillna(method='ffill', inplace=True)  # Forward fill
    df_copy[column].fillna(method='bfill', inplace=True)  # Backward fill

    result = seasonal_decompose(df_copy[column].dropna(), model=model)
    return result

# Function to save combined plots for additive and multiplicative models
def save_combined_plots(analysis, title, plot_dir, file, model):
    fig = go.Figure()

    for column, result in analysis.items():
        fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name=f'{column} - {model} - Observed'))
        fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name=f'{column} - {model} - Trend'))
        fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name=f'{column} - {model} - Seasonal'))
        fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name=f'{column} - {model} - Residual'))

    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Values')
    plot_path = os.path.join(plot_dir, f"{file.rstrip('.csv')}_{model}_combined.html")
    fig.write_html(plot_path)
    print(f"Saved plot: {plot_path}")

# Function to plot and save autocorrelation plots
def save_acf_plots(df, column, plot_dir, file):
    fig, ax = plt.subplots()
    plot_acf(df[column].dropna(), ax=ax)
    plt.title(f'Autocorrelation Function for {column}')
    plot_path = os.path.join(plot_dir, f"{file.rstrip('.csv')}_{column}_acf.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved ACF plot: {plot_path}")

# Main Script
def main():
    warnings.filterwarnings("ignore")  # Suppress warnings for a cleaner output
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, '../datasets/completes/')
    plot_dir = os.path.join(current_dir, '../plots/trends_seasonality_residuals/')
    
    os.makedirs(plot_dir, exist_ok=True)
    
    datasets = import_datasets(data_dir, 'filtered_dataset.csv')

    columns_to_analyze = [
        'HOURLY_MINUTES_OF_WORK_TIME', 
        'HOURLY_MINUTES_OF_TRAVEL_TIME', 
        'HOURLY_MINUTES_OF_HOME_TRAVEL_TIME',
        'TOTAL_HOURLY_MINUTES'
    ]
    
    decomposed_results = {}
    
    for file, df in datasets.items():
        decomposed_results[file] = {'additive': {}, 'multiplicative': {}}
        
        for column in columns_to_analyze:
            if column in df.columns:
                print(f"Analyzing {column} in {file}")

                # Handle zeros and subzero values
                handle_zeros_and_subzeros(df, column)

                # Save Autocorrelation Plot
                save_acf_plots(df, column, plot_dir, file)

                try:
                    decomposed_results[file]['additive'][column] = decompose_series(df, column, model='additive')
                    decomposed_results[file]['multiplicative'][column] = decompose_series(df, column, model='multiplicative')
                    print(f"Successfully analyzed {column} in {file}")
                except KeyError as e:
                    print(f"Skipping {column} in {file}: {e}")
                except ValueError as e:
                    print(f"Error during decomposition of {column} in {file}: {e}")
    
    for file, analyses in decomposed_results.items():
        for model, analysis in analyses.items():
            if analysis:  # Ensure there's at least one valid analysis for the model
                title = f"{file} - {model.capitalize()} Decomposition Analysis"
                save_combined_plots(analysis, title, plot_dir, file, model)

if __name__ == "__main__":
    main()

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