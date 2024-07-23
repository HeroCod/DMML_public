import os
import json
import pandas as pd
import plotly.express as px

# Set base directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Define columns
columns = ['HOURLY_MINUTES_OF_WORK_TIME', 'HOURLY_MINUTES_OF_TRAVEL_TIME', 'HOURLY_MINUTES_OF_HOME_TRAVEL_TIME', 'TOTAL_HOURLY_MINUTES']

# Step 0.1: Read datasets
datasets_dir = os.path.join(current_dir, '../datasets/training/')
datasets = [os.path.splitext(file)[0] for file in os.listdir(datasets_dir) if file.endswith('.csv')]

# Create empty dictionaries to store results
mae_data = {}
mape_data = {}

plots_dir = os.path.join(current_dir, '../plots')
folders = [folder for folder in os.listdir(plots_dir) if folder not in ['compare_models', 'preliminary_analysis', 'trends_seasonality_residuals']]

# Function to recursively find keys in nested dictionaries (case insensitive)
def find_key(d, key):
    for k, v in d.items():
        if k.lower() == key.lower():
            return v
        if isinstance(v, dict):
            result = find_key(v, key)
            if result is not None:
                return result
    return None

# Step 1 and 2: Read JSON files and extract MAE and MAPE values
for folder in folders:
    mae_data[folder] = {}
    mape_data[folder] = {}
    subfolder_path = os.path.join(plots_dir, folder, 'params')

    # Create default entries with 0 for every expected combination
    for dataset in datasets:
        for col in columns:
            mae_data[folder][f"{dataset}_{col}"] = 0
            mape_data[folder][f"{dataset}_{col}"] = 0
    
    if not os.path.exists(subfolder_path):
        continue

    json_files = [file for file in os.listdir(subfolder_path) if file.endswith('.json')]

    for json_file in json_files:
        # Get dataset and column name. The filename format has 4 parts, not 3.
        parts = json_file.split('_')
        dataset = parts[0]
        col = '_'.join(parts[1:-1]).upper()  # Join the parts that make up the column, excluding the last part
        # Remove the .csv from the column name
        col = col.replace('.csv', '')
        col = col.replace('.CSV', '')

        with open(os.path.join(subfolder_path, json_file), 'r') as f:
            data = json.load(f)

        mae = find_key(data, 'mae') or 0
        mape = find_key(data, 'mape') or 0

        mae_data[folder][f"{dataset}_{col}"] = mae
        mape_data[folder][f"{dataset}_{col}"] = mape

# Convert data to DataFrames for easier plotting
mae_df = pd.DataFrame(mae_data).T
mape_df = pd.DataFrame(mape_data).T

# Step 3: Create and save interactive plots
compare_models_path = os.path.join(plots_dir, 'compare_models')
os.makedirs(compare_models_path, exist_ok=True)

mae_fig = px.line(mae_df.T, title='MAE Comparison Across Models', labels={'index': 'Dataset_Column', 'value': 'MAE'})
mae_fig.write_html(os.path.join(compare_models_path, 'mae_comparison.html'))

mape_fig = px.line(mape_df.T, title='MAPE Comparison Across Models', labels={'index': 'Dataset_Column', 'value': 'MAPE'})
mape_fig.write_html(os.path.join(compare_models_path, 'mape_comparison.html'))

# Step 5: Ranking models
ranked_mae = mae_df.mean(axis=1).sort_values().index.tolist()
ranked_mape = mape_df.mean(axis=1).sort_values().index.tolist()

print("Ranking of models by MAE (lowest to highest):")
for i, model in enumerate(ranked_mae):
    print(f"{i + 1}. {model}")

print("\nRanking of models by MAPE (lowest to highest):")
for i, model in enumerate(ranked_mape):
    print(f"{i + 1}. {model}")

# Step 6: Create summary PNG graphs
mae_summary_fig = px.bar(mae_df.mean(axis=1), title='Average MAE by Model', labels={'index': 'Model', 'value': 'Average MAE'})
mae_summary_fig.write_image(os.path.join(compare_models_path, 'average_mae.png'))

mape_summary_fig = px.bar(mape_df.mean(axis=1), title='Average MAPE by Model', labels={'index': 'Model', 'value': 'Average MAPE'})
mape_summary_fig.write_image(os.path.join(compare_models_path, 'average_mape.png'))

# Also save the summary graphs as HTML
mae_summary_html_fig = px.bar(mae_df.mean(axis=1), title='Average MAE by Model', labels={'index': 'Model', 'value': 'Average MAE'})
mae_summary_html_fig.write_html(os.path.join(compare_models_path, 'average_mae.html'))

mape_summary_html_fig = px.bar(mape_df.mean(axis=1), title='Average MAPE by Model', labels={'index': 'Model', 'value': 'Average MAPE'})
mape_summary_html_fig.write_html(os.path.join(compare_models_path, 'average_mape.html'))

# Step 7: Statistical Aggregations
stat_data = {
    'Model': mae_df.index,
    'Mean_MAE': mae_df.mean(axis=1),
    'Median_MAE': mae_df.median(axis=1),
    'Std_MAE': mae_df.std(axis=1),
    'Mean_MAPE': mape_df.mean(axis=1),
    'Median_MAPE': mape_df.median(axis=1),
    'Std_MAPE': mape_df.std(axis=1)
}
stat_df = pd.DataFrame(stat_data)

stat_fig = px.bar(stat_df, x='Model', y=['Mean_MAE', 'Median_MAE', 'Std_MAE', 'Mean_MAPE', 'Median_MAPE', 'Std_MAPE'], 
                  title='Statistical Aggregation of Error Metrics', barmode='group')
stat_fig.write_html(os.path.join(compare_models_path, 'statistical_aggregation.html'))

# Create Podium Graphs for Best Models

# Function to create podium charts
def create_podium_chart(df, metric_name, metric_label, output_file):
    top_models = df.mean(axis=1).sort_values()[:3]
    podium_df = pd.DataFrame(
        {
            'Model': top_models.index,
            metric_label: top_models.values,
            'Rank': ['1st', '2nd', '3rd']
        }
    )
    podium_fig = px.bar(podium_df, x='Rank', y=metric_label, text='Model',
                        title=f'Top 3 Models by {metric_label}', labels={'Rank': 'Podium Position'})
    podium_fig.write_image(os.path.join(compare_models_path, output_file))

create_podium_chart(mae_df, 'Mean_MAE', 'Average MAE', 'mae_podium.png')
create_podium_chart(mape_df, 'Mean_MAPE', 'Average MAPE', 'mape_podium.png')

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