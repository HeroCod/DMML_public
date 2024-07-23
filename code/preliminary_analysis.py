import os
import pandas as pd
import plotly.express as px
import json

# Ensure the needed paths and directories are defined and exist
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../datasets/completes/')
plots_dir = os.path.join(current_dir, '../plots/preliminary_analysis/')
subfolders = ['hourly_averages', 'correlations', 'incidents_per_region', 'timeseries_analysis']

# Creating the plots directory and necessary subfolders if they do not exist
os.makedirs(plots_dir, exist_ok=True)
for folder in subfolders:
    os.makedirs(os.path.join(plots_dir, folder), exist_ok=True)

# Defining the paths
filtered_dataset_name = 'filtered_dataset.csv'

# Function to import dataset
def import_dataset(file_name):
    file_path = os.path.join(data_dir, file_name)
    print(f"Importing dataset from {file_path}")
    return pd.read_csv(file_path)

# Function to describe dataset and save description as JSON
def describe_dataset(df, file_name):
    print("Dataset description:")
    description = df.describe().to_dict()
    
    # Adding the sum of each column
    sums = df.sum(numeric_only=True).to_dict()
    description['sum'] = sums
    
    print(description)
    
    # Save dataset description to JSON
    description_path = os.path.join(plots_dir, "dataset_descriptions")
    os.makedirs(description_path, exist_ok=True)
    with open(os.path.join(description_path, f"{file_name}_description.json"), 'w') as f:
        json.dump(description, f)
    print(f"Saved dataset description to {file_name}_description.json")

# Function to save plots
def save_plot(fig, plot_type, plot_name):
    plot_path = os.path.join(plots_dir, plot_type, plot_name)
    fig.write_html(plot_path)
    print(f"Saved plot to {plot_path}")

# Function to analyze the filtered dataset with hourly averages and incidents per region
def analyze_filtered_dataset(df):
    print("Analyzing filtered dataset...")

    # Convert TASK_CREATION_TIME to datetime
    df['TASK_CREATION_TIME'] = pd.to_datetime(df['TASK_CREATION_TIME'])
    
    # Filtering date range between 01/2020 and 04/2024
    start_date = '2020-01-01'
    end_date = '2024-04-30'
    df = df[(df['TASK_CREATION_TIME'] >= start_date) & (df['TASK_CREATION_TIME'] <= end_date)]

    # Extracting the hour from TASK_CREATION_TIME
    df['HOUR'] = df['TASK_CREATION_TIME'].dt.floor('h')
    
    # List of metrics to analyze
    metrics = ['MEASURED_TRAVEL_TIME', 'MEASURED_WORK_TIME', 'MEASURED_HOME_TRAVEL_TIME']

    # Plot number of incidents by region
    incident_counts = df['REGION_ID'].value_counts().reset_index()
    incident_counts.columns = ['REGION_ID', 'NUMBER_OF_INCIDENTS']
    fig = px.bar(incident_counts, x='REGION_ID', y='NUMBER_OF_INCIDENTS', title='Number of Incidents per Region ID')
    save_plot(fig, "incidents_per_region", "incidents_per_region.html")
        
    region_ids = df['REGION_ID'].unique()
    for region_id in region_ids:
        region_data = df[df['REGION_ID'] == region_id]
        
        # Group by HOUR and calculate mean for the metrics
        hourly_avg = region_data.groupby('HOUR')[metrics].mean().reset_index()
        
        # Plot hourly averages
        fig = px.line(hourly_avg, x='HOUR', y=metrics,
                      title=f'Hourly Averages for Region {region_id}')
        save_plot(fig, "hourly_averages", f"region_{region_id}_hourly_averages.html")

# Function to create correlation matrix and save one-hot encodings for the filtered dataset
def create_correlation_matrix(df):
    print("Creating correlation matrix...")

    # One-hot encode categorical columns
    categorical_columns = ['INCIDENT_TYPE', 'PRIMARY_FAULT', 'VEHICLE_TYPE', 'VEHICLE_BRAND', 'VEHICLE_MODEL', 'REGION_NAME']
    available_categorical_columns = [col for col in categorical_columns if col in df.columns]

    if not available_categorical_columns:
        print("No categorical columns available for one-hot encoding.")
    else:
        print("Categorical columns available for one-hot encoding:", available_categorical_columns)

        # Store the unique values before one-hot encoding
        encodings = {col: df[col].unique().tolist() for col in available_categorical_columns}

        df = pd.get_dummies(df, columns=available_categorical_columns)
        print("Categorical columns one-hot encoded.")

        # Save one-hot encodings
        with open(os.path.join(current_dir, 'one_hot_encodings.json'), 'w') as f:
            json.dump(encodings, f)
        print("Saved one-hot encodings to one_hot_encodings.json")
    
    # Create correlation matrix
    numeric_cols = df.select_dtypes(include=[int, float])
    correlation_matrix = numeric_cols.corr()
    fig = px.imshow(correlation_matrix, text_auto=True)
    save_plot(fig, "correlations", "correlation_matrix_filtered_dataset.html")

# Function to create correlation matrix and save one-hot encodings for the region datasets
def create_correlation_matrix_regions(df, file_name=None):
    print("Creating correlation matrix for " + file_name + " ...")
    
    # Exclude date columns for correlation matrix
    numeric_cols = df.select_dtypes(include=[int, float])
    
    # Create correlation matrix
    correlation_matrix = numeric_cols.corr()
    fig = px.imshow(correlation_matrix, text_auto=True)
    save_plot(fig, "correlations", "correlation_matrix_" + file_name + ".html")

# Function to plot over time
def plot_over_time(df, file_name=None):
    print("Plotting datasets over time...")
    if file_name is None:
        file_name = "filtered_dataset"
        fig = px.line(df, x='TASK_CREATION_TIME', y=['MEASURED_TRAVEL_TIME', 'MEASURED_WORK_TIME', 'MEASURED_HOME_TRAVEL_TIME'], title='Timeseries Analysis')
    else:
        fig = px.line(df, x='TASK_CREATION_HOUR', y=['NUMBER_OF_INCIDENTS', 'HOURLY_MINUTES_OF_WORK_TIME', 'HOURLY_MINUTES_OF_TRAVEL_TIME',
                                                     'HOURLY_MINUTES_OF_HOME_TRAVEL_TIME', 'TOTAL_HOURLY_MINUTES'], title='Timeseries Analysis')
    save_plot(fig, "timeseries_analysis", "timeseries_analysis_" + file_name + ".html")

# Main function to process all datasets
def main():
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            df = import_dataset(file_name)
            describe_dataset(df, file_name)
            
            if file_name == filtered_dataset_name:
                analyze_filtered_dataset(df)
                create_correlation_matrix(df)
                plot_over_time(df)
                print(f"No further action for {filtered_dataset_name}.")
            else:
                plot_over_time(df, file_name)
                create_correlation_matrix_regions(df, file_name)

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