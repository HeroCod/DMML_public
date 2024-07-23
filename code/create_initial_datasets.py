import pandas as pd
import os
from glob import glob
import numpy as np
import holidays

# Directory paths
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../datasets/originals/')
output_dir = os.path.join(current_dir, '../datasets/completes/')
training_dir = os.path.join(current_dir, '../datasets/training/')
test_dir = os.path.join(current_dir, '../datasets/test/')
validation_dir = os.path.join(current_dir, '../datasets/validation/')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

def read_csv_with_error_handling(file_path, delimiter=';', encoding='utf-8'):
    """Reads a CSV file with error handling."""
    try:
        data = pd.read_csv(file_path, sep=delimiter, encoding=encoding, on_bad_lines='skip', engine='python')
        print(f"Loaded {file_path} successfully.")
    except pd.errors.ParserError as e:
        print(f"Error reading {file_path}: {e}")
        return None
    return data

def is_weekend(date):
    return int(date.weekday() >= 5)

def is_holiday(date, holiday_list):
    return int(date in holiday_list)

def get_season(date):
    m = date.month
    if m in [12, 1, 2]:  # December, January, February
        return 0  # Winter
    elif m in [3, 4, 5]:  # March, April, May
        return 1  # Spring
    elif m in [6, 7, 8]:  # June, July, August
        return 2  # Summer
    else:  # September, October, November
        return 3  # Autumn

def add_missing_hours(df, start_time, end_time, holiday_list, time_col='TASK_CREATION_HOUR'):
    """Ensure every hour from start_time to end_time has a row, interpolating missing hours."""
    all_hours = pd.date_range(start=start_time, end=end_time, freq='h')
    df_all_hours = pd.DataFrame({time_col: all_hours})
    df = pd.merge(df_all_hours, df, how='left', on=time_col)

    # Interpolate missing values for numeric columns and correct logical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear')
        df[col] = df[col].fillna(0)
    
    # Correct IS_HOLIDAY, IS_WEEKEND, and SEASON
    df['IS_HOLIDAY'] = df[time_col].apply(lambda x: is_holiday(x, holiday_list))
    df['IS_WEEKEND'] = df[time_col].apply(is_weekend)
    df['SEASON'] = df[time_col].apply(get_season)

    return df

# Read and concatenate CSV files
all_files = glob(os.path.join(data_dir, '*.csv'))
df_list = [read_csv_with_error_handling(file, delimiter=';') for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# Select relevant columns
columns_to_keep = ['REGION_ID', 'REGION_NAME', 'TASK_CREATION_TIME', 'INCIDENT_TYPE', 'PRIMARY_FAULT', 
                   'VEHICLE_TYPE', 'VEHICLE_BRAND', 'VEHICLE_MODEL', 'MEASURED_TRAVEL_TIME', 
                   'MEASURED_WORK_TIME', 'MEASURED_HOME_TRAVEL_TIME', 'MEASURED_START_WORK_TIME', 
                   'MEASURED_END_WORK_TIME', 'FORWARDED_TIME']
df = df[columns_to_keep]

# Convert TASK_CREATION_TIME, MEASURED_START_WORK_TIME, MEASURED_END_WORK_TIME, and FORWARDED_TIME to datetime
time_columns = ['TASK_CREATION_TIME', 'MEASURED_START_WORK_TIME', 'MEASURED_END_WORK_TIME']
for col in time_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Convert measured columns to numeric, forcing errors to NaN
measured_columns = ['MEASURED_TRAVEL_TIME', 'MEASURED_WORK_TIME', 'MEASURED_HOME_TRAVEL_TIME']
for col in measured_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows based on the specified conditions
df = df[
    df['FORWARDED_TIME'].isna() &
    ~(
        df['MEASURED_WORK_TIME'].isna() & 
        df['MEASURED_START_WORK_TIME'].isna() & 
        df['MEASURED_END_WORK_TIME'].isna()
    )
]

# Fill MEASURED_WORK_TIME where it's empty
df.loc[df['MEASURED_WORK_TIME'].isna(), 'MEASURED_WORK_TIME'] = (
    df['MEASURED_END_WORK_TIME'] - df['MEASURED_START_WORK_TIME']
).dt.total_seconds() / 60

# Drop invalid MEASURED_WORK_TIME rows
def is_valid_work_time(x):
    return not pd.isna(x) and x > 0 and x <= 10000

df = df[df['MEASURED_WORK_TIME'].apply(is_valid_work_time)]

# Drop the columns for start work time and end work time
df.drop(columns=['MEASURED_START_WORK_TIME', 'MEASURED_END_WORK_TIME', 'FORWARDED_TIME'], inplace=True)

# Filter outliers in MEASURED_WORK_TIME outside the 99.5th percentile
percentile_99 = df['MEASURED_WORK_TIME'].quantile(0.995)
df = df[df['MEASURED_WORK_TIME'] <= percentile_99]

# Add IS_HOLIDAY, IS_WEEKEND, and SEASON columns
holiday_list = holidays.CountryHoliday('CH', years=range(df['TASK_CREATION_TIME'].dt.year.min(), df['TASK_CREATION_TIME'].dt.year.max()+1))

df['IS_HOLIDAY'] = df['TASK_CREATION_TIME'].apply(lambda x: is_holiday(x, holiday_list))
df['IS_WEEKEND'] = df['TASK_CREATION_TIME'].apply(is_weekend)
df['SEASON'] = df['TASK_CREATION_TIME'].apply(get_season)

# Save the filtered dataset
filtered_dataset_path = os.path.join(output_dir, 'filtered_dataset.csv')
df.to_csv(filtered_dataset_path, index=False)

# Create hourly dataset aggregated by TASK_CREATION_TIME aligned to the start of the hour
df['TASK_CREATION_HOUR'] = df['TASK_CREATION_TIME'].dt.floor('h')

# Make sure INCIDENT_TYPE is numeric (counting incidents)
df['INCIDENT_TYPE'] = df['INCIDENT_TYPE'].apply(lambda x: 1 if x is not np.nan else 0)

hourly_agg = df.groupby('TASK_CREATION_HOUR').agg({
    'INCIDENT_TYPE': 'sum',
    'MEASURED_WORK_TIME': 'sum',
    'MEASURED_TRAVEL_TIME': 'sum',
    'MEASURED_HOME_TRAVEL_TIME': 'sum',
    'IS_HOLIDAY': 'max',
    'IS_WEEKEND': 'max',
    'SEASON': 'max'
}).rename(columns={
    'INCIDENT_TYPE': 'NUMBER_OF_INCIDENTS',
    'MEASURED_WORK_TIME': 'HOURLY_MINUTES_OF_WORK_TIME',
    'MEASURED_TRAVEL_TIME': 'HOURLY_MINUTES_OF_TRAVEL_TIME',
    'MEASURED_HOME_TRAVEL_TIME': 'HOURLY_MINUTES_OF_HOME_TRAVEL_TIME'
}).reset_index()

# Add TOTAL_HOURLY_MINUTES column
hourly_agg['TOTAL_HOURLY_MINUTES'] = (
    hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'] +
    hourly_agg['HOURLY_MINUTES_OF_TRAVEL_TIME'] +
    hourly_agg['HOURLY_MINUTES_OF_HOME_TRAVEL_TIME']
)

# Complete the hourly dataset
start_time = hourly_agg['TASK_CREATION_HOUR'].min()
end_time = hourly_agg['TASK_CREATION_HOUR'].max()

hourly_agg = add_missing_hours(hourly_agg, start_time, end_time, holiday_list)

# Fill zeroes with cubic interpolation for the work time column
hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'] = hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'].replace(0, np.nan)
hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'] = hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'].interpolate(method='cubic').fillna(0)

# Recalculate TOTAL_HOURLY_MINUTES after interpolation
hourly_agg['TOTAL_HOURLY_MINUTES'] = (
    hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'] +
    hourly_agg['HOURLY_MINUTES_OF_TRAVEL_TIME'] +
    hourly_agg['HOURLY_MINUTES_OF_HOME_TRAVEL_TIME']
)

# Save the total dataset
total_dataset_path = os.path.join(output_dir, 'all_regions.csv')
hourly_agg.to_csv(total_dataset_path, index=False)

# Create region-wise datasets and save them
region_ids = df['REGION_ID'].unique()
for region_id in region_ids:
    region_df = df[df['REGION_ID'] == region_id]
    region_hourly_agg = region_df.groupby('TASK_CREATION_HOUR').agg({
        'INCIDENT_TYPE': 'sum',
        'MEASURED_WORK_TIME': 'sum',
        'MEASURED_TRAVEL_TIME': 'sum',
        'MEASURED_HOME_TRAVEL_TIME': 'sum',
        'IS_HOLIDAY': 'max',
        'IS_WEEKEND': 'max',
        'SEASON': 'max'
    }).rename(columns={
        'INCIDENT_TYPE': 'NUMBER_OF_INCIDENTS',
        'MEASURED_WORK_TIME': 'HOURLY_MINUTES_OF_WORK_TIME',
        'MEASURED_TRAVEL_TIME': 'HOURLY_MINUTES_OF_TRAVEL_TIME',
        'MEASURED_HOME_TRAVEL_TIME': 'HOURLY_MINUTES_OF_HOME_TRAVEL_TIME'
    }).reset_index()

    # Add TOTAL_HOURLY_MINUTES column
    region_hourly_agg['TOTAL_HOURLY_MINUTES'] = (
        region_hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'] +
        region_hourly_agg['HOURLY_MINUTES_OF_TRAVEL_TIME'] +
        region_hourly_agg['HOURLY_MINUTES_OF_HOME_TRAVEL_TIME']
    )
    
    # Complete the region hourly dataset
    region_start_time = region_hourly_agg['TASK_CREATION_HOUR'].min()
    region_end_time = region_hourly_agg['TASK_CREATION_HOUR'].max()
    
    region_hourly_agg = add_missing_hours(region_hourly_agg, region_start_time, region_end_time, holiday_list)
    
    # Fill zeroes with cubic interpolation for the work time column
    region_hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'] = region_hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'].replace(0, np.nan)
    region_hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'] = region_hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'].interpolate(method='cubic').fillna(0)
    
    # Recalculate TOTAL_HOURLY_MINUTES after interpolation
    region_hourly_agg['TOTAL_HOURLY_MINUTES'] = (
        region_hourly_agg['HOURLY_MINUTES_OF_WORK_TIME'] +
        region_hourly_agg['HOURLY_MINUTES_OF_TRAVEL_TIME'] +
        region_hourly_agg['HOURLY_MINUTES_OF_HOME_TRAVEL_TIME']
    )
    
    region_file_path = os.path.join(output_dir, f'region_{region_id}.csv')
    region_hourly_agg.to_csv(region_file_path, index=False)

# Divide datasets into training, test, and validation sets
def partition_datasets(data_frame, train_start, train_end, test_start, test_end, val_start, val_end, base_name):
    train_df = data_frame[(data_frame['TASK_CREATION_HOUR'] >= train_start) & (data_frame['TASK_CREATION_HOUR'] < train_end)]
    val_df = data_frame[(data_frame['TASK_CREATION_HOUR'] >= val_start) & (data_frame['TASK_CREATION_HOUR'] < val_end)]
    test_df = data_frame[(data_frame['TASK_CREATION_HOUR'] >= test_start) & (data_frame['TASK_CREATION_HOUR'] < test_end)]

    train_df.to_csv(os.path.join(training_dir, f'{base_name}.csv'), index=False)
    val_df.to_csv(os.path.join(validation_dir, f'{base_name}.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, f'{base_name}.csv'), index=False)

# Define date ranges
train_start = '2023-01-01'
train_end = '2023-12-31'
val_start = '2024-01-01'
val_end = '2024-02-19'
test_start = '2024-02-20'
test_end = '2024-03-20'

# Total dataset
partition_datasets(hourly_agg, train_start, train_end, test_start, test_end, val_start, val_end, 'all_regions')

# Region-specific datasets
for region_id in region_ids:
    region_file_path = os.path.join(output_dir, f'region_{region_id}.csv')
    region_hourly_agg = read_csv_with_error_handling(region_file_path, delimiter = ',')
    partition_datasets(region_hourly_agg, train_start, train_end, test_start, test_end, val_start, val_end, f'region_{region_id}')
    
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