from datetime import datetime
import pandas as pd
import os
from datetime import datetime, timedelta

# Global Variables
SAMPLE_NAME = 'Sample1'  # Prefix for the sample
NUM_SUBSAMPLES = 3       # Number of subsamples to create
TIME_DELTA_INITIAL = 60  # Initial time delta between scans (in seconds)
# Time delta for appending after the first split (e.g., 5 minutes or 300 seconds)
TIME_DELTA_APPEND = 300

# Folder path where the files are located
FOLDER_PATH = 'Raw_Data\\'+SAMPLE_NAME
EXPORT_PATH = 'Datasets'

# List all files that start with the SAMPLE_NAME and have a .csv extension
files = [f for f in os.listdir(FOLDER_PATH) if f.startswith(
    SAMPLE_NAME) and f.endswith('.csv')]

# Sort files by timestamp in the filename, handling only .csv extensions
files = sorted(files, key=lambda x: datetime.strptime(
    x.replace('.csv', '').split('_')[4], '%H%M%S'))

# Function to extract sample information from the filename


def extract_sample_info(filename):
    # Remove the .csv extension from the filename
    filename = filename.replace('.csv', '')

    parts = filename.split('_')
    sample_name = parts[0]
    protocol = parts[1]
    scan_number = parts[2]
    scan_date = parts[3]
    scan_time = parts[4]

    # Parse the datetime of the scan
    scan_datetime_str = scan_date + scan_time
    scan_datetime = datetime.strptime(scan_datetime_str, '%y%m%d%H%M%S')

    return sample_name, protocol, scan_number, scan_datetime


# Create empty subsample containers
subsample_dfs = {i: pd.DataFrame() for i in range(1, NUM_SUBSAMPLES + 1)}

# Initialize variables for tracking subsample indices and timepoints
timepoints = []
current_subsample_idx = 1

# Process each file
for i, file in enumerate(files):
    file_path = os.path.join(FOLDER_PATH, file)

    # Extract the datetime from the filename
    sample_name, protocol, scan_number, scan_datetime = extract_sample_info(
        file)

    # Read the CSV file, skipping the first 21 lines and using the 22nd line as headers
    df = pd.read_csv(file_path, skiprows=21)

    # If this is the first file, initialize the timepoints list
    if i == 0:
        timepoints.append(scan_datetime)

    # Assign files to subsamples based on NUM_SUBSAMPLES and TIME_DELTA
    if len(subsample_dfs[current_subsample_idx]) == 0:
        # For the first scan, add the "Wavelength (nm)" and "Absorbance (AU)" columns
        subsample_dfs[current_subsample_idx] = df[[
            'Wavelength (nm)', 'Absorbance (AU)']].copy()
        subsample_dfs[current_subsample_idx].rename(
            columns={'Absorbance (AU)': f'Absorbance (AU) {scan_datetime}'}, inplace=True)
    else:
        # Check if the time difference between this scan and the first scan exceeds the TIME_DELTA_APPEND
        if scan_datetime - timepoints[current_subsample_idx - 1] >= timedelta(seconds=TIME_DELTA_APPEND):
            # Append new absorbance column to the current subsample
            subsample_dfs[current_subsample_idx][f'Absorbance (AU) {scan_datetime}'] = df[
                'Absorbance (AU)'].values

            # Update the timepoint for this subsample
            timepoints[current_subsample_idx - 1] = scan_datetime

        # Move to the next subsample if we've reached the scan limit for the current one
        if len(subsample_dfs[current_subsample_idx].columns) > NUM_SUBSAMPLES:
            current_subsample_idx += 1

            # Add the current scan to the next subsample
            if current_subsample_idx <= NUM_SUBSAMPLES:
                subsample_dfs[current_subsample_idx] = df[[
                    'Wavelength (nm)', 'Absorbance (AU)']].copy()
                subsample_dfs[current_subsample_idx].rename(
                    columns={'Absorbance (AU)': f'Absorbance (AU) {scan_datetime}'}, inplace=True)

# After processing, save each subsample to a CSV file
for subsample_idx, subsample_df in subsample_dfs.items():
    subsample_name = f"{SAMPLE_NAME}_{subsample_idx}.csv"
    export_file_path = os.path.join(EXPORT_PATH, subsample_name)
    subsample_df.to_csv(export_file_path, index=False)
    print(f"Saved subsample: {export_file_path}")
