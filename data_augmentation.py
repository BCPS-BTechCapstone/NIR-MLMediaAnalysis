from re import S
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import argparse

# Function to extract sample information from the filename
def extract_sample_info(filename):
    filename = filename.replace('.csv', '')
    parts = filename.split('_')
    sample_name = parts[0]
    protocol = parts[1]
    scan_number = parts[2]
    scan_date = parts[3]  # Format is YYYYMMDD (e.g., 20240912)
    scan_time = parts[4]  # Format is HHMMSS (e.g., 155429)

    scan_datetime_str = scan_date + scan_time
    try:
        scan_datetime = datetime.strptime(scan_datetime_str, '%Y%m%d%H%M%S')
        return sample_name, protocol, scan_number, scan_datetime
    except ValueError as e:
        print(f"\tError occurred while parsing: {e}")
        raise

# Function to calculate the total number of scans and the total duration of the scans
def calculate_total_scans_and_duration(files):
    scan_datetimes = []
    
    for file in files:
        _, _, _, scan_datetime = extract_sample_info(file)
        scan_datetimes.append(scan_datetime)
    
    # Sort the datetime objects
    scan_datetimes.sort()

    # Calculate total number of scans
    total_scans = len(scan_datetimes)
    
    # Calculate the duration between the first and the last scan
    if total_scans > 1:
        total_duration = scan_datetimes[-1] - scan_datetimes[0]
    else:
        total_duration = timedelta(0)

    return total_scans, total_duration

# Function to delete existing subsample files for the current sample
def delete_existing_subsample_files(sample_name, export_path, auto_confirm):
    # List all files in the export path that start with the sample name
    files_to_delete = [f for f in os.listdir(export_path) if f.startswith(sample_name+'_') and f.endswith('.csv')]
    
    # If there are no files to delete, return
    if not files_to_delete:
        return

    # Ask for confirmation if auto_confirm is False
    if not auto_confirm:
        confirm = input(f"Are you sure you want to delete {len(files_to_delete)} subsample file(s) for '{sample_name}'? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("\tAborting deletion.")
            return

    # Delete each of these files
    #print('Deleting old files:')
    for file in files_to_delete:
        file_path = os.path.join(export_path, file)
        try:
            os.remove(file_path)
            print(f"\tDeleted old subsample file: {file_path}")
        except Exception as e:
            print(f"\tError deleting file {file_path}: {e}")
    print('')

def main(args, sample):
    SAMPLE_NAME = sample
    NUM_SUBSAMPLES = args.subsamples
    TIME_DELTA_INITIAL = args.initial_delta
    TIME_DELTA_APPEND = args.append_delta
    AUTO_CONFIRM = args.yes

    print(f'\nProcessing {SAMPLE_NAME}:\n')

    # Get the absolute paths relative to the script directory
    FOLDER_PATH = os.path.join(args.folder_path, SAMPLE_NAME)
    EXPORT_PATH = os.path.join(args.export_path)

    if not os.path.exists(EXPORT_PATH):
        os.makedirs(EXPORT_PATH)

    # List all .csv files that start with the SAMPLE_NAME from the folder path
    files = [f for f in os.listdir(FOLDER_PATH) if f.startswith(SAMPLE_NAME) and f.endswith('.csv')]

    # Check if there are any valid input files
    if not files:
        print(f"\tNo input files found for sample name '{SAMPLE_NAME}' in folder '{FOLDER_PATH}'. Please check the file paths.")
        return

    # Calculate the total number of scans and total duration
    total_scans, total_duration = calculate_total_scans_and_duration(files)
    #print('Scan Info:')
    print(f"\tTotal number of scans: {total_scans}")
    print(f"\tTotal duration of scans: {total_duration}")
    print('')

    # Delete existing subsample files before exporting new ones
    delete_existing_subsample_files(SAMPLE_NAME, EXPORT_PATH, AUTO_CONFIRM)

    # If NUM_SUBSAMPLES is 0, combine all data into one file without augmentation
    if NUM_SUBSAMPLES == 0:
        combined_df = None
        for file in files:
            file_path = os.path.join(FOLDER_PATH, file)
            df = pd.read_csv(file_path, skiprows=21)
            if combined_df is None:
                combined_df = df[['Wavelength (nm)', 'Absorbance (AU)']].copy()
                combined_df.rename(columns={'Absorbance (AU)': f'Absorbance (AU) {extract_sample_info(file)[3]}'}, inplace=True)
            else:
                combined_df[f'Absorbance (AU) {extract_sample_info(file)[3]}'] = df['Absorbance (AU)'].astype(float).values
        
        export_file_path = os.path.join(EXPORT_PATH, f"{SAMPLE_NAME}.csv")
        combined_df.to_csv(export_file_path, index=False)
        print(f"\tExported combined original data without augmentation: {export_file_path}")
        return

    # Sort files by timestamp in the filename
    files = sorted(files, key=lambda x: datetime.strptime(x.replace('.csv', '').split('_')[3] + x.replace('.csv', '').split('_')[4], '%Y%m%d%H%M%S'))


    subsample_dfs = {i: pd.DataFrame() for i in range(1, NUM_SUBSAMPLES + 1)}
    scan_counts = {i: 0 for i in range(1, NUM_SUBSAMPLES + 1)}  # To track the number of scans added
    start_times = {i: None for i in range(1, NUM_SUBSAMPLES + 1)}  # Track start times
    end_times = {i: None for i in range(1, NUM_SUBSAMPLES + 1)}  # Track end times
    timepoints = [None] * NUM_SUBSAMPLES  # Initialize timepoints for each subsample
    current_offsets = [i * TIME_DELTA_INITIAL for i in range(NUM_SUBSAMPLES)]  # Initial offsets

    # Initialize global max and min values
    global_max = float('-inf')
    global_min = float('inf')

    # Process each scan
    for i, file in enumerate(files):
        file_path = os.path.join(FOLDER_PATH, file)
        sample_name, protocol, scan_number, scan_datetime = extract_sample_info(file)

        df = pd.read_csv(file_path, skiprows=21)

        # Check if the DataFrame is empty
        if df.empty:
            print(f"\tWarning: The input file '{file}' contains no data after skipping rows. Skipping this file.")
            continue

        # Determine which subsample this scan belongs to by using current offset for each subsample
        for subsample_idx in range(1, NUM_SUBSAMPLES + 1):
            current_offset = current_offsets[subsample_idx - 1]  # Get the current offset for this subsample

            # Calculate the difference in seconds if there is an existing timepoint
            if timepoints[subsample_idx - 1] is None or (scan_datetime - timepoints[subsample_idx - 1]).total_seconds() >= current_offset:
                # Collect new data for the subsample in a list
                new_columns = []

                if subsample_dfs[subsample_idx].empty:
                    # Create the initial DataFrame with 'Wavelength (nm)' and the first Absorbance column
                    subsample_dfs[subsample_idx] = df[['Wavelength (nm)']].copy()
                    new_columns.append(pd.DataFrame({f'Absorbance (AU) {scan_datetime}': df['Absorbance (AU)'].astype(float).values}))
                else:
                    # Append new absorbance column to the list
                    new_columns.append(pd.DataFrame({f'Absorbance (AU) {scan_datetime}': df['Absorbance (AU)'].astype(float).values}))

                # Concatenate all new columns at once
                if new_columns:
                    new_columns_df = pd.concat(new_columns, axis=1)
                    subsample_dfs[subsample_idx] = pd.concat([subsample_dfs[subsample_idx], new_columns_df], axis=1).copy()

                # Update timepoints, scan count, and start/end times
                if start_times[subsample_idx] is None:
                    start_times[subsample_idx] = scan_datetime  # Set start time for this subsample
                end_times[subsample_idx] = scan_datetime  # Update the end time for this subsample
                timepoints[subsample_idx - 1] = scan_datetime
                scan_counts[subsample_idx] += 1
                current_offsets[subsample_idx - 1] = TIME_DELTA_APPEND  # Switch to appending with the append delta
                break

    # Update global max and min values after all subsamples are created
    for subsample_df in subsample_dfs.values():
        if not subsample_df.empty:
            current_max = subsample_df.iloc[:, 1:].max().max()  # Find max across all columns except 'Wavelength (nm)'
            current_min = subsample_df.iloc[:, 1:].min().min()  # Find min across all columns except 'Wavelength (nm)'
            if current_max > global_max:
                global_max = current_max
            if current_min < global_min:
                global_min = current_min

    # Print global max and min values before normalization
    print(f"\tGlobal max value: {global_max}")
    print(f"\tGlobal min value: {global_min}")
    print('\tNormalizing the data...\n')

    # Save each subsample to a CSV file in the EXPORT_PATH
    #print("Saving Subsamples:")
    method = args.method
    for subsample_idx, subsample_df in subsample_dfs.items():
        if method is not None:
            method = args.method.strip().lower()
            subsample_name = f"{SAMPLE_NAME}_{subsample_idx}-{method}.csv"
        else:
            subsample_name = f"{SAMPLE_NAME}_{subsample_idx}.csv"

        export_file_path = os.path.join(EXPORT_PATH, subsample_name)
        if not subsample_df.empty:
            # Normalize the data before exporting using the specified method
            if method == 'n':
                # Custom normalization
                subsample_df.iloc[:, 1:] = subsample_df.iloc[:, 1:].apply(lambda x: (x - global_min) / (global_max - global_min))
            elif method == 's':
                # Scikit-learn MinMaxScaler
                scaler = MinMaxScaler()
                subsample_df.iloc[:, 1:] = scaler.fit_transform(subsample_df.iloc[:, 1:])

            subsample_df.to_csv(export_file_path, index=False)
            # Calculate total time delta for the subsample
            total_time_delta = end_times[subsample_idx] - start_times[subsample_idx] if start_times[subsample_idx] and end_times[subsample_idx] else timedelta(0)
            print(f"\tSaved subsample: {subsample_name} with {scan_counts[subsample_idx]} scan(s) and total time delta of {total_time_delta}.")
        else:
            print(f"\tWarning: Subsample {subsample_idx} contains no data and will not be saved.")
    print('')

if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser(
        description="Process time-series CSV files for a specific sample.")

    # Positional argument for sample name
    parser.add_argument('sample_name', type=str, nargs='+',
                        help='The sample name (e.g., "Sample1")')

    # Optional arguments (with defaults)
    parser.add_argument('-s', '--subsamples', type=int, default=3,
                        help='Number of subsamples to create (default: 3)')
    parser.add_argument('-i', '--initial-delta', type=int, default=60,
                        help='Initial time delta between scans in seconds (default: 60)')
    parser.add_argument('-d', '--append-delta', type=int, default=900,
                        help='Time delta for appending in seconds after the first split (default: 900)')
    parser.add_argument('-m', '--method', type=str, default='n', choices=['n','s'], 
                        help='Normalization method: n for normalization, s for MinMaxScaler')
    parser.add_argument('-f', '--folder-path', type=str, default='Raw_Data',
                        help='Relative path to the folder containing the files (default: Raw_Data)')
    parser.add_argument('-e', '--export-path', type=str, default='Datasets',
                        help='Relative path to export the processed subsamples (default: Datasets)')
    parser.add_argument('-y', '--yes', action='store_true', help='Automatically confirm deletion of existing subsample files')

    args = parser.parse_args()
    
    main(args)
