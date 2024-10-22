import pandas as pd
import os
from datetime import datetime, timedelta
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
        print(f"Error occurred while parsing: {e}")
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
    files_to_delete = [f for f in os.listdir(export_path) if f.startswith(sample_name) and f.endswith('.csv')]
    
    # If there are no files to delete, return
    if not files_to_delete:
        return

    # Ask for confirmation if auto_confirm is False
    if not auto_confirm:
        confirm = input(f"Are you sure you want to delete {len(files_to_delete)} subsample file(s) for '{sample_name}'? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Aborting deletion.")
            return

    # Delete each of these files
    for file in files_to_delete:
        file_path = os.path.join(export_path, file)
        try:
            os.remove(file_path)
            print(f"Deleted old subsample file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def main():
    # Setup argparse
    parser = argparse.ArgumentParser(
        description="Process time-series CSV files for a specific sample.")

    # Positional argument for sample name
    parser.add_argument('sample_name', type=str,
                        help='The sample name (e.g., "Sample1")')

    # Optional arguments (with defaults)
    parser.add_argument('-s', '--subsamples', type=int, default=3,
                        help='Number of subsamples to create (default: 3)')
    parser.add_argument('-i', '--initial_delta', type=int, default=60,
                        help='Initial time delta between scans in seconds (default: 60)')
    parser.add_argument('-d', '--append_delta', type=int, default=900,
                        help='Time delta for appending in seconds after the first split (default: 900)')
    parser.add_argument('-p', '--folder_path', type=str, default='Raw_Data',
                        help='Relative path to the folder containing the files (default: Raw_Data)')
    parser.add_argument('-e', '--export_path', type=str, default='Datasets',
                        help='Relative path to export the processed subsamples (default: Datasets)')
    parser.add_argument('-y', '--yes', action='store_true', help='Automatically confirm deletion of existing subsample files')

    args = parser.parse_args()

    SAMPLE_NAME = args.sample_name
    NUM_SUBSAMPLES = args.subsamples
    TIME_DELTA_INITIAL = args.initial_delta
    TIME_DELTA_APPEND = args.append_delta
    AUTO_CONFIRM = args.yes

    # Get the absolute paths relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    FOLDER_PATH = os.path.join(script_dir, args.folder_path, SAMPLE_NAME)
    EXPORT_PATH = os.path.join(script_dir, args.export_path)

    if not os.path.exists(EXPORT_PATH):
        os.makedirs(EXPORT_PATH)

    # List all .csv files that start with the SAMPLE_NAME
    files = [f for f in os.listdir(FOLDER_PATH) if f.startswith(SAMPLE_NAME) and f.endswith('.csv')]

    # Calculate the total number of scans and total duration
    total_scans, total_duration = calculate_total_scans_and_duration(files)
    print(f"Total number of scans: {total_scans}")
    print(f"Total duration of scans: {total_duration}")

    # Delete existing subsample files before exporting new ones
    delete_existing_subsample_files(SAMPLE_NAME, EXPORT_PATH, AUTO_CONFIRM)

    # Sort files by timestamp in the filename
    files = sorted(files, key=lambda x: datetime.strptime(x.replace('.csv', '').split('_')[4], '%H%M%S'))

    subsample_dfs = {i: pd.DataFrame() for i in range(1, NUM_SUBSAMPLES + 1)}
    scan_counts = {i: 0 for i in range(1, NUM_SUBSAMPLES + 1)}  # To track the number of scans added
    start_times = {i: None for i in range(1, NUM_SUBSAMPLES + 1)}  # Track start times
    end_times = {i: None for i in range(1, NUM_SUBSAMPLES + 1)}  # Track end times
    timepoints = [None] * NUM_SUBSAMPLES  # Initialize timepoints for each subsample
    current_offsets = [i * TIME_DELTA_INITIAL for i in range(NUM_SUBSAMPLES)]  # Initial offsets

    # Process each scan
    for i, file in enumerate(files):
        file_path = os.path.join(FOLDER_PATH, file)
        sample_name, protocol, scan_number, scan_datetime = extract_sample_info(file)

        df = pd.read_csv(file_path, skiprows=21)

        # Determine which subsample this scan belongs to by using current offset for each subsample
        for subsample_idx in range(1, NUM_SUBSAMPLES + 1):
            current_offset = current_offsets[subsample_idx - 1]  # Get the current offset for this subsample

            # Check if this scan matches the current offset for the subsample
            if timepoints[subsample_idx - 1] is None or scan_datetime - timepoints[subsample_idx - 1] >= timedelta(seconds=current_offset):
                if subsample_dfs[subsample_idx].empty:
                    subsample_dfs[subsample_idx] = df[['Wavelength (nm)', 'Absorbance (AU)']].copy()
                    subsample_dfs[subsample_idx].rename(columns={'Absorbance (AU)': f'Absorbance (AU) {scan_datetime}'}, inplace=True)
                else:
                    subsample_dfs[subsample_idx][f'Absorbance (AU) {scan_datetime}'] = df['Absorbance (AU)'].values

                # Update timepoints, scan count, and start/end times
                if start_times[subsample_idx] is None:
                    start_times[subsample_idx] = scan_datetime  # Set start time for this subsample
                end_times[subsample_idx] = scan_datetime  # Update the end time for this subsample
                timepoints[subsample_idx - 1] = scan_datetime
                scan_counts[subsample_idx] += 1
                current_offsets[subsample_idx - 1] = TIME_DELTA_APPEND  # Switch to appending with the append delta
                break

    # Save each subsample to a CSV file in the EXPORT_PATH
    for subsample_idx, subsample_df in subsample_dfs.items():
        subsample_name = f"{SAMPLE_NAME}_{subsample_idx}.csv"
        export_file_path = os.path.join(EXPORT_PATH, subsample_name)
        subsample_df.to_csv(export_file_path, index=False)
        
        # Calculate total time delta for the subsample
        total_time_delta = end_times[subsample_idx] - start_times[subsample_idx] if start_times[subsample_idx] and end_times[subsample_idx] else timedelta(0)
        print(f"Saved subsample: {subsample_name} with {scan_counts[subsample_idx]} scan(s) and total time delta of {total_time_delta}.")


if __name__ == "__main__":
    main()
