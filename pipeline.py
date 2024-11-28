import argparse
import os
import pandas as pd
from data_augmentation import main as data_augmentation_main  # Import the main function from data_augmentation
from plotter import plot_data  # Import the function from plotter.py

def main(args):
    for num in range(args.sample_start, args.sample_end+1):
        sample = args.sample_name + str(num)
        # Run the data augmentation process with the args Namespace
        data_augmentation_main(args, sample)

        # Set up the plot directory
        plots_dir = 'Plots'
        os.makedirs(plots_dir, exist_ok=True)

        # Iterate over all the subsample files generated by data augmentation
        subsample_files = [f for f in os.listdir(args.export_path) if f.startswith(sample+"_") and f.endswith('.csv')]
        for subsample_file in subsample_files:
            subsample_path = os.path.join(args.export_path, subsample_file)
            
            # Check if the subsample file is empty and skip if it is
            if os.path.getsize(subsample_path) == 0:
                print(f"Skipping empty subsample file: {subsample_file}")
                continue

            data = pd.read_csv(subsample_path)
            
            # If data is empty after reading, skip further processing
            if data.empty:
                print(f"Skipping empty subsample file: {subsample_file}")
                continue

            # Generate output filename for each subsample plot
            base_filename = os.path.basename(subsample_path).replace('.csv', '_plot.'+args.type)
            output_filename = os.path.join(plots_dir, base_filename)

            elev, azim = map(float, [angle.strip() for angle in args.angles.split(',')])

            # Call the plot_data function for each subsample
            if args.no_plot is not True:
                plot_data(data, output_filename, elev, azim, args.view, args.method)

if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser(
        description="Process time-series CSV files for a specific sample.")

    # Positional argument for sample name
    parser.add_argument('sample_name', type=str,
                        help='The sample name (e.g., "Sample1")')
    parser.add_argument('sample_start', type=int, help='Start value for sample numbers')
    parser.add_argument('sample_end', type=int, help='End value for sample numbers ')

    # Optional arguments (with defaults)
    parser.add_argument('-s', '--subsamples', type=int, default=3,
                        help='Number of subsamples to create (default: 3)')
    parser.add_argument('-i', '--initial-delta', type=int, default=60,
                        help='Initial time delta between scans in seconds (default: 60)')
    parser.add_argument('-d', '--append-delta', type=int, default=900,
                        help='Time delta for appending in seconds after the first split (default: 900)')
    parser.add_argument('-m', '--method', type=str, default=None, choices=['n','s'], 
                        help='Normalization method: n for normalization, s for MinMaxScaler')
    parser.add_argument('-p', '--folder-path', type=str, default='Raw_Data',
                        help='Relative path to the folder containing the files (default: Raw_Data)')
    parser.add_argument('-e', '--export-path', type=str, default='Datasets',
                        help='Relative path to export the processed subsamples (default: Datasets)')
    parser.add_argument('-y', '--yes', action='store_true', help='Automatically confirm deletion of existing subsample files')
    parser.add_argument('--noise', action='store_true', help='Automatically confirm deletion of existing subsample files')

    parser.add_argument('-a', '--angles', type=str, default='0,-90',
                        help='Elevation and azimuth angles for the 3D plot view as a comma-separated list (default: "0,-90" , recommended: "5,-140")')
    parser.add_argument('-v', '--view', action='store_true',
                        help='Show the plot instead of saving it')
    parser.add_argument('-t', '--type', type=str, default='png', choices=['png','pgf'], 
                        help='File type of the plots')
    parser.add_argument('--no-plot', action='store_true',
                        help='Don\'t perform plotting operation')
    
    args = parser.parse_args()
    main(args)
    print('')
