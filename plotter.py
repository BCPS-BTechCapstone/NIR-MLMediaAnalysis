import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import datetime
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import matplotlib.colors as mcolors
import os
import argparse

def plot_data(data, output_filename):
    # Set seaborn theme
    sns.set_theme(style="whitegrid")

    # Extract timestamps from the original headers
    time_columns = data.columns[1:]
    timestamps = [datetime.datetime.strptime(col.split()[-2] + ' ' + col.split()[-1], '%Y-%m-%d %H:%M:%S') for col in time_columns]

    # Calculate time deltas relative to the first timestamp
    time_deltas = [(ts - timestamps[0]).total_seconds() / 60 for ts in timestamps]  # Convert to minutes

    # Update column names to reflect the correct time deltas without the "Absorbance (AU)" text
    new_columns = ['Wavelength (nm)'] + [f'{delta:.2f} min' for delta in time_deltas]
    data.columns = new_columns

    # Plot the data in 3D
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(top=1.0,
                        bottom=0.0,
                        left=0.0,
                        right=0.845,
                        hspace=0.2,
                        wspace=0.2)  # Adjust to reduce whitespace

    # Extracting data for plotting
    X = np.array(data['Wavelength (nm)'])
    Z = np.array(time_deltas)

    # Normalizing the color map
    norm = mcolors.Normalize(vmin=min(Z), vmax(max(Z)))
    cmap = cm.viridis

    # Loop to ensure dimensions match for plotting
    for i, col in enumerate(data.columns[1:]):
        Y = np.array(data[col])
        if len(X) == len(Y):
            ax.plot(X, [Z[i]] * len(X), Y, color=cmap(norm(Z[i])), alpha=0.7)

    # Set axis labels and initial viewing angle
    ax.set_xlabel('Wavelength (nm)', labelpad=10)
    ax.set_ylabel('Time (min)', labelpad=10)
    ax.set_zlabel('Absorbance (AU)', labelpad=10)
    ax.view_init(elev=0, azim=-90)  # Set initial view so that time axis is prominent and labels appear on the right

    # Add color bar to indicate time progression
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, fraction=0.02, pad=0.1, label='Time (min)')

    # Save the plot to the output directory
    plt.savefig(output_filename, dpi=600)
    plt.close()
    print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting Script for 3D Visualization of Absorbance Data")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file (e.g., "Sample1.csv")')
    parser.add_argument('--output_dir', type=str, default='Plots', help='Directory to save the plot (default: Plots)')

    args = parser.parse_args()

    # Load the data from the input file
    data = pd.read_csv(args.input_file)

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename based on the input filename
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0] + '_3d_plot.png'
    output_filename = os.path.join(args.output_dir, base_filename)

    # Call the plot_data function
    plot_data(data, output_filename)
