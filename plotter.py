import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import datetime
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import matplotlib.colors as mcolors

# Load the data from the uploaded file
file_path = 'Datasets/Sample1.csv'
data = pd.read_csv(file_path)

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
norm = mcolors.Normalize(vmin=min(Z), vmax=max(Z))
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

plt.show()
'''
import os
output_filename = os.path.splitext(os.path.basename(file_path))[0] + '_plot.png'
plt.savefig(output_filename, dpi=600)
plt.close()
'''
