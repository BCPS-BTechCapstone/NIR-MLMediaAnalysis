import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Declare the X and Y axis columns
X_AXIS = "Wavelength (nm)"
Y_AXIS = "Absorbance (AU)"

# Global variables for target PGA gain and blank subtraction
TARGET_PGA_GAIN = 64  # Adjust this as needed
USE_BLANK_SUBTRACTION = False  # Toggle blank subtraction on/off

# Global variables for file prefixes
ETHANOL_PREFIX = 'Ethanol'  # Prefix to filter ethanol files
BLANKS_PREFIX = 'Quartz_'   # Prefix to filter blank files

# Folder paths where the files are located
ETHANOL_FOLDER_PATH = 'Ethanol_Standards'
BLANKS_FOLDER_PATH = 'Blanks'

# Function to normalize the absorbance based on the PGA gain
def normalize_by_pga(df, pga_gain):
    normalization_factor = pga_gain / TARGET_PGA_GAIN
    print(df.head())
    df[Y_AXIS] = df[Y_AXIS] * normalization_factor
    return df

# List all files in the ethanol folder that match the ETHANOL_PREFIX
ethanol_files = [f for f in os.listdir(ETHANOL_FOLDER_PATH) if f.startswith(ETHANOL_PREFIX)]

# List all files in the blanks folder that match the BLANKS_PREFIX
blanks_files = [f for f in os.listdir(BLANKS_FOLDER_PATH) if f.startswith(BLANKS_PREFIX)]

# Initialize an empty dictionary to hold the dataframes by sample
ethanol_data = {}
blanks_data = []

# Process each ethanol file
for file in ethanol_files:
    file_path = os.path.join(ETHANOL_FOLDER_PATH, file)

    # Extract the ethanol sample number and replicate information from the filename
    filename_parts = file.split('_')
    sample_name = filename_parts[0]

    # Read the CSV file starting at row 22
    df = pd.read_csv(file_path, skiprows=21)

    # Extract the PGA gain from row 20 (index 19 in zero-indexed)
    with open(file_path, 'r') as f:
        metadata = f.readlines()
    pga_gain = float(metadata[19].split(',')[1])

    # Normalize the data using the sample's PGA gain
    df = normalize_by_pga(df, pga_gain)

    # Append the dataframe to the correct ethanol sample group
    if sample_name not in ethanol_data:
        ethanol_data[sample_name] = []
    ethanol_data[sample_name].append(df)

# Process each blanks file if USE_BLANK_SUBTRACTION is enabled
if USE_BLANK_SUBTRACTION:
    for file in blanks_files:
        file_path = os.path.join(BLANKS_FOLDER_PATH, file)

        # Read the CSV file starting at row 22
        df = pd.read_csv(file_path, skiprows=21)

        # Extract the PGA gain from row 20 (index 19 in zero-indexed)
        with open(file_path, 'r') as f:
            metadata = f.readlines()
        pga_gain = float(metadata[19].split(',')[1])

        # Normalize the data using the sample's PGA gain
        df = normalize_by_pga(df, pga_gain)

        # Append the dataframe to the blanks list
        blanks_data.append(df)

# Averaging the blanks data if blank subtraction is enabled
if USE_BLANK_SUBTRACTION and blanks_data:
    combined_blanks_df = pd.concat([df.set_index(X_AXIS)[Y_AXIS] for df in blanks_data], axis=1)
    averaged_blanks_df = combined_blanks_df.mean(axis=1).reset_index()  # Averaging the Y_AXIS values
    averaged_blanks_df.rename(columns={0: Y_AXIS}, inplace=True)

# Averaging, shifting by blanks, and normalizing the data for each sample
averaged_data = {}

for sample_name, dfs in ethanol_data.items():
    # Check if X_AXIS exists in the data
    if all(X_AXIS in df.columns and Y_AXIS in df.columns for df in dfs):
        
        # Concatenate all replicates on the X_AXIS column and average the Y_AXIS values
        combined_df = pd.concat([df.set_index(X_AXIS)[Y_AXIS] for df in dfs], axis=1)
        averaged_df = combined_df.mean(axis=1).reset_index()  # Averaging the Y_AXIS across replicates
        
        # Rename the averaged Y_AXIS column
        averaged_df.rename(columns={0: Y_AXIS}, inplace=True)

        # Shift the ethanol data by the averaged blanks (blanks act as zero baseline) if blank subtraction is enabled
        if USE_BLANK_SUBTRACTION and not averaged_blanks_df.empty:
            averaged_df[Y_AXIS] = averaged_df[Y_AXIS] - averaged_blanks_df[Y_AXIS]

        # Normalize the shifted Y_AXIS values between 0 and 1
        scaler = MinMaxScaler()
        averaged_df['Normalized ' + Y_AXIS] = scaler.fit_transform(averaged_df[[Y_AXIS]])
        
        # Store the averaged and normalized data
        averaged_data[sample_name] = averaged_df

# Plotting the averaged and normalized data for each sample
plt.figure(figsize=(12, 8))

for sample_name, df in averaged_data.items():
    if 'Normalized ' + Y_AXIS in df.columns:
        x_values = df[X_AXIS]
        y_values = df['Normalized ' + Y_AXIS]
        
        # Plot the normalized averaged data
        plt.plot(x_values, y_values, label=f'{sample_name} - Averaged')

# Customizing the plot
plt.title(f'Normalized Averaged {Y_AXIS} vs {X_AXIS} for Ethanol Samples')
plt.xlabel(X_AXIS)
plt.ylabel(f'Normalized {Y_AXIS}')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
