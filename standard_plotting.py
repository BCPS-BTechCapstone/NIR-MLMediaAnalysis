import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Declare the X and Y axis columns
X_AXIS = "Wavelength (nm)"
Y_AXIS = "Absorbance (AU)"

# Folder path where the files are located
FOLDER_PATH = 'Ethanol_Standards'

# List all files in the folder
files = [f for f in os.listdir(FOLDER_PATH) if f.startswith('Ethanol')]

print("Files in folder:", files)

# Initialize an empty dictionary to hold the dataframes by sample
ethanol_data = {}

# Process each file
for file in files:
    file_path = os.path.join(FOLDER_PATH, file)

    # Extract the ethanol sample number and replicate information from the filename
    filename_parts = file.split('_')
    sample_name = filename_parts[0]

    # Read the CSV file starting at row 22
    print(f"Processing file: {file}, extracted sample name: {sample_name}")
    df = pd.read_csv(file_path, skiprows=21)

    # Append the dataframe to the correct ethanol sample group
    if sample_name not in ethanol_data:
        ethanol_data[sample_name] = []
    ethanol_data[sample_name].append(df)

# Averaging and normalizing the data for each sample
averaged_data = {}

for sample_name, dfs in ethanol_data.items():
    # Check if X_AXIS exists in the data
    if all(X_AXIS in df.columns and Y_AXIS in df.columns for df in dfs):

        # Concatenate all replicates on the X_AXIS column and average the Y_AXIS values
        combined_df = pd.concat([df.set_index(X_AXIS)[Y_AXIS] for df in dfs], axis=1)
        averaged_df = combined_df.mean(axis=1).reset_index()  # Averaging the Y_AXIS across replicates

        # Rename the averaged Y_AXIS column
        averaged_df.rename(columns={0: Y_AXIS}, inplace=True)

        # Print the columns in the averaged_df to confirm the correct renaming
        print(f"Sample: {sample_name}, Columns in averaged_df: {averaged_df.columns}")

        # Normalize the averaged Y_AXIS values between 0 and 1
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