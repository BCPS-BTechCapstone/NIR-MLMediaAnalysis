# Online NIR Cell Culture Analysis Using 1D-CNN

## Project Information

### Full Title

Non-intrusive Testing of Liquid Culture Medium using Online NIR Spectroscopy and Machine Learning for Qualitative Analysis

### Contributers

Benjamin Samuel\
Connor Reintjes\
Paola Gonzalez Perez\
Shiza Hassan

### Supervisor

Dr. Amin R. Rajabzadeh

### Association

Bachelor of Technology in Bioinformatics:\
Capstone Project for 4TR1 & 4TR3

McMaster Univeristy - W. Booth School of Engineering Practice and Technology

## Purpose

Implement  NIR spectroscopy and machine learning for the detection and real-time analysis of media within cultures for continuous quality testing.

## Overview

This project implements a 1D Convolutional Neural Network (CNN) to detect contamination in Near-Infrared (NIR) spectroscopy data. The model is trained using time-series data from CSV files, with the aim of differentiating between contaminated and non-contaminated samples. The dataset is augmented and preprocessed before being fed into the CNN for training and evaluation.

The pipeline tool is designed to automate the processing, augmentation, and visualization of time-series CSV data. It takes input sample names, performs data augmentation to generate subsamples, and generates 3D plots for each subsample. This script is ideal for analyzing time-series datasets and visualizing the processed results.

### Features

- **Data Augmentation**: Creates subsamples from time-series data based on user-defined parameters.
- **Data Normalization**: Supports optional data normalization using different techniques.
- **3D Data Visualization**: Plots the subsampled data in 3D for better visualization.
- **Automated Workflow**: Automates file management, processing, and plotting.
- **Filename Parsing**: Extracts sample information such as sample name, protocol, scan number, and scan timestamp from the filenames.
- **Scan Statistics**: Calculates the total number of scans and the total duration of the scans.
- **Subsample Deletion**: Deletes existing subsample files for a specific sample to ensure the latest processing results are saved.
- **Data Combination**: Combines all scans into one file without augmentation if specified by the user.
- **Global Normalization**: Computes global maximum and minimum values for normalization across all subsamples.
- **Data Loading and Preprocessing**: Loads NIR spectroscopy data from CSV files and preprocesses the data by padding and normalizing.
- **1D Convolutional Neural Network**: Implements a 1D CNN for binary classification to distinguish between contaminated and non-contaminated samples.
- **K-Fold Cross-Validation**: Uses Stratified K-Fold cross-validation to evaluate model performance on different subsets of the data.
- **Model Checkpoints**: Saves the best model during training using the validation loss as a metric.
- **Visualization**: Plots validation loss, accuracy, confusion matrix, and Receiver Operating Characteristic (ROC) curve to evaluate the model.

### Requirements

- Python 3.x
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow/Keras
- etc...

Create a Conda environment using the provided NIRenv.yml file by running:

```bash
conda env create -f NIRenv.yml 
```

### Model Architecture

- **Input Layer**: Accepts time-series input data.
- **Conv1D Layers**: Two convolutional layers with ReLU activation for feature extraction.
- **MaxPooling1D Layers**: Downsamples the output to reduce dimensionality.
- **Dropout Layers**: Reduces overfitting by randomly setting input units to zero during training.
- **Flatten Layer**: Flattens the output to prepare for dense layers.
- **Dense Layers**: Two dense layers, including a final sigmoid layer for binary classification.

### Pipeline Usage

The script can be executed using the following command:

```bash
python pipeline.py <sample_name> [options]
```

#### Positional Argument

- `<sample_name>`: One or more sample names for processing. Example: `Sample1` or `Sample1 Sample2`

#### Optional Arguments

- `-s`, `--subsamples`: Number of subsamples to create (default: 3).
- `-i`, `--initial-delta`: Initial time delta between scans in seconds (default: 60).
- `-d`, `--append-delta`: Time delta for appending after the first split in seconds (default: 900).
- `-m`, `--method`: Normalization method: `n` for normalization, `s` for MinMaxScaler (default: None).
- `-p`, `--folder-path`: Relative path to the folder containing the raw data files (default: `Raw_Data`).
- `-e`, `--export-path`: Relative path to export the processed subsamples (default: `Datasets`).
- `-y`, `--yes`: Automatically confirm deletion of existing subsample files.
- `-a`, `--angles`: Elevation and azimuth angles for the 3D plot view as a comma-separated list (default: `0,-90`).
- `-v`, `--view`: Show the plot instead of saving it.

#### Example

To process a sample named `Sample1` and create 3D plots:

```bash
python pipeline.py Sample1 -s 5 -i 120 -d 1800 -m n -p Data -e Processed -a 5,-140
```

This command will:

- Process the sample named `Sample1`
- Create 5 subsamples
- Use 120 seconds for the initial time delta and 1800 seconds for appending time delta
- Apply normalization (`-m n`)
- Use the folder `Data` for raw data and `Processed` for exporting results
- Set the 3D plot angles to elevation 5 and azimuth -140

### File Structure

- **Raw_Data**: Folder containing the raw time-series CSV files.
- **Datasets**: Folder to export the processed subsamples.
- **Plots**: Folder where the generated plots will be saved.

## Notes

- Empty subsample files are automatically skipped during processing.
- You can view plots instead of saving them by using the `-v` flag.
- The script extracts sample information from filenames, including sample name, protocol, scan number, and timestamp.
- The script calculates the total number of scans and the total duration of the scans for each sample.
- Existing subsample files are deleted before exporting new ones to ensure consistency.
- If the number of subsamples (`-s`) is set to 0, all scans are combined into a single file without augmentation.
- Global normalization is performed across all subsamples using either a custom normalization method or Scikit-learn's MinMaxScaler.
- The model is trained with binary cross-entropy loss and the Adam optimizer.
- The best model is saved as best_model_overall.keras based on validation loss.
- The script generates multiple plots to help evaluate the model's performance, including validation accuracy, loss, confusion matrix, and ROC curve.
