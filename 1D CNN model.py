import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import os
import pandas as pd

# Load the NIR data from CSV files
# Assuming each CSV file contains all the timepoints for that run

# Directory containing all sample runs
base_dir = './Datasets'

print(f"Checking directory: {base_dir}")
if not os.path.exists(base_dir):
    raise ValueError(f"The directory {base_dir} does not exist.")

# Lists to hold data and labels
X_data = []
Y_data = []

print("Listing folders in the base directory:")
for folder_name in os.listdir(base_dir):
    print(f"Found folder: {folder_name}")

# Loop through each CSV file in the base directory
for csv_file in os.listdir(base_dir):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(base_dir, csv_file)
        print(f"Loading file: {csv_path}")

        # Load the CSV and append to data list
        df = pd.read_csv(csv_path)
        time_series_array = df.values  # Assuming the data in the CSV is correctly structured as (num_time_steps, num_features)

        X_data.append(time_series_array)

        # Assign label based on file name
        if '_1' in csv_file.lower() or '_2' in csv_file.lower() or '_3' in csv_file.lower():
            Y_data.append(0)  # Non-contaminated (augmented data)

# Determine the maximum timepoints and features
max_timepoints = max([x.shape[0] for x in X_data])
max_features = max([x.shape[1] for x in X_data])

# Pad each sample to have the same number of timepoints and features
X_padded = []
for x in X_data:
    padded = np.zeros((max_timepoints, max_features))
    padded[:x.shape[0], :x.shape[1]] = x
    X_padded.append(padded)

# Convert lists to NumPy arrays
if len(X_padded) == 0:
    raise ValueError("No data found. Please ensure the base directory contains correctly named folders and CSV files.")

X = np.array(X_padded)  # Shape: (num_samples, max_timepoints, max_features)
Y = np.array(Y_data)  # Shape: (num_samples,)

# Preprocessing the data
if X.size == 0:
    raise ValueError("The dataset is empty. Please check the data loading process.")

scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Define 1D CNN model function
def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Check data balance
print(f"Number of non-contaminated samples: {np.sum(Y == 0)}")
print(f"Number of contaminated samples: {np.sum(Y == 1)}")

# K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold = 1

best_checkpoint = ModelCheckpoint('best_model_overall.keras', monitor='val_loss', save_best_only=True, verbose=1)

val_accuracies = []
for train_index, val_index in kf.split(X, Y):
    print(f'Training on Fold {fold}...')
    
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]
    
    # Create a new instance of the model
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train the model
    history = model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_val, Y_val), callbacks=[best_checkpoint])
    
    # Plot validation loss and accuracy over training epochs for this fold
    plt.plot(history.history['val_loss'], label=f'Validation Loss Fold {fold}')
    plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy Fold {fold}')
    
    # Record validation accuracy for each fold
    val_accuracy = history.history['val_accuracy'][-1]
    val_accuracies.append(val_accuracy)
    
    fold += 1

# Show validation loss and accuracy plot
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Validation Loss and Accuracy Over Epochs for Each Fold')
plt.legend()
plt.show()

# Show validation accuracy for each fold
plt.figure()
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', linestyle='-', label='Validation Accuracy per Fold')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy per Fold')
plt.legend()
plt.show()

# Assuming final model is trained on the entire dataset for evaluation purposes
model = create_model(input_shape=(X.shape[1], X.shape[2]))
model.fit(X, Y, epochs=50, batch_size=16, callbacks=[best_checkpoint])

# Predict on the entire dataset (or use a held-out test set)
Y_pred = (model.predict(X) > 0.5).astype('int32')

# Plot confusion matrix
cm = confusion_matrix(Y, Y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Contaminated', 'Contaminated'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Compute ROC curve and AUC
Y_prob = model.predict(X).ravel()
fpr, tpr, _ = roc_curve(Y, Y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
