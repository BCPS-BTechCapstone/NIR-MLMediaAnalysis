import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import os
import pandas as pd

# Load the NIR data from CSV files
# Assuming each timepoint is a CSV file and each folder represents a sample run

# Directory containing all sample runs
base_dir = 'path/to/your/data'

# Lists to hold data and labels
X_data = []
Y_data = []

# Loop through each run folder
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        # Collect timepoint data for this run
        time_series_data = []
        for csv_file in sorted(os.listdir(folder_path)):
            if csv_file.endswith('.csv'):
                csv_path = os.path.join(folder_path, csv_file)
                # Load CSV and append to time_series_data list
                df = pd.read_csv(csv_path)
                time_series_data.append(df.values)
        
        # Stack the timepoints into a single NumPy array for this run
        time_series_array = np.stack(time_series_data, axis=0)  # Shape: (num_time_steps, num_features)
        
        # Assign label based on folder name
        if 'sample' in folder_name.lower():
            if '_1' in folder_name.lower() or '_2' in folder_name.lower() or '_3' in folder_name.lower():
                X_data.append(time_series_array)
                Y_data.append(0)  # Non-contaminated (augmented data)

# Convert lists to NumPy arrays
X = np.array(X_data)  # Shape: (num_samples, num_time_steps, num_features)
Y = np.array(Y_data)  # Shape: (num_samples,)

# Preprocessing the data
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

# K-Fold Cross-Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold = 1

best_checkpoint = ModelCheckpoint('best_model_overall.keras', monitor='val_loss', save_best_only=True, verbose=1)

val_accuracies = []
for train_index, val_index in kf.split(X):
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

# Evaluate the model on the entire dataset (or use a held-out test set)
loss, accuracy = model.evaluate(X, Y)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Predict on the entire dataset (or use a held-out test set)
Y_pred = (model.predict(X) > 0.5).astype('int32')

# Plot confusion matrix
cm = confusion_matrix(Y, Y_pred)
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


