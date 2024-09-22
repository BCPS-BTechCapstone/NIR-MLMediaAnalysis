import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Sample data shape: (num_samples, time_points)
# Assuming you have 35 samples, each with 24 time points
# X_non_contaminated: (20, 24), X_contaminated: (15, 24)
# Concatenating data and creating labels
X_non_contaminated = np.random.rand(20, 24)  # Replace with actual data
X_contaminated = np.random.rand(15, 24)  # Replace with actual data

X = np.concatenate([X_non_contaminated, X_contaminated], axis=0)
y = np.concatenate([np.zeros(20), np.ones(15)])  # 0 for non-contaminated, 1 for contaminated

# Preprocess the data (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input to 3D for Conv1D: (samples, time_steps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the 1D CNN model
model = Sequential()

# First Conv1D layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Second Conv1D layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Flatten and fully connected layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Predict and generate classification report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
