import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the NIR data
# Assuming X_train contains data from 20 non-contaminated fermentation runs (time series data)
# Assuming X_test contains data from 10 contaminated fermentation runs (time series data)
# Assuming Y_train and Y_test contain the corresponding labels (0 for non-contaminated, 1 for contaminated)

# Placeholder: Replace these with the actual loading code
X_train = np.load('X_train.npy')
Y_train = np.zeros(X_train.shape[0])  # Non-contaminated data labeled as 0

X_test = np.load('X_test.npy')
Y_test = np.ones(X_test.shape[0])  # Contaminated data labeled as 1

# Combine train and test datasets for splitting later
X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)

# Preprocessing the data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define 1D CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_val, Y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, Y_val)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Test the model on test data
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
