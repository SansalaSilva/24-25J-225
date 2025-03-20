import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Set paths
DATASET_PATH = "ESC-50-master/audio/"
CSV_FILE = "ESC-50-master/meta/esc50.csv"

# Define the security-related classes (based on your dataset)
SECURITY_CLASSES = ["door_wood_knock", "door_wood_creaks", "glass_breaking", "footsteps", "crackling_fire"]

# Load CSV metadata to check class names
metadata = np.genfromtxt(CSV_FILE, delimiter=",", dtype=str, skip_header=1)
file_names = metadata[:, 0]
labels = metadata[:, 3]

# Filter out only the security-related sounds
filtered_files = []
filtered_labels = []

for i, label in enumerate(labels):
    if label in SECURITY_CLASSES:
        filtered_files.append(file_names[i])
        filtered_labels.append(label)

# Check if any security-related sounds were found
if len(filtered_files) == 0:
    print("No matching security sounds found. Check dataset path and class names.")
    exit()

# Load audio files and their labels
X = []
y = []

for file, label in zip(filtered_files, filtered_labels):
    # Load the audio file using librosa
    file_path = os.path.join(DATASET_PATH, file)
    audio, sr = librosa.load(file_path, sr=None)
    
    # Extract features (e.g., MFCC)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)  # Take the mean across time
    
    X.append(mfcc)
    y.append(label)

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels as integers (for categorical classification)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = models.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(SECURITY_CLASSES), activation='softmax')  # Output layer (number of classes)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Save the model
model.save("security_sound_classifier.h5")
