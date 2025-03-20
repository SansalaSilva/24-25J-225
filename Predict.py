import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Set paths
MODEL_PATH = "security_sound_classifier.h5"
DATASET_PATH = "ESC-50-master/audio/"

# Define the security-related classes (based on your dataset)
SECURITY_CLASSES = ["door_wood_knock", "door_wood_creaks", "glass_breaking", "footsteps", "crackling_fire"]

# Load the trained model
model = load_model(MODEL_PATH)

# Load the LabelEncoder to map predictions back to class names (assuming you saved this during training)
# If you saved the LabelEncoder, load it here. Otherwise, re-fit it on the same classes used during training.
label_encoder = LabelEncoder()
label_encoder.fit(SECURITY_CLASSES)

# Function to load and preprocess the audio file
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC features from the audio
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)  # Take the mean across time
    
    # Reshape to match model input
    return np.reshape(mfcc, (1, -1))

# Test on a new audio file
test_audio_file = "Security_Sounds/glass_breaking/1.wav"  # Replace with actual file path
processed_audio = preprocess_audio(test_audio_file)

# Make prediction
predicted_class_index = np.argmax(model.predict(processed_audio))
predicted_class = label_encoder.inverse_transform([predicted_class_index])

# Print the predicted class
print(f"Predicted Class: {predicted_class[0]}")
