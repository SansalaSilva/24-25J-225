import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO
import os
import numpy as np
import librosa
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, Response
import cv2
import time
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model


# Load YOLO model
model = YOLO("runs/detect/train2/weights/best.pt")

# Define labels and thresholds
threshold = 0.5
unknown_threshold = 0.85
class_name_dict = {
    0: 'Known Person',
    1: 'Hidden Person',
}

# Initialize counters
known_person_count = 0
hidden_person_count = 0
unknown_person_count = 0

# Open a text file to save detection counts
with open("detection_counts.txt", "a") as file:
    file.write("Detection Counts (Every 60 Seconds)\n")
    file.write("------------------------------\n")

# Webcam setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()


# Function to capture and process video
def gen_frames():
    global known_person_count, hidden_person_count, unknown_person_count
    start_time = time.time()
    previous_known_person_count = 0
    previous_unknown_person_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        current_known_person_count = 0
        current_unknown_person_count = 0

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                if score >= unknown_threshold:
                    label = f"{class_name_dict[int(class_id)]} ({score:.2f})"
                    color = (0, 255, 0)

                    if int(class_id) == 0:
                        current_known_person_count += 1
                    elif int(class_id) == 1:
                        current_hidden_person_count += 1
                else:
                    label = f"Unknown Face ({score:.2f})"
                    color = (0, 0, 255)
                    current_unknown_person_count += 1

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

        known_person_count += current_known_person_count
        unknown_person_count += current_unknown_person_count

        # Update counts every 60 seconds
        current_time = time.time()
        if current_time - start_time >= 60:
            with open("detection_counts.txt", "a") as file:
                file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}\n")
                file.write(f"Known Person Count: {known_person_count - previous_known_person_count}\n")
                file.write(f"Unknown Person Count: {unknown_person_count - previous_unknown_person_count}\n")
                file.write("------------------------------\n")

            # Update previous counts
            previous_known_person_count = known_person_count
            previous_unknown_person_count = unknown_person_count
            start_time = current_time

        # Draw the detection count on the frame
        cv2.putText(frame, f"Known Person: {known_person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Unknown Person: {unknown_person_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert the frame to JPEG format for web display
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')


# Route for the video stream
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to get live detection counts
@app.route('/get_counts')
def get_counts():
    return jsonify({
        'known_person_count': known_person_count,
        'unknown_person_count': unknown_person_count
    })


# Home route to serve the web page
@app.route('/')
def index():
    return render_template('index.html')


######## Audio
MODEL_PATH = "security_sound_classifier.h5"
UPLOAD_FOLDER = 'static/uploads/'
SECURITY_CLASSES = ["door_wood_knock", "door_wood_creaks", "glass_breaking", "footsteps", "crackling_fire"]

# Load the trained model
model2 = load_model(MODEL_PATH)

# Load the LabelEncoder to map predictions back to class names
label_encoder = LabelEncoder()
label_encoder.fit(SECURITY_CLASSES)


# Function to load and preprocess the audio file
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.reshape(mfcc, (1, -1))


# Define allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/audiopage')
def audio():
    return render_template('audio.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('audio.html')  # Show form on GET request

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    processed_audio = preprocess_audio(file_path)
    predicted_class_index = np.argmax(model2.predict(processed_audio))
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

    print(predicted_class)

    return render_template('audio.html', prediction=predicted_class)


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
