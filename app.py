from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load the scikit-learn model
model = joblib.load('model.pkl')

# Cost per unit in Sri Lankan Rupees (LKR)
COST_PER_UNIT_LKR = 50  # Replace with the actual cost per kWh in LKR

# In-memory storage for sensor data
sensor_data = {
    "voltage": 0.0,
    "current": 0.0,
    "timestamp": None
}

# In-memory storage for machine learning predictions
ml_predictions = {
    "predicted_energy_usage": 0.0,
    "total_price_lkr": 0.0,
    "hourly_data": {  # Example hourly data for the graph
        "labels": ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00"],
        "values": [0.5, 0.7, 0.6, 0.8, 0.9, 1.0]
    },
    "timestamp": None
}

# Route to serve the frontend
@app.route('/')
def home():
    return render_template('index.html')

# Route to receive sensor data from Arduino
@app.route('/api/sensor-data', methods=['POST'])
def receive_sensor_data():
    global sensor_data

    # Parse JSON data from Arduino
    data = request.json
    voltage = data.get('voltage')
    current = data.get('current')

    # Validate and store sensor data
    if voltage is not None and current is not None:
        sensor_data = {
            "voltage": voltage,
            "current": current,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        print(f"Received sensor data - Voltage: {voltage} V, Current: {current} A")
        return jsonify({"status": "success", "message": "Sensor data received"})
    else:
        return jsonify({"status": "error", "message": "Invalid sensor data"}), 400

# Route to fetch real-time sensor data for the frontend
@app.route('/api/sensor-data', methods=['GET'])
def get_sensor_data():
    return jsonify(sensor_data)

# Route to make predictions using the ML model
@app.route('/api/ml-predictions', methods=['POST'])
def make_ml_predictions():
    global ml_predictions

    # Parse input data for prediction
    data = request.json
    input_data = [data.get('voltage'), data.get('current')]

    # Convert input data to a 2D array
    input_array = np.array(input_data).reshape(1, -1)

    # Make prediction using the scikit-learn model
    try:
        prediction = model.predict(input_array)[0]
        total_price_lkr = prediction * COST_PER_UNIT_LKR

        ml_predictions = {
            "predicted_energy_usage": float(prediction),
            "total_price_lkr": float(total_price_lkr),
            "hourly_data": ml_predictions["hourly_data"],  # Include hourly data
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        print(f"ML Prediction: {prediction} kWh, Total Price: {total_price_lkr} LKR")
        return jsonify({"status": "success", "prediction": prediction, "total_price_lkr": total_price_lkr, "hourly_data": ml_predictions["hourly_data"]})
    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({"status": "error", "message": "Prediction failed"}), 500

# Route to fetch machine learning predictions for the frontend
@app.route('/api/ml-predictions', methods=['GET'])
def get_ml_predictions():
    return jsonify(ml_predictions)

# Route to control lights
@app.route('/api/lights', methods=['POST'])
def control_lights():
    data = request.json
    action = data.get('action')

    if action == "on":
        print("Lights turned on")
        return jsonify({"status": "success", "message": "Lights turned on"})
    elif action == "off":
        print("Lights turned off")
        return jsonify({"status": "success", "message": "Lights turned off"})
    else:
        return jsonify({"status": "error", "message": "Invalid action"}), 400

# Route to control fan
@app.route('/api/fan', methods=['POST'])
def control_fan():
    data = request.json
    action = data.get('action')

    if action == "on":
        print("Fan turned on")
        return jsonify({"status": "success", "message": "Fan turned on"})
    elif action == "off":
        print("Fan turned off")
        return jsonify({"status": "success", "message": "Fan turned off"})
    else:
        return jsonify({"status": "error", "message": "Invalid action"}), 400

# Route to control fridge
@app.route('/api/fridge', methods=['POST'])
def control_fridge():
    data = request.json
    action = data.get('action')

    if action == "on":
        print("Fridge turned on")
        return jsonify({"status": "success", "message": "Fridge turned on"})
    elif action == "off":
        print("Fridge turned off")
        return jsonify({"status": "success", "message": "Fridge turned off"})
    elif action == "hibernate":
        print("Fridge in hibernate mode")
        return jsonify({"status": "success", "message": "Fridge in hibernate mode"})
    else:
        return jsonify({"status": "error", "message": "Invalid action"}), 400

# Route to toggle night mode
@app.route('/api/night-mode', methods=['POST'])
def toggle_night_mode():
    data = request.json
    action = data.get('action')

    if action == "toggle":
        print("Night mode toggled")
        return jsonify({"status": "success", "message": "Night mode toggled"})
    else:
        return jsonify({"status": "error", "message": "Invalid action"}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)