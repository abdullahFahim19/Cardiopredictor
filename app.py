import pandas as pd
import requests
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Paths to the model and scaler
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Load the model and scaler with error handling
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
except Exception as e:
    print("Error loading scaler:", e)
    scaler = None

# Ensure that model and scaler are loaded before continuing
if model is None or scaler is None:
    raise ValueError("Model or scaler failed to load. Ensure the files are available and accessible.")

# Load and prepare a sample of the dataset for feature alignment
data = pd.read_csv('cardio_train.csv', delimiter=';').sample(n=20000, random_state=42)
X = data.drop(columns=['id', 'cardio'])
y = data['cardio']
X = pd.get_dummies(X, columns=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])

# Train-test split (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy on test data and print it to four decimal places
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data: {accuracy:.4f}')

# Function to fetch ThinkSpeak data
def fetch_thinkspeak_data():
    CHANNEL_ID = '2720455'
    READ_API_KEY = '56QHX9QZLRIWROGK'
    url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        latest_entry = data['feeds'][0]
        return {
            'cholesterol': int(latest_entry['field1']),
            'glucose': int(latest_entry['field2']),
            'ap_hi': int(latest_entry['field3']),
            'ap_lo': int(latest_entry['field4'])
        }
    except Exception as err:
        print("Error fetching ThinkSpeak data:", err)
        return None

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    user_data = request.get_json()
    user_data['age'] = user_data['age'] * 365  # Convert age from years to days
    user_data['height'] = user_data['height'] * 2.54  # Convert height from inches to cm

    # Fetch ThinkSpeak data
    sensor_data = fetch_thinkspeak_data()
    if not sensor_data:
        return jsonify({"error": "Failed to retrieve sensor data"}), 500

    # Combine user input and sensor data
    data_input = pd.DataFrame({
        'age': [user_data['age']],
        'gender': [user_data['gender']],
        'height': [user_data['height']],
        'weight': [user_data['weight']],
        'ap_hi': [sensor_data['ap_hi']],
        'ap_lo': [sensor_data['ap_lo']],
        'cholesterol': [sensor_data['cholesterol']],
        'gluc': [sensor_data['glucose']],
        'smoke': [user_data['smoke']],
        'alco': [user_data['alco']],
        'active': [user_data['active']]
    })

    # Convert categorical variables to dummies
    data_input = pd.get_dummies(data_input, columns=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
    data_input = data_input.reindex(columns=X.columns, fill_value=0)

    # Scale the input data
    try:
        data_scaled = scaler.transform(data_input)
    except Exception as e:
        return jsonify({"error": f"Error scaling input data: {e}"}), 500

    # Predict the result and calculate probability
    try:
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0][prediction] * 100
        result = {
            "prediction": int(prediction),
            "message": f"You have a {probability:.2f}% chance of having heart disease." if prediction == 1 else f"You have a {100 - probability:.2f}% chance of not having heart disease."
        }
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {e}"}), 500

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
