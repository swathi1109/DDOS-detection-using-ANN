from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('trained_model.h5')

# Load the scaler parameters
scaler_mean = np.loadtxt('scaler_mean.txt')
scaler_scale = np.loadtxt('scaler_scale.txt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the uploaded file in binary mode
            input_data = np.array([[float(x) for x in line.strip().decode().split(',')] for line in file])
            # Preprocess the input data using the loaded scaler parameters
            input_data_scaled = (input_data - scaler_mean) / scaler_scale
            # Make predictions using the loaded model
            prediction = model.predict(input_data_scaled)
            # Convert prediction probabilities to labels
            prediction_label = np.argmax(prediction)
            if prediction_label == 1:
                result = 'Warning: This is a DDOS attack. Action required.'
            else:
                result = 'Normal traffic: No action required.'
            return render_template('results.html', result=result)
        else:
            return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True)
