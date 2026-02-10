from flask import Blueprint, render_template, flash, request
import numpy as np
import joblib
import os

routes = Blueprint('routes', __name__, template_folder='../templates')

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'static', 'model', 'RandomForestClassifier_model')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'static', 'model', 'scaler')

FEATURES = [
    'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
    'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity',
    'Extent', 'Roundness', 'Aspect_Ration', 'Compactness'
]

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    scaler = None
    print(f"Error loading scaler: {e}")

@routes.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@routes.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    if model is None or scaler is None:
        flash('Error: Model or scaler not loaded.', 'error')
        return render_template('predict.html')
    
    try:
        input_data = []
        for feature in FEATURES:
            value = request.form.get(feature)
            if not value or value.strip() == '':
                flash(f'Missing value for {feature}.', 'error')
                return render_template('predict.html')
            try:
                num_value = float(value)
                if num_value < 0:
                    flash(f'{feature} must be non-negative.', 'error')
                    return render_template('predict.html')
                input_data.append(num_value)
            except ValueError:
                flash(f'Invalid value for {feature}. Please enter a number.', 'error')
                return render_template('predict.html')
        
        input_array = np.array([input_data])
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]
        input_summary = {feature: value for feature, value in zip(FEATURES, input_data)}
        
        return render_template('result.html', prediction=prediction, input_summary=input_summary)
    
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return render_template('predict.html')