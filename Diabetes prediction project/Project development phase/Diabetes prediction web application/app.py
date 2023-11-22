from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Mapping for categorical features
categorical_mapping = {
    'HighBP': {'No': 0, 'Yes': 1},
    'HighChol': {'No': 0, 'Yes': 1},
    'CholCheck': {'No': 0, 'Yes': 1},
    'BMI': {},  # No mapping for BMI as it's a numeric value
    'Smoker': {'No': 0, 'Yes': 1},
    'Stroke': {'No': 0, 'Yes': 1},
    'HeartDiseaseorAttack': {'No': 0, 'Yes': 1},
    'PhysActivity': {'No': 0, 'Yes': 1},
    'Fruits': {'No': 0, 'Yes': 1},
    'Veggies': {'No': 0, 'Yes': 1},
    'HvyAlcoholConsump': {'No': 0, 'Yes': 1},
    'AnyHealthcare': {'No': 0, 'Yes': 1},
    'NoDocbcCost': {'No': 0, 'Yes': 1},
    'GenHlth': {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5},
    'MentHlth': {},  # No mapping for MentHlth as it's a numeric value
    'PhysHlth': {},  # No mapping for PhysHlth as it's a numeric value
    'DiffWalk': {'No': 0, 'Yes': 1},
    'Sex': {'Female': 0, 'Male': 1},
    'Age': {},  # No mapping for Age as it's a numeric value
    'Education': {},  # No mapping for Education as it's a numeric value
    'Income': {}  # No mapping for Income as it's a numeric value
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    features = {}
    for key, value in request.form.items():
        features[key] = value

    # Convert categorical values to numeric using the mapping
    for feature, mapping in categorical_mapping.items():
        if not mapping:  # If there's no mapping, it's a numeric value
            features[feature] = float(features[feature])
        else:
            features[feature] = mapping[features[feature]]

    # Convert features to a numpy array for prediction
    input_features = np.array([list(features.values())], dtype=float)

    # Make a prediction using the loaded model
    prediction = model.predict(input_features)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
