from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Load the trained model and preprocessing objects
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
smoking_history_encoder = joblib.load('smoking_history_encoder.pkl')

# Define the feature names used during training
TRAINING_FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 
                     'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        hba1c_level = float(request.form['hba1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        gender = request.form['gender']
        smoking_history = request.form['smoking_history']
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])

        # Encode categorical data
        gender_encoded = gender_encoder.transform([gender])[0]
        smoking_history_encoded = smoking_history_encoder.transform([smoking_history])[0]

        # Create input data as a DataFrame with the expected column order
        input_data = pd.DataFrame(
            [[gender_encoded, age, hypertension, heart_disease, 
              smoking_history_encoded, bmi, hba1c_level, blood_glucose_level]],
            columns=TRAINING_FEATURES
        )

        # Standardize numerical features
        numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # Predict using the trained model
        prediction = model.predict(input_data)

        # Generate response
        result = "You have diabetes." if prediction[0] == 1 else "You do not have diabetes."
        return render_template('index.html', result=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
