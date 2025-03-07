

#### **Diabetes Prediction Web Application using Machine Learning**

This project involves building a **web application** to predict diabetes based on a variety of user input features using **Machine Learning**. The application uses a **Gradient Boosting Classifier** to predict whether a person has diabetes or not based on their demographic and health data, such as age, BMI, blood glucose levels, and more.

The project follows a complete pipeline from data preprocessing, model training, to web deployment using Flask. Here's a breakdown of the steps:

---

#### **Key Features of the Project:**

1. **Data Preprocessing**:
   - **Categorical Feature Encoding**: The `gender` and `smoking_history` columns are encoded using `LabelEncoder` to convert categorical data into numerical values.
   - **Standardization**: The numerical features, such as `age`, `bmi`, `HbA1c_level`, and `blood_glucose_level`, are standardized using `StandardScaler` to ensure they are on the same scale for model training.
   - **Saving Preprocessing Objects**: The pre-trained encoders and scalers are saved using `joblib` for reuse in the prediction phase.

2. **Model Training**:
   - **Gradient Boosting Classifier**: The machine learning model used to predict diabetes is a **Gradient Boosting Classifier**, which is trained using the preprocessed data.
   - **Model Evaluation**: The model is evaluated using metrics like **accuracy**, **F1 score**, **ROC AUC score**, and **confusion matrix**.

3. **Web Application**:
   - The trained model and preprocessing objects are loaded in a Flask web application.
   - **User Input**: The web interface allows users to input their data (e.g., age, BMI, blood glucose level, etc.) via a form.
   - **Prediction**: Upon form submission, the web application preprocesses the input data, applies the trained model, and outputs a prediction: whether the user has diabetes or not.
   - **Error Handling**: Appropriate error handling to manage invalid or missing data entries.

4. **Model and Preprocessing Object Persistence**:
   - The **trained model**, **scaler**, and **encoders** are saved to disk using `joblib`, ensuring that the model can be reused without retraining each time the application is run.

---

#### **Project Files:**

- **Model Training and Evaluation**: The first part of the project involves loading the diabetes dataset, preprocessing the data, training the Gradient Boosting Classifier, and evaluating the model using multiple metrics.
- **Flask Web Application**: A simple Flask web application that interacts with the user and returns predictions based on the trained model.
- **Joblib Saved Files**: The `model.pkl`, `scaler.pkl`, and encoders for categorical features (`gender_encoder.pkl`, `smoking_history_encoder.pkl`) are saved for later use in the web application.

---

#### **How to Run the Project Locally:**

1. **Clone the Repository**:
   ```
   git clone https://github.com/Zakeer2811/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Install the Dependencies**:
   Install the required Python libraries:
   ```
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   Download the diabetes prediction dataset present in kaggle and save it in the project folder as `diabetes_prediction_dataset.csv`.

4. **Train the Model** (if not already trained):
   Run the `diabetes.ipynb` script to train the Gradient Boosting Classifier:
   
5. **Run the Flask Application**:
   Start the Flask web server:
   ```
   python app.py
   ```
   The application will be available at `http://127.0.0.1:5000/`.

6. **Interact with the Web App**:
   Open a browser and navigate to the local server URL to input your data and get a diabetes prediction.

---

#### **Technologies Used**:

- **Flask**: Web framework for creating the web application.
- **Scikit-learn**: For preprocessing the data, training the machine learning model, and evaluating its performance.
- **Pandas**: For handling and processing the dataset.
- **Joblib**: For saving and loading machine learning models and preprocessing objects.
- **HTML/CSS**: For building the user interface.

---

