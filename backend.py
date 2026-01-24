from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)


# Load trained model and scaler


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb")) 


mapping = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        Gender = request.form['Gender']
        Married = request.form['Married']
        Education = request.form['Education']
        Self_Employed = request.form['Self_Employed']
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = float(request.form['Credit_History'])
        Property_Area = request.form['Property_Area']
        Dependents = request.form['Dependents']

        
        # One-hot encoding Property_Area
       
        Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
        Property_Area_Urban = 1 if Property_Area == "Urban" else 0

       
        # One-hot encoding Dependents
        
        Dependents_1 = 1 if Dependents == "1" else 0
        Dependents_2 = 1 if Dependents == "2" else 0
        Dependents_3plus = 1 if Dependents == "3+" else 0

        
        # Convert categorical variables using mapping
       
        Gender_num = mapping['Gender'][Gender]
        Married_num = mapping['Married'][Married]
        Education_num = mapping['Education'][Education]
        Self_Employed_num = mapping['Self_Employed'][Self_Employed]

       
        # Create feature array in correct order (14 features)
        
        features = [
            Gender_num, Married_num, Education_num, Self_Employed_num,
            ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
            Credit_History,
            Property_Area_Semiurban, Property_Area_Urban,
            Dependents_1, Dependents_2, Dependents_3plus
        ]

        X_new = np.array([features])  # shape (1,14)

       
        # Scale using the loaded StandardScaler
        
        X_scaled = scaler.transform(X_new)

        
        # Predict
        
        prediction = model.predict(X_scaled)
        result = "Approved" if prediction[0] == 1 else "Rejected"

        return render_template("index.html", prediction_text=f"Loan Prediction Result: {result}")

    except Exception as e:
        return f"Error occurred: {e}"

import os

if __name__ == "__main__":
    # Render provides a PORT environment variable; if not found, it defaults to 5000
    port = int(os.environ.get("PORT", 5000))
    # host="0.0.0.0" is the critical fix for the "No open ports" error
    app.run(host="0.0.0.0", port=port)

