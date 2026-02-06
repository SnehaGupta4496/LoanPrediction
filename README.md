# Loan Prediction System

## Overview

The **Loan Prediction System** is a machine learning–based project that predicts whether a loan applicant is eligible for loan approval based on demographic, financial, and credit-related attributes. The project reflects a real-world banking use case where automated decision-support systems help reduce manual effort, improve consistency, and enhance decision accuracy.

---

## Problem Statement

Banks and financial institutions receive a large number of loan applications every day. Manual evaluation of each application is time-consuming and may introduce bias or inconsistency.

The objective of this project is to:

* Automate the loan approval decision process
* Predict loan approval status using historical data
* Support data-driven decision-making in financial institutions

---

## Dataset Description

The dataset consists of historical loan application records with the following features:

| Feature           | Description                               |
| ----------------- | ----------------------------------------- |
| Loan_ID           | Unique loan identifier                    |
| Gender            | Applicant gender                          |
| Married           | Marital status                            |
| Dependents        | Number of dependents                      |
| Education         | Education level                           |
| Self_Employed     | Self-employment status                    |
| ApplicantIncome   | Applicant’s income                        |
| CoapplicantIncome | Co-applicant’s income                     |
| LoanAmount        | Loan amount requested                     |
| Loan_Amount_Term  | Loan repayment term                       |
| Credit_History    | Credit history (0 = No, 1 = Yes)          |
| Property_Area     | Urban / Semiurban / Rural                 |
| Loan_Status       | Target variable (Approved / Not Approved) |

---

## Technologies Used

* **Programming Language:** Python
* **Libraries:**

  * NumPy
  * Pandas
  * Matplotlib / Seaborn
  * Scikit-learn
* **Machine Learning Models:**

  * Logistic Regression
  * Decision Tree
  * kNN to remove missing values

---

## Project Workflow

1. Data collection
2. Data cleaning and preprocessing
3. Exploratory data analysis (EDA)
4. Feature encoding and scaling
5. Model training
6. Model evaluation
7. Prediction on unseen data

---

## Machine Learning Approach

* **Learning Type:** Supervised learning
* **Problem Type:** Binary classification
* **Target Variable:** Loan_Status

### Evaluation Metrics

* Accuracy
* Confusion matrix
* Precision, recall, and F1-score

---

## Results and Observations

The trained model predicts loan approval with good accuracy. Credit history and applicant income were identified as key factors influencing loan approval decisions.

---

## How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/loan-prediction-project.git

# Navigate to the project directory
cd loan-prediction-project

# Install required dependencies
pip install -r requirements.txt

# Run the program
python loan_prediction.py
```

---

## Future Enhancements

* Hyperparameter tuning for improved performance
* Deployment using Flask or Streamlit
* Model explainability using SHAP or LIME
* Improved handling of class imbalance

---

## Conclusion

This project demonstrates the practical application of machine learning techniques in the financial domain. It highlights the complete pipeline from data preprocessing to model evaluation and provides a strong foundation for further enhancements and deployment.

