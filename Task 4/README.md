Customer Churn Prediction with XGBoost
Project Overview
This project implements a machine learning pipeline to predict customer churn for a telecommunications company using the XGBoost classifier. The model processes both numerical and categorical features to classify whether a customer will churn or not.

Dataset
The dataset used is WA_Fn-UseC_-Telco-Customer-Churn.csv, which contains information about telecom customers including:

Demographic information (gender, senior citizen status)

Account information (tenure, contract type)

Service details (phone service, internet service, streaming services)

Billing information (monthly charges, total charges, payment method)

Target variable: Churn (Yes/No)

Data Preprocessing
Data Type Conversion: Converted TotalCharges from object to numeric type, handling errors with coercion

Missing Values: Removed rows with missing values after conversion

Feature Separation:

Numerical features: SeniorCitizen, tenure, MonthlyCharges, TotalCharges

Categorical features: All other features except customerID and Churn

Target Encoding: Converted Churn column to binary (0/1) using LabelEncoder

Model Pipeline
The project uses a scikit-learn pipeline with the following components:

Preprocessing:

StandardScaler for numerical features

OneHotEncoder for categorical features

Classifier: XGBoost with parameters:

n_estimators=300

learning_rate=0.05

max_depth=4

Model Performance
The model was evaluated on a 80-20 train-test split with the following results:

Precision Score: 0.633

Recall Score: 0.508

F1 Score: 0.564

Comparison with Logistic Regression
Logistic Regression: Slightly better recall (catches more churners)

XGBoost: Slightly better precision and F1 score (more reliable predictions)

Overall: Both models are comparable, but XGBoost has a slight edge with better F1 score

Requirements
pandas

scikit-learn

xgboost

Usage
Load the dataset

Preprocess the data (type conversion, missing values)

Separate features and target

Build and train the pipeline

Evaluate model performance
