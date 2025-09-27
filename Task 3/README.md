# Customer Churn Prediction

This project implements a machine learning model to predict customer churn for a telecommunications company using the Telco Customer Churn dataset.

## Project Overview

The goal of this project is to build a classification model that can predict whether a customer will churn (leave the service) based on various customer attributes and usage patterns.

## Dataset

The dataset used is `WA_Fn-UseC_-Telco-Customer-Churn.csv`, which contains information about 7,043 customers with 21 features including:

- **Demographic information**: gender, senior citizen status, partner, dependents
- **Account information**: tenure, contract type, paperless billing, payment method
- **Service information**: phone service, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV, streaming movies
- **Charges**: monthly charges, total charges
- **Target variable**: Churn (whether the customer left or not)

## Project Structure

The notebook follows this workflow:

1. **Import Libraries**: Essential Python libraries for data processing and machine learning
2. **Data Loading and Exploration**: Load the dataset and examine its structure
3. **Data Preprocessing**:
   - Handle missing values in TotalCharges column
   - Separate features and target variable
   - Identify numerical and categorical features
4. **Feature Engineering**:
   - Standard scaling for numerical features
   - One-hot encoding for categorical features
5. **Model Building**:
   - Create a pipeline with preprocessing and Logistic Regression
   - Split data into training and testing sets
6. **Model Evaluation**:
   - Calculate accuracy score
   - Generate confusion matrix
   - Visualize results

## Key Results

The Logistic Regression model achieved:
- **Accuracy**: 78.75%
- **Confusion Matrix**:
  - True Negatives: 915
  - False Positives: 118
  - False Negatives: 181
  - True Positives: 193

## Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning library
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Development environment

