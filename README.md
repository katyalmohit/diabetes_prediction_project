# Diabetes Prediction Model

## Overview
This project develops machine learning models to predict diabetes using the PIMA Indians Diabetes Dataset. The implementation compares multiple classification algorithms to identify the most effective approach for early diabetes detection.

## Dataset
The PIMA Indians Diabetes Dataset contains medical data from female patients of Pima Indian heritage, with the following features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: Diabetes pedigree function (a function of diabetes history in relatives)
- Age: Age in years
- Outcome: Class variable (0: Non-diabetic, 1: Diabetic)

## Project Structure
- `diabetes_prediction.ipynb`: Jupyter notebook containing all code and analysis
- `diabetes.csv`: Dataset file containing patient records
- `README.md`: This file containing project information

## Methods Implemented
The notebook implements and compares the following classification algorithms:
1. Support Vector Machine (SVM)
2. K-Nearest Neighbors (KNN)
3. Naive Bayes
4. Logistic Regression
5. Random Forest

## Model Performance Comparison

### Before Grid Search Optimization

| Algorithm           | Accuracy | Precision | Recall | F1-Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| Support Vector Machine | 0.773   | 0.67      | 0.69   | 0.68     | N/A  |
| K-Nearest Neighbors (K=11) | 0.766   | 0.65      | 0.71   | 0.68     | 0.82 |
| Naive Bayes         | 0.773   | 0.68      | 0.68   | 0.68     | N/A  |
| Logistic Regression | 0.760   | 0.65      | 0.70   | 0.67     | 0.83 |
| Random Forest       | 0.721   | 0.62      | 0.64   | 0.63     | 0.81 |

### After Grid Search Optimization

| Algorithm           | Accuracy | Precision | Recall | F1-Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| Support Vector Machine | 0.792   | 0.72      | 0.71   | 0.71     | 0.84 |
| K-Nearest Neighbors | 0.779   | 0.69      | 0.73   | 0.71     | 0.83 |
| Naive Bayes         | 0.779   | 0.70      | 0.69   | 0.69     | 0.82 |
| Logistic Regression | 0.785   | 0.71      | 0.71   | 0.71     | 0.85 |
| Random Forest       | 0.766   | 0.68      | 0.69   | 0.68     | 0.83 |

Note: 
- Precision, Recall, and F1-Score values are weighted averages across classes.
- Grid Search significantly improved model performance across all algorithms.
- Logistic Regression achieved the highest AUC score (0.85) after optimization.
- SVM demonstrated the best overall accuracy (0.792) after hyperparameter tuning.

## Analysis Workflow
1. Data Collection & Exploration
   - Loading and examining the dataset
   - Statistical analysis
   - Checking for missing values and duplicates

2. Data Visualization
   - Class distribution analysis
   - Feature outlier detection
   - Correlation analysis

3. Data Preprocessing
   - Feature standardization
   - Train-test split

4. Model Training & Evaluation
   - Implementation of multiple classification algorithms
   - Performance evaluation using accuracy, classification report, and confusion matrix
   - ROC curve comparison
   - Training time comparison

5. Hyperparameter Optimization
   - Grid Search for optimal parameters for each algorithm
   - Comparison of model performance before and after optimization
   - ROC curve analysis of optimized models

6. Predictive System
   - Sample prediction system for new patient data

## Key Findings
- Hyperparameter optimization substantially improved model performance
- SVM and Logistic Regression showed the best performance after optimization
- ROC curve analysis indicates good discrimination ability across optimized models
- Feature importance analysis revealed glucose level as the most predictive feature for diabetes
- Evaluation of model efficiency and accuracy trade-offs showed SVM as the best compromise

## Requirements
- Python 3.x
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

## Usage
To run this project:
1. Ensure all required libraries are installed
2. Open the Jupyter notebook `diabetes_prediction.ipynb`
3. Run all cells to see the complete analysis and model comparison

## Date
Last updated: April 2025