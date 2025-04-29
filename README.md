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

5. Predictive System
   - Sample prediction system for new patient data

## Key Findings
- Comparison of different model performances for diabetes prediction
- Analysis of feature importance for diabetes risk
- Evaluation of model efficiency and accuracy trade-offs

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