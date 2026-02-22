# 🏦 Loan Default Risk Prediction using Random Forest

## 📌 Project Overview

This Machine Learning project predicts whether a loan applicant is likely to default using a Random Forest Classifier.

The dataset contains 100,000 applicant records with mixed numerical and categorical features. 
The goal is to build a robust classification model that can accurately identify high-risk applicants while handling class imbalance.

---

## 🎯 Problem Statement

Financial institutions must evaluate loan applications efficiently to minimize default risk.

Manual assessment:
- Is time-consuming
- Can introduce bias
- Does not scale well for large datasets

This project builds a data-driven predictive model to automate loan default risk classification.

---

## 📂 Dataset Information

Dataset: `Applicant_details.csv.zip`

Total Records: 100,000

Target Variable:
- `Loan_Default_Risk`
    - 0 → No Default
    - 1 → Default

### Features Used:

- Annual_Income
- Applicant_Age
- Work_Experience
- Marital_Status
- House_Ownership
- Vehicle_Ownership
- Occupation
- Residence_City
- Residence_State
- Years_in_Current_Employment
- Years_in_Current_Residence

Dropped:
- Applicant_ID (Identifier only)

---

## 🔎 Why Random Forest?

Random Forest was chosen because:

- Handles large datasets efficiently
- Works with mixed feature types
- Captures non-linear relationships
- Reduces overfitting compared to a single decision tree
- Performs well on imbalanced classification problems
- Provides feature importance insights

Given:
- 100,000 rows
- Mixed numerical & categorical features
- Non-linear patterns

Random Forest was an appropriate and powerful choice.

---

## 🛠 Project Workflow

### 1️⃣ Data Understanding
- Checked missing values
- Analyzed class imbalance
- Reviewed data types
- Explored statistical summary

### 2️⃣ Data Preprocessing
- Dropped unnecessary column (Applicant_ID)
- Separated features and target
- Applied One-Hot Encoding for categorical variables
- Train-Test Split (80% / 20%)
- Random state = 42 for reproducibility

### 3️⃣ Model Building
Baseline Model:
- RandomForestClassifier
- n_estimators = 100

### 4️⃣ Hyperparameter Tuning
Used GridSearchCV with:
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- class_weight

Cross-validation: 3-fold  
Evaluation metric: F1-score (to address class imbalance)

---

## 📊 Model Evaluation

Confusion Matrix:

[[16485   962]  
 [  431  2122]]

### Classification Report:

Class 0 (No Default):
- Precision: 0.97
- Recall: 0.94
- F1-score: 0.96

Class 1 (Default):
- Precision: 0.69
- Recall: 0.83
- F1-score: 0.75

Overall Accuracy: **93%**

Macro Avg F1-score: 0.86  
Weighted Avg F1-score: 0.93  

---

## ⚖ Handling Imbalanced Data

The dataset distribution:
- 87% Non-default
- 13% Default

To improve recall for defaulters:
- Used class_weight="balanced"
- Increased n_estimators to 200

This improved the model’s ability to detect high-risk applicants.

---

## 📈 Feature Importance

Random Forest provides feature importance ranking.

This helped identify:
- Most influential factors affecting loan default
- Key risk drivers for financial institutions

---

## 🚀 How to Run the Project

1. Clone the repository:
git clone https://github.com/LohitRaj-001/loan-default-risk-prediction.git

2. Install required libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


3. Open the notebook in:
- Google Colab
- Jupyter Notebook

4. Run all cells

---

## 🧠 Skills Demonstrated

- Data Cleaning & Preprocessing
- Handling Imbalanced Datasets
- One-Hot Encoding
- Random Forest Modeling
- Hyperparameter Tuning (GridSearchCV)
- Model Evaluation (Accuracy, Precision, Recall, F1-score)
- Feature Importance Analysis

---

## 📌 Future Improvements

- SMOTE for advanced imbalance handling
- Model comparison (XGBoost, LightGBM)
- Deployment using Flask / Streamlit
- API integration for real-time predictions

---

## 👨‍💻 Author

Lohit Raj  
Machine Learning 