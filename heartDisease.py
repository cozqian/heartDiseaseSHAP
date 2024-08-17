import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load the dataset (assuming it's in the same directory)
df = pd.read_csv("heart.csv")

# Preprocessing
X = df.drop("target", axis=1)
y = df["target"]

# Mapping of short column names to full descriptions
column_mapping = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol Level (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
    'restecg': 'Resting Electrocardiographic Results',
    'thalach': 'Maximum Heart Rate Achieved',
    'exang': 'Exercise-Induced Angina',
    'oldpeak': 'ST Depression Induced by Exercise',
    'slope': 'Slope of the Peak Exercise ST Segment',
    'ca': 'Number of Major Vessels Colored by Fluoroscopy',
    'thal': 'Thalassemia'
}

# Rename the columns in X for display purposes
X_display = X.rename(columns=column_mapping)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model training
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Streamlit app
st.title("SHAP Analysis for Heart Disease Prediction")

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")
st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

# Summary plot
st.subheader("Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test.rename(columns=column_mapping), show=False)
st.pyplot(fig)

# Summary plot for class 0 (No Heart Disease)
st.subheader("Summary Plot for Class 0 (No Heart Disease)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values[0], X_test.rename(columns=column_mapping), show=False)
st.pyplot(fig)

# Summary plot for class 1 (Heart Disease)
st.subheader("Summary Plot for Class 1 (Heart Disease)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values[1], X_test.rename(columns=column_mapping), show=False)
st.pyplot(fig)

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Mappings for categorical variables
sex_mapping = {0: "Female", 1: "Male"}
cp_mapping = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}
fbs_mapping = {0: "False", 1: "True"}
restecg_mapping = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
exang_mapping = {0: "No", 1: "Yes"}
slope_mapping = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
thal_mapping = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}

# Input fields for features with better user interface
input_data = {}
input_data['age'] = st.slider("Age:", min_value=int(X['age'].min()), max_value=int(X['age'].max()), value=int(X['age'].mean()))
input_data['sex'] = st.selectbox("Sex:", options=sex_mapping.values())
input_data['cp'] = st.selectbox("Chest Pain Type:", options=cp_mapping.values())
input_data['trestbps'] = st.slider("Resting Blood Pressure (mm Hg):", min_value=int(X['trestbps'].min()), max_value=int(X['trestbps'].max()), value=int(X['trestbps'].mean()))
input_data['chol'] = st.slider("Serum Cholesterol Level (mg/dl):", min_value=int(X['chol'].min()), max_value=int(X['chol'].max()), value=int(X['chol'].mean()))
input_data['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl:", options=fbs_mapping.values())
input_data['restecg'] = st.selectbox("Resting Electrocardiographic Results:", options=restecg_mapping.values())
input_data['thalach'] = st.slider("Maximum Heart Rate Achieved:", min_value=int(X['thalach'].min()), max_value=int(X['thalach'].max()), value=int(X['thalach'].mean()))
input_data['exang'] = st.selectbox("Exercise-Induced Angina:", options=exang_mapping.values())
input_data['oldpeak'] = st.slider("ST Depression Induced by Exercise:", min_value=float(X['oldpeak'].min()), max_value=float(X['oldpeak'].max()), value=float(X['oldpeak'].mean()))
input_data['slope'] = st.selectbox("Slope of the Peak Exercise ST Segment:", options=slope_mapping.values())
input_data['ca'] = st.slider("Number of Major Vessels Colored by Fluoroscopy:", min_value=int(X['ca'].min()), max_value=int(X['ca'].max()), value=int(X['ca'].mean()))
input_data['thal'] = st.selectbox("Thalassemia:", options=thal_mapping.values())

# Convert categorical inputs to numerical values as needed
input_df = pd.DataFrame(input_data, index=[0])

# Convert categorical features to match model training
input_df['sex'] = list(sex_mapping.keys())[list(sex_mapping.values()).index(input_df['sex'][0])]
input_df['cp'] = list(cp_mapping.keys())[list(cp_mapping.values()).index(input_df['cp'][0])]
input_df['fbs'] = list(fbs_mapping.keys())[list(fbs_mapping.values()).index(input_df['fbs'][0])]
input_df['restecg'] = list(restecg_mapping.keys())[list(restecg_mapping.values()).index(input_df['restecg'][0])]
input_df['exang'] = list(exang_mapping.keys())[list(exang_mapping.values()).index(input_df['exang'][0])]
input_df['slope'] = list(slope_mapping.keys())[list(slope_mapping.values()).index(input_df['slope'][0])]
input_df['thal'] = list(thal_mapping.keys())[list(thal_mapping.values()).index(input_df['thal'][0])]

# Make prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]  # Probability of heart disease

# Display prediction
st.write(f"**Prediction:** {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
st.write(f"**Heart Disease Probability:** {probability:.2f}")

# SHAP explanation for the input
shap_values_input = explainer.shap_values(input_df)

# Force plot
st.subheader("Force Plot")
st_shap(shap.force_plot(explainer.expected_value[1], shap_values_input[1], input_df), height=400, width=1000)

# Decision plot
st.subheader("Decision Plot")
st_shap(shap.decision_plot(explainer.expected_value[1], shap_values_input[1], X_test.rename(columns=column_mapping).columns))
