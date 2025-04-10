import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kaggle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Download the dataset using Kaggle API
def download_data():
    st.text("Downloading the dataset...")
    
    # Use Kaggle API to download the dataset
    try:
        kaggle.api.dataset_download_files('redwankarimsony/heart-disease-data', path='datasets/', unzip=True)
        st.text("Dataset downloaded successfully.")
        return 'datasets/heart_disease_uci.csv' 
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Run the download process when the user clicks the button
if st.button('Download Dataset'):
    dataset_path = download_data()

    if dataset_path:
        # Load the dataset once it's downloaded
        @st.cache_data
        def load_data(path):
            df = pd.read_csv(path)
            return df

        df = load_data(dataset_path)

        # Displaying the first few rows of the data
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Data Cleaning and Feature Engineering
        data = df.copy()

        # Renaming and encoding categorical columns
        data['thal'].replace({'fixed defect': 'fixed_defect', 'reversable defect': 'reversable_defect'}, inplace=True)
        data['cp'].replace({'typical angina': 'typical_angina', 'atypical angina': 'atypical_angina'}, inplace=True)
        data['restecg'].replace({'normal': 'normal', 'st-t abnormality': 'ST-T_wave_abnormality', 'lv hypertrophy': 'left_ventricular_hypertrophy'}, inplace=True)

        # Creating new dataset with necessary columns
        data_1 = data[['age', 'sex', 'cp', 'dataset', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
        
        # Transforming the target variable
        data_1['target'] = ((data['num'] > 0) * 1).copy()

        # Encoding categorical variables
        data_1['sex'] = (data['sex'] == 'Male') * 1
        data_1['fbs'] = (data['fbs']) * 1
        data_1['exang'] = (data['exang']) * 1
        
        # Renaming columns
        data_1.columns = ['age', 'sex', 'chest_pain_type', 'country', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'Restecg', 
                          'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target']
        
        # Convert categorical columns to numeric using Label Encoding
        label_encoder = LabelEncoder()
        categorical_columns = ['chest_pain_type', 'Restecg', 'thalassemia_type', 'country']  # List the categorical columns

        for col in categorical_columns:
            data_1[col] = label_encoder.fit_transform(data_1[col])

        # Display the cleaned data
        st.subheader("Cleaned Data")
        st.write(data_1.head())

        # Feature Histograms
        st.subheader("Feature Histograms")
        fig, ax = plt.subplots(figsize=(12, 8))
        data_1.hist(bins=20, ax=ax)
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        
        # Select only numeric columns for correlation calculation
        numeric_data = data_1.select_dtypes(include=[np.number])

        # Generate the correlation matrix for numeric columns only
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Split data for modeling
        X = data_1.drop('target', axis=1)
        y = data_1['target']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # Ensure that no non-numeric columns are present in X_train before scaling
        X_train = X_train.select_dtypes(include=[np.number])  # Selecting only numeric columns
        X_test = X_test.select_dtypes(include=[np.number])  # Selecting only numeric columns

        # Scaling the data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and evaluate Random Forest Model
        @st.cache_resource
        def train_random_forest():
            rf_model = RandomForestClassifier(random_state=0, class_weight='balanced')
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train_scaled, y_train)
            return grid_search.best_estimator_, grid_search.best_params_

        # Train Random Forest Model
        best_rf_model, rf_best_params = train_random_forest()

        # Evaluate Random Forest Model
        rf_y_pred = best_rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_y_pred)

        st.subheader("Random Forest Model Statistics")
        st.write(f"Best Hyperparameters: {rf_best_params}")
        st.write(f"Accuracy: {rf_accuracy:.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, rf_y_pred))
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(confusion_matrix(y_test, rf_y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # Train and evaluate XGBoost Model
        @st.cache_resource
        def train_xgb_classifier():
            xgb_model = XGBClassifier(random_state=0)
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 1, 2]
            }
            grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train_scaled, y_train)
            return grid_search.best_estimator_, grid_search.best_params_

        # Train XGBoost Model
        best_xgb_model, xgb_best_params = train_xgb_classifier()

        # Evaluate XGBoost Model
        xgb_y_pred = best_xgb_model.predict(X_test_scaled)
        xgb_accuracy = accuracy_score(y_test, xgb_y_pred)

        st.subheader("XGBoost Model Statistics")
        st.write(f"Best Hyperparameters: {xgb_best_params}")
        st.write(f"Accuracy: {xgb_accuracy:.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, xgb_y_pred))
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(confusion_matrix(y_test, xgb_y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
