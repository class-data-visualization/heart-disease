import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kaggle
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import GridSearchCV

# Create directories for saving artifacts
os.makedirs('saved_models', exist_ok=True)
os.makedirs('preprocessing', exist_ok=True)

# Download the dataset using Kaggle API
def download_data():
    st.text("Downloading the dataset...")
    try:
        kaggle.api.dataset_download_files('redwankarimsony/heart-disease-data', 
                                         path='datasets/', unzip=True)
        st.text("Dataset downloaded successfully.")
        return 'datasets/heart_disease_uci.csv'
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Load and cache data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Cache preprocessing and training artifacts
@st.cache_resource
def load_artifacts():
    artifacts = {
        'scaler': joblib.load('preprocessing/scaler.pkl'),
        'label_encoders': joblib.load('preprocessing/label_encoders.pkl'),
        'rf_model': joblib.load('saved_models/rf_model.pkl'),
        'xgb_model': joblib.load('saved_models/xgb_model.pkl'),
        'rf_params': joblib.load('saved_models/rf_params.pkl'),
        'xgb_params': joblib.load('saved_models/xgb_params.pkl')
    }
    return artifacts

def save_artifacts(model, params, model_name, scaler, label_encoders):
    joblib.dump(model, f'saved_models/{model_name}_model.pkl')
    joblib.dump(params, f'saved_models/{model_name}_params.pkl')
    joblib.dump(scaler, 'preprocessing/scaler.pkl')
    joblib.dump(label_encoders, 'preprocessing/label_encoders.pkl')

# Initialize session state
if 'artifacts_loaded' not in st.session_state:
    st.session_state.artifacts_loaded = False

# Preprocess data: update the list of categorical columns to include st_slope_type.
# The actual categorical columns (names from the dataset after renaming) are:
# - 'chest_pain_type' (from original 'cp')
# - 'country'         (from original 'dataset')
# - 'Restecg'
# - 'st_slope_type'
# - 'thalassemia_type' (from original 'thal')
categorical_columns = ['chest_pain_type', 'country', 'Restecg', 'st_slope_type', 'thalassemia_type']

def preprocess_data(df):
    data = df.copy()
    label_encoders = {}
    
    # Label encode each categorical column, if it exists in the data.
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        else:
            st.warning(f"Column '{col}' not found in data, skipping encoding for it.")
    return data, label_encoders

# Main processing and model training
if st.button('Download Dataset'):
    dataset_path = download_data()
    
    if dataset_path:
        with st.spinner("Preprocessing the data and training models, please wait..."):
            df = load_data(dataset_path)
            
            # Data preprocessing: fix values and rename columns as required
            data = df.copy()
            data['thal'].replace({'fixed defect': 'fixed_defect', 'reversable defect': 'reversable_defect'}, inplace=True)
            data['cp'].replace({'typical angina': 'typical_angina', 'atypical angina': 'atypical_angina'}, inplace=True)
            data['restecg'].replace({'normal': 'normal', 'st-t abnormality': 'ST-T_wave_abnormality', 
                                     'lv hypertrophy': 'left_ventricular_hypertrophy'}, inplace=True)
            
            # Select and rename columns
            data_1 = data[['age', 'sex', 'cp', 'dataset', 'trestbps', 'chol', 'fbs', 
                           'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
            data_1['target'] = ((data['num'] > 0) * 1).copy()
            
            # Encode binary features
            data_1['sex'] = (data['sex'] == 'Male') * 1
            data_1['fbs'] = (data['fbs']) * 1
            data_1['exang'] = (data['exang']) * 1
            
            # Rename columns for clarity
            data_1.columns = ['age', 'sex', 'chest_pain_type', 'country', 'resting_blood_pressure',
                              'cholesterol', 'fasting_blood_sugar', 'Restecg', 'max_heart_rate_achieved',
                              'exercise_induced_angina', 'st_depression', 'st_slope_type', 
                              'num_major_vessels', 'thalassemia_type', 'target']
            
            # Label encode the categorical columns
            # (This replaces the original string values with numbers, including "flat" in st_slope_type)
            data_encoded, label_encoders = preprocess_data(data_1)
            
            # Train-test split
            X = data_encoded.drop('target', axis=1)
            y = data_encoded['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            
            # Scale all features (now all are numeric)
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Check for existing models; if not, train and save them.
            if not all([os.path.exists(f'saved_models/{m}_model.pkl') for m in ['rf', 'xgb']]):
                # Train Random Forest
                rf_model = RandomForestClassifier(random_state=0, class_weight='balanced')
                rf_grid = {'n_estimators': [50, 100, 150],
                           'max_depth': [None, 10, 20],
                           'min_samples_split': [2, 5, 10],
                           'min_samples_leaf': [1, 2, 4]}
                rf_search = GridSearchCV(rf_model, rf_grid, cv=5, scoring='accuracy')
                rf_search.fit(X_train_scaled, y_train)
                
                # Train XGBoost
                xgb_model = XGBClassifier(random_state=0)
                xgb_grid = {'n_estimators': [50, 100, 150],
                            'max_depth': [3, 5, 7],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'subsample': [0.8, 1.0],
                            'colsample_bytree': [0.8, 1.0],
                            'gamma': [0, 1, 2]}
                xgb_search = GridSearchCV(xgb_model, xgb_grid, cv=5, scoring='accuracy')
                xgb_search.fit(X_train_scaled, y_train)
                
                # Save the trained models and preprocessing artifacts
                save_artifacts(rf_search.best_estimator_, rf_search.best_params_, 'rf', scaler, label_encoders)
                save_artifacts(xgb_search.best_estimator_, xgb_search.best_params_, 'xgb', scaler, label_encoders)
            
            st.session_state.artifacts_loaded = True

# After training is complete or if models are already saved, load artifacts and display tabs.
if st.session_state.artifacts_loaded or all([os.path.exists(f'saved_models/{m}_model.pkl') for m in ['rf', 'xgb']]):
    artifacts = load_artifacts()
    
    # For visualization, load raw data and also re-preprocess it if necessary.
    df = load_data('datasets/heart_disease_uci.csv')
    # We'll use the same renaming but then read raw again for exploration.
    data_1_vis = pd.read_csv('datasets/heart_disease_uci.csv')
    # Apply renaming as above for consistent column names.
    data_1_vis['thal'].replace({'fixed defect': 'fixed_defect', 'reversable defect': 'reversable_defect'}, inplace=True)
    data_1_vis['cp'].replace({'typical angina': 'typical_angina', 'atypical angina': 'atypical_angina'}, inplace=True)
    data_1_vis['restecg'].replace({'normal': 'normal', 'st-t abnormality': 'ST-T_wave_abnormality', 
                                   'lv hypertrophy': 'left_ventricular_hypertrophy'}, inplace=True)
    data_1_vis = data_1_vis[['age', 'sex', 'cp', 'dataset', 'trestbps', 'chol', 'fbs', 
                              'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
    data_1_vis['target'] = ((data_1_vis['num'] > 0) * 1).copy()
    data_1_vis['sex'] = (data_1_vis['sex'] == 'Male') * 1
    data_1_vis['fbs'] = (data_1_vis['fbs']) * 1
    data_1_vis['exang'] = (data_1_vis['exang']) * 1
    data_1_vis.columns = ['age', 'sex', 'chest_pain_type', 'country', 'resting_blood_pressure',
                           'cholesterol', 'fasting_blood_sugar', 'Restecg', 'max_heart_rate_achieved',
                           'exercise_induced_angina', 'st_depression', 'st_slope_type', 
                           'num_major_vessels', 'thalassemia_type', 'target']
    
    tab1, tab2, tab3 = st.tabs(["Data Exploration", "Model Performance", "Live Prediction"])
    
    with tab1:
        st.subheader("Interactive Data Exploration")
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Dataset Preview")
            st.dataframe(df.head())
        with col2:
            st.write("### Column Distributions")
            selected_feature = st.selectbox("Select feature:", data_1_vis.columns)
            fig = px.histogram(data_1_vis, x=selected_feature, nbins=20, 
                               title=f'Distribution of {selected_feature}')
            st.plotly_chart(fig)
        st.write("### Correlation Matrix")
        corr_matrix = data_1_vis.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                        color_continuous_scale='coolwarm')
        st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Model Performance Analysis")
        # Prepare test data using the saved scaler (using numeric columns)
        X_test_scaled = artifacts['scaler'].transform(data_1_vis.drop('target', axis=1).select_dtypes(include=[np.number]))
        y_test = data_1_vis['target']
        
        st.write("## Random Forest Classifier")
        rf_pred = artifacts['rf_model'].predict(X_test_scaled)
        rf_proba = artifacts['rf_model'].predict_proba(X_test_scaled)[:, 1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, rf_pred):.2f}")
        col2.metric("ROC AUC", f"{roc_auc_score(y_test, rf_proba):.2f}")
        col3.metric("Precision", f"{precision_score(y_test, rf_pred):.2f}")
        col4.metric("Recall", f"{recall_score(y_test, rf_pred):.2f}")
        
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, rf_pred)
        fig = ff.create_annotated_heatmap(z=cm, x=['Predicted 0', 'Predicted 1'],
                                          y=['Actual 0', 'Actual 1'], colorscale='Blues')
        st.plotly_chart(fig)
        
        st.write("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, rf_proba)
        fig = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc_score(y_test, rf_proba):.2f})')
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig)
        
        st.write("### Feature Importance")
        importance = pd.DataFrame({
            'Feature': data_1_vis.drop('target', axis=1).columns,
            'Importance': artifacts['rf_model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig)
        
        # (Similar sections for XGBoost can be added here)
    
    with tab3:
        st.subheader("Live Prediction")
        # Create input widgets
        inputs = {}
        col1, col2 = st.columns(2)
        with col1:
            inputs['age'] = st.number_input("Age", min_value=18, max_value=100, value=50)
            inputs['sex'] = st.selectbox("Sex", options=['Male', 'Female'])
            # Use human-friendly choices for chest pain type
            inputs['chest_pain_type'] = st.selectbox("Chest Pain Type", options=['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
        with col2:
            inputs['resting_blood_pressure'] = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
            inputs['cholesterol'] = st.number_input("Cholesterol", min_value=100, max_value=500, value=200)
            inputs['exang'] = st.selectbox("Exercise Induced Angina", options=[True, False])
        
        # For other features not gathered, you may assign default values
        # Here we fill with median or typical values as an example:
        inputs['fasting_blood_sugar'] = 1
        inputs['Restecg'] = 'normal'
        inputs['max_heart_rate_achieved'] = 150
        inputs['exercise_induced_angina'] = 0
        inputs['st_depression'] = 1.0
        inputs['st_slope_type'] = 'flat'
        inputs['num_major_vessels'] = 0
        inputs['thalassemia_type'] = 'normal'
        inputs['country'] = 'Cleveland'
        
        # Convert input dictionary to DataFrame
        inputs_df = pd.DataFrame([inputs])
        
        # Encode the categorical features using artifacts label encoders
        # Note: The keys here must match those used during preprocessing.
        # For example, for chest pain type:
        inputs_df['chest_pain_type'] = artifacts['label_encoders']['chest_pain_type'].transform(inputs_df['chest_pain_type'])
        inputs_df['country'] = artifacts['label_encoders']['country'].transform(inputs_df['country'])
        # For the remaining categorical features, if provided as string, you can perform encoding:
        for cat in ['Restecg', 'st_slope_type', 'thalassemia_type']:
            # If the input is not already numeric, transform it.
            if isinstance(inputs_df[cat].iloc[0], str):
                inputs_df[cat] = artifacts['label_encoders'][cat].transform(inputs_df[cat])
        
        # Encode binary: sex (already in UI converted) and exang (as boolean)
        inputs_df['sex'] = (inputs_df['sex'] == 'Male') * 1
        inputs_df['exang'] = (inputs_df['exang']) * 1
        
        # Ensure the DataFrame column order matches training set's order:
        feature_order = ['age', 'sex', 'chest_pain_type', 'country', 'resting_blood_pressure',
                         'cholesterol', 'fasting_blood_sugar', 'Restecg', 'max_heart_rate_achieved',
                         'exercise_induced_angina', 'st_depression', 'st_slope_type', 
                         'num_major_vessels', 'thalassemia_type']
        inputs_df = inputs_df[feature_order]
        
        # Scale input data
        input_scaled = artifacts['scaler'].transform(inputs_df)
        
        # Make predictions using the Random Forest model (and similarly for XGBoost if desired)
        rf_prediction = artifacts['rf_model'].predict(input_scaled)[0]
        st.write("### Prediction:")
        st.write("Heart Disease" if rf_prediction == 1 else "No Heart Disease")
