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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve,
                             precision_score, recall_score)

os.makedirs('saved_models', exist_ok=True)
os.makedirs('preprocessing', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

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

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_artifacts():
    return {
        'scaler': joblib.load('preprocessing/scaler.pkl'),
        'label_encoders': joblib.load('preprocessing/label_encoders.pkl'),
        'rf_model': joblib.load('saved_models/rf_model.pkl'),
        'xgb_model': joblib.load('saved_models/xgb_model.pkl'),
        'rf_params': joblib.load('saved_models/rf_params.pkl'),
        'xgb_params': joblib.load('saved_models/xgb_params.pkl')
    }

def save_artifacts(model, params, model_name, scaler, label_encoders):
    joblib.dump(model, f'saved_models/{model_name}_model.pkl')
    joblib.dump(params, f'saved_models/{model_name}_params.pkl')
    joblib.dump(scaler, 'preprocessing/scaler.pkl')
    joblib.dump(label_encoders, 'preprocessing/label_encoders.pkl')

categorical_columns = ['chest_pain_type', 'country', 'Restecg', 'st_slope_type', 'thalassemia_type']

def preprocess_data(df):
    data = df.copy()
    label_encoders = {}
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        else:
            st.warning(f"Column '{col}' not found in data.")
    return data, label_encoders

if 'artifacts_loaded' not in st.session_state:
    st.session_state.artifacts_loaded = False

if st.button('Download Dataset'):
    dataset_path = download_data()
    if dataset_path:
        with st.spinner("Preprocessing and training models..."):
            df = load_data(dataset_path)
            df['thal'].replace({'fixed defect': 'fixed_defect', 'reversable defect': 'reversable_defect'}, inplace=True)
            df['cp'].replace({'typical angina': 'typical_angina', 'atypical angina': 'atypical_angina'}, inplace=True)
            df['restecg'].replace({'st-t abnormality': 'ST-T_wave_abnormality',
                                   'lv hypertrophy': 'left_ventricular_hypertrophy'}, inplace=True)

            target = ((df['num'] > 0) * 1).copy()

            data_1 = df[['age', 'sex', 'cp', 'dataset', 'trestbps', 'chol', 'fbs', 
                         'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
            data_1['target'] = target

            data_1['sex'] = (df['sex'] == 'Male') * 1
            data_1['fbs'] = df['fbs'] * 1
            data_1['exang'] = df['exang'] * 1

            data_1.columns = ['age', 'sex', 'chest_pain_type', 'country', 'resting_blood_pressure',
                              'cholesterol', 'fasting_blood_sugar', 'Restecg', 'max_heart_rate_achieved',
                              'exercise_induced_angina', 'st_depression', 'st_slope_type',
                              'num_major_vessels', 'thalassemia_type', 'target']

            data_encoded, label_encoders = preprocess_data(data_1)
            X = data_encoded.drop('target', axis=1)
            y = data_encoded['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            if not all([os.path.exists(f'saved_models/{m}_model.pkl') for m in ['rf', 'xgb']]):
                # Train Random Forest
                rf = RandomForestClassifier(random_state=0, class_weight='balanced')
                rf_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10],
                           'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
                rf_search = GridSearchCV(rf, rf_grid, cv=3, scoring='accuracy')
                rf_search.fit(X_train_scaled, y_train)

                # Train XGBoost
                xgb = XGBClassifier(random_state=0)
                xgb_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5],
                            'learning_rate': [0.01, 0.1], 'subsample': [0.8, 1.0]}
                xgb_search = GridSearchCV(xgb, xgb_grid, cv=3, scoring='accuracy')
                xgb_search.fit(X_train_scaled, y_train)

                save_artifacts(rf_search.best_estimator_, rf_search.best_params_, 'rf', scaler, label_encoders)
                save_artifacts(xgb_search.best_estimator_, xgb_search.best_params_, 'xgb', scaler, label_encoders)

            st.session_state.artifacts_loaded = True

if st.session_state.artifacts_loaded or all([os.path.exists(f'saved_models/{m}_model.pkl') for m in ['rf', 'xgb']]):
    artifacts = load_artifacts()
    df = load_data('datasets/heart_disease_uci.csv')
    df['thal'].replace({'fixed defect': 'fixed_defect', 'reversable defect': 'reversable_defect'}, inplace=True)
    df['cp'].replace({'typical angina': 'typical_angina', 'atypical angina': 'atypical_angina'}, inplace=True)
    df['restecg'].replace({'st-t abnormality': 'ST-T_wave_abnormality',
                           'lv hypertrophy': 'left_ventricular_hypertrophy'}, inplace=True)

    data_1_vis = df[['age', 'sex', 'cp', 'dataset', 'trestbps', 'chol', 'fbs',
                     'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
    data_1_vis['target'] = ((df['num'] > 0) * 1).copy()
    data_1_vis['sex'] = (df['sex'] == 'Male') * 1
    data_1_vis['fbs'] = df['fbs'] * 1
    data_1_vis['exang'] = df['exang'] * 1
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
            fig = px.histogram(data_1_vis, x=selected_feature, nbins=20)
            st.plotly_chart(fig)

        st.write("### Correlation Matrix")
        numeric_df = data_1_vis.select_dtypes(include=[np.number]) 
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
        
        with st.expander("Additional Visualizations"):
            st.write("#### Target Distribution")
            fig_target = px.histogram(data_1_vis, x="target", text_auto=True, nbins=2, 
                                      title="Distribution of Heart Disease Status", labels={"target": "Heart Disease"})
            st.plotly_chart(fig_target)
            
            st.write("#### Age Distribution by Heart Disease Status")
            fig_box = px.box(data_1_vis, x="target", y="age", 
                             title="Age by Heart Disease Status", labels={"target": "Heart Disease", "age": "Age"})
            st.plotly_chart(fig_box)
            
            st.write("#### Cholesterol vs. Resting Blood Pressure")
            fig_scatter = px.scatter(data_1_vis, x="cholesterol", y="resting_blood_pressure", color="target",
                                     title="Cholesterol vs. Resting Blood Pressure", labels={"target": "Heart Disease"})
            st.plotly_chart(fig_scatter)
            
            st.write("#### Scatter Matrix")
            fig_matrix = px.scatter_matrix(data_1_vis, dimensions=["age", "resting_blood_pressure", "cholesterol", "max_heart_rate_achieved"], 
                                           color="target", title="Scatter Matrix of Selected Features")
            st.plotly_chart(fig_matrix)

    with tab2:
        st.subheader("Model Performance Analysis")
        # Prepare data for evaluation
        data_vis_encoded = data_1_vis.copy()
        for col, le in artifacts['label_encoders'].items():
            if col in data_vis_encoded.columns:
                data_vis_encoded[col] = le.transform(data_vis_encoded[col])
        data_vis_encoded['exercise_induced_angina'] = data_vis_encoded['exercise_induced_angina'] * 1

        X_vis = data_vis_encoded.drop('target', axis=1)
        y_true = data_vis_encoded['target']
        X_scaled = artifacts['scaler'].transform(X_vis)

        # Random Forest Performance
        st.write("## Random Forest Classifier")
        rf_pred = artifacts['rf_model'].predict(X_scaled)
        rf_proba = artifacts['rf_model'].predict_proba(X_scaled)[:, 1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_true, rf_pred):.2f}")
        col2.metric("ROC AUC", f"{roc_auc_score(y_true, rf_proba):.2f}")
        col3.metric("Precision", f"{precision_score(y_true, rf_pred):.2f}")
        col4.metric("Recall", f"{recall_score(y_true, rf_pred):.2f}")

        st.write("### Random Forest - Confusion Matrix")
        cm_rf = confusion_matrix(y_true, rf_pred)
        fig_rf_cm = ff.create_annotated_heatmap(z=cm_rf, x=['Pred 0', 'Pred 1'], y=['Actual 0', 'Actual 1'], colorscale='Blues')
        st.plotly_chart(fig_rf_cm)

        st.write("### Random Forest - Feature Importance")
        fi_rf = pd.DataFrame({
            'Feature': X_vis.columns,
            'Importance': artifacts['rf_model'].feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.plotly_chart(px.bar(fi_rf, x='Importance', y='Feature', orientation='h'))

        st.write("### Random Forest - ROC Curve")
        fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_proba)
        fig_rf_roc = px.area(x=fpr_rf, y=tpr_rf, title="ROC Curve - Random Forest", 
                             labels={"x": "False Positive Rate", "y": "True Positive Rate"})
        fig_rf_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline")
        st.plotly_chart(fig_rf_roc)

        st.markdown("---")
        # XGBoost Performance
        st.write("## XGBoost Classifier")
        xgb_pred = artifacts['xgb_model'].predict(X_scaled)
        xgb_proba = artifacts['xgb_model'].predict_proba(X_scaled)[:, 1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_true, xgb_pred):.2f}")
        col2.metric("ROC AUC", f"{roc_auc_score(y_true, xgb_proba):.2f}")
        col3.metric("Precision", f"{precision_score(y_true, xgb_pred):.2f}")
        col4.metric("Recall", f"{recall_score(y_true, xgb_pred):.2f}")

        st.write("### XGBoost - Confusion Matrix")
        cm_xgb = confusion_matrix(y_true, xgb_pred)
        fig_xgb_cm = ff.create_annotated_heatmap(z=cm_xgb, x=['Pred 0', 'Pred 1'], y=['Actual 0', 'Actual 1'], colorscale='Blues')
        st.plotly_chart(fig_xgb_cm)

        st.write("### XGBoost - Feature Importance")
        fi_xgb = pd.DataFrame({
            'Feature': X_vis.columns,
            'Importance': artifacts['xgb_model'].feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.plotly_chart(px.bar(fi_xgb, x='Importance', y='Feature', orientation='h'))

        st.write("### XGBoost - ROC Curve")
        fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_proba)
        fig_xgb_roc = px.area(x=fpr_xgb, y=tpr_xgb, title="ROC Curve - XGBoost", 
                              labels={"x": "False Positive Rate", "y": "True Positive Rate"})
        fig_xgb_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline")
        st.plotly_chart(fig_xgb_roc)

    with tab3:
        st.subheader("Live Prediction")
        input_data = {}
        col1, col2 = st.columns(2)
        with col1:
            input_data['age'] = st.number_input("Age", 18, 100, 50)
            input_data['sex'] = st.selectbox("Sex", ['Male', 'Female'])
            input_data['chest_pain_type'] = st.selectbox("Chest Pain Type", ['typical_angina', 'atypical_angina', 'non-anginal', 'asymptomatic'])
        with col2:
            input_data['resting_blood_pressure'] = st.number_input("Resting BP", 80, 200, 120)
            input_data['cholesterol'] = st.number_input("Cholesterol", 100, 500, 200)
            input_data['exang'] = st.selectbox("Exercise Induced Angina", [True, False])

        input_data.update({
            'fasting_blood_sugar': 1,
            'Restecg': 'normal',
            'max_heart_rate_achieved': 150,
            'exercise_induced_angina': 0,
            'st_depression': 1.0,
            'st_slope_type': 'flat',
            'num_major_vessels': 0,
            'thalassemia_type': 'normal',
            'country': 'Cleveland'
        })

        input_df = pd.DataFrame([input_data])
        for col in categorical_columns:
            if col in input_df and isinstance(input_df[col].iloc[0], str):
                input_df[col] = artifacts['label_encoders'][col].transform(input_df[col])
        input_df['sex'] = (input_df['sex'] == 'Male') * 1
        input_df['exang'] = input_df['exang'] * 1

        input_df = input_df[X_vis.columns]
        input_scaled = artifacts['scaler'].transform(input_df)
        prediction = artifacts['rf_model'].predict(input_scaled)[0]
        st.write("### Prediction:", "Heart Disease" if prediction == 1 else "No Heart Disease")
