# Heart Disease Prediction Model Performance Report  

## Project Overview  
This project evaluates and compares the performance of **Random Forest** and **XGBoost Classifier** models for predicting heart disease using the Cleveland Heart Disease dataset. Both models were fine-tuned through hyperparameter optimization to achieve optimal classification performance.

## Dataset Description  
The **Cleveland Heart Disease Dataset** is a multivariate dataset containing clinical features for cardiovascular disease prediction. Key characteristics:

- **Type**: Multivariate (numerical/categorical)  
- **Attributes**: 14 clinically relevant features (subset from original 76)  
- **Target**: Presence of heart disease (binary classification)  
- **Size**: 303 instances  

### Key Features:
1. Demographic:  
   - Age  
   - Sex  

2. Clinical Measurements:  
   - Chest pain type (4 values)  
   - Resting blood pressure  
   - Serum cholesterol  
   - Fasting blood sugar  
   - Resting electrocardiographic results  

3. Exercise-Induced Metrics:  
   - Maximum heart rate achieved  
   - Exercise-induced angina  
   - ST depression (oldpeak)  
   - Slope of peak exercise ST segment  

4. Cardiac Test Results:  
   - Number of major vessels  
   - Thalassemia  

*Note: This is the most widely used version in ML research, containing only 14 of the original 76 attributes.*

## Key Findings  

### Model Performance  
| **Model**          | **Test Accuracy** | **Key Strengths** |
|--------------------|------------------|------------------|
| **Random Forest**  | **84%**          | Robust, resistant to overfitting, reliable performance |
| **XGBoost**        | **86%**          | Higher accuracy, captures complex relationships |

### Comparative Analysis  
- **Random Forest**: Performs well with clinical data structure and provides strong resistance to overfitting  
- **XGBoost**: Achieves better accuracy by modeling complex relationships between clinical indicators  

### Hyperparameter Configurations  
| **Hyperparameter**       | **Random Forest** | **XGBoost**       |
|--------------------------|-------------------|-------------------|
| Maximum Depth            | 10                | 3                 |
| Learning Rate            | -                 | 0.1               |
| Number of Estimators     | 100               | 50                |
| Gamma                    | -                 | 2                 |
| Subsample Ratio          | -                 | 1.0               |
| Column Sample by Tree    | -                 | 0.8               |
| Minimum Samples per Leaf | 4                 | -                 |
| Minimum Samples for Split| 2                 | -                 |

## Recommendations  
- **Use Random Forest** when:  
  - Clinical interpretability is crucial  
  - Working with limited patient data  
  - Need robust performance across populations  

- **Use XGBoost** when:  
  - Maximizing diagnostic accuracy is critical  
  - Working with complete patient profiles  
  - Can handle slightly longer training times  

## Conclusion  
Both models demonstrate strong performance in heart disease prediction, with **XGBoost** (86% accuracy) slightly outperforming **Random Forest** (84%). The choice depends on clinical requirements - whether prioritizing model stability (Random Forest) or predictive accuracy (XGBoost).

---  
dataseturl https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data



blogPost url https://medium.com/@pjjames095/heart-disease-prediction-using-machine-learning-a-step-by-step-guide-23b953f931d2
