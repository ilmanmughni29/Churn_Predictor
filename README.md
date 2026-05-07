# Telco Customer Churn Prediction

An end-to-end Machine Learning project to predict customer churn in a telecommunications company using classification algorithms, feature engineering, explainable AI (SHAP), and interactive deployment with Streamlit.

---

# Project Overview

Customer churn is one of the biggest challenges in the telecommunications industry. Losing customers can significantly reduce company revenue and increase acquisition costs.

This project aims to:

- Predict whether a customer will churn or not
- Identify important factors influencing churn
- Provide explainable predictions using SHAP
- Deploy the model into an interactive Streamlit dashboard

---

# Business Problem

Telecommunication companies need to identify customers with high churn risk before they leave the service.

By predicting churn early, companies can:

- Improve customer retention
- Reduce revenue loss
- Create targeted retention campaigns
- Optimize customer service strategy

---

# Dataset

Dataset: **Telco Customer Churn Dataset**

Target Variable:

```python
Churn
```

Class Distribution:
- 0 → Customer Retained
- 1 → Customer Churned

---

# Tech Stack

## Programming & Deployment
- Python
- Streamlit

## Data Analysis & Visualization
- Pandas
- NumPy
- Matplotlib
- SHAP

## Machine Learning
- Scikit-learn

---

# Machine Learning Workflow

## 1. Data Cleaning
- Handle missing values
- Fix inconsistent categories
- Remove unnecessary columns

## 2. Exploratory Data Analysis (EDA)
Analyze:
- Customer demographics
- Service usage behavior
- Churn distribution
- Numerical feature distributions
- Correlation between variables

## 3. Feature Engineering
Custom features created:
- `tenure_group`
- `has_internet`
- `num_addon_services`
- `charge_category`

## 4. Preprocessing
- Binary Encoding
- One-Hot Encoding
- Ordinal Encoding
- Robust Scaling

## 5. Handling Imbalanced Data
- SMOTE Oversampling

## 6. Model Training
Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- F2-Score
- ROC-AUC

## 7. Hyperparameter Tuning
- GridSearchCV
- Stratified Cross Validation

## 8. Explainable AI
- SHAP Waterfall Plot
- Feature Importance Analysis

---

# Best Model

Best performing model:
```python
Logistic Regression
```

Main optimization metric:
```python
F2-Score
```

F2-Score prioritizes Recall because:
- False Negative is costly
- Missing churn customers is more dangerous than false alarms

---

# Streamlit Dashboard Features

* Interactive customer input  
* Real-time churn prediction  
* Churn probability score  
* Risk assessment system  
* SHAP explainability visualization  
* Responsive dashboard layout  

---

# SHAP Explainability

This project uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

The SHAP waterfall plot shows:
- Which features increase churn probability
- Which features decrease churn probability
- Contribution magnitude of each feature

Example:
- High monthly charges → increase churn risk
- Long tenure → reduce churn risk
- Month-to-month contract → increase churn risk

---

# Project Structure

```bash
churn_predictor/
│
├── streamlit_app.py
├── requirements.txt
├── runtime.txt
├── telco_churn_pipeline.pkl
├── telco_churn_metadata.pkl
└── README.md
```

---

# Run Locally

## 1. Clone Repository

```bash
git clone <https://github.com/ilmanmughni29/Churn_Predictor/tree/main>
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# Streamlit Cloud Deployment

## Deployment Steps

1. Upload project to GitHub
2. Open Streamlit Cloud
3. Connect GitHub repository
4. Select:
   - Branch
   - Main file:
   
```python
streamlit_app.py
```

5. Deploy

---

# Requirements

```txt
pandas==2.3.3
streamlit==1.56.0
scikit-learn==1.8.0
feature_engine==1.9.4
numpy==1.26.4
matplotlib==3.9.2
shap==0.45.0
```

---

# Future Improvements

Potential future enhancements:
- Add customer retention recommendation system
- Integrate database connection
- Add batch prediction feature
- Add model monitoring
- Deploy with Docker
- Add CI/CD pipeline
- Use CatBoost or LightGBM
- Improve SHAP visualization interactivity

---

# Author

**M. Ilman Mughni**
 
Data Science Enthusiast  
Machine Learning & Analytics Projects

---

# Key Takeaways

This project demonstrates:
- End-to-end ML workflow
- Business-oriented problem solving
- Explainable AI implementation
- Interactive ML deployment
- Custom preprocessing pipelines
- Imbalanced classification handling

---

# Contact

Feel free to connect for discussion, collaboration, or opportunities.

LinkedIn: *[Linkedin Profile](https://www.linkedin.com/in/milmanmughni/)*  
GitHub: *[Github Profile](https://github.com/ilmanmughni29)*
