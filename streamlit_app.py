import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title='Telco Customer Churn Predictor',
    page_icon='📉',
    layout='wide'
)

# =====================================================
# CUSTOM TRANSFORMERS
# =====================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Recreates all feature-engineering steps from training."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # tenure_group
        def tenure_group(t):
            if t <= 12:
                return '0-1 Year'
            elif t <= 24:
                return '1-2 Years'
            elif t <= 48:
                return '2-4 Years'
            else:
                return '4+ Years'

        X['tenure_group'] = X['tenure'].apply(tenure_group)

        # has_internet
        X['has_internet'] = (X['InternetService'] != 'No').astype(int)

        # num_addon_services
        addon_cols = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport'
        ]

        X['num_addon_services'] = X[addon_cols].apply(
            lambda row: (row == 'Yes').sum(), axis=1
        )

        # charge_category
        X['charge_category'] = pd.cut(
            X['MonthlyCharges'],
            bins=[0, 35, 65, 95, 120],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):

    BINARY_COLS = ['Dependents', 'PaperlessBilling']

    BINARY_MAP = {
        'Yes': 1,
        'No': 0
    }

    CONTRACT_MAP = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }

    TENURE_MAP = {
        '0-1 Year': 0,
        '1-2 Years': 1,
        '2-4 Years': 2,
        '4+ Years': 3
    }

    CHARGE_MAP = {
        'Low': 0,
        'Medium': 1,
        'High': 2,
        'Very High': 3
    }

    OHE_COLS = [
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport'
    ]

    def fit(self, X, y=None):

        X_temp = X.copy()

        X_temp = pd.get_dummies(
            X_temp,
            columns=self.OHE_COLS,
            drop_first=True
        )

        self.all_columns_ = X_temp.columns.tolist()

        return self

    def transform(self, X):

        X = X.copy()

        # binary encoding
        for col in self.BINARY_COLS:
            X[col] = X[col].map(self.BINARY_MAP)

        # ordinal encoding
        X['Contract'] = X['Contract'].map(self.CONTRACT_MAP)
        X['tenure_group'] = X['tenure_group'].map(self.TENURE_MAP)
        X['charge_category'] = X['charge_category'].map(self.CHARGE_MAP)

        # one hot encoding
        X = pd.get_dummies(
            X,
            columns=self.OHE_COLS,
            drop_first=True
        )

        # align columns
        for col in self.all_columns_:
            if col not in X.columns:
                X[col] = 0

        X = X[self.all_columns_]

        # convert bool to int
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)

        return X


class FeatureScaler(BaseEstimator, TransformerMixin):

    SCALE_COLS = ['tenure', 'MonthlyCharges']

    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.SCALE_COLS])
        return self

    def transform(self, X):

        X = X.copy()

        X[self.SCALE_COLS] = self.scaler.transform(
            X[self.SCALE_COLS]
        )

        return X


# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource

def load_model():

    with open('model/artifacts/telco_churn_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model/artifacts/telco_churn_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    return model, metadata


model, metadata = load_model()

# =====================================================
# HEADER
# =====================================================
st.title('📉 Telco Customer Churn Predictor')
st.markdown('Predict customer churn probability using Machine Learning.')

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header('Customer Input')

# numerical

tenure = st.sidebar.slider(
    'Tenure (Months)',
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.sidebar.slider(
    'Monthly Charges',
    min_value=0.0,
    max_value=120.0,
    value=70.0
)

# categorical

dependents = st.sidebar.selectbox(
    'Dependents',
    ['Yes', 'No']
)

paperless = st.sidebar.selectbox(
    'Paperless Billing',
    ['Yes', 'No']
)

contract = st.sidebar.selectbox(
    'Contract',
    ['Month-to-month', 'One year', 'Two year']
)

internet_service = st.sidebar.selectbox(
    'Internet Service',
    ['DSL', 'Fiber optic', 'No']
)

online_security = st.sidebar.selectbox(
    'Online Security',
    ['Yes', 'No']
)

online_backup = st.sidebar.selectbox(
    'Online Backup',
    ['Yes', 'No']
)

device_protection = st.sidebar.selectbox(
    'Device Protection',
    ['Yes', 'No']
)

tech_support = st.sidebar.selectbox(
    'Tech Support',
    ['Yes', 'No']
)

# =====================================================
# INPUT DATAFRAME
# =====================================================
input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'Dependents': [dependents],
    'PaperlessBilling': [paperless],
    'Contract': [contract],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support]
})

# =====================================================
# SHOW INPUT
# =====================================================
st.subheader('📋 Customer Profile')
st.dataframe(input_df, use_container_width=True)

# =====================================================
# PREDICTION
# =====================================================
if st.button('Predict Churn'):

    probability = model.predict_proba(input_df)[0][1]

    threshold = metadata.get('optimal_threshold', 0.5)

    prediction = int(probability >= threshold)

    st.subheader('🎯 Prediction Result')

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            'Churn Probability',
            f'{probability:.2%}'
        )

    with col2:
        if prediction == 1:
            st.error('⚠️ Customer Potentially Churn')
        else:
            st.success('✅ Customer Likely Retained')

    # =====================================================
    # RISK LEVEL
    # =====================================================
    st.subheader('📊 Risk Assessment')

    if probability >= 0.75:
        st.error('HIGH RISK')

    elif probability >= 0.50:
        st.warning('MEDIUM RISK')

    else:
        st.success('LOW RISK')

    # =====================================================
    # SHAP EXPLANATION
    # =====================================================
    st.subheader('🧠 Model Explainability (SHAP)')

    try:

        # ambil classifier
        classifier = model.named_steps['classifier']

        # preprocessing manually
        transformed_data = model.named_steps['feature_engineer'].transform(input_df)

        transformed_data = model.named_steps['encoder'].transform(transformed_data)

        transformed_data = model.named_steps['scaler'].transform(transformed_data)

        # dataframe agar feature names muncul
        transformed_df = pd.DataFrame(
            transformed_data,
            columns=classifier.feature_names_in_
        )

        # shap explainer
        explainer = shap.LinearExplainer(
            classifier,
            transformed_df
        )

        shap_values = explainer(transformed_df)

        # waterfall plot
        fig = plt.figure(figsize=(10, 5))

        shap.plots.waterfall(
            shap_values[0],
            max_display=10,
            show=False
        )

        st.pyplot(fig)

    except Exception as e:
        st.warning(f'SHAP visualization unavailable: {e}')
        
    

# =====================================================
# FOOTER
# =====================================================
st.markdown('---')
st.caption('Built with Streamlit + Machine Learning')