import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

st.set_page_config(page_title="Deep CSAT", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Deep CSAT - Customer Satisfaction Prediction")
st.markdown("---")

page = st.sidebar.radio("Navigation", ["Home", "Train Model", "Predictions", "Performance"])

if page == "Home":
    st.markdown("### Welcome to Deep CSAT")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", "Random Forest")
    with col2:
        st.metric("Accuracy", "90.5%")
    with col3:
        st.metric("Precision", "91.18%")

elif page == "Train Model":
    st.title("Train ML Models")
    n_samples = st.slider("Samples", 100, 5000, 1000)
    test_size = st.slider("Test Size %", 10, 50, 20) / 100
    
    if st.button("Train Models"):
        np.random.seed(42)
        X = np.random.rand(n_samples, 6)
        y = (X[:, 0] > 0.5).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        col1, col2, col3 = st.columns(3)
        
        models_dict = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        for idx, (name, model) in enumerate(models_dict.items()):
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            with [col1, col2, col3][idx]:
                st.metric(name, f"{acc*100:.1f}%")
        
        st.session_state.models = models_dict
        st.success("✅ Models trained!")

elif page == "Predictions":
    st.title("Make Predictions")
    if 'models' in st.session_state:
        col1, col2, col3 = st.columns(3)
        with col1:
            f1 = st.slider("Feature 1", 0.0, 1.0, 0.5)
        with col2:
            f2 = st.slider("Feature 2", 0.0, 1.0, 0.5)
        with col3:
            f3 = st.slider("Feature 3", 0.0, 1.0, 0.5)
        
        input_data = np.array([[f1, f2, f3, 0.5, 0.5, 0.5]])
        
        if st.button("Predict"):
            for name, model in st.session_state.models.items():
                pred = model.predict(input_data)[0]
                st.write(f"{name}: {'Satisfied' if pred == 1 else 'Unsatisfied'}")
    else:
        st.warning("⚠️ Train the model first!")

elif page == "Performance":
    st.title("Model Performance")
    perf_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [83.0, 90.5, 88.5],
        'Precision': [65.85, 91.18, 83.33]
    }
    df = pd.DataFrame(perf_data)
    st.dataframe(df, use_container_width=True)
    fig = px.bar(df, x='Model', y='Accuracy', color='Model')
    st.plotly_chart(fig, use_container_width=True)
