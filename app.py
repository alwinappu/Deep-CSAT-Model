import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import time

# Page Config
st.set_page_config(page_title="🤖 Deep CSAT", page_icon="🤖", layout="wide")

# Persistent Caching for Models
@st.cache_resource
def train_models_cached(n_samples, test_size):
    np.random.seed(42)
    data = {
        'Experience': np.random.uniform(1, 10, n_samples),
        'Interaction': np.random.uniform(1, 10, n_samples),
        'Quality': np.random.uniform(1, 10, n_samples),
        'Sentiment': np.random.uniform(-1, 1, n_samples),
    }
    df = pd.DataFrame(data)
    df['Target'] = (0.3*df['Experience'] + 0.3*df['Interaction'] + 0.4*df['Quality'] + 2*df['Sentiment'] > 6).astype(int)
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    
    lr = LogisticRegression().fit(X_train, y_train)
    rf = RandomForestClassifier().fit(X_train, y_train)
    xgb = XGBClassifier().fit(X_train, y_train)
    
    metrics = {
        'LR': lr.score(X_test, y_test),
        'RF': rf.score(X_test, y_test),
        'XGB': xgb.score(X_test, y_test)
    }
    
    return {'LR': lr, 'RF': rf, 'XGB': xgb}, metrics

# Initialize state
if 'last_params' not in st.session_state:
    st.session_state.last_params = None

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; background: linear-gradient(45deg, #FF512F, #DD2476); color: white; border: none; font-weight: bold; }
    .reportview-container .main .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📋 Navigation")
    page = st.radio("Go to", ["🏠 Home", "⚙️ Train Model", "🔮 Predictions", "📊 Performance"])
    st.info("Tip: Caching is enabled. Models will persist even after refresh!")

if page == "🏠 Home":
    st.title("🤖 Deep CSAT Dashboard")
    st.markdown("### Intelligent Customer Satisfaction Analysis")
    st.write("Welcome to the next generation of CSAT prediction. Use the sidebar to train models and run real-time predictions.")
    st.image("https://img.freepik.com/free-vector/customer-feedback-concept-illustration_114360-1496.jpg", width=600)

elif page == "⚙️ Train Model":
    st.title("⚙️ Model Training Factory")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        n_samples = st.slider("Number of Samples", 500, 5000, 1000)
        test_size = st.slider("Test Size %", 10, 50, 20)
        
        if st.button("🚀 Train Models"):
            with st.spinner("Training in progress..."):
                models, metrics = train_models_cached(n_samples, test_size)
                st.session_state.last_params = (n_samples, test_size)
                st.success("✅ Models trained and cached successfully!")
                st.balloons()
    
    with col2:
        # Check if cache exists for current params
        cached_models, cached_metrics = train_models_cached(n_samples, test_size)
        if cached_metrics:
            st.markdown("#### 🏆 Training Results")
            for m, acc in cached_metrics.items():
                st.metric(m, f"{acc*100:.1f}%")
            
            fig = px.bar(x=list(cached_metrics.keys()), y=list(cached_metrics.values()), 
                         labels={'x':'Model', 'y':'Accuracy'}, title="Model Comparison",
                         color=list(cached_metrics.keys()))
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predictions":
    st.title("🔮 Make Predictions")
    
    # Try to get models from cache with default or last params
    params = st.session_state.last_params if st.session_state.last_params else (1000, 20)
    models, metrics = train_models_cached(*params)
    
    if models:
        c1, c2 = st.columns(2)
        with c1:
            exp = st.number_input("Experience Score (1-10)", 1.0, 10.0, 5.0)
            inter = st.number_input("Interaction Score (1-10)", 1.0, 10.0, 5.0)
            qual = st.number_input("Product Quality (1-10)", 1.0, 10.0, 5.0)
            text = st.text_area("Customer Review", "I really loved the service!")
            sentiment = TextBlob(text).sentiment.polarity
            
        with c2:
            st.write(f"**Sentiment Score:** {sentiment:.2f}")
            if st.button("🎯 Predict"):
                input_data = [[exp, inter, qual, sentiment]]
                results = {}
                for name, model in models.items():
                    res = "Satisfied 😊" if model.predict(input_data)[0] == 1 else "Unsatisfied 😞"
                    results[name] = res
                
                st.json(results)
    else:
        st.warning("⚠️ Please train models first on the 'Train Model' page!")

elif page == "📊 Performance":
    st.title("📊 Model Performance Analytics")
    params = st.session_state.last_params if st.session_state.last_params else (1000, 20)
    models, metrics = train_models_cached(*params)
    
    if metrics:
        fig = go.Figure(data=[
            go.Table(header=dict(values=['Model', 'Accuracy Score']),
                     cells=dict(values=[list(metrics.keys()), list(metrics.values())]))
        ])
        st.plotly_chart(fig)
    else:
        st.info("Train models to see depth analysis.")

st.markdown("---")
st.caption("Deep CSAT v2.0 | Persistent Analytics Engine")
