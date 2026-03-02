import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob

# Page Configuration
st.set_page_config(
    page_title="🤖 Deep CSAT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        font-size: 16px;
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .model-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    h1 {
        color: #667eea;
        font-size: 3em;
        text-align: center;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2 {
        color: #764ba2;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    h3 {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 style="color: white; margin: 0; text-shadow: none;">🤖 Deep CSAT</h1>
    <p style="text-align: center; font-size: 18px; margin: 10px 0 0 0;">Customer Satisfaction Prediction using Machine Learning & NLP</p>
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "📋 Navigation",
    ["🏠 Home", "🤖 Train Model", "🔮 Predictions", "📊 Performance"],
    label_visibility="collapsed"
)

# Home Page
if "Home" in page:
    st.markdown("### 👋 Welcome to Deep CSAT")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="stat-card">
        <h3 style="margin: 0; color: white;">🏆 Best Model</h3>
        <p style="margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">Random Forest</p>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="stat-card">
        <h3 style="margin: 0; color: white;">🎯 Accuracy</h3>
        <p style="margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">90.5%</p>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""<div class="stat-card">
        <h3 style="margin: 0; color: white;">📈 Precision</h3>
        <p style="margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">91.18%</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 Key Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ✅ **Multi-Model Training**
        - Logistic Regression
        - Random Forest
        - XGBoost
        """)
    with col2:
        st.markdown("""
        ✅ **Advanced Analysis**
        - Real-time Predictions
        - Performance Metrics
        - Model Comparison
        """)
    
    st.markdown("---")
    st.markdown("### 🎯 How It Works")
    with st.expander("📖 Learn More", expanded=False):
        st.markdown("""
        1. **Train Models**: Configure your dataset and train multiple ML models
        2. **Make Predictions**: Use the trained models to predict customer satisfaction
        3. **Analyze Performance**: Compare model metrics and visualize results
        """)

# Train Model Page
elif "Train" in page:
    st.markdown("### 🤖 Train ML Models")
    st.markdown("Configure your training parameters and train multiple models")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("📊 Number of Samples", 100, 5000, 1000, step=100)
    with col2:
        test_size = st.slider("🧪 Test Size %", 10, 50, 20, step=5) / 100
    
    if st.button("🚀 Train Models", use_container_width=True, key="train_btn"):
        with st.spinner("⏳ Training models..."):
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
            
            results = {}
            for idx, (name, model) in enumerate(models_dict.items()):
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                results[name] = acc
                with [col1, col2, col3][idx]:
                    st.markdown(f"""<div class="model-card">
                    <h3 style="margin: 0; color: white;">{name}</h3>
                    <p style="margin: 10px 0 0 0; font-size: 28px; font-weight: bold;">{acc*100:.1f}%</p>
                    </div>""", unsafe_allow_html=True)
            
            st.session_state.models = models_dict
            st.markdown("""<div class="success-card">
            ✅ Models trained successfully!
            </div>""", unsafe_allow_html=True)
            
            # Training summary
            st.markdown("### 📊 Training Summary")
            summary_data = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy (%)': [v*100 for v in results.values()]
            })
            st.dataframe(summary_data, use_container_width=True)

# Predictions Page
elif "Prediction" in page:
    st.markdown("### 🔮 Make Predictions")
    st.markdown("Input customer features and get predictions from multiple models")
    st.markdown("---")
    
    if 'models' in st.session_state:
        st.markdown("#### 📝 Customer Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            f1 = st.slider("📊 Feature 1", 0.0, 1.0, 0.5, key="f1")
        with col2:
            f2 = st.slider("📊 Feature 2", 0.0, 1.0, 0.5, key="f2")
        with col3:
            f3 = st.slider("📊 Feature 3", 0.0, 1.0, 0.5, key="f3")
        
        input_data = np.array([[f1, f2, f3, 0.5, 0.5, 0.5]])
        
        if st.button("🎯 Predict Satisfaction", use_container_width=True, key="predict_btn"):
            st.markdown("---")
            st.markdown("#### 🎯 Prediction Results")
            col1, col2, col3 = st.columns(3)
            
            for idx, (name, model) in enumerate(st.session_state.models.items()):
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1] * 100
                status = '😊 Satisfied' if pred == 1 else '😞 Unsatisfied'
                
                with [col1, col2, col3][idx]:
                    st.markdown(f"""<div class="model-card">
                    <h4 style="margin: 0; color: white;">{name}</h4>
                    <p style="margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">{status}</p>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">Confidence: {prob:.1f}%</p>
                    </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please train models first on the 'Train Model' page!")

# Performance Page
elif "Performance" in page:
    st.markdown("### 📊 Model Performance Analysis")
    st.markdown("Compare and analyze model metrics and visualizations")
    st.markdown("---")
    
    perf_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [83.0, 90.5, 88.5],
        'Precision': [65.85, 91.18, 83.33],
        'Recall': [57.45, 65.96, 63.83],
        'F1-Score': [61.33, 76.54, 72.29]
    }
    df = pd.DataFrame(perf_data)
    
    st.markdown("#### 📈 Performance Metrics")
    st.dataframe(df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Accuracy Comparison")
        fig_acc = px.bar(df, x='Model', y='Accuracy', 
                        color='Accuracy',
                        color_continuous_scale='Viridis',
                        title="Model Accuracy Comparison")
        fig_acc.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        st.markdown("#### Precision vs Recall")
        fig_pr = px.scatter(df, x='Precision', y='Recall', 
                           size='Accuracy',
                           color='Model',
                           title="Precision vs Recall Analysis",
                           hover_data=['Accuracy'])
        fig_pr.update_layout(height=400)
        st.plotly_chart(fig_pr, use_container_width=True)
    
    st.markdown("#### All Metrics Comparison")
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig_radar = go.Figure()
    
    for idx, model in enumerate(df['Model']):
        fig_radar.add_trace(go.Scatterpolar(
            r=df.loc[idx, metrics_cols].tolist(),
            theta=metrics_cols,
            fill='toself',
            name=model
        ))
    
    fig_radar.update_layout(height=500, title="Comprehensive Model Comparison")
    st.plotly_chart(fig_radar, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #667eea; padding: 20px;">
    <p>🚀 Deep CSAT - Advanced Customer Satisfaction Prediction</p>
    <p style="font-size: 12px;">Powered by Streamlit | Built with ❤️</p>
</div>
""", unsafe_allow_html=True)
