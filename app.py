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
st.set_page_config(page_title="🌈 Deep CSAT v2.0", page_icon="🤖", layout="wide")

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

# Initialize state at the very top
if 'last_params' not in st.session_state or st.session_state.last_params is None:
    st.session_state.last_params = (1000, 20)

# Custom CSS for Vibrant Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
    }

    /* Gradient Title */
    .rainbow-text {
        background: linear-gradient(to right, #ff00cc, #3333ff, #00ffff, #33ff33, #ffff00, #ff00cc);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        animation: rainbow 5s linear infinite;
        margin-bottom: 0px;
    }

    @keyframes rainbow {
        to { background-position: 200% center; }
    }

    /* Cards with neon border */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 20px rgba(0, 198, 255, 0.1);
        transition: all 0.4s ease;
        margin-bottom: 2rem;
    }
    
    .glass-card:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(0, 198, 255, 0.3);
        border: 1px solid rgba(0, 198, 255, 0.5);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(48, 43, 99, 0.8) 0%, rgba(36, 36, 62, 0.8) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
        
    /* Sidebar text and labels - Enhanced visibility */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }
    
    /* Radio buttons styling */
    [data-testid="stSidebar"] [role="radio"] {
        accent-color: #ff00cc !important;
    }
    
    /* Make sidebar text more visible */
    [data-testid="stSidebar"] {
        color: #ffffff !important;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 50px;
        padding: 1rem 2.5rem;
        font-weight: 800;
        border: none;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(245, 87, 108, 0.6);
    }
    
    h1, h2, h3 {
        color: #00d2ff !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #ff00cc !important;'>🕹️ MENU</h1>", unsafe_allow_html=True)
    page = st.radio("Select Module", ["🏠 Dashboard", "🚀 Model Factory", "🎯 AI Predictions", "📊 Deep Analytics"])
    st.markdown("---")
    st.success("✨ Persistent Core Online")
    st.info("The ML brain is cached and ready for instant insights!")

# Helper to get params safely
def get_safe_params():
    params = st.session_state.get('last_params', (1000, 20))
    if params is None or len(params) != 2:
        return (1000, 20)
    return params

# Home Page
if page == "🏠 Dashboard":
    st.markdown("<h1 class='rainbow-text'>DEEP CSAT v2.0</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #e0e0e0 !important;'>Future-Ready Customer Satisfaction Intelligence</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'><h2 style='color:#ff00cc !important'>⚡ Neural Speed</h2><p>Sub-millisecond processing using optimized XGBoost and Random Forest architectures.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h2 style='color:#00ffff !important'>🌈 Neon UI</h2><p>Vibrant glassmorphism interface designed for maximum engagement and clarity.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h2 style='color:#33ff33 !important'>🧠 Ensemble AI</h2><p>Combines Logistic Regression with advanced ensemble methods for robust accuracy.</p></div>", unsafe_allow_html=True)
    
    st.image("https://img.freepik.com/free-vector/abstract-techno-background-with-connecting-dots-lines_1048-11812.jpg", use_column_width=True)

# Training Page
elif page == "🚀 Model Factory":
    st.markdown("<h1 style='text-align: center;'>🏭 Model Factory</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("### ⚙️ Build Parameters")
        n_samples = st.slider("Dataset Population", 500, 5000, 1000)
        test_size = st.slider("Validation Split (%)", 10, 50, 20)
        
        if st.button("🚀 IGNITE TRAINING"):
            with st.status("🛠️ Building Neural Networks...", expanded=True) as status:
                models, metrics = train_models_cached(n_samples, test_size)
                st.session_state.last_params = (n_samples, test_size)
                time.sleep(1)
                st.balloons()
                status.update(label="✅ Success! Models Synchronized.", state="complete")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("### 🏆 Active Performance")
        params = get_safe_params()
        models, metrics = train_models_cached(*params)
        if metrics:
            for m, acc in metrics.items():
                st.write(f"**{m} Core**: `{acc*100:.2f}%` Accuracy")
                st.progress(acc)
        st.markdown("</div>", unsafe_allow_html=True)

# Predictions Page
elif page == "🎯 AI Predictions":
    st.markdown("<h1 style='text-align: center;'>🎯 AI Prediction Hub</h1>", unsafe_allow_html=True)
    params = get_safe_params()
    models, metrics = train_models_cached(*params)
    
    if models:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("### 📝 Customer Profile")
            exp = st.slider("Product Experience", 1, 10, 5)
            inter = st.slider("Support Interaction", 1, 10, 5)
            qual = st.slider("Build Quality", 1, 10, 5)
            review = st.text_area("Customer Sentiment (Text)", "The experience was absolutely brilliant and I'm very happy!")
            sentiment = TextBlob(review).sentiment.polarity
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("### 🔮 Predicted Outcome")
            st.metric("NLP Sentiment Score", f"{sentiment:.2f}")
            if st.button("🎯 RUN PREDICTION"):
                input_data = [[exp, inter, qual, sentiment]]
                results = {name: ("Satisfied 😊" if model.predict(input_data)[0] == 1 else "Unsatisfied 😞") for name, model in models.items()}
                
                for name, res in results.items():
                    if "Satisfied" in res:
                        st.success(f"**{name}**: {res}")
                    else:
                        st.error(f"**{name}**: {res}")
                st.snow()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Factory Offline. Please train models first!")

# Performance Page
elif page == "📊 Deep Analytics":
    st.markdown("<h1 style='text-align: center;'>📊 Deep Performance Analytics</h1>", unsafe_allow_html=True)
    params = get_safe_params()
    models, metrics = train_models_cached(*params)
    
    if metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()), color=list(metrics.keys()), 
                         title="Accuracy Benchmarks", template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            fig_radar = go.Figure(go.Scatterpolar(r=list(metrics.values()), theta=list(metrics.keys()), fill='toself', line_color='#ff00cc'))
            fig_radar.update_layout(template="plotly_dark", title="Neural Radar Map")
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #888; margin-top: 5rem;'>Built with 💓 | DEEP CSAT v2.0 Persistent Core</p>", unsafe_allow_html=True)
