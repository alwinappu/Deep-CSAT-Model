# 🤖 Deep CSAT - Streamlit Web Application

## Overview
An interactive web application for Customer Satisfaction Prediction using Machine Learning. This Streamlit app allows real-time predictions and model training.

## Features
- **Home Page**: Dashboard with key metrics
- **Train Model**: Configurable model training with synthetic data
- **Make Predictions**: Real-time satisfaction predictions
- **Performance Analytics**: Visual model comparison and metrics
- **Multiple Models**: Logistic Regression, Random Forest, XGBoost

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
git clone https://github.com/alwinappu/Deep-CSAT-Model.git
cd Deep-CSAT-Model
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. **Home**: View project overview and key metrics
2. **Train Model**: Configure dataset size and train models
3. **Predictions**: Make real-time predictions with custom input
4. **Performance**: View detailed model metrics and comparisons

## Model Performance
- Random Forest: 90.5% accuracy (Best)
- XGBoost: 88.5% accuracy
- Logistic Regression: 83.0% accuracy

## Technologies
- Streamlit (Frontend)
- scikit-learn, XGBoost (ML)
- Pandas, NumPy (Data Processing)
- Plotly (Visualization)

## Running the App
```bash
streamlit run app.py
```
Access at http://localhost:8501

## License
MIT License
