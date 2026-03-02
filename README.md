# Deep CSAT: Customer Satisfaction Prediction Model

🤖 **Deep CSAT** - An AI-powered system for predicting E-commerce Customer Satisfaction (CSAT) using Machine Learning and NLP.

## Project Overview

This project predicts customer satisfaction scores using advanced machine learning models combining structured data and text analytics.

### Key Features
- **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost
- **Sentiment Analysis**: TextBlob for analyzing customer remarks
- **Hybrid Modeling**: Combines numeric, categorical, and text features
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score metrics

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 83.00% | 0.6585 | 0.5745 | 0.6133 |
| **Random Forest** | **90.50%** | **0.9118** | **0.6596** | **0.7654** |
| XGBoost | 88.50% | 0.8333 | 0.6383 | 0.7229 |

**Best Model: Random Forest with 90.5% accuracy**

## Dataset
- **Size**: 1000 customer service interactions
- **Training Set**: 800 samples (80%)
- **Testing Set**: 200 samples (20%)
- **Features**: 6 (Response Time, Sentiment Score, Shift, Channel, Tenure, Remarks Length)
- **Target**: Binary Classification (Satisfied/Unsatisfied)

## Technologies Used
- Python 3.10+
- scikit-learn
- XGBoost
- Pandas & NumPy
- TextBlob for NLP
- Matplotlib & Seaborn

## Files
- `Deep_CSAT_Model.ipynb` - Main Jupyter Notebook with full analysis and model training
- `README.md` - This file

## Author
**alwinappu** - Machine Learning Enthusiast

## License
MIT License
