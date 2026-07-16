# 📈 Marketing Campaign Conversion Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-006400)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![Feature Engineering](https://img.shields.io/badge/Feature-Engineering-success)
![Hyperparameter Tuning](https://img.shields.io/badge/Hyperparameter-Tuning-blueviolet)
![EDA](https://img.shields.io/badge/EDA-Completed-orange)


> **An end-to-end Machine Learning project that predicts customer conversion using demographic, campaign, and engagement data. The project integrates business-driven feature engineering, model comparison, explainable AI, and an interactive Streamlit dashboard to support data-driven marketing decisions.**

---

# 🚀 Key Highlights

- Built a complete Machine Learning pipeline from data preprocessing to deployment.
- Performed comprehensive Exploratory Data Analysis (EDA) to identify customer behavior and campaign trends.
- Engineered multiple business-driven features to improve predictive performance.
- Compared multiple classification models and selected the best-performing model based on quantitative evaluation metrics.
- Performed hyperparameter tuning to optimize model performance.
- Applied SHAP Explainable AI to interpret feature contributions and improve model transparency.
- Developed an interactive Streamlit dashboard for real-time customer conversion prediction.
- Integrated business KPI calculations to bridge Machine Learning predictions with marketing insights.
- Generated actionable business recommendations based on customer behavior and campaign performance.

---

# 🎯 Business Problem

Digital marketing campaigns generate large volumes of customer interaction data. Identifying customers who are most likely to convert enables organizations to:

- Improve campaign effectiveness
- Optimize advertising budget allocation
- Increase conversion rates
- Improve customer targeting
- Support data-driven marketing decisions

This project predicts whether a customer is likely to convert based on demographic information, campaign characteristics, and engagement metrics.

---

# 🔄 Project Workflow

```text
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Streamlit Deployment
```

---

# 📊 Dataset

The dataset contains customer demographics, campaign information, and engagement metrics.

### Customer Information

- Age
- Gender
- Income
- Previous Purchases
- Loyalty Points

### Campaign Information

- Campaign Channel
- Campaign Type
- Ad Spend
- Click Through Rate (CTR)
- Conversion Rate

### Customer Engagement

- Website Visits
- Pages Per Visit
- Time On Site
- Social Shares
- Email Opens
- Email Clicks

---

# ⚙️ Feature Engineering

Rather than relying solely on raw features, several business-oriented metrics were engineered to improve predictive performance.

### Engagement Score

```text
WebsiteVisits
+ PagesPerVisit
+ TimeOnSite
+ SocialShares
+ EmailOpens
+ EmailClicks
```

### Estimated Clicks

```text
WebsiteVisits × ClickThroughRate
```

### Estimated Cost Per Click (CPC)

```text
AdSpend / Estimated_Clicks
```

### Email Engagement Rate

```text
EmailClicks / EmailOpens
```

These engineered features provide richer representations of customer engagement and campaign performance.

---

# 📈 Exploratory Data Analysis

Comprehensive EDA was performed to understand customer behavior and campaign effectiveness.

Analysis included:

- Missing Value Analysis
- Distribution Analysis
- Correlation Heatmap
- Campaign Performance Analysis
- Customer Behaviour Analysis
- Conversion Distribution
- Feature Relationship Analysis

---

# 🤖 Machine Learning Pipeline

### Data Preprocessing

- Missing Value Handling
- One-Hot Encoding
- Feature Engineering
- Train-Test Split

### Model Optimization

- Hyperparameter Tuning
- Cross Validation
- Performance Comparison
- Best Model Selection

### Models Evaluated

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|---------:|----------:|--------:|---------:|---------:|
| Logistic Regression | 70.19% | 92.08% | 72.18% | 80.93% | 0.676 |
| Random Forest | 88.50% | 88.54% | 99.79% | 93.83% | 0.801 |
| **XGBoost** | **91.31%** | **91.96%** | **98.72%** | **95.22%** | **0.807** |

---

# 🏆 Final Model

After comparing multiple machine learning algorithms, **XGBoost Classifier** was selected as the final deployment model due to its superior overall performance.

| Metric | Score |
|---------|-------|
| **Accuracy** | **91.31%** |
| **Precision** | **91.96%** |
| **Recall** | **98.72%** |
| **F1 Score** | **95.22%** |
| **ROC-AUC** | **0.807** |

The model achieves a strong balance between identifying potential customer conversions and minimizing misclassification.

---

# 🔍 Explainable AI

To improve model interpretability, **SHAP (SHapley Additive Explanations)** was used.

The project includes:

- Global Feature Importance
- Local Prediction Explanation
- Feature Contribution Analysis

---

# 📊 Business KPIs

The Streamlit dashboard automatically computes key business metrics.

- Engagement Score
- Estimated Clicks
- Estimated Cost Per Click (CPC)
- Email Engagement Rate

The dashboard also displays the formulas used to compute each KPI, improving transparency for end users.

---

# 💡 Business Recommendations

Based on the analysis and model predictions, the following recommendations can help improve campaign performance:

- **Increase investment in high-performing campaign channels** that consistently achieve higher conversion rates.
- **Prioritize highly engaged customers**, as engagement-related features were strong indicators of conversion.
- **Optimize advertising spend** by monitoring Estimated CPC and reallocating budget toward cost-effective campaigns.
- **Strengthen email marketing strategies** to improve Email Engagement Rate through personalized content and targeted campaigns.
- **Focus on customer retention**, rewarding loyal customers with personalized offers and loyalty incentives.
- **Leverage customer segmentation** to design campaign-specific marketing strategies for different customer groups.
- **Continuously monitor campaign KPIs** such as Engagement Score, Estimated Clicks, CPC, and Email Engagement Rate to support data-driven marketing decisions.

---

# 💻 Streamlit Dashboard

The interactive dashboard enables users to:

- Enter customer and campaign details
- Predict customer conversion in real time
- View conversion probability
- Explore KPI calculation formulas


</p>

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Pickle

---

# 📂 Repository Structure

```text
marketing-campaign-conversion-prediction
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── model
│   ├── model.pkl
│   └── encoder.pkl
│
├── notebook
│   └── Marketing_Campaign_Analysis.ipynb
│
├── data
│   └── digital_marketing_campaign_dataset.csv
│
└── images
    ├── dashboard.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.png
    └── correlation_heatmap.png
```

---

# ▶️ Installation

```bash
git clone https://github.com/your-username/marketing-campaign-conversion-prediction.git

cd marketing-campaign-conversion-prediction

pip install -r requirements.txt

streamlit run app.py
```

## Live Link : https://marketing-campaign-ml-5qh2di2mtl863pdfc9zkrz.streamlit.app/
---

# 📌 Future Improvements

- Integrate real-time marketing campaign data.
- Add model monitoring for production usage.

---

# 👩‍💻 Author

**Apoorva Shree**

B.Tech Student | AI & Machine Learning Enthusiast

Passionate about building practical machine learning solutions that combine predictive modeling with business insights.

---

## ⭐ If you found this project interesting, consider giving it a star!
