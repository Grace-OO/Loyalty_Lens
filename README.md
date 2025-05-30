# 🔍 Loyalty Lens: Understanding Customer Retention Patterns
A business-focused data science project aimed at understanding customer churn behavior in a telecom company, identifying high-risk segments, and providing actionable recommendations to improve customer retention.

---

## 📌 Project Overview

Customer churn is a critical issue for telecom companies, as retaining existing customers is more cost-effective than acquiring new ones. This project analyzes customer demographic, service usage, and support interaction data to:

- Identify patterns associated with churn
- Quantify revenue risk from lost customers
- Predict churn using machine learning
- Provide actionable business insights and recommendations

---

## 📈 Business Goals

- Understand **who is churning** and **why**
- Identify **high-risk customer segments**
- Quantify **revenue lost to churn**
- Suggest **strategies to reduce churn**

---

## 🧾 Dataset Summary

The dataset contains information about telecom customers, including:

- **Customer Info**: `gender`, `seniorcitizen`, `partner`, `dependents`
- **Service Details**: `phoneservice`, `multiplelines`, `internetservice`, `techsupport`, etc.
- **Support Tickets**: `numadmintickets`, `numtechtickets`
- **Billing Info**: `monthlycharges`, `totalcharges`, `paymentmethod`
- **Contract Info**: `contract`, `paperlessbilling`
- **Target Variable**: `churn`

---

## 🔍 Key Questions Answered

- What is the overall churn rate and how much revenue is being lost?
- What services and customer types are associated with higher churn?
- Are certain ticket types (admin or tech) linked to churn?
- Can we predict churn and identify at-risk customers early?

---

## 📊 Analysis & Insights

### Exploratory Data Analysis (EDA)
- Churn by contract type, payment method, and tenure
- Service usage patterns and churn behavior
- Admin/tech support ticket frequency and customer dissatisfaction

### Visualization Highlights
- Count plots for churn across service types
- Boxplots showing ticket frequency differences
- Correlation matrix for all encoded features

### Predictive Modeling 
- Logistic Regression and Random Forest
- Feature importance to support business decisions
- Evaluation metrics: accuracy, precision, recall

---

## 🧠 Business Recommendations

| Insight | Recommendation | Expected Impact |
|--------|----------------|------------------|
| Month-to-month contracts have higher churn | Offer discounts for annual contracts | Improved retention |
| Customers without tech support churn more | Bundle tech support in standard plans | Reduce service-related churn |
| Electronic check users churn more | Incentivize card or auto-pay use | Reduce friction at payment |

---

## 🛠 Tech Stack

- **Python** (pandas, matplotlib, seaborn, scikit-learn)
- **Jupyter Notebook**
-  **Streamlit** for dashboard

---

## 🚀 How to Run

1. Clone this repo:
```bash
git clone https://github.com/yourusername/telecom-churn-analysis.git
