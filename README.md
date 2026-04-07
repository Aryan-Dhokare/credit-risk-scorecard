# CreditIQ — Credit Risk Scorecard & Loan Default Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?style=flat-square)
![AUC](https://img.shields.io/badge/AUC--ROC-0.871-green?style=flat-square)

---

## Problem Statement

Banks lose crores of rupees every year approving loans to applicants who later default. The challenge: how do you identify risky applicants **before** approving their loan, without wrongly rejecting good customers?

This project builds a complete credit risk scoring system that:
- Assigns every applicant a credit score between **300–850**
- Segments them into **Low / Medium / High Risk** bands
- Gives the bank a clear **Approve / Review / Reject** decision
- Saves the bank approximately **₹14.2 Crores annually** per 10,000 applications

---

## Live App

Run locally:
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-scorecard
cd credit-risk-scorecard
pip install -r requirements.txt
streamlit run app.py
```

---

## Business Impact

| Scenario | Annual Loss |
|---|---|
| Without model (approve everyone) | ₹20.1 Crores |
| With our model | ₹5.9 Crores |
| **Net Annual Savings** | **₹14.2 Crores** |

> Calculated on 10,000 loan applications · Average loan ₹5,00,000 · Loss given default 60%

---

## Dataset

**Give Me Some Credit** — Kaggle Competition  
- 150,000 loan applicants  
- 11 raw features including payment history, income, debt ratio, age  
- Target: Did this person default within 2 years?  
- Class imbalance: only **6.7% defaulted**

---

## Project Structure

```
credit-risk-scorecard/
├── app.py                              # Streamlit live app
├── 01_eda.ipynb                        # EDA + Data Cleaning
├── 02_feature_engineering.ipynb        # Feature Engineering
├── 03_modeling.ipynb                   # Model Building
├── 04_scorecard_business_impact.ipynb  # Scorecard + Business Impact
├── 05_export_dashboard_data.ipynb      # Dashboard Data Export
├── 05_dashboard.ipynb                  # Interactive Charts
├── credit_risk_dashboard.html          # Standalone Dashboard
├── requirements.txt                    # Dependencies
└── README.md
```

---

## Key Findings

1. **Age group 25–35 has the highest default rate** — bank should apply stricter scrutiny here
2. **Defaulters earn ~40% less monthly income** — income is a top predictor
3. **Payment history is the strongest signal** — even 1 missed payment significantly raises default probability
4. **6.7% default rate** creates severe class imbalance — handled with SMOTE

---

## Methodology

### Data Cleaning
- Removed 609 duplicate records
- Removed records with age < 18 (cannot legally take loans)
- Filled 19.8% missing income values with median
- Capped extreme outliers using IQR method

### Feature Engineering
6 new domain-specific features created:

| Feature | Why It Matters |
|---|---|
| DebtToIncomeRatio | Classic banking metric |
| IncomePerDependent | Financial pressure per family member |
| TotalLatePayments | Aggregated payment behavior |
| WeightedLateScore | 90-day miss counts 3x more than 30-day |
| UtilizationCategory | Banks think in bands not percentages |
| HasLatePayment | Any miss is a red flag |

### Handling Class Imbalance
Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance training data. Without this, model predicts "no default" always and claims 93% accuracy — completely useless.

### Models

| Model | AUC-ROC |
|---|---|
| Logistic Regression | ~0.84 |
| **XGBoost (final)** | **0.871** |

**Why AUC-ROC over accuracy?**  
With 6.7% default rate, accuracy is misleading. AUC-ROC correctly measures how well the model separates defaulters from non-defaulters.

### Credit Scorecard

| Risk Band | Score | Action | Default Rate |
|---|---|---|---|
| Low Risk | 700–850 | Auto Approve | ~2.4% |
| Medium Risk | 550–699 | Manual Review | ~14.3% |
| High Risk | 300–549 | Reject | ~37.6% |

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, xgboost, imbalanced-learn
- **Visualization** — matplotlib, seaborn, plotly
- **App** — Streamlit
- **Environment** — Jupyter Notebook, VS Code

---

## How to Run

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/credit-risk-scorecard
cd credit-risk-scorecard

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebooks in order
# 01_eda.ipynb → 02_feature_engineering.ipynb → 03_modeling.ipynb → 04_scorecard_business_impact.ipynb

# 5. Launch app
streamlit run app.py
```

---

## Recommendations to the Bank

1. **Deploy the XGBoost scorecard** for all new loan applications
2. **Auto-approve** scores above 700 to reduce processing time
3. **Assign loan officers** to manually review 550–699 band
4. **Offer secured loans only** to sub-550 applicants
5. **Re-score existing customers quarterly** — good customers can become risky

---

## What Makes This Project Different

Most students stop at building a model and reporting accuracy. This project goes further:

- ✅ Domain-specific feature engineering
- ✅ SMOTE for real-world class imbalance
- ✅ Credit scorecard (what banks actually deploy)
- ✅ Business impact in rupees — not just accuracy %
- ✅ Actionable playbook for loan officers
- ✅ Live interactive web app
- ✅ Professional analytics dashboard

---

*Aryan Dhokare · BTech CSE · [Your College Name]*
