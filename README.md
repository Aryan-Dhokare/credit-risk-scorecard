# Credit Risk Scorecard — Loan Default Prediction

## Problem Statement
Banks lose crores of rupees every year approving loans to applicants who later default. The challenge: how do you identify risky applicants **before** approving their loan, without wrongly rejecting good customers?

This project builds an end-to-end credit risk scoring system that assigns every applicant a credit score between 300–850, segments them into risk bands, and gives the bank a clear action playbook.

---

## Business Impact
> **Our model saves a bank approximately ₹14+ Crores per year on a portfolio of 10,000 loan applications**

| Scenario | Annual Loss |
|---|---|
| Without model (approve everyone) | ₹20.1 Crores |
| With our model | ₹6.0 Crores |
| **Net savings** | **₹14+ Crores** |

---

## Dataset
**Give Me Some Credit** — Kaggle Competition Dataset  
- 150,000 loan applicants  
- 11 features including payment history, income, debt ratio, age  
- Target: `SeriousDlqin2yrs` (did this person default within 2 years?)  
- Class imbalance: only 6.7% defaulted

---

## Project Structure

```
credit-risk-project/
├── 01_eda.ipynb                     # Exploratory Data Analysis + Data Cleaning
├── 02_feature_engineering.ipynb     # Feature Engineering (6 new features)
├── 03_modeling.ipynb                # Logistic Regression + XGBoost + SMOTE
├── 04_scorecard_business_impact.ipynb  # Scorecard + Business Impact
└── plots/                           # All generated charts
```

---

## Key Findings

1. **Age group 25–35 has the highest default rate** — bank should apply stricter scrutiny here
2. **Defaulters earn ~40% less monthly income** than non-defaulters — income is a top predictor
3. **Payment history is the strongest signal** — even 1 missed payment significantly raises default probability
4. **6.7% default rate** creates severe class imbalance — a naive model gets 93.3% accuracy while being completely useless

---

## Methodology

### 1. Data Cleaning
- Removed 609 duplicate records
- Removed 14 records with age < 18 (cannot legally take loans)
- Filled 19.8% missing income values with median (robust to outliers)
- Capped extreme outliers using IQR method (e.g., utilization of 50,000 → capped at 1.0)

### 2. Feature Engineering
Created 6 new domain-specific features:

| Feature | Formula | Why it matters |
|---|---|---|
| DebtToIncomeRatio | debt × income / (income+1) | Classic banking metric |
| IncomePerDependent | income / (dependents+1) | Financial pressure measure |
| TotalLatePayments | sum of all late payment columns | Aggregated payment behavior |
| WeightedLateScore | 30-day×1 + 60-day×2 + 90-day×3 | Recent/severe misses count more |
| UtilizationCategory | bucketed utilization (0-3) | Banks think in categories |
| HasLatePayment | binary flag | Any miss is a red flag |

### 3. Handling Class Imbalance
Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the training set.  
Without this, the model predicts "no default" for everyone and claims 93% accuracy — completely useless.

### 4. Models Built

| Model | AUC-ROC | Notes |
|---|---|---|
| Logistic Regression | ~0.84 | Interpretable, used by real banks |
| **XGBoost** | **~0.87** | Best performer, final model |

**Why AUC-ROC and not accuracy?**  
With 6.7% default rate, accuracy is misleading. AUC-ROC measures how well the model separates defaulters from non-defaulters regardless of threshold.

**Why recall matters:**  
Missing a defaulter costs the bank money. We optimize for catching as many defaulters as possible while keeping false positives manageable.

### 5. Credit Scorecard
Converted model probability into a 300–850 score (like a real CIBIL score):

| Risk Band | Score Range | Action | Default Rate |
|---|---|---|---|
| Low Risk | 700–850 | Auto Approve | ~2% |
| Medium Risk | 550–699 | Manual Review | ~8% |
| High Risk | 300–549 | Reject | ~25%+ |

---

## Tech Stack
- **Python** — pandas, numpy, scikit-learn, xgboost, imbalanced-learn
- **Visualization** — matplotlib, seaborn
- **Environment** — Jupyter Notebook, VS Code

---

## How to Run

```bash
# 1. Clone the repo and navigate to folder
cd credit-risk-project

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter ipykernel

# 4. Setup folders
python 00_setup.py

# 5. Run notebooks in order
# 01_eda.ipynb → 02_feature_engineering.ipynb → 03_modeling.ipynb → 04_scorecard_business_impact.ipynb
```

---

## Recommendations to the Bank

1. **Deploy the XGBoost scorecard** for all new loan applications
2. **Auto-approve** scores above 700 to reduce processing time and cost
3. **Assign dedicated loan officers** to manually review 550–699 band
4. **Offer secured loans only** to sub-550 applicants (collateral required)
5. **Re-score existing customers quarterly** — a good customer can become risky
6. **Focus retention efforts** on high-value customers showing score decline

---

## What Makes This Project Different

Most students stop at building a model and reporting accuracy.  
This project goes further:

- ✅ Domain-specific feature engineering (not just raw columns)
- ✅ SMOTE to handle real-world class imbalance
- ✅ Scorecard output (what banks actually deploy)
- ✅ Business impact in rupees (not just accuracy %)
- ✅ Actionable playbook for loan officers
- ✅ Professional case-study documentation

---

*Project by: Aryan Dhokare | BTech CSE | MIT WPU PUNE*
