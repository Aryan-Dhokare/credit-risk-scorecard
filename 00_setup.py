import os

# Create plots folder if it doesn't exist
os.makedirs('plots', exist_ok=True)
print("plots/ folder created — all chart images will save here")
print("\nProject structure:")
print("  credit-risk-project/")
print("  ├── cs-training.csv          ← your downloaded dataset")
print("  ├── 00_setup.py              ← run this first")
print("  ├── 01_eda.ipynb             ← Phase 1: Exploration")
print("  ├── 02_feature_engineering.ipynb  ← Phase 2: Features")
print("  ├── 03_modeling.ipynb        ← Phase 3: ML Models")
print("  ├── 04_scorecard_business_impact.ipynb  ← Phase 4: Scorecard")
print("  └── plots/                   ← all charts saved here")
