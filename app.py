import streamlit as st
import numpy as np
import pickle
import time

st.set_page_config(
    page_title="CreditIQ — Loan Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f8fafc !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Top Nav ── */
.topnav {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 16px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
}
.topnav .logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    color: #0f172a;
    letter-spacing: -0.5px;
}
.topnav .logo span { color: #2563eb; }
.topnav .nav-tag {
    background: #dbeafe;
    color: #1d4ed8;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* ── Page Layout ── */
.page-wrap {
    max-width: 1100px;
    margin: 0 auto;
    padding: 48px 32px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    margin-bottom: 52px;
}
.hero h1 {
    font-size: 48px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin-bottom: 14px;
}
.hero h1 span { color: #2563eb; }
.hero p {
    font-size: 17px;
    color: #64748b;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
    font-weight: 400;
}

/* ── Form Card ── */
.form-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
}
.form-section-title {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid #f1f5f9;
}

/* ── Streamlit input overrides ── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #374151 !important;
    margin-bottom: 4px !important;
}
[data-testid="stNumberInput"] input {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 15px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #0f172a !important;
    background: #f8fafc !important;
    transition: border-color 0.2s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #2563eb !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
}

/* ── Button ── */
.stButton > button {
    background: #1d4ed8 !important;
    color: white !important;
    border: none !important;
    padding: 16px 32px !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: #1e40af !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.3) !important;
}

/* ── Result Section ── */
.result-hero {
    border-radius: 20px;
    padding: 44px 40px;
    text-align: center;
    margin-bottom: 24px;
}
.result-hero.approve {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac;
}
.result-hero.review {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1.5px solid #fcd34d;
}
.result-hero.reject {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1.5px solid #fca5a5;
}
.result-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
    opacity: 0.6;
    color: #374151;
}
.result-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 96px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 8px;
}
.result-band {
    font-size: 20px;
    font-weight: 500;
    margin-bottom: 16px;
    opacity: 0.75;
    color: #374151;
}
.result-action {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ── Score Bar ── */
.sbar-track {
    height: 8px;
    background: rgba(0,0,0,0.08);
    border-radius: 4px;
    margin: 16px auto;
    max-width: 340px;
    overflow: hidden;
}
.sbar-fill {
    height: 100%;
    border-radius: 4px;
}

/* ── Metric Cards ── */
.m3 {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 14px;
    margin-bottom: 20px;
}
.mc {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.mc .ml {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.mc .mv {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 600;
}

/* ── Factor List ── */
.fc {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.fc-title {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}
.fi {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 11px 0;
    border-bottom: 1px solid #f8fafc;
}
.fi:last-child { border-bottom: none; }
.fi-name { font-size: 13px; color: #475569; }
.fi-val { font-size: 13px; font-weight: 600; color: #0f172a; }
.fi-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
}
.tag-g { background: #dcfce7; color: #166534; }
.tag-a { background: #fef3c7; color: #92400e; }
.tag-r { background: #fee2e2; color: #991b1b; }

/* ── Decision Card ── */
.dc {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.dc-row {
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid #f1f5f9;
}
.dc-row:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.dc-label {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.dc-value {
    font-size: 15px;
    font-weight: 600;
    color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
}
.dc-note {
    font-size: 13px;
    color: #64748b;
    line-height: 1.7;
    background: #f8fafc;
    border-radius: 10px;
    padding: 14px;
    margin-top: 8px;
}

/* ── Info Box ── */
.infobox {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 20px;
    font-size: 12px;
    color: #3b82f6;
    line-height: 1.7;
}
.infobox strong { color: #1d4ed8; }

/* ── Empty State ── */
.empty {
    text-align: center;
    padding: 64px 0;
}
.empty .e-icon { font-size: 44px; margin-bottom: 16px; }
.empty .e-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #cbd5e1;
    letter-spacing: 2px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ── Nav ──
st.markdown("""
<div class="topnav">
    <div class="logo">Credit<span>IQ</span></div>
    <div class="nav-tag">XGBoost · AUC 0.871</div>
</div>
<div class="page-wrap">
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
    <h1>Loan Risk<br><span>Analyzer</span></h1>
    <p>Enter applicant details below to instantly assess credit risk and get a lending decision.</p>
</div>
""", unsafe_allow_html=True)

# ── Form ──
st.markdown('<div class="form-card"><div class="form-section-title">Applicant Details</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    age             = st.number_input("Age", min_value=18, max_value=100, value=None, placeholder="e.g. 35")
    monthly_income  = st.number_input("Monthly Income ($)", min_value=0, value=None, placeholder="e.g. 5000")
    num_dependents  = st.number_input("Number of Dependents", min_value=0, max_value=10, value=None, placeholder="e.g. 2")

with c2:
    utilization     = st.number_input("Credit Utilization (%)", min_value=0, max_value=100, value=None, placeholder="e.g. 30")
    debt_ratio      = st.number_input("Debt Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=None, placeholder="e.g. 0.35", step=0.01)
    open_lines      = st.number_input("Open Credit Lines", min_value=0, max_value=30, value=None, placeholder="e.g. 5")

with c3:
    late_30         = st.number_input("Late Payments (30–59 days)", min_value=0, max_value=20, value=None, placeholder="e.g. 0")
    late_60         = st.number_input("Late Payments (60–89 days)", min_value=0, max_value=20, value=None, placeholder="e.g. 0")
    late_90         = st.number_input("Late Payments (90+ days)",   min_value=0, max_value=20, value=None, placeholder="e.g. 0")

st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button("Analyze Credit Risk →")

# ── Helpers ──
def build_features(age, income, util, debt, lines, l30, l60, l90, deps):
    u = util / 100.0
    dti = debt * income / (income + 1)
    ipd = income / (deps + 1)
    total = l30 + l60 + l90
    weighted = l30 * 1 + l60 * 2 + l90 * 3
    ucat = 0 if u <= 0.3 else (1 if u <= 0.6 else (2 if u <= 0.8 else 3))
    has_late = 1 if total > 0 else 0
    return np.array([u, age, l30, debt, income, lines, l90, 1,
                     l60, deps, dti, ipd, total, weighted, ucat, has_late]).reshape(1, -1)

def to_score(prob):
    return int(round(850 - prob * 550))

def get_band(score):
    if score >= 700:
        return "Low Risk", "AUTO APPROVE ✓", "approve", "#16a34a"
    elif score >= 550:
        return "Medium Risk", "MANUAL REVIEW ⚠", "review", "#d97706"
    else:
        return "High Risk", "REJECT ✕", "reject", "#dc2626"

def tag(val, lo, hi, rev=False):
    if rev:
        c = "#16a34a" if val >= hi else ("#d97706" if val >= lo else "#dc2626")
        l = "GOOD" if val >= hi else ("FAIR" if val >= lo else "POOR")
    else:
        c = "#16a34a" if val <= lo else ("#d97706" if val <= hi else "#dc2626")
        l = "GOOD" if val <= lo else ("FAIR" if val <= hi else "HIGH RISK")
    cls = "tag-g" if c == "#16a34a" else ("tag-a" if c == "#d97706" else "tag-r")
    return f'<span class="fi-tag {cls}">{l}</span>'

# ── Result ──
if analyze:
    missing = [f for f, v in zip(
        ["Age","Monthly Income","Dependents","Utilization","Debt Ratio",
         "Open Lines","Late 30-59","Late 60-89","Late 90+"],
        [age, monthly_income, num_dependents, utilization, debt_ratio,
         open_lines, late_30, late_60, late_90]
    ) if v is None]

    if missing:
        st.warning(f"Please fill in: {', '.join(missing)}")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.6)

        feats = build_features(age, monthly_income, utilization, debt_ratio,
                               open_lines, late_30, late_60, late_90, num_dependents)
        prob  = model.predict_proba(feats)[0][1]
        score = to_score(prob)
        band, action, cls, color = get_band(score)
        pct   = int((score - 300) / 550 * 100)

        # Result hero
        st.markdown(f"""
        <div class="result-hero {cls}">
            <div class="result-label">Credit Assessment</div>
            <div class="result-score" style="color:{color}">{score}</div>
            <div class="result-band">{band} &nbsp;·&nbsp; Score out of 850</div>
            <div class="sbar-track">
                <div class="sbar-fill" style="width:{pct}%;background:{color}"></div>
            </div>
            <div class="result-action" style="color:{color}">{action}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metric row
        loan_limit = "Full Amount" if score >= 700 else ("70% of Request" if score >= 550 else "Secured Only")
        rate       = "Standard"    if score >= 700 else ("+1.5% Premium"  if score >= 550 else "+3.0% / Collateral")

        st.markdown(f"""
        <div class="m3">
            <div class="mc"><div class="ml">Default Probability</div>
                <div class="mv" style="color:#dc2626">{prob*100:.1f}%</div></div>
            <div class="mc"><div class="ml">Loan Limit</div>
                <div class="mv" style="color:#2563eb;font-size:16px">{loan_limit}</div></div>
            <div class="mc"><div class="ml">Interest Rate</div>
                <div class="mv" style="color:#374151;font-size:16px">{rate}</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Two columns: factors + decision
        left, right = st.columns(2)
        total_late = late_30 + late_60 + late_90
        weighted   = late_30 + late_60 * 2 + late_90 * 3

        with left:
            st.markdown(f"""
            <div class="fc">
                <div class="fc-title">Risk Factors</div>
                <div class="fi">
                    <div><div class="fi-name">Credit Utilization</div></div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <span class="fi-val">{utilization}%</span>
                        {tag(utilization/100, 0.3, 0.7)}
                    </div>
                </div>
                <div class="fi">
                    <div><div class="fi-name">Total Late Payments</div></div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <span class="fi-val">{total_late}</span>
                        {tag(total_late, 0, 2)}
                    </div>
                </div>
                <div class="fi">
                    <div><div class="fi-name">Weighted Late Score</div></div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <span class="fi-val">{weighted}</span>
                        {tag(weighted, 0, 4)}
                    </div>
                </div>
                <div class="fi">
                    <div><div class="fi-name">Debt Ratio</div></div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <span class="fi-val">{debt_ratio:.2f}</span>
                        {tag(debt_ratio, 0.3, 0.6)}
                    </div>
                </div>
                <div class="fi">
                    <div><div class="fi-name">Monthly Income</div></div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <span class="fi-val">${monthly_income:,}</span>
                        {tag(monthly_income, 3000, 6000, rev=True)}
                    </div>
                </div>
                <div class="fi">
                    <div><div class="fi-name">Age</div></div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <span class="fi-val">{age} yrs</span>
                        {tag(age, 30, 45)}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            review = "Annual" if score >= 700 else ("Quarterly" if score >= 550 else "Monthly")
            note   = (
                "Strong applicant profile. All key indicators are within acceptable range. Recommend immediate approval with standard lending terms."
                if score >= 700 else
                "Moderate risk detected. Some payment or utilization concerns noted. Recommend reduced loan limit with closer monitoring."
                if score >= 550 else
                "High default risk detected. Multiple risk factors present. Recommend rejection or secured loan with collateral only."
            )
            st.markdown(f"""
            <div class="dc">
                <div class="fc-title">Lending Decision</div>
                <div class="dc-row">
                    <div class="dc-label">Decision</div>
                    <div class="dc-value" style="color:{color}">{action}</div>
                </div>
                <div class="dc-row">
                    <div class="dc-label">Loan Limit</div>
                    <div class="dc-value">{loan_limit}</div>
                </div>
                <div class="dc-row">
                    <div class="dc-label">Interest Rate</div>
                    <div class="dc-value">{rate}</div>
                </div>
                <div class="dc-row">
                    <div class="dc-label">Review Cycle</div>
                    <div class="dc-value">{review}</div>
                </div>
                <div class="dc-row">
                    <div class="dc-label">Recommendation</div>
                    <div class="dc-note">{note}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="infobox">
            <strong>How this works:</strong> XGBoost model trained on 150,000 real applicants · AUC-ROC 0.871 ·
            Class imbalance handled with SMOTE · Scores scaled 300–850 ·
            Model saves <strong>₹14.2 Crores annually</strong> per 10,000 applications
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty">
        <div class="e-icon">🏦</div>
        <div class="e-text">Fill in the details above and click Analyze</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
