import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Proactive Retention Dashboard",
    page_icon="🚨",
    layout="wide"
)

st.title(" Proactive Retention Dashboard")
st.markdown("""
This tool uses our trained XGBoost Machine Learning model to predict the probability of an agreement-level churn event. 
Adjust the customer metrics in the sidebar to see how the AI evaluates their flight risk in real-time.
""")



# ==========================================
# 2. LOAD THE MODEL & SCALER
# ==========================================
@st.cache_resource
def load_assets():
    # Load BOTH the model and the scaler you used in Jupyter
    model = joblib.load('churn_xgboost_model.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    return model, scaler

model, scaler = load_assets()



# ==========================================
# 3. SIDEBAR: USER INPUTS
# ==========================================
st.sidebar.header("🔧 Adjust Customer Metrics")

# Category A: Support Frustration 
st.sidebar.subheader("Support Experience")
resolved_time_sec = st.sidebar.number_input("Resolution Time (Seconds)", min_value=0, max_value=1000000, value=7200)
days_since_last_ticket = st.sidebar.number_input("Days Since Last Ticket", min_value=0, max_value=365, value=294)

# Category B: Complexity & Lifecycle
st.sidebar.subheader("Account Complexity & Lifecycle")
agreement_count = st.sidebar.number_input("Total Agreements", min_value=1, max_value=50, value=7)
lob_diversity_score = st.sidebar.number_input("LOB Diversity Score", min_value=1, max_value=10, value=1)
contract_completion_pct = st.sidebar.number_input("Contract Completion %", min_value=0.0, max_value=1.0, value=0.9997, format="%.4f")
remaining_days = st.sidebar.number_input("Remaining Days on Contract", min_value=0, max_value=1000, value=0)
tenure_days = st.sidebar.number_input("Customer Tenure (Days)", min_value=0, max_value=5000, value=3504)
service_interval = st.sidebar.number_input("Service Interval (Days)", min_value=0.0, max_value=365.0, value=19.92)

# Category C: Financials
st.sidebar.subheader("Financial Value")
total_bob_raw = st.sidebar.number_input("Total Book of Business ($)", min_value=0.0, max_value=20000000.0, value=2165.0)
revenue_per_agreement = st.sidebar.number_input("Revenue Per Agreement ($)", min_value=0.0, max_value=1000000.0, value=310.0)
revenue_vs_peer_average = st.sidebar.number_input("Revenue vs Peer Average", min_value=-5.0, max_value=5.0, value=0.221, format="%.3f")
company_sizing = st.sidebar.selectbox("Company Sizing Tier (Encoded)", options=[0, 1, 2, 3, 4], index=3)



# ==========================================
# 2. LOAD THE MODEL & SCALER
# ==========================================
@st.cache_resource
def load_assets():
    # Load BOTH the model and the scaler you used in Jupyter
    model = joblib.load('churn_xgboost_model.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# ... (Keep your sidebar code the same as the last update) ...

# ==========================================
# 4. DATA TRANSFORMATION & PREDICTION
# ==========================================
if model and scaler:
    # 1. Apply the log transform to the dollar amounts (just like in Jupyter)
    total_bob_log = np.log1p(total_bob_raw)
    revenue_per_agreement_log = np.log1p(revenue_per_agreement)

    # 2. Put the raw/logged inputs into a DataFrame IN THE EXACT ORDER XGBoost expects
    input_data = pd.DataFrame({
        'company_sizing': [company_sizing],
        'total_bob': [total_bob_log],
        'revenue_per_agreement': [revenue_per_agreement_log],
        'revenue_vs_peer_average': [revenue_vs_peer_average],
        'lob_diversity_score': [lob_diversity_score],
        'service_interval': [service_interval],
        'tenure_days': [tenure_days],
        'remaining_days': [remaining_days],
        'contract_completion_pct': [contract_completion_pct],
        'agreement_count': [agreement_count],
        'resolved_time_sec': [resolved_time_sec],
        'days_since_last_ticket': [days_since_last_ticket]
    })

    # 3. ⚠️ THE MAGIC FIX: Scale the data so it matches the decimals the model trained on!
    # Note: If you only scaled numerical features and NOT categorical ones in Jupyter, 
    # you must apply this scaler ONLY to the columns you scaled in Jupyter.
    scaled_input = scaler.transform(input_data)

    # Generate Probability
    churn_probability = model.predict_proba(scaled_input)[0][1]


    
    # ==========================================
    # 5. MAIN DASHBOARD DISPLAY
    # ==========================================
    st.markdown("###  AI Risk Assessment")
    
    # Create two columns for a clean layout
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Probability of Churn", value=f"{churn_probability * 100:.1f}%")
        
    with col2:
        # Apply your optimal threshold logic (0.30)
        if churn_probability > 0.30:
            st.error(" HIGH FLIGHT RISK - Immediate Action Required")
        else:
            st.success(" SAFE - Customer is stable")

    st.markdown("---")
    
    # Dynamic Business Advice based on the inputs
    st.markdown("###  Recommended Action Plan")
    if churn_probability > 0.30:
        if resolved_time_sec > 10000 and days_since_last_ticket < 7:
            st.warning("**Support Intervention:** This customer is currently experiencing severe support delays. Escalate their open tickets to Tier 2 immediately.")
        elif contract_completion_pct > 0.90 or remaining_days < 30:
            st.warning("**Renewal Danger:** Contract is almost complete. Deploy a Customer Success Manager immediately with a personalized renewal incentive.")
        elif agreement_count > 3:
            st.warning("**Complexity Overload:** This VIP client is struggling to integrate multiple products. Deploy a Technical Account Manager for an architecture review.")
        else:
            st.warning("Customer crossed the 30% risk threshold. Initiate standard proactive retention protocols.")
    else:
        st.info("No immediate intervention needed. Continue standard service delivery.")