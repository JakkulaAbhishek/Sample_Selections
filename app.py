import streamlit as st
import pandas as pd
import plotly.express as px
import holidays
import io

# Initialize India Holidays for 2026
ind_holidays = holidays.India(years=2026)

# --- TAX LOGIC ---
TDS_THRESHOLDS = {
    "Contract (194C)": 30000,
    "Professional (194J)": 30000,
    "Rent (194I)": 240000,
    "Commission (194H)": 15000
}

def get_selection_reason(row, materiality):
    reasons = []
    dt = pd.to_datetime(row['Invoice Date'])
    # Holiday/Sunday Checks
    if dt.weekday() == 6: reasons.append("Sunday")
    if dt in ind_holidays: reasons.append(f"Holiday ({ind_holidays.get(dt)})")
    
    # Value Checks
    if row['Amount'] >= materiality: reasons.append("Above Materiality")
    
    # TDS Threshold Check
    for sec, limit in TDS_THRESHOLDS.items():
        if sec.split(" ")[0].lower() in str(row['Transaction Type']).lower() and row['Amount'] >= limit:
            reasons.append(f"TDS Trigger ({sec})")
            
    return " | ".join(reasons) if reasons else "Random Sample"

st.set_page_config(page_title="CA Audit & Tax Pro", layout="wide")

# --- SESSION STATE ---
if 'audit_data' not in st.session_state:
    st.session_state.audit_data = None

st.title("🛡️ Audit Sampler + TDS/GST Compliance Tracker")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Audit Parameters")
    materiality = st.number_input("Materiality (Key Items)", value=100000)
    gst_rate = st.selectbox("Standard GST Rate (%)", [5, 12, 18, 28], index=2)
    uploaded_file = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
    
    # Template
    template = pd.DataFrame({
        'Invoice Date':['2026-04-01'], 'Party Name':['Tech Solutions'], 
        'Transaction Type':['Professional (194J)'], 'Amount':[35000], 
        'GST Amount':[6300], 'Invoice No':['INV-99']
    })
    st.download_button("📥 Download Sample Template", template.to_csv(index=False), "template.csv")

if uploaded_file:
    if st.session_state.audit_data is None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
        
        # 1. Flag Compliance Issues
        df['Selection Reason'] = df.apply(lambda x: get_selection_reason(x, materiality), axis=1)
        
        # 2. GST Variance Check (Amount * Rate vs Uploaded GST)
        if 'GST Amount' in df.columns:
            expected_gst = df['Amount'] * (gst_rate / 100)
            df['GST Variance'] = (df['GST Amount'] - expected_gst).abs()
            df.loc[df['GST Variance'] > 1, 'Selection Reason'] += " | GST Mismatch"

        df['Vouching Status'] = "Pending"
        df['Auditor Remarks'] = ""
        st.session_state.audit_data = df

    df = st.session_state.audit_data

    # --- SAMPLING ---
    st.subheader("🎯 Selection Filters")
    types = st.multiselect("Filter Transaction Types", df['Transaction Type'].unique(), default=df['Transaction Type'].unique())
    num_samples = st.slider("Additional Random Samples", 1, 50, 10)

    if st.button("Generate Smart Samples"):
        priority = df[(df['Transaction Type'].isin(types)) & (df['Selection Reason'] != "Random Sample")]
        random_pool = df[(df['Transaction Type'].isin(types)) & (df['Selection Reason'] == "Random Sample")]
        st.session_state.selected_samples = pd.concat([priority, random_pool.sample(n=min(len(random_pool), num_samples))])

    # --- VOUCHING DASHBOARD ---
    if 'selected_samples' in st.session_state:
        st.divider()
        st.subheader("📝 Vouching & Compliance Dashboard")
        
        # Progress Tracking
        samples = st.session_state.selected_samples
        verified_count = len(samples[samples['Vouching Status'] == 'Verified'])
        st.metric("Audit Progress", f"{verified_count} / {len(samples)}", f"{int(verified_count/len(samples)*100)}% Complete")

        edited_df = st.data_editor(
            samples,
            column_config={
                "Selection Reason": st.column_config.TextColumn("Audit Flag", width="medium"),
                "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Pending", "Verified", "Query", "TDS Missing", "GST Mismatch"]),
                "Auditor Remarks": st.column_config.TextColumn("Remarks", width="large")
            },
            disabled=["Invoice Date", "Party Name", "Amount", "GST Amount", "Selection Reason"],
            hide_index=True, use_container_width=True
        )
        st.session_state.selected_samples = edited_df

        # --- COMPLIANCE CHARTS ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write("### 🚩 Risk Breakdown")
            st.plotly_chart(px.histogram(edited_df, x='Selection Reason', color='Selection Reason', barmode='group'))
        with c2:
            st.write("### 💰 Sample Value Distribution")
            st.plotly_chart(px.box(edited_df, y='Amount', points="all", color='Transaction Type'))

        st.download_button("💾 Export Final Audit Paper", edited_df.to_csv(index=False), "Audit_Working_Paper.csv")
