import streamlit as st
import pandas as pd
import plotly.express as px
import holidays
import io
import numpy as np

# Initialize India Holidays for 2026
ind_holidays = holidays.India(years=2026)

# --- TAX & AUDIT LOGIC ---
TDS_THRESHOLDS = {
    "Contract (194C)": 30000,
    "Professional (194J)": 30000,
    "Rent (194I)": 240000,
    "Commission (194H)": 15000
}

def get_selection_reason(row, materiality):
    reasons = []
    dt = pd.to_datetime(row['Invoice Date'])
    
    # 1. Date Risks
    if dt.weekday() == 6: reasons.append("Sunday")
    if dt in ind_holidays: reasons.append(f"Holiday ({ind_holidays.get(dt)})")
    
    # 2. Value Risks
    amt = row['Amount']
    if amt >= materiality: reasons.append("Above Materiality")
    if amt > 0 and amt % 1000 == 0: reasons.append("Round Number Case")
    
    # 3. TDS Logic
    for sec, limit in TDS_THRESHOLDS.items():
        if sec.split(" ")[0].lower() in str(row['Transaction Type']).lower() and amt >= limit:
            reasons.append(f"TDS Trigger ({sec})")
            
    return " | ".join(reasons) if reasons else "Random Sample"

st.set_page_config(page_title="CA Audit Pro 2026", layout="wide")

# --- SESSION STATE MANAGEMENT ---
if 'audit_data' not in st.session_state:
    st.session_state.audit_data = None
if 'selected_samples' not in st.session_state:
    st.session_state.selected_samples = None

st.title("🛡️ CA Audit Pro: Sampling & Tax Compliance")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Audit Config")
    materiality = st.number_input("Materiality Threshold (INR)", value=100000, step=10000)
    conf_level = st.select_slider("Statistical Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
    
    uploaded_file = st.file_uploader("Upload Client Data", type=['csv', 'xlsx'])
    
    # --- ENHANCED TEMPLATE ---
    st.subheader("📥 Data Template")
    template = pd.DataFrame({
        'Invoice Date': ['2026-04-01'], 'Invoice No': ['INV-001'], 'Party Name': ['Example Corp'],
        'Transaction Type': ['Professional (194J)'], 'Amount': [50000],
        'CGST': [4500], 'SGST': [4500], 'IGST': [0], 'Cess': [0], 'Total Value': [59000]
    })
    st.download_button("Download CSV Template", template.to_csv(index=False), "audit_template.csv")

# --- FILE PROCESSING ---
if uploaded_file:
    # Reset state if new file is uploaded
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.audit_data = None
        st.session_state.selected_samples = None
        st.session_state.last_file = uploaded_file.name

    if st.session_state.audit_data is None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
        
        # 1. Run Risk Engine
        df['Selection Reason'] = df.apply(lambda x: get_selection_reason(x, materiality), axis=1)
        
        # 2. Tax Validation (Sum of GST vs Total)
        if all(col in df.columns for col in ['Amount', 'CGST', 'SGST', 'IGST']):
            df['Calc_Total'] = df['Amount'] + df['CGST'] + df['SGST'] + df['IGST'] + df.get('Cess', 0)
            if 'Total Value' in df.columns:
                df.loc[(df['Total Value'] - df['Calc_Total']).abs() > 1, 'Selection Reason'] += " | Tax Math Error"

        df['Vouching Status'] = "Pending"
        df['Auditor Remarks'] = ""
        st.session_state.audit_data = df

    df = st.session_state.audit_data

    # --- TOP METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", len(df))
    m2.metric("Total Audit Value", f"₹{df['Amount'].sum():,.0f}")
    m3.metric("High-Risk Items", len(df[df['Selection Reason'] != "Random Sample"]))
    # Margin of Error Calculation (Simplified for display)
    moe = (1.96 * (0.5 / np.sqrt(len(df)))) * 100 # Standard error formula
    m4.metric("Pop. Margin of Error", f"{moe:.2f}%")

    # --- SAMPLING CONTROLS ---
    st.divider()
    st.subheader("🎯 Sample Selection Criteria")
    col1, col2 = st.columns(2)
    with col1:
        types = st.multiselect("Select Transaction Types", df['Transaction Type'].unique(), default=df['Transaction Type'].unique())
    with col2:
        num_samples = st.number_input("Additional Random Samples to Draw", min_value=1, value=10)

    if st.button("🚀 Generate Audit Working Paper"):
        priority = df[(df['Transaction Type'].isin(types)) & (df['Selection Reason'] != "Random Sample")]
        random_pool = df[(df['Transaction Type'].isin(types)) & (df['Selection Reason'] == "Random Sample")]
        st.session_state.selected_samples = pd.concat([priority, random_pool.sample(n=min(len(random_pool), num_samples))])

    # --- VOUCHING TABLE ---
    if st.session_state.selected_samples is not None:
        st.divider()
        st.subheader("📋 Vouching Dashboard")
        
        # Progress Tracking
        samples = st.session_state.selected_samples
        done = len(samples[samples['Vouching Status'].isin(['Verified', 'GST Mismatch'])])
        st.progress(done/len(samples))
        
        edited_df = st.data_editor(
            samples,
            column_config={
                "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Pending", "Verified", "Query", "TDS Issue", "GST Mismatch"]),
                "Selection Reason": st.column_config.TextColumn("Risk Flags", width="medium"),
                "Amount": st.column_config.NumberColumn("Base Amt", format="₹%d"),
                "Auditor Remarks": st.column_config.TextColumn("Remarks", width="large")
            },
            disabled=["Invoice Date", "Party Name", "Amount", "CGST", "SGST", "IGST", "Selection Reason", "Invoice No"],
            hide_index=True, use_container_width=True
        )
        st.session_state.selected_samples = edited_df

        # --- VISUAL ANALYTICS ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Sample Risk Profile")
            st.plotly_chart(px.pie(edited_df, names='Selection Reason', hole=0.4), use_container_width=True)
        with c2:
            st.write("### Party-wise Exposure")
            st.plotly_chart(px.bar(edited_df, x='Party Name', y='Amount', color='Vouching Status'), use_container_width=True)

        st.download_button("💾 Export Workpaper to Excel", edited_df.to_csv(index=False), "Audit_Working_Paper_2026.csv")
