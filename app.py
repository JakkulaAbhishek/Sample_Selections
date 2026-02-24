import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- 1. PAGE CONFIG & UI ---
st.set_page_config(page_title="Audit Analytics & TDS Tool", layout="wide")
st.title("🛡️ Audit Sampling & TDS Compliance Dashboard")

# --- 2. TDS RULES ENGINE ---
# Thresholds and Section Logic
TDS_RULES = {
    '194C': {'name': 'Contractors', 'threshold_single': 30000, 'threshold_agg': 100000, 'keywords': ['contract', 'labor', 'maintenance', 'civil', 'repair']},
    '194J': {'name': 'Professional/Tech', 'threshold_single': 30000, 'threshold_agg': 30000, 'keywords': ['audit', 'legal', 'consultant', 'technical', 'software']},
    '194H': {'name': 'Commission/Brokerage', 'threshold_single': 15000, 'threshold_agg': 15000, 'keywords': ['commission', 'brokerage', 'agent']},
    '194I': {'name': 'Rent', 'threshold_single': 240000, 'threshold_agg': 240000, 'keywords': ['rent', 'lease', 'office rent']}
}

def identify_tds_section(particulars, amount):
    part_lower = str(particulars).lower()
    for sec, rule in TDS_RULES.items():
        if any(key in part_lower for key in rule['keywords']):
            return sec
    return "Others/NA"

# --- 3. SIDEBAR: SAMPLING CONFIG ---
st.sidebar.header("📋 Sampling Parameters")
materiality = st.sidebar.number_input("Enter Materiality Level (₹)", value=50000)
method = st.sidebar.selectbox("Select Sampling Method", [
    "Simple Random Sampling", "Systematic Sampling", "Monetary Unit Sampling (MUS)", 
    "Haphazard Sampling", "Judgmental (High Value)", "Stratified Sampling"
])

# --- 4. DATA UPLOAD & SAMPLE FILE ---
st.subheader("1. Data Ingestion")
col1, col2 = st.columns([2, 1])

with col2:
    # Create a dummy template based on user image
    template_df = pd.DataFrame(columns=[
        'Date', 'Particulars', 'Voucher Ref. No.', 'Gross Total', 
        'Maintenance & Spares', 'Input CGST', 'Input SGST', 'Input IGST', 'Round Off', 'TDS 194C'
    ])
    buffer = BytesIO()
    template_df.to_excel(buffer, index=False)
    st.download_button("📥 Download Upload Template", buffer.getvalue(), "audit_template.xlsx")

uploaded_file = st.file_uploader("Upload your Raw Ledger File (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    
    # --- 5. TDS CALCULATION ENGINE (CELL BREAKDOWN) ---
    st.subheader("2. TDS Applicability & Section Breakdown")
    
    # Apply logic to create the new "Section" column
    df['Detected Section'] = df.apply(lambda x: identify_tds_section(x['Particulars'], x['Gross Total']), axis=1)
    
    # Check for non-compliance (Threshold reached but TDS is 0)
    def check_compliance(row):
        sec = row['Detected Section']
        if sec in TDS_RULES:
            if row['Gross Total'] >= TDS_RULES[sec]['threshold_single'] and (row['TDS 194C'] == 0):
                return "⚠️ Potential Non-Compliance"
        return "✅ OK"

    df['Compliance Status'] = df.apply(check_compliance, axis=1)

    # Display Table with Section Breakdown
    st.dataframe(df[['Date', 'Particulars', 'Gross Total', 'Detected Section', 'TDS 194C', 'Compliance Status']].style.applymap(
        lambda x: 'background-color: #ffcccc' if x == "⚠️ Potential Non-Compliance" else '', subset=['Compliance Status']
    ))

    # --- 6. VISUAL ANALYTICS (DASHBOARD) ---
    st.divider()
    st.subheader("3. Audit Dashboard")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        # Party-wise Expenditure
        party_exp = df.groupby('Particulars')['Gross Total'].sum().sort_values(ascending=False).head(10).reset_index()
        fig1 = px.bar(party_exp, x='Gross Total', y='Particulars', title="Top 10 Party-wise Expenditure", orientation='h', color='Gross Total')
        st.plotly_chart(fig1, use_container_width=True)

    with d_col2:
        # Section-wise Breakdown Pie Chart
        sec_dist = df['Detected Section'].value_counts().reset_index()
        fig2 = px.pie(sec_dist, values='count', names='Detected Section', title="TDS Section Distribution", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    # --- 7. SAMPLE OUTPUT GENERATION ---
    st.divider()
    st.subheader("4. Generated Samples for Testing")
    
    if method == "Monetary Unit Sampling (MUS)":
        df['Cum_Sum'] = df['Gross Total'].cumsum()
        interval = df['Gross Total'].sum() / 10 # Default 10 samples
        sample_df = df[df['Cum_Sum'] % interval <= (interval * 0.1)].head(10)
    elif method == "Judgmental (High Value)":
        sample_df = df[df['Gross Total'] > materiality]
    else:
        sample_df = df.sample(frac=0.1 if len(df)>10 else 0.5)

    st.write(f"Selected **{len(sample_df)}** samples using **{method}**.")
    st.table(sample_df[['Voucher Ref. No.', 'Particulars', 'Gross Total', 'Detected Section']])

    # Export Final Audit File
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Full_Analysis', index=False)
        sample_df.to_excel(writer, sheet_name='Selected_Samples', index=False)
    
    st.download_button("📤 Download Final Audit Report", output.getvalue(), "Audit_Report.xlsx")

else:
    st.info("Please upload a file to begin the analysis.")
