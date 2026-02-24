import streamlit as st
import pandas as pd
import plotly.express as px
import holidays
import io
import numpy as np

# --- 1. CONFIGURATION & TAX MASTERS ---
TDS_MASTER = {
    "192": {"desc": "Salary", "limit": 250000},
    "194A": {"desc": "Interest", "limit": 5000},
    "194C": {"desc": "Contractors", "limit": 30000},
    "194H": {"desc": "Commission", "limit": 15000},
    "194I": {"desc": "Rent", "limit": 240000},
    "194J": {"desc": "Professional", "limit": 30000},
    "194Q": {"desc": "Purchase of Goods", "limit": 5000000}
}

ind_holidays = holidays.India(years=[2025, 2026])

def get_fiscal_quarter(date):
    if pd.isna(date): return "Invalid Date"
    month = date.month
    if month in [4, 5, 6]: return "Q1 (Apr-Jun)"
    if month in [7, 8, 9]: return "Q2 (Jul-Sep)"
    if month in [10, 11, 12]: return "Q3 (Oct-Dec)"
    return "Q4 (Jan-Mar)"

def get_selection_reason(row, materiality):
    reasons = []
    # ERROR PROTECTION: Force numeric comparison
    amt = float(row.get('Amount', 0))
    dt = pd.to_datetime(row.get('Invoice Date'), errors='coerce')
    
    if not pd.isna(dt):
        if dt.weekday() == 6: reasons.append("Sunday")
        if dt in ind_holidays: reasons.append("National Holiday")
    
    if amt >= materiality: reasons.append("Above Materiality")
    if amt > 0 and amt % 1000 == 0: reasons.append("Round Sum Case")
    
    sec = str(row.get('TDS Section', ''))
    if sec in TDS_MASTER and amt >= TDS_MASTER[sec]['limit']:
        reasons.append(f"TDS Threshold {sec}")
    return " | ".join(reasons) if reasons else "Routine Sample"

st.set_page_config(page_title="CA Audit Intelligence", layout="wide")

# --- 2. DATA LOADING & CLEANING ---
if 'audit_data' not in st.session_state:
    st.session_state.audit_data = None

st.title("🛡️ Professional Audit Sampler (Error-Proof Version)")

with st.sidebar:
    materiality = st.number_input("Materiality Threshold", value=100000)
    uploaded_file = st.file_uploader("Upload Data (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    # This block fixes the TypeError by cleaning the data immediately after upload
    if st.session_state.audit_data is None or st.session_state.get('last_file') != uploaded_file.name:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        
        # CLEANING: Remove commas and force to numeric
        df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        df = df.dropna(subset=['Invoice Date'])
        
        # ANALYSIS
        df['Quarter'] = df['Invoice Date'].apply(get_fiscal_quarter)
        df['Selection Reason'] = df.apply(lambda x: get_selection_reason(x, materiality), axis=1)
        
        # GST CHECK
        cols = ['CGST', 'SGST', 'IGST']
        for c in cols: 
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
        df['Calc_Total'] = df['Amount'] + df['CGST'] + df['SGST'] + df['IGST']
        df['Vouching Status'] = "Pending"
        df['Auditor Remarks'] = ""
        
        st.session_state.audit_data = df
        st.session_state.last_file = uploaded_file.name

    df = st.session_state.audit_data

    # --- 3. DASHBOARD ---
    st.subheader("📊 Spending Trends")
    q_chart = px.bar(df.groupby('Quarter')['Amount'].sum().reset_index(), x='Quarter', y='Amount', color='Quarter')
    st.plotly_chart(q_chart, use_container_width=True)

    # --- 4. SAMPLING ---
    if st.button("Generate Samples"):
        priority = df[df['Selection Reason'] != "Routine Sample"]
        randoms = df[df['Selection Reason'] == "Routine Sample"].sample(n=min(10, len(df)))
        st.session_state.selected_samples = pd.concat([priority, randoms])

    if st.session_state.get('selected_samples') is not None:
        st.subheader("📋 Audit Workpaper")
        edited_df = st.data_editor(st.session_state.selected_samples, hide_index=True)
        st.download_button("Export Results", edited_df.to_csv(index=False), "Audit_Report.csv")
