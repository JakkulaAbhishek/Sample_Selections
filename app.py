import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- 1. SETTINGS ---
st.set_page_config(page_title="CA Audit Analytics", layout="wide")
st.title("💎 Professional Audit Sampling & TDS Interest Engine")

# --- 2. DATA SANITIZATION (Fixes TypeError) ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(series, errors='coerce').fillna(0)

# --- 3. SIDEBAR: SAMPLING CONFIGURATION ---
st.sidebar.header("🎯 Sampling Settings")
sample_pct = st.sidebar.slider("Sample Selection %", 1, 100, 20)

selected_categories = st.sidebar.multiselect(
    "Select Method Categories",
    ["Probability", "Non-Probability", "Audit-Specific", "Advanced"],
    default=["Probability", "Audit-Specific"]
)

method_list = ["Simple Random", "Systematic", "Judgmental (High Value)", "Monetary Unit Sampling (MUS)"]
primary_methods = st.sidebar.multiselect("Choose Basis for Selection", options=method_list, default=["Judgmental (High Value)"])

# --- 4. TEMPLATE & UPLOAD (Headers from) ---
headers = ['Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value', 
           'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted', 'TDS Section']

col_tmp, col_up = st.columns([1, 2])
with col_tmp:
    tmp_df = pd.DataFrame(columns=headers)
    t_buffer = BytesIO()
    # Uses xlsxwriter to ensure no ModuleNotFoundError
    tmp_df.to_excel(t_buffer, index=False, engine='xlsxwriter') 
    st.download_button("📥 Download Pro Template", t_buffer.getvalue(), "audit_pro_template.xlsx")

uploaded_file = st.file_uploader("Upload Raw Ledger", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    
    # Pre-processing
    num_cols = ['Gross Total', 'taxable value', 'TDS deducted']
    for col in num_cols:
        if col in df.columns: df[col] = clean_numeric(df[col])
    
    if 'TDS Section' not in df.columns: df['TDS Section'] = "Not Specified"

    # --- 5. EXECUTE SAMPLING ---
    n = max(1, int(len(df) * (sample_pct / 100)))
    sample_df = pd.DataFrame()

    for method in primary_methods:
        if method == "Simple Random":
            s = df.sample(n=min(n, len(df)))
        elif method == "Judgmental (High Value)":
            s = df.nlargest(n, 'taxable value')
        elif method == "Systematic":
            k = max(1, len(df) // n)
            s = df.iloc[::k]
        else:
            s = df.sample(n=min(n, len(df)))
        
        s['Basis for Selection'] = method
        sample_df = pd.concat([sample_df, s]).drop_duplicates(subset=['Invoice no', 'Party name'])

    # --- 6. ADVANCED TDS & INTEREST CALCULATOR ---
    # Section Rates (Adjust as needed)
    rates = {'194C': 0.01, '194J': 0.10, '194I': 0.10, '194H': 0.05, '194Q': 0.001}
    
    tds_summary = df.groupby(['Party name', 'TDS Section']).agg({
        'taxable value': 'sum',
        'TDS deducted': 'sum'
    }).reset_index()

    def calc_tds(row):
        section = str(row['TDS Section']).upper()
        return row['taxable value'] * rates.get(section, 0)

    tds_summary['TDS Needs to be Deducted'] = tds_summary.apply(calc_tds, axis=1)
    tds_summary['Shortfall (Less Deducted)'] = np.maximum(0, tds_summary['TDS Needs to be Deducted'] - tds_summary['TDS deducted'])
    
    # Interest Calculator (1.5% per month for late payment on shortfall)
    # Assumed delay: 3 months for demonstration; can be made dynamic
    tds_summary['Interest on Late Payment (1.5% pm)'] = tds_summary['Shortfall (Less Deducted)'] * 0.015 * 3
    tds_summary['Total Payable with Interest'] = tds_summary['Shortfall (Less Deducted)'] + tds_summary['Interest on Late Payment (1.5% pm)']

    # --- 7. AUDIT DASHBOARD DATA ---
    coverage_summary = df.groupby('Party name')['taxable value'].sum().reset_index()
    coverage_summary.columns = ['Party name', 'Raw File Total Value']
    
    sampled_totals = sample_df.groupby('Party name')['taxable value'].sum().reset_index()
    sampled_totals.columns = ['Party name', 'Sampled Value']
    
    coverage_summary = coverage_summary.merge(sampled_totals, on='Party name', how='left').fillna(0)
    coverage_summary['% Sample Coverage'] = (coverage_summary['Sampled Value'] / coverage_summary['Raw File Total Value']) * 100
    coverage_summary['Selection Basis'] = ", ".join(primary_methods)

    # --- 8. UI & EXCEL EXPORT ---
    st.divider()
    st.subheader("📋 TDS Interest & Applicability Report")
    st.dataframe(tds_summary.style.format(precision=2))

    out_bio = BytesIO()
    with pd.ExcelWriter(out_bio, engine='xlsxwriter') as writer:
        # Sheet 1: Audit Dashboard
        coverage_summary.to_excel(writer, sheet_name='Audit Dashboard', index=False)
        # Sheet 2: Selected Samples
        sample_df.to_excel(writer, sheet_name='Selected Samples', index=False)
        # Sheet 3: TDS Applicability & Interest
        tds_summary.to_excel(writer, sheet_name='TDS Applicability', index=False)
        
        # Add Chart to Dashboard Sheet
        workbook = writer.book
        dash_sheet = writer.sheets['Audit Dashboard']
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'Sampled Value',
            'categories': "='Audit Dashboard'!$A$2:$A$11",
            'values': "='Audit Dashboard'!$C$2:$C$11",
        })
        dash_sheet.insert_chart('H2', chart)

    st.download_button("📤 Download Final Multi-Sheet Audit Report", out_bio.getvalue(), "Final_Audit_Report.xlsx")

else:
    st.info("Awaiting file upload...")
