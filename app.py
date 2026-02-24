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
        series = series.str.replace(r'[^\d.]', '', regex=True) #
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
           'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted', 'TDS Section'] #

col_tmp, col_up = st.columns([1, 2])
with col_tmp:
    tmp_df = pd.DataFrame(columns=headers)
    t_buffer = BytesIO()
    # Uses xlsxwriter to ensure no ModuleNotFoundError
    tmp_df.to_excel(t_buffer, index=False, engine='xlsxwriter') #
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
    # Section Rates Formula
    rates = {'194C': 0.01, '194J': 0.10, '194I': 0.10, '194H': 0.05, '194Q': 0.001}
    
    tds_summary = df.groupby(['Party name', 'TDS Section']).agg({
        'taxable value': 'sum',
        'TDS deducted': 'sum'
    }).reset_index()

    def calc_tds(row):
        section = str(row['TDS Section']).upper()
        # FORMULA: Taxable Value * Section Rate
        return row['taxable value'] * rates.get(section, 0)

    tds_summary['TDS Needs to be Deducted'] = tds_summary.apply(calc_tds, axis=1)
    tds_summary['Shortfall (Less Deducted)'] = np.maximum(0, tds_summary['TDS Needs to be Deducted'] - tds_summary['TDS deducted'])
    
    tds_summary['Interest on Late Payment (1.5% pm)'] = tds_summary['Shortfall (Less Deducted)'] * 0.015 * 3
    tds_summary['Total Payable with Interest'] = tds_summary['Shortfall (Less Deducted)'] + tds_summary['Interest on Late Payment (1.5% pm)']

    # --- 7. AUDIT DASHBOARD DATA ---
    coverage_summary = df.groupby('Party name')['taxable value'].sum().reset_index()
    coverage_summary.columns = ['Party name', 'Raw File Total Value']
    
    sampled_totals = sample_df.groupby('Party name')['taxable value'].sum().reset_index()
    sampled_totals.columns = ['Party name', 'Sampled Value']
    
    coverage_summary = coverage_summary.merge(sampled_totals, on='Party name', how='left').fillna(0)
    # Selection Percentage Calculation
    coverage_summary['% Sample Selection'] = (coverage_summary['Sampled Value'] / coverage_summary['Raw File Total Value']) * 100
    coverage_summary['Selection Basis'] = ", ".join(primary_methods)

    # --- 8. UI & EXCEL EXPORT ---
    st.divider()
    st.subheader("📋 TDS Interest & Applicability Report")
    st.dataframe(tds_summary.style.format(precision=2))

    out_bio = BytesIO()
    with pd.ExcelWriter(out_bio, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Define Sheets to include Subtotals at Top and Pie Charts
        sheet_data = {
            'Audit Dashboard': coverage_summary,
            'Selected Samples': sample_df,
            'TDS Applicability': tds_summary
        }

        for sheet_name, data in sheet_data.items():
            # Write Subtotals at the Top (Row 0)
            if not data.empty:
                # Calculate numeric totals only
                numeric_only = data.select_dtypes(include=[np.number]).sum()
                subtotal_row = pd.DataFrame([["SUBTOTALS"] + [""] * (data.shape[1]-1)], columns=data.columns)
                for col in numeric_only.index:
                    subtotal_row[col] = numeric_only[col]
                
                subtotal_row.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)

                # Add Pie Chart to every sheet
                ws = writer.sheets[sheet_name]
                pie_chart = workbook.add_chart({'type': 'pie'})
                # Points to the first numeric column for distribution
                pie_chart.add_series({
                    'name': f'{sheet_name} Distribution',
                    'categories': f"='{sheet_name}'!$A$4:$A$13",
                    'values': f"='{sheet_name}'!$B$4:$B$13",
                })
                ws.insert_chart('K2', pie_chart)

    st.download_button("📤 Download Final Multi-Sheet Audit Report", out_bio.getvalue(), "Final_Audit_Report.xlsx")

else:
    st.info("Awaiting file upload...")
