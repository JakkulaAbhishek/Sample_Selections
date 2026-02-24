import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- 1. SETTINGS ---
st.set_page_config(page_title="CA Audit Analytics", layout="wide")
st.title("💎 Professional Audit Sampling & TDS Analytics")

# --- 2. DATA SANITIZATION ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.str.replace(r'[^\d.]', '', regex=True) # Fixes TypeError
    return pd.to_numeric(series, errors='coerce').fillna(0)

# --- 3. SIDEBAR: SAMPLING ---
st.sidebar.header("🎯 Sampling Configuration")
sample_pct = st.sidebar.slider("Sample Selection %", 1, 100, 20)

selected_categories = st.sidebar.multiselect(
    "Select Method Categories",
    ["Probability", "Non-Probability", "Audit-Specific", "Advanced"],
    default=["Probability", "Audit-Specific"]
)

# Multi-select for Primary Methods
method_list = ["Simple Random", "Systematic", "Judgmental (High Value)", "Monetary Unit Sampling (MUS)"]
primary_methods = st.sidebar.multiselect("Choose Basis for Selection", options=method_list, default=["Judgmental (High Value)"])

# --- 4. TEMPLATE & UPLOAD ---
# Updated headers include 'TDS Section' as requested
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
    
    # Pre-processing & Cleaning
    num_cols = ['Gross Total', 'taxable value', 'TDS deducted']
    for col in num_cols:
        if col in df.columns: df[col] = clean_numeric(df[col])
    
    if 'TDS Section' not in df.columns: df['TDS Section'] = "Not Specified"

    # --- 5. EXECUTE SAMPLING WITH "BASIS" ---
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
        
        s['Basis for Selection'] = method # Added Basis Column
        sample_df = pd.concat([sample_df, s]).drop_duplicates(subset=['Invoice no', 'Party name'])

    # --- 6. TDS APPLICABILITY BY PARTY ---
    tds_summary = df.groupby(['Party name', 'TDS Section']).agg({
        'taxable value': 'sum',
        'TDS deducted': 'sum'
    }).reset_index()
    
    # Logic: Flag if aggregate > 1,00,000 (194C) or 30,000 (194J) but TDS is 0
    tds_summary['Status'] = np.where(
        (tds_summary['taxable value'] > 30000) & (tds_summary['TDS deducted'] == 0), 
        "⚠️ Potential Default", "✅ Compliant"
    )

    # --- 7. DASHBOARD ---
    st.divider()
    st.subheader("📊 Audit & Sampling Insights")
    
    d1, d2 = st.columns(2)
    with d1:
        # Expenditure vs Sample Coverage
        comp_data = df.groupby('Party name')['taxable value'].sum().nlargest(10).reset_index()
        samp_data = sample_df.groupby('Party name')['taxable value'].sum().reset_index()
        merged = comp_data.merge(samp_data, on='Party name', how='left', suffixes=('_Total', '_Sampled')).fillna(0)
        
        fig1 = px.bar(merged, x='Party name', y=['taxable value_Total', 'taxable value_Sampled'], 
                      title="Sample Coverage by Party", barmode='group', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        st.plotly_chart(fig1, use_container_width=True)

    with d2:
        # TDS Section Breakdown
        fig2 = px.pie(tds_summary, values='taxable value', names='TDS Section', title="TDS Section-wise Exposure", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    # --- 8. PROFESSIONAL EXCEL EXPORT ---
    st.divider()
    st.subheader("📋 Review Selected Samples")
    st.dataframe(sample_df[['Basis for Selection', 'Party name', 'Invoice no', 'taxable value', 'TDS Section']])

    out_bio = BytesIO()
    with pd.ExcelWriter(out_bio, engine='xlsxwriter') as writer:
        # Sheet 1: Raw Analysis
        df.to_excel(writer, sheet_name='1. Full Data Analysis', index=False)
        # Sheet 2: Samples with "Basis"
        sample_df.to_excel(writer, sheet_name='2. Selected Samples', index=False)
        # Sheet 3: Party-wise TDS Applicability
        tds_summary.to_excel(writer, sheet_name='3. TDS Applicability', index=False)
        
        # Add Charts to Excel (Impressive Feature)
        workbook = writer.book
        worksheet = writer.sheets['3. TDS Applicability']
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'Taxable Value',
            'categories': "='3. TDS Applicability'!$A$2:$A$10",
            'values': "='3. TDS Applicability'!$C$2:$C$10",
        })
        worksheet.insert_chart('F2', chart)

    st.download_button("📤 Download Multi-Sheet Audit Report", out_bio.getvalue(), "Final_Audit_Report.xlsx")

else:
    st.info("Awaiting file upload...")
