import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- 1. SETTINGS & UI ---
st.set_page_config(page_title="CA Audit Tool", layout="wide")
st.title("🛡️ Advanced Audit Sampling & TDS Analytics")

# --- 2. DATA SANITIZATION (Fixes TypeError) ---
def clean_numeric_data(series):
    # Converts to string, removes non-numeric chars except decimals, then to float
    if series.dtype == 'object':
        series = series.str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(series, errors='coerce').fillna(0)

# --- 3. SIDEBAR: MULTI-METHOD SELECTION ---
st.sidebar.header("🎯 Sampling Settings")

selected_categories = st.sidebar.multiselect(
    "Select Method Categories",
    ["Probability", "Non-Probability", "Audit-Specific", "Advanced"],
    default=["Probability", "Audit-Specific", "Advanced"]
)

method_map = {
    "Probability": ["Simple Random", "Systematic", "Stratified", "Cluster", "PPS"],
    "Non-Probability": ["Convenience", "Judgmental", "Quota", "Haphazard"],
    "Audit-Specific": ["Statistical", "Monetary Unit Sampling (MUS)", "Block Sampling"],
    "Advanced": ["Bootstrap", "Bayesian", "Sequential"]
}

# Combine available methods from selected categories
available_methods = []
for cat in selected_categories:
    available_methods.extend(method_map[cat])

# UPDATED: Select one or more primary sampling methods
primary_methods = st.sidebar.multiselect(
    "Choose Primary Sampling Method(s)", 
    options=available_methods,
    default=[available_methods[0]] if available_methods else []
)

sample_pct = st.sidebar.slider("Sample Percentage (%)", 1, 100, 20)

# --- 4. DATA TEMPLATE & UPLOAD ---
st.subheader("1. Data Ingestion")
col_tmp, col_up = st.columns([1, 2])

with col_tmp:
    # Column headers based on your image requirement
    headers = ['Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value', 
               'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted']
    tmp_df = pd.DataFrame(columns=headers)
    buffer = BytesIO()
    # Note: Ensure xlsxwriter is installed in your environment
    tmp_df.to_excel(buffer, index=False, engine='xlsxwriter') 
    st.download_button("📥 Download Audit Template", buffer.getvalue(), "audit_template.xlsx")

uploaded_file = st.file_uploader("Upload your Ledger", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    
    # --- DATA CLEANING (Prevents TypeError) ---
    numeric_cols = ['Gross Total', 'taxable value', 'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_data(df[col])
        else:
            df[col] = 0.0

    # --- 5. TDS COMPLIANCE ANALYSIS ---
    def get_tds_section(name):
        name = str(name).lower()
        if 'rent' in name: return "194I"
        if any(x in name for x in ['prof', 'legal', 'audit']): return "194J"
        if any(x in name for x in ['contract', 'repair', 'civil']): return "194C"
        return "Manual Review Required"

    df['TDS Section'] = df['Party name'].apply(get_tds_section)
    
    # Safe numerical comparison for compliance
    df['Compliance'] = np.where(
        (df['taxable value'] > 30000) & (df['TDS deducted'] == 0), 
        "⚠️ Action Required", "✅ OK"
    )

    # --- 6. MULTI-METHOD SAMPLING EXECUTION ---
    n = max(1, int(len(df) * (sample_pct / 100)))
    combined_samples = pd.DataFrame()

    for method in primary_methods:
        if method == "Simple Random":
            s = df.sample(n=min(n, len(df)))
        elif method == "Judgmental":
            s = df.sort_values(by='taxable value', ascending=False).head(n)
        elif method == "Systematic":
            k = max(1, len(df) // n)
            s = df.iloc[::k]
        else:
            s = df.sample(n=min(n, len(df)))
        
        s['Sampling Method Used'] = method
        combined_samples = pd.concat([combined_samples, s]).drop_duplicates(subset=['Invoice no', 'Party name'])

    # --- 7. DASHBOARD & VISUALS ---
    st.divider()
    st.subheader("2. Audit Dashboard")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", len(df))
    m2.metric("Total Taxable Value", f"₹{df['taxable value'].sum():,.0f}")
    m3.metric("Final Sample Count", len(combined_samples))

    c1, c2 = st.columns(2)
    with c1:
        top_parties = df.groupby('Party name')['taxable value'].sum().nlargest(10).reset_index()
        fig1 = px.bar(top_parties, x='taxable value', y='Party name', title="Top 10 Parties (by Value)", orientation='h', color='taxable value')
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        gst_sums = pd.DataFrame({
            'GST Type': ['CGST', 'SGST', 'IGST'],
            'Amount': [df['Input CGST'].sum(), df['Input SGST'].sum(), df['Input IGST'].sum()]
        })
        fig2 = px.pie(gst_sums, values='Amount', names='GST Type', title="GST Input Distribution", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    # --- 8. OUTPUT ---
    st.divider()
    st.subheader(f"3. Sampled Output ({', '.join(primary_methods)})")
    st.dataframe(combined_samples)

    # Excel Export with Multiple Sheets
    out_bio = BytesIO()
    try:
        with pd.ExcelWriter(out_bio, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Full_Analysis', index=False)
            combined_samples.to_excel(writer, sheet_name='Samples_Selected', index=False)
        st.download_button("📤 Download Final Audit Report", out_bio.getvalue(), "Audit_Report.xlsx")
    except ModuleNotFoundError:
        st.error("Error: 'xlsxwriter' not found. Please add it to your requirements.txt")

else:
    st.info("Please upload a file to begin.")
