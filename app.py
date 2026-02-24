import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- 1. SETTINGS ---
st.set_page_config(page_title="Audit Dashboard", layout="wide")
st.title("📊 Audit Sampling & TDS Analytics Dashboard")

# --- 2. DATA SANITIZATION ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.str.replace(r'[^\d.]', '', regex=True) # Fixes TypeError
    return pd.to_numeric(series, errors='coerce').fillna(0)

# --- 3. SIDEBAR: SAMPLING ---
st.sidebar.header("🎯 Sampling Settings")
sample_pct = st.sidebar.slider("Sample Selection Percentage (%)", 1, 100, 20)

selected_categories = st.sidebar.multiselect(
    "Select Method Categories",
    ["Probability", "Non-Probability", "Audit-Specific", "Advanced"],
    default=["Probability", "Audit-Specific"]
)

method_map = {
    "Probability": ["Simple Random", "Systematic", "Stratified", "Cluster", "PPS"],
    "Non-Probability": ["Convenience", "Judgmental", "Quota", "Haphazard"],
    "Audit-Specific": ["Statistical", "Monetary Unit Sampling (MUS)", "Block Sampling"],
    "Advanced": ["Bootstrap", "Bayesian", "Sequential"]
}

available_methods = []
for cat in selected_categories:
    available_methods.extend(method_map[cat])

primary_methods = st.sidebar.multiselect("Choose Primary Method(s)", options=available_methods, default=["Simple Random"])

# --- 4. TEMPLATE & UPLOAD ---
# Column headers based on your image
headers = ['Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value', 
           'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted']

# Generate Sample Excel for download
tmp_df = pd.DataFrame(columns=headers)
t_buffer = BytesIO()
tmp_df.to_excel(t_buffer, index=False, engine='xlsxwriter')
st.download_button("📥 Download Template Excel", t_buffer.getvalue(), "audit_template.xlsx")

uploaded_file = st.file_uploader("Upload Raw Ledger File", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    
    # Numeric Cleaning to prevent app crash
    for col in ['Gross Total', 'taxable value', 'TDS deducted']:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # --- 5. TDS SECTION-WISE BREAKDOWN LOGIC ---
    def get_section(name):
        name = str(name).lower()
        if 'rent' in name: return "194I"
        if any(x in name for x in ['prof', 'legal', 'audit']): return "194J"
        if any(x in name for x in ['contract', 'repair', 'civil']): return "194C"
        return "Manual Section Check"

    df['TDS Section'] = df['Party name'].apply(get_section)

    # --- 6. SAMPLING EXECUTION ---
    n = max(1, int(len(df) * (sample_pct / 100)))
    sample_df = pd.DataFrame()

    for method in primary_methods:
        if method == "Simple Random":
            s = df.sample(n=min(n, len(df)))
        elif method == "Judgmental":
            s = df.sort_values(by='taxable value', ascending=False).head(n)
        else:
            s = df.sample(n=min(n, len(df)))
        sample_df = pd.concat([sample_df, s]).drop_duplicates(subset=['Invoice no', 'Party name'])

    # --- 7. DASHBOARD VISUALS ---
    st.divider()
    st.subheader("📈 Sampling Coverage Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Comparison: Total vs Sampled per Party
        total_party = df.groupby('Party name')['taxable value'].sum().nlargest(10).reset_index()
        sample_party = sample_df.groupby('Party name')['taxable value'].sum().reset_index()
        
        comparison = total_party.merge(sample_party, on='Party name', how='left', suffixes=('_Total', '_Sampled')).fillna(0)
        
        fig1 = px.bar(comparison, x='Party name', y=['taxable value_Total', 'taxable value_Sampled'], 
                      title="Expenditure Coverage: Total vs Sampled", barmode='group')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # TDS Section Pie Chart
        sec_dist = df['TDS Section'].value_counts().reset_index()
        fig2 = px.pie(sec_dist, values='count', names='TDS Section', title="TDS Section-wise Distribution", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    # --- 8. FINAL OUTPUT ---
    st.divider()
    st.subheader("📋 Final Selected Samples")
    st.dataframe(sample_df)

    # Exporting Excel with specific sheets
    out_bio = BytesIO()
    with pd.ExcelWriter(out_bio, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Raw_Data_Full', index=False)
        sample_df.to_excel(writer, sheet_name='Selected_Samples', index=False)
        # Unique sheet for TDS Section Breakdown
        df[['Party name', 'taxable value', 'TDS Section']].to_excel(writer, sheet_name='TDS_Applicability', index=False)
    
    st.download_button("📤 Download Final Audit File", out_bio.getvalue(), "Audit_Report_Final.xlsx")

else:
    st.info("Please upload the raw ledger file to generate the dashboard and samples.")
