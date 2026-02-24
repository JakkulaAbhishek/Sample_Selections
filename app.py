import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- 1. SETTINGS & UI ---
st.set_page_config(page_title="CA Audit Tool", layout="wide")
st.title("🛡️ Advanced Audit Sampling & TDS Compliance")

# --- 2. DATA CLEANING HELPER ---
def clean_numeric(series):
    # Converts to numeric, turns errors/strings into NaN, then fills with 0
    return pd.to_numeric(series, errors='coerce').fillna(0)

# --- 3. SIDEBAR: MULTI-METHOD SELECTION ---
st.sidebar.header("🎯 Sampling Settings")

# Allow selecting multiple categories
selected_categories = st.sidebar.multiselect(
    "Select Method Categories",
    ["Probability", "Non-Probability", "Audit-Specific", "Advanced"],
    default=["Probability"]
)

method_map = {
    "Probability": ["Simple Random", "Systematic", "Stratified", "Cluster", "PPS"],
    "Non-Probability": ["Convenience", "Judgmental", "Quota", "Haphazard"],
    "Audit-Specific": ["Statistical", "Monetary Unit Sampling (MUS)", "Block Sampling"],
    "Advanced": ["Bootstrap", "Bayesian", "Sequential"]
}

# Flatten available methods based on chosen categories
available_methods = []
for cat in selected_categories:
    available_methods.extend(method_map[cat])

final_method = st.sidebar.selectbox("Choose Primary Sampling Method", available_methods) if available_methods else None
sample_pct = st.sidebar.slider("Sample Percentage (%)", 1, 100, 20)

# --- 4. TEMPLATE & UPLOAD ---
st.subheader("1. Data Ingestion")
col_template, col_upload = st.columns([1, 2])

with col_template:
    # Based on your image headers
    headers = ['Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value', 
               'Input CGST', 'Input SGST', 'Input IGST', 'TDS Deducted']
    tmp_df = pd.DataFrame(columns=headers)
    buffer = BytesIO()
    tmp_df.to_excel(buffer, index=False)
    st.download_button("📥 Download Audit Template", buffer.getvalue(), "audit_template.xlsx")

uploaded_file = st.file_uploader("Upload your Ledger", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    
    # --- FIX: DATA TYPE CONVERSION (Prevents your TypeError) ---
    numeric_cols = ['Gross Total', 'taxable value', 'Input CGST', 'Input SGST', 'Input IGST', 'TDS Deducted']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
        else:
            df[col] = 0.0

    # --- 5. TDS COMPLIANCE LOGIC ---
    # Section Detection (Basic Keywords)
    def get_section(name):
        name = str(name).lower()
        if 'rent' in name: return "194I"
        if 'prof' in name or 'legal' in name: return "194J"
        if 'contract' in name or 'repair' in name: return "194C"
        return "Check Section"

    df['Detected Section'] = df['Party name'].apply(get_section)
    
    # Safe comparison to prevent TypeError
    df['Compliance Status'] = np.where(
        (df['taxable value'] > 30000) & (df['TDS Deducted'] == 0), 
        "⚠️ Non-Compliant", "✅ OK"
    )

    # --- 6. SAMPLING EXECUTION ---
    n = max(1, int(len(df) * (sample_pct / 100)))
    
    if final_method == "Simple Random":
        sample_df = df.sample(n=n)
    elif final_method == "Judgmental":
        sample_df = df.sort_values(by='taxable value', ascending=False).head(n)
    elif final_method == "Systematic":
        k = max(1, len(df) // n)
        sample_df = df.iloc[::k]
    else:
        sample_df = df.sample(n=n) # Fallback

    # --- 7. DASHBOARD & CHARTS ---
    st.divider()
    st.subheader("2. Analytics Dashboard")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", len(df))
    m2.metric("Total Taxable Value", f"₹{df['taxable value'].sum():,.2f}")
    m3.metric("Exceptions Found", len(df[df['Compliance Status'] == "⚠️ Non-Compliant"]))

    c1, c2 = st.columns(2)
    with c1:
        # Party-wise Expenditure Chart
        top_parties = df.groupby('Party name')['taxable value'].sum().nlargest(10).reset_index()
        fig1 = px.bar(top_parties, x='taxable value', y='Party name', title="Top 10 Parties", orientation='h', color='taxable value')
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        # GST Composition
        gst_totals = pd.DataFrame({
            'Tax Type': ['CGST', 'SGST', 'IGST'],
            'Amount': [df['Input CGST'].sum(), df['Input SGST'].sum(), df['Input IGST'].sum()]
        })
        fig2 = px.pie(gst_totals, values='Amount', names='Tax Type', title="GST Input Distribution", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    # --- 8. OUTPUT ---
    st.divider()
    st.subheader(f"3. Selected Samples ({final_method})")
    st.dataframe(sample_df)

    # Export
    out_bio = BytesIO()
    with pd.ExcelWriter(out_bio, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Full_Analysis', index=False)
        sample_df.to_excel(writer, sheet_name='Sample_List', index=False)
    
    st.download_button("📤 Download Final Audit Report", out_bio.getvalue(), "Audit_Report.xlsx")

else:
    st.info("Please download the template, fill it, and upload here.")
