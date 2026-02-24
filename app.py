import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="Pro Audit Sampler", layout="wide")
st.title("📊 Advanced Audit Sampling & TDS Analytics")

# --- 2. TDS LOGIC & THRESHOLDS ---
TDS_RULES = {
    '194C (Contractor)': {'keywords': ['contract', 'labor', 'maintenance', 'civil', 'repair'], 'threshold': 30000},
    '194J (Professional)': {'keywords': ['audit', 'legal', 'consultant', 'technical', 'professional'], 'threshold': 30000},
    '194H (Commission)': {'keywords': ['commission', 'brokerage'], 'threshold': 15000},
    '194I (Rent)': {'keywords': ['rent', 'lease'], 'threshold': 240000},
    '194Q (Goods)': {'keywords': ['purchase', 'goods', 'raw material'], 'threshold': 5000000}
}

def detect_section(party_name, taxable_val):
    name = str(party_name).lower()
    for sec, rule in TDS_RULES.items():
        if any(key in name for key in rule['keywords']):
            return sec
    return "Others / Check Manually"

# --- 3. SIDEBAR: SAMPLING PARAMETERS ---
st.sidebar.header("🎯 Sampling Configuration")

# Category Selection
cat_type = st.sidebar.selectbox("Category Group", [
    "Probability Sampling", "Non-Probability Sampling", 
    "Audit-Specific Methods", "Advanced / Special Methods"
])

# Method Selection based on Category
methods = {
    "Probability Sampling": ["Simple Random", "Systematic", "Stratified", "Cluster", "PPS"],
    "Non-Probability Sampling": ["Convenience", "Judgmental", "Quota", "Haphazard", "Snowball"],
    "Audit-Specific Methods": ["Statistical", "Non-Statistical", "Monetary Unit Sampling (MUS)", "Block Sampling"],
    "Advanced / Special Methods": ["Sequential", "Bootstrap", "Bayesian", "Reservoir"]
}
selected_method = st.sidebar.selectbox("Specific Method", methods[cat_type])

# Percentage Selection
sample_pct = st.sidebar.slider("Sample Percentage (%)", 1, 100, 10)
materiality = st.sidebar.number_input("Materiality Level (Threshold)", value=100000)

# --- 4. DATA INGESTION & TEMPLATE ---
st.subheader("1. Data Upload & Template")
col_a, col_b = st.columns([3, 1])

with col_b:
    # UPDATED: Template now includes columns from BOTH images
    cols = ['Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value', 
            'Input CGST', 'Input SGST', 'Input IGST', 'TDS Deducted', 'Round Off']
    template_df = pd.DataFrame(columns=cols)
    
    t_buffer = BytesIO()
    template_df.to_excel(t_buffer, index=False)
    st.download_button("📥 Download Excel Template", t_buffer.getvalue(), "audit_template.xlsx")

uploaded_file = st.file_uploader("Upload Raw File", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    
    # Fill missing columns if any for stability
    for col in cols:
        if col not in df.columns: df[col] = 0

    # --- 5. TDS ANALYSIS (CELL BREAKDOWN) ---
    df['TDS Section'] = df.apply(lambda x: detect_section(x['Party name'], x['taxable value']), axis=1)
    df['Compliance'] = np.where((df['taxable value'] > 30000) & (df['TDS Deducted'] == 0), "⚠️ Action Required", "✅ OK")

    # --- 6. SAMPLING LOGIC ---
    num_samples = int(len(df) * (sample_pct / 100))
    if num_samples < 1: num_samples = 1

    if selected_method == "Simple Random":
        sample_df = df.sample(n=num_samples)
    elif selected_method == "Judgmental":
        sample_df = df[df['taxable value'] >= materiality].sort_values(by='taxable value', ascending=False)
    elif selected_method == "Systematic":
        step = len(df) // num_samples
        sample_df = df.iloc[::step] if step > 0 else df
    else:
        # Default fallback for complex methods in this demo
        sample_df = df.sample(n=num_samples)

    # --- 7. DASHBOARD ---
    st.divider()
    st.subheader("2. Dashboard & Visuals")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Total Vouchers", len(df))
    with c2:
        st.metric("Total Taxable Value", f"₹{df['taxable value'].sum():,.2f}")
    with c3:
        st.metric("Samples Selected", len(sample_df))

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        # Party-wise Expenditure
        party_data = df.groupby('Party name')['taxable value'].sum().reset_index().sort_values(by='taxable value', ascending=False).head(10)
        fig1 = px.bar(party_data, x='taxable value', y='Party name', orientation='h', title="Top 10 Parties by Expenditure", color='taxable value')
        st.plotly_chart(fig1, use_container_width=True)
    
    with chart_col2:
        # TDS Section Breakdown
        sec_data = df['TDS Section'].value_counts().reset_index()
        fig2 = px.pie(sec_data, names='TDS Section', values='count', title="TDS Section Distribution", hole=0.3)
        st.plotly_chart(fig2, use_container_width=True)

    # --- 8. OUTPUT ---
    st.divider()
    st.subheader("3. Sampled Output")
    st.dataframe(sample_df[['Date', 'Party name', 'Invoice no', 'taxable value', 'TDS Section', 'Compliance']])

    # Final Export
    out_buffer = BytesIO()
    with pd.ExcelWriter(out_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Raw_Data_Analysis', index=False)
        sample_df.to_excel(writer, sheet_name='Sampled_Vouchers', index=False)
    
    st.download_button("📤 Download Final Audit Report", out_buffer.getvalue(), "Audit_Report.xlsx")

else:
    st.info("Waiting for file upload...")
