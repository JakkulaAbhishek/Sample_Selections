import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# --- 1. STYLISH UI CONFIG ---
st.set_page_config(page_title="Ultra-Audit Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .subtotal-box { background-color: #1e3d59; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .materiality-box { background-color: #ff6f61; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("💎 Ultra-Audit Pro: Advanced Sampling, TDS & Materiality Engine")

# --- 2. ROBUST DATA CLEANING ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(series, errors='coerce').fillna(0)

# --- 3. SIDEBAR: FULL SAMPLING METHODS ---
st.sidebar.header("🎯 Sampling Settings")

method_categories = {
    "🔹 Probability Sampling": [
        "Simple Random Sampling", "Systematic Sampling", "Stratified Sampling", 
        "Cluster Sampling", "Multistage Sampling", "Multiphase Sampling", 
        "Area Sampling", "Probability Proportional to Size (PPS) Sampling"
    ],
    "🔹 Non-Probability Sampling": [
        "Convenience Sampling", "Judgmental Sampling", "Purposive Sampling", 
        "Quota Sampling", "Snowball Sampling", "Volunteer Sampling", 
        "Haphazard Sampling", "Consecutive Sampling"
    ],
    "🔹 Audit-Specific Sampling": [
        "Statistical Sampling", "Non-Statistical Sampling", 
        "Monetary Unit Sampling (MUS)", "Block Sampling"
    ],
    "🔹 Advanced / Special Methods": [
        "Sequential Sampling", "Adaptive Sampling", "Reservoir Sampling", 
        "Acceptance Sampling", "Bootstrap Sampling", "Bayesian Sampling"
    ]
}

selected_cats = st.sidebar.multiselect("Select Method Categories", list(method_categories.keys()), default=["🔹 Audit-Specific Sampling"])

available_methods = []
for cat in selected_cats:
    available_methods.extend(method_categories[cat])

primary_methods = st.sidebar.multiselect("Choose Basis for Selection", options=available_methods, default=[available_methods[0]] if available_methods else [])
sample_pct = st.sidebar.slider("Sample Selection %", 1, 100, 20)

# --- 4. MATERIALITY SETTINGS ---
st.sidebar.header("📊 Materiality Settings")
materiality_pct = st.sidebar.slider("Materiality Threshold (%)", 0.5, 10.0, 2.0)

# --- 5. DATA INGESTION ---
headers = ['Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value', 'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted', 'TDS Section']

uploaded_file = st.file_uploader("Upload Raw Ledger", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    
    # Pre-processing
    num_cols = ['Gross Total', 'taxable value', 'TDS deducted', 'Input CGST', 'Input SGST', 'Input IGST']
    for col in num_cols:
        if col in df.columns: df[col] = clean_numeric(df[col])
    
    if 'TDS Section' not in df.columns: df['TDS Section'] = "NA"

    # --- 6. EXECUTE SAMPLING ---
    n = max(1, int(len(df) * (sample_pct / 100)))
    sample_df = pd.DataFrame()
    for method in primary_methods:
        if method == "Judgmental Sampling": s = df.nlargest(n, 'taxable value')
        elif method == "Systematic Sampling": s = df.iloc[::max(1, len(df)//n)]
        else: s = df.sample(n=min(n, len(df)))
        s['Basis for Selection'] = method
        sample_df = pd.concat([sample_df, s]).drop_duplicates(subset=['Invoice no', 'Party name'])

    # --- 7. TDS & DASHBOARD CALCULATIONS ---
    rates = {'194C': 0.01, '194J': 0.10, '194I': 0.10, '194H': 0.05, '194Q': 0.001}
    tds_summary = df.groupby(['Party name', 'TDS Section']).agg({'taxable value': 'sum', 'TDS deducted': 'sum'}).reset_index()
    tds_summary['TDS Needs to be Deducted'] = tds_summary.apply(lambda r: r['taxable value'] * rates.get(str(r['TDS Section']).upper(), 0), axis=1)
    tds_summary['Shortfall'] = np.maximum(0, tds_summary['TDS Needs to be Deducted'] - tds_summary['TDS deducted'])
    tds_summary['Interest (1.5% pm)'] = tds_summary['Shortfall'] * 0.015 * 3

    # Audit Coverage Data
    raw_totals = df.groupby('Party name')['taxable value'].sum().reset_index().rename(columns={'taxable value': 'Raw File Total'})
    samp_totals = sample_df.groupby('Party name')['taxable value'].sum().reset_index().rename(columns={'taxable value': 'Sampled Value'})
    dashboard_df = raw_totals.merge(samp_totals, on='Party name', how='left').fillna(0)
    dashboard_df['% Sample Selection'] = (dashboard_df['Sampled Value'] / dashboard_df['Raw File Total']) * 100

    # --- 8. MATERIALITY CALCULATION ---
    overall_total = df['taxable value'].sum()
    materiality_threshold = overall_total * (materiality_pct / 100)
    materiality_df = df[df['taxable value'] >= materiality_threshold].copy()
    materiality_df['Materiality Flag'] = "Above Threshold"

    st.markdown(f"""
        <div class="materiality-box">
        Materiality Threshold: {materiality_pct}% of total = {materiality_threshold:,.2f}  
        Transactions above threshold: {len(materiality_df)}
        </div>
    """, unsafe_allow_html=True)

    # --- 9. EXCEL EXPORT ---
    out_bio = BytesIO()
    try:
        import xlsxwriter
        with pd.ExcelWriter(out_bio, engine='xlsxwriter') as writer:
            workbook = writer.book
            header_format = workbook.add_format({'bold': True, 'bg_color': '#1e3d59', 'font_color': 'white', 'border': 1})
            num_format = workbook.add_format({'num_format': '#,##0.00'})

            sheet_map = {
                'Audit Dashboard': dashboard_df,
                'TDS Applicability': tds_summary,
                'Selected Samples': sample_df,
                'Materiality Analysis': materiality_df
            }

            for sheet_name, data in sheet_map.items():
                data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
                ws = writer.sheets[sheet_name]
                
                for i, col in enumerate(data.columns):
                    ws.write(1, i, col, header_format)
                    if data[col].dtype in [np.float64, np.int64]:
                        ws.write_formula(0, i, f"=SUM({xlsxwriter.utility.xl_col_to_name(i)}3:{xlsxwriter.utility.xl_col_to_name(i)}{len(data)+2})", num_format)
                
                ws.write(0, 0, "SUBTOTALS (AUTO)", header_format)

                chart = workbook.add_chart({'type': 'pie'})
                chart.add_series({
                    'name': f'{sheet_name} Distribution',
                    'categories': f"='{sheet_name}'!$A$3:$A$12",
                    'values': f"='{sheet_name}'!$B$3:$B$12",
                })
                chart.set_style(10)
                ws.insert_chart('K2', chart)

        st.success("✨ Ultra-Stylish Audit Report Generated with Materiality!")
        st.download_button("📤 Download Multi-Sheet Audit Report", out_bio.getvalue(), "Pro_Audit_Report.xlsx")
    except NameError:
        st.error("Error: Please ensure 'xlsxwriter' is added to your requirements.txt file.")

else:
    st.info("👋 Welcome! Please upload your ledger to begin the Ultra-Audit process.")
