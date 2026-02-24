import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import xlsxwriter

# ---------------------------------------------------
# 1️⃣ PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Ultra-Audit Pro | Materiality Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# 2️⃣ PREMIUM UI
# ---------------------------------------------------
st.markdown("""
<style>
.main {background: linear-gradient(to right, #f8f9fa, #eef2f7);}
.stMetric {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #1e3d59;
}
.materiality-box {
    background-color: #1e3d59;
    color: white;
    padding: 18px;
    border-radius: 12px;
    font-size: 18px;
}
.highrisk {
    background-color: #ffe6e6;
}
</style>
""", unsafe_allow_html=True)

st.title("💎 Ultra-Audit Pro – Advanced Sampling + Materiality Engine")

# ---------------------------------------------------
# 3️⃣ CLEAN NUMERIC FUNCTION
# ---------------------------------------------------
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(series, errors='coerce').fillna(0)

# ---------------------------------------------------
# 4️⃣ SIDEBAR – SAMPLING + MATERIALITY
# ---------------------------------------------------
st.sidebar.header("🎯 Audit Controls")

materiality_base = st.sidebar.selectbox(
    "Materiality Base",
    ["Total Revenue", "Total Taxable Value", "Gross Total"]
)

materiality_pct = st.sidebar.slider("Overall Materiality %", 0.5, 10.0, 5.0)

performance_pct = st.sidebar.slider("Performance Materiality % of OM", 40, 90, 75)

sample_pct = st.sidebar.slider("Sample % (Base Selection)", 1, 100, 20)

sampling_method = st.sidebar.selectbox(
    "Sampling Method",
    ["Simple Random Sampling",
     "Systematic Sampling",
     "Judgmental Sampling (High Value)",
     "Monetary Unit Sampling (MUS)"]
)

# ---------------------------------------------------
# 5️⃣ FILE UPLOAD
# ---------------------------------------------------
headers = ['Date', 'Party name', 'Invoice no', 'Gross Total',
           'taxable value', 'Input CGST', 'Input SGST',
           'Input IGST', 'TDS deducted', 'TDS Section']

uploaded_file = st.file_uploader("📂 Upload Ledger (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded_file:

    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)

    num_cols = ['Gross Total', 'taxable value', 'TDS deducted',
                'Input CGST', 'Input SGST', 'Input IGST']

    for col in num_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # ---------------------------------------------------
    # 6️⃣ MATERIALITY CALCULATION
    # ---------------------------------------------------
    if materiality_base == "Total Revenue":
        base_value = df['Gross Total'].sum()
    elif materiality_base == "Total Taxable Value":
        base_value = df['taxable value'].sum()
    else:
        base_value = df['Gross Total'].sum()

    overall_materiality = base_value * (materiality_pct / 100)
    performance_materiality = overall_materiality * (performance_pct / 100)

    st.markdown(f"""
    <div class="materiality-box">
    📊 Overall Materiality: ₹ {overall_materiality:,.2f} <br>
    🔎 Performance Materiality: ₹ {performance_materiality:,.2f}
    </div>
    """, unsafe_allow_html=True)

    # Flag high risk
    df['High Risk (Above PM)'] = df['taxable value'] >= performance_materiality

    # ---------------------------------------------------
    # 7️⃣ SAMPLING LOGIC
    # ---------------------------------------------------
    n = max(1, int(len(df) * sample_pct / 100))

    if sampling_method == "Simple Random Sampling":
        sample_df = df.sample(n=min(n, len(df)))

    elif sampling_method == "Systematic Sampling":
        step = max(1, len(df)//n)
        sample_df = df.iloc[::step]

    elif sampling_method == "Judgmental Sampling (High Value)":
        sample_df = df.nlargest(n, 'taxable value')

    elif sampling_method == "Monetary Unit Sampling (MUS)":
        df_sorted = df.sort_values('taxable value', ascending=False)
        sample_df = df_sorted.head(n)

    # Add mandatory high-risk items
    high_risk_df = df[df['High Risk (Above PM)']]
    sample_df = pd.concat([sample_df, high_risk_df]).drop_duplicates()

    # ---------------------------------------------------
    # 8️⃣ TDS ENGINE
    # ---------------------------------------------------
    rates = {'194C': 0.01, '194J': 0.10, '194I': 0.10,
             '194H': 0.05, '194Q': 0.001}

    df['Expected TDS'] = df.apply(
        lambda r: r['taxable value'] * rates.get(str(r['TDS Section']).upper(), 0),
        axis=1
    )

    df['Shortfall'] = np.maximum(0, df['Expected TDS'] - df['TDS deducted'])
    df['Interest (3 months @1.5%)'] = df['Shortfall'] * 0.015 * 3

    # ---------------------------------------------------
    # 9️⃣ DASHBOARD
    # ---------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(df))
    col2.metric("Selected Samples", len(sample_df))
    col3.metric("High Risk Transactions", df['High Risk (Above PM)'].sum())

    fig = px.pie(
        names=["Sampled", "Remaining"],
        values=[len(sample_df), len(df)-len(sample_df)],
        title="Sampling Coverage"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight high risk
    st.subheader("📋 Sampled Transactions")
    st.dataframe(sample_df.style.apply(
        lambda x: ['background-color: #ffe6e6' if x['High Risk (Above PM)'] else '' for _ in x],
        axis=1
    ))

    # ---------------------------------------------------
    # 🔟 EXCEL EXPORT
    # ---------------------------------------------------
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:

        df.to_excel(writer, sheet_name='Full Data', index=False)
        sample_df.to_excel(writer, sheet_name='Sampled Data', index=False)

        materiality_df = pd.DataFrame({
            "Base Value": [base_value],
            "Overall Materiality": [overall_materiality],
            "Performance Materiality": [performance_materiality]
        })

        materiality_df.to_excel(writer, sheet_name='Materiality', index=False)

        workbook = writer.book
        chart = workbook.add_chart({'type': 'column'})

        chart.add_series({
            'name': 'Shortfall',
            'categories': ['Full Data', 1, 1, len(df), 1],
            'values': ['Full Data', 1, df.columns.get_loc('Shortfall'),
                       len(df), df.columns.get_loc('Shortfall')],
        })

        worksheet = writer.sheets['Full Data']
        worksheet.insert_chart('L2', chart)

    st.download_button(
        "📥 Download Ultra Audit Report",
        output.getvalue(),
        file_name="Ultra_Audit_With_Materiality.xlsx"
    )

else:
    st.info("👋 Upload your ledger to activate the Audit Engine.")
