import streamlit as st
import pandas as pd
import plotly.express as px
import holidays
import io
import numpy as np

# --- 1. CONFIGURATION & TAX MASTERS (FY 2025-26) ---
TDS_MASTER = {
    "192": {"desc": "Salary", "limit": 250000},
    "194A": {"desc": "Interest", "limit": 5000},
    "194C": {"desc": "Contractors", "limit": 30000},
    "194H": {"desc": "Commission", "limit": 15000},
    "194I": {"desc": "Rent", "limit": 240000},
    "194J": {"desc": "Professional", "limit": 30000},
    "194Q": {"desc": "Purchase of Goods", "limit": 5000000},
    "194R": {"desc": "Business Perquisites", "limit": 20000},
    "194S": {"desc": "VDA/Crypto", "limit": 10000}
}

# India Holidays for 2025-2026 to cover full audit cycle
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
    # Ensure Amount is treated as a float for comparison
    try:
        amt = float(row['Amount'])
    except:
        amt = 0.0

    dt = pd.to_datetime(row['Invoice Date'], errors='coerce')
    
    if not pd.isna(dt):
        if dt.weekday() == 6: reasons.append("Sunday")
        if dt in ind_holidays: reasons.append(f"Holiday")
    
    if amt >= materiality: reasons.append("High Value")
    if amt > 0 and amt % 1000 == 0: reasons.append("Round Sum")
    
    # TDS Section Matcher
    sec = str(row.get('TDS Section', ''))
    if sec in TDS_MASTER and amt >= TDS_MASTER[sec]['limit']:
        reasons.append(f"TDS Threshold {sec}")
        
    return " | ".join(reasons) if reasons else "Routine"

st.set_page_config(page_title="CA Audit Command Center", layout="wide")

# --- 2. SESSION STATE MANAGEMENT ---
if 'audit_data' not in st.session_state:
    st.session_state.audit_data = None
if 'selected_samples' not in st.session_state:
    st.session_state.selected_samples = None

st.title("🛡️ CA Audit Pro: Intelligence & Compliance Dashboard")

# --- 3. SIDEBAR: SETUP & TEMPLATE ---
with st.sidebar:
    st.header("⚙️ Audit Parameters")
    materiality = st.number_input("Materiality Threshold (INR)", value=100000, step=10000)
    uploaded_file = st.file_uploader("Upload Transaction File", type=['csv', 'xlsx'])
    
    st.subheader("📥 Data Template")
    template = pd.DataFrame({
        'Invoice Date': ['2026-03-31'], 'Invoice No': ['INV/26/001'], 'Party Name': ['Audit Client'],
        'TDS Section': ['194J'], 'Amount': [100000], 'CGST': [9000], 'SGST': [9000], 'IGST': [0],
        'Cess': [0], 'Total Value': [118000]
    })
    st.download_button("Download CSV Template", template.to_csv(index=False), "audit_template.csv")

# --- 4. DATA PROCESSING ENGINE ---
if uploaded_file:
    # Reset state if new file is uploaded
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.audit_data = None
        st.session_state.selected_samples = None
        st.session_state.last_file = uploaded_file.name

    if st.session_state.audit_data is None:
        with st.spinner("Analyzing data for risks..."):
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
            
            # --- Bulletproof Cleaning ---
            df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
            df = df.dropna(subset=['Invoice Date'])
            
            # Feature Engineering
            df['Quarter'] = df['Invoice Date'].apply(get_fiscal_quarter)
            df['Selection Reason'] = df.apply(lambda x: get_selection_reason(x, materiality), axis=1)
            
            # GST Math Integrity
            tax_cols = ['CGST', 'SGST', 'IGST', 'Cess']
            for col in tax_cols:
                if col not in df.columns: df[col] = 0
            
            df['Calc_Total'] = df['Amount'] + df['CGST'] + df['SGST'] + df['IGST'] + df['Cess']
            if 'Total Value' in df.columns:
                df['Total Value'] = pd.to_numeric(df['Total Value'], errors='coerce').fillna(0)
                df.loc[(df['Total Value'] - df['Calc_Total']).abs() > 2, 'Selection Reason'] += " | GST Math Error"

            df['Vouching Status'] = "Pending"
            df['Auditor Remarks'] = ""
            st.session_state.audit_data = df

    df = st.session_state.audit_data

    # --- 5. EXECUTIVE METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", len(df))
    m2.metric("Total Book Value", f"₹{df['Amount'].sum():,.2f}")
    q4_val = df[df['Quarter'] == "Q4 (Jan-Mar)"]['Amount'].sum()
    m3.metric("Q4 Exposure", f"₹{q4_val:,.2f}", delta="Risk High" if q4_val > (df['Amount'].sum()*0.3) else "Normal")
    m4.metric("Flagged Risks", len(df[df['Selection Reason'] != "Routine"]))

    # --- 6. QUARTERLY TREND ANALYSIS ---
    st.divider()
    st.subheader("📅 Expenditure Periodicity (Quarterly Trends)")
    q_data = df.groupby('Quarter')['Amount'].sum().reset_index()
    fig_q = px.bar(q_data, x='Quarter', y='Amount', color='Quarter', text_auto='.2s', title="Spending Distribution by Fiscal Quarter")
    st.plotly_chart(fig_q, use_container_width=True)

    # --- 7. SAMPLING ENGINE ---
    st.divider()
    st.subheader("🎯 Audit Sample Selection")
    col_a, col_b = st.columns(2)
    with col_a:
        sel_q = st.multiselect("Scrutinize specific Quarters", df['Quarter'].unique(), default=df['Quarter'].unique())
    with col_b:
        num_rand = st.number_input("Add Random Samples (Statistical Coverage)", min_value=1, value=15)

    if st.button("🔍 Generate Audit Working Paper"):
        sub_df = df[df['Quarter'].isin(sel_q)]
        priority = sub_df[sub_df['Selection Reason'] != "Routine"]
        randoms = sub_df[sub_df['Selection Reason'] == "Routine"]
        st.session_state.selected_samples = pd.concat([priority, randoms.sample(n=min(len(randoms), num_rand))])

    # --- 8. VOUCHING DASHBOARD ---
    if st.session_state.get('selected_samples') is not None:
        st.divider()
        st.subheader("📝 Vouching Workpaper & Tracking")
        samples = st.session_state.selected_samples
        
        edited_df = st.data_editor(
            samples,
            column_config={
                "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Pending", "Verified", "Query", "TDS Issue", "GST Mismatch"]),
                "Selection Reason": st.column_config.TextColumn("Audit Flag", width="medium"),
                "Amount": st.column_config.NumberColumn("Base Amount", format="₹%.2f"),
                "Auditor Remarks": st.column_config.TextColumn("Remarks", width="large")
            },
            disabled=["Invoice Date", "Party Name", "Amount", "TDS Section", "CGST", "SGST", "IGST", "Quarter", "Selection Reason", "Total Value"],
            hide_index=True, use_container_width=True
        )
        st.session_state.selected_samples = edited_df

        # --- 9. OUTPUT ANALYTICS ---
        st.divider()
        st.subheader("📊 Audit Evidence Charts")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Risk Concentration (Sections)**")
            st.plotly_chart(px.sunburst(edited_df, path=['TDS Section', 'Selection Reason'], values='Amount'), use_container_width=True)
        with c2:
            st.write("**Vouching Status Summary**")
            st.plotly_chart(px.pie(edited_df, names="Vouching Status", hole=0.5), use_container_width=True)
        with c3:
            st.write("**High-Value Outlier Map**")
            st.plotly_chart(px.scatter(edited_df, x="Amount", y="Party Name", color="Vouching Status", size="Amount"), use_container_width=True)

        st.download_button("💾 Export Verified Workpaper", edited_df.to_csv(index=False), "Audit_Quarterly_Verified_Report.csv")
