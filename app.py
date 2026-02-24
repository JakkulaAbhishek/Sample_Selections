import streamlit as st
import pandas as pd
import plotly.express as px
import holidays
import io
import numpy as np

# --- 1. CONFIGURATION & MASTERS ---
TDS_MASTER = {
    "192": {"desc": "Salary", "limit": 250000},
    "194A": {"desc": "Interest", "limit": 5000},
    "194C": {"desc": "Contractors", "limit": 30000},
    "194H": {"desc": "Commission", "limit": 15000},
    "194I": {"desc": "Rent", "limit": 240000},
    "194J": {"desc": "Professional", "limit": 30000},
    "194Q": {"desc": "Purchase of Goods", "limit": 5000000}
}

ind_holidays = holidays.India(years=[2025, 2026])

def get_fiscal_quarter(date):
    month = date.month
    if month in [4, 5, 6]: return "Q1 (Apr-Jun)"
    if month in [7, 8, 9]: return "Q2 (Jul-Sep)"
    if month in [10, 11, 12]: return "Q3 (Oct-Dec)"
    return "Q4 (Jan-Mar)"

def get_selection_reason(row, materiality):
    reasons = []
    dt = pd.to_datetime(row['Invoice Date'])
    if dt.weekday() == 6: reasons.append("Sunday")
    if dt in ind_holidays: reasons.append(f"Holiday")
    if row['Amount'] >= materiality: reasons.append("High Value")
    if row['Amount'] > 0 and row['Amount'] % 1000 == 0: reasons.append("Round Sum")
    
    # Section Check
    sec = str(row['TDS Section'])
    if sec in TDS_MASTER and row['Amount'] >= TDS_MASTER[sec]['limit']:
        reasons.append(f"TDS Threshold {sec}")
    return " | ".join(reasons) if reasons else "Routine"

st.set_page_config(page_title="Audit Intelligence 2026", layout="wide")

# --- 2. SIDEBAR SETUP ---
with st.sidebar:
    st.header("📌 Audit Configuration")
    materiality = st.number_input("Materiality (INR)", value=100000)
    uploaded_file = st.file_uploader("Upload Transaction File", type=['csv', 'xlsx'])
    
    # Professional Template
    st.subheader("📥 Standard Template")
    template = pd.DataFrame({
        'Invoice Date': ['2026-03-25'], 'Invoice No': ['INV/25/001'], 'Party Name': ['Audit Client'],
        'TDS Section': ['194J'], 'Amount': [100000], 'CGST': [9000], 'SGST': [9000], 'IGST': [0],
        'Cess': [0], 'Total Value': [118000]
    })
    st.download_button("Download Template", template.to_csv(index=False), "audit_template.csv")

# --- 3. DATA PROCESSING ---
if uploaded_file:
    if 'audit_data' not in st.session_state or st.session_state.get('filename') != uploaded_file.name:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
        df['Quarter'] = df['Invoice Date'].apply(get_fiscal_quarter)
        df['Selection Reason'] = df.apply(lambda x: get_selection_reason(x, materiality), axis=1)
        
        # GST Integrity
        df['Calc_Total'] = df['Amount'] + df['CGST'] + df['SGST'] + df['IGST'] + df.get('Cess', 0)
        df.loc[(df['Total Value'] - df['Calc_Total']).abs() > 2, 'Selection Reason'] += " | GST Error"
        
        df['Vouching Status'] = "Pending"
        df['Auditor Remarks'] = ""
        st.session_state.audit_data = df
        st.session_state.filename = uploaded_file.name

    df = st.session_state.audit_data

    # --- 4. EXECUTIVE SUMMARY ---
    st.title("📊 Audit Intelligence Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Population Size", len(df))
    m2.metric("Total Turnover/Exp", f"₹{df['Amount'].sum():,.0f}")
    q4_val = df[df['Quarter'] == "Q4 (Jan-Mar)"]['Amount'].sum()
    m3.metric("Q4 Exposure", f"₹{q4_val:,.0f}", delta=f"Risk: High" if q4_val > (df['Amount'].sum()*0.3) else "Normal")
    m4.metric("Flagged Items", len(df[df['Selection Reason'] != "Routine"]))

    # --- 5. QUARTERLY TRENDS ---
    st.divider()
    st.subheader("📅 Quarterly Trend & Periodicity Analysis")
    q_data = df.groupby('Quarter')['Amount'].sum().reset_index()
    fig_q = px.bar(q_data, x='Quarter', y='Amount', color='Quarter', title="Expense Distribution by Fiscal Quarter")
    st.plotly_chart(fig_q, use_container_width=True)

    # --- 6. SAMPLING & VOUCHING ---
    st.divider()
    st.subheader("🔍 Sampling Engine")
    col_a, col_b = st.columns(2)
    with col_a:
        sel_q = st.multiselect("Filter by Quarter", df['Quarter'].unique(), default=df['Quarter'].unique())
    with col_b:
        num_rand = st.number_input("Additional Random Samples", value=10)

    if st.button("Generate Working Paper"):
        sub_df = df[df['Quarter'].isin(sel_q)]
        priority = sub_df[sub_df['Selection Reason'] != "Routine"]
        randoms = sub_df[sub_df['Selection Reason'] == "Routine"]
        st.session_state.selected_samples = pd.concat([priority, randoms.sample(n=min(len(randoms), num_rand))])

    if st.session_state.get('selected_samples') is not None:
        samples = st.session_state.selected_samples
        
        st.write("### 📝 Vouching Dashboard")
        edited_df = st.data_editor(
            samples,
            column_config={
                "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Pending", "Verified", "Query", "TDS Issue", "MSR Mismatch"]),
                "Selection Reason": st.column_config.TextColumn("Audit Flags", width="medium"),
            },
            disabled=["Invoice Date", "Party Name", "Amount", "TDS Section", "CGST", "SGST", "IGST", "Quarter", "Selection Reason"],
            hide_index=True, use_container_width=True
        )
        st.session_state.selected_samples = edited_df

        # --- 7. FINAL ANALYTICS & EXPORT ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Section-wise Audit Progress**")
            st.plotly_chart(px.strip(edited_df, x="Quarter", y="Amount", color="Vouching Status", hover_data=["Party Name"]))
        with c2:
            st.write("**Risk Reason Breakdown**")
            st.plotly_chart(px.pie(edited_df, names="Selection Reason", hole=0.5))

        st.download_button("💾 Export Verified Audit File", edited_df.to_csv(index=False), "Audit_Quarterly_Verified.csv")
