import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import holidays
import io
import numpy as np
from datetime import datetime

# --- 1. GLOBAL SETTINGS & STYLING ---
st.set_page_config(page_title="CA Audit Intelligence Suite", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004b95; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MASTER DATA & LOGIC ---
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

# --- 3. SESSION STATE INITIALIZATION ---
if 'db' not in st.session_state: st.session_state.db = None
if 'samples' not in st.session_state: st.session_state.samples = None
if 'reset' not in st.session_state: st.session_state.reset = False

def get_fiscal_quarter(date):
    if pd.isna(date): return "N/A"
    m = date.month
    if m in [4,5,6]: return "Q1 (AMJ)"
    if m in [7,8,9]: return "Q2 (JAS)"
    if m in [10,11,12]: return "Q3 (OND)"
    return "Q4 (JFM)"

def run_risk_engine(row, materiality):
    flags = []
    amt = float(row.get('Amount', 0))
    dt = pd.to_datetime(row.get('Invoice Date'), errors='coerce')
    
    if not pd.isna(dt):
        if dt.weekday() == 6: flags.append("Sunday")
        if dt in ind_holidays: flags.append("Public Holiday")
    
    if amt >= materiality: flags.append("High Value")
    if amt > 0 and amt % 1000 == 0: flags.append("Round Amount")
    
    sec = str(row.get('TDS Section', ''))
    if sec in TDS_MASTER and amt >= TDS_MASTER[sec]['limit']:
        flags.append(f"TDS Threshold {sec}")
        
    return " | ".join(flags) if flags else "Routine"

# --- 4. SIDEBAR & DATA INPUT ---
with st.sidebar:
    st.title("🛡️ Audit Control")
    materiality = st.number_input("Performance Materiality (INR)", value=100000, step=10000)
    sample_rate = st.slider("Sampling Confidence Rate (%)", 5, 100, 25)
    
    # --- AUTOMATED TEMPLATE GENERATOR ---
    st.subheader("📋 Setup Template")
    buffer = io.BytesIO()
    example_data = pd.DataFrame({
        'Invoice Date': ['2026-03-31', '2025-12-25', '2026-01-26'],
        'Invoice No': ['INV/001', 'INV/002', 'INV/003'],
        'Party Name': ['Reliance Ind', 'TCS Ltd', 'Local Vendor'],
        'TDS Section': ['194Q', '194J', '194C'],
        'Amount': [6000000, 45000, 15000],
        'CGST': [0, 4050, 1350], 'SGST': [0, 4050, 1350], 'IGST': [1080000, 0, 0],
        'Total Value': [7080000, 53100, 17700]
    })
    example_data.to_csv(buffer, index=False)
    st.download_button("📥 Download Sample Excel", buffer.getvalue(), "audit_template.csv", "text/csv")
    
    st.divider()
    uploaded_file = st.file_uploader("Upload Client Ledger", type=['csv', 'xlsx'])

# --- 5. MAIN APPLICATION LOGIC ---
if uploaded_file:
    # Load and Clean
    if st.session_state.db is None or st.sidebar.button("Reload Data"):
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        
        # Numeric Force
        for col in ['Amount', 'CGST', 'SGST', 'IGST', 'Total Value']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        df = df.dropna(subset=['Invoice Date'])
        
        # Enrich
        df['Quarter'] = df['Invoice Date'].apply(get_fiscal_quarter)
        df['Risk Flags'] = df.apply(lambda x: run_risk_engine(x, materiality), axis=1)
        df['Vouching Status'] = "Pending"
        df['Error Amount'] = 0.0
        
        st.session_state.db = df

    db = st.session_state.db

    # --- TOP METRICS ---
    st.title("📊 Financial Ledger Intelligence")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Count", len(db))
    m2.metric("Total Value", f"₹{db['Amount'].sum():,.0f}")
    m3.metric("Critical Alerts", len(db[db['Risk Flags'] != "Routine"]))
    m4.metric("Avg. Invoice Size", f"₹{db['Amount'].mean():,.0f}")

    # --- TABS FOR WORKFLOW ---
    t1, t2, t3 = st.tabs(["📈 Population Analytics", "🎯 Smart Sampling", "🧮 Final Report"])

    with t1:
        st.subheader("Population Distribution")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.box(db, x='Quarter', y='Amount', color='Quarter', title="Expense Variance per Quarter"), use_container_width=True)
        with c2:
            st.plotly_chart(px.pie(db, names='Risk Flags', title="Risk Profile Concentration"), use_container_width=True)
            
        st.subheader("Top Parties by Expenditure")
        party_sum = db.groupby('Party Name')['Amount'].sum().sort_values(ascending=False).reset_index().head(10)
        st.plotly_chart(px.bar(party_sum, x='Party Name', y='Amount', color='Amount'), use_container_width=True)

    with t2:
        st.subheader("Selection Engine")
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            q_filter = st.multiselect("Target Quarters", db['Quarter'].unique(), default=db['Quarter'].unique())
        
        if st.button("🚀 Execute Strategic Sampling"):
            # Stratified Logic
            working_set = db[db['Quarter'].isin(q_filter)]
            key_items = working_set[working_set['Risk Flags'] != "Routine"].copy()
            key_items['Basis'] = "Risk-Based Selection"
            
            remain = working_set[working_set['Risk Flags'] == "Routine"]
            sample_n = int(len(remain) * (sample_rate / 100))
            stat_samples = remain.sample(n=min(len(remain), sample_n)).copy()
            stat_samples['Basis'] = f"Statistical Sample ({sample_rate}%)"
            
            st.session_state.samples = pd.concat([key_items, stat_samples])

        if st.session_state.samples is not None:
            st.write(f"### Digital Workpaper ({len(st.session_state.samples)} Samples)")
            
            edited_samples = st.data_editor(
                st.session_state.samples,
                column_config={
                    "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Verified", "Pending", "Query", "Error Found"]),
                    "Error Amount": st.column_config.NumberColumn("Actual Misstatement", help="Enter 0 if no error"),
                    "Risk Flags": st.column_config.TextColumn("Risk Detected", width="medium"),
                },
                disabled=["Invoice Date", "Party Name", "Amount", "Quarter", "Basis", "Risk Flags"],
                hide_index=True, use_container_width=True
            )
            st.session_state.samples = edited_samples

    with t3:
        if st.session_state.samples is not None:
            st.header("🧮 Materiality Projection & Conclusion")
            
            # Math
            pop_total = db['Amount'].sum()
            sample_total = st.session_state.samples['Amount'].sum()
            actual_error = st.session_state.samples['Error Amount'].sum()
            
            projected_error = (actual_error / sample_total) * pop_total if sample_total > 0 else 0
            
            r1, r2 = st.columns(2)
            r1.metric("Projected Total Misstatement", f"₹{projected_error:,.2f}")
            status = "FAIL" if projected_error > materiality else "PASS"
            r2.metric("Audit Conclusion", status, delta="Exceeds Materiality" if status == "FAIL" else "Safe", delta_color="inverse")
            
            # Final Chart
            report_fig = go.Figure()
            report_fig.add_trace(go.Bar(name='Materiality', x=['Threshold'], y=[materiality], marker_color='black'))
            report_fig.add_trace(go.Bar(name='Projected Error', x=['Threshold'], y=[projected_error], marker_color='red'))
            st.plotly_chart(report_fig, use_container_width=True)
            
            # Export
            st.download_button("💾 Export Verified Audit Working Paper", edited_samples.to_csv(index=False), "Audit_Report_Final.csv")

else:
    st.info("Please upload a file to begin the Audit Suite.")
    st.image("https://cdn.pixabay.com/photo/2016/10/11/09/26/office-1730939_1280.jpg")
