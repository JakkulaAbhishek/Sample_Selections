import streamlit as st
import pandas as pd
import plotly.express as px
import holidays
import io
import numpy as np

# --- 1. CONFIGURATION & COMPLIANCE MASTER (FY 2025-26) ---
TDS_MASTER = {
    "192": {"name": "Salary", "rate": 0.0, "limit": 250000},
    "194A": {"name": "Interest", "rate": 0.10, "limit": 5000},
    "194C": {"name": "Contractors", "rate": 0.02, "limit": 30000},
    "194H": {"name": "Commission", "rate": 0.05, "limit": 15000},
    "194I": {"name": "Rent (P&M)", "rate": 0.02, "limit": 240000},
    "194J": {"name": "Professional", "rate": 0.10, "limit": 30000},
    "194Q": {"name": "Purchase of Goods", "rate": 0.001, "limit": 5000000}
}

ind_holidays = holidays.India(years=[2025, 2026])

st.set_page_config(page_title="Audit Master Suite Pro", layout="wide")

if 'db' not in st.session_state: st.session_state.db = None
if 'audit_samples' not in st.session_state: st.session_state.audit_samples = None

# --- 2. DATA CLEANING & TDS ENGINE ---
def process_ledger(df, materiality):
    # Fix for TypeError: Force numeric conversion and remove commas
    numeric_cols = ['Amount', 'CGST', 'SGST', 'IGST', 'Total Value']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Invoice Date'])

    # TDS Calculation Logic
    def calculate_tds(row):
        sec = str(row.get('TDS Section', ''))
        if sec in TDS_MASTER:
            return float(row['Amount']) * TDS_MASTER[sec]['rate']
        return 0.0

    def get_flags(row):
        f = []
        if row['Invoice Date'].weekday() == 6: f.append("Sunday")
        if row['Invoice Date'] in ind_holidays: f.append("Holiday")
        if float(row['Amount']) >= materiality: f.append("Material Item")
        return " | ".join(f) if f else "Routine"

    df['TDS Deducted Amount'] = df.apply(calculate_tds, axis=1)
    df['Risk Flags'] = df.apply(get_flags, axis=1)
    df['Vouching Status'] = "Pending"
    return df

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Audit Control Center")
    materiality = st.number_input("Performance Materiality", value=100000)
    uploaded_file = st.file_uploader("Upload Ledger", type=['csv', 'xlsx'])
    
    # Mandatory Template for the user
    tpl_buf = io.BytesIO()
    pd.DataFrame({
        'Invoice Date': ['31-03-2026'], 'Invoice No': ['V-001'], 'Party Name': ['Reliance Ind'],
        'TDS Section': ['194Q'], 'Amount': [6000000], 'CGST': [0], 'SGST': [0],
        'IGST': [1080000], 'Total Value': [7080000]
    }).to_csv(tpl_buf, index=False)
    st.download_button("Download Mandatory Template", tpl_buf.getvalue(), "audit_template.csv")

# --- 4. MAIN APPLICATION ---
if uploaded_file:
    if st.session_state.db is None or st.sidebar.button("Reload Ledger"):
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        st.session_state.db = process_ledger(raw, materiality)

    db = st.session_state.db

    tab1, tab2, tab3 = st.tabs(["📉 Analytics Dashboard", "🎯 Sampling Engine", "📋 Final Workpaper"])

    with tab1:
        st.subheader("Section-wise TDS Applicability & Risk")
        
        # Dashboard Charts
        party_sec = db.groupby(['Party Name', 'TDS Section'])['Amount'].sum().reset_index()
        st.plotly_chart(px.bar(party_sec, x='Party Name', y='Amount', color='TDS Section', title="Party-wise Section Exposure"), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if 'TDS Deducted Amount' in db.columns:
                st.plotly_chart(px.pie(db, values='TDS Deducted Amount', names='TDS Section', title="TDS Value Distribution"), use_container_width=True)
        with c2:
            st.plotly_chart(px.sunburst(db, path=['TDS Section', 'Risk Flags'], values='Amount', title="Risk Strategy Matrix"), use_container_width=True)

    with tab2:
        st.header("Select Audit Methodology")
        methods = st.multiselect("Sampling Methods", ["Simple Random", "Systematic", "Stratified", "MUS", "PPS", "Haphazard"])
        coverage = st.slider("Cumulative Selection Rate (%)", 5, 100, 25)

        if st.button("🚀 Run Sampling Engine"):
            results = db[db['Risk Flags'].str.contains("Material")].copy()
            results['Selection Basis'] = "Key Item (100% Test)"
            
            pool = db[~db.index.isin(results.index)]
            needed = int(len(db) * (coverage / 100))
            
            if methods:
                per_m = max(1, needed // len(methods))
                for m in methods:
                    if len(pool) > 0:
                        s = pool.sample(n=min(len(pool), per_m))
                        s['Selection Basis'] = m
                        results = pd.concat([results, s])
                        pool = pool[~pool.index.isin(s.index)]
            
            st.session_state.audit_samples = results.drop_duplicates()
            st.success(f"Generated {len(st.session_state.audit_samples)} Audit Samples.")

    with tab3:
        if st.session_state.audit_samples is not None:
            st.subheader("Interactive Audit Workpaper")
            edited = st.data_editor(
                st.session_state.audit_samples,
                column_config={
                    "TDS Deducted Amount": st.column_config.NumberColumn("Estimated TDS", format="₹%.2f"),
                    "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Verified", "Pending", "Query", "Exception"]),
                },
                disabled=["Invoice Date", "Amount", "TDS Section", "Party Name", "Selection Basis", "TDS Deducted Amount"],
                hide_index=True, use_container_width=True
            )
            st.session_state.audit_samples = edited

            # MULTI-SHEET EXCEL EXPORT
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Sheet 1: Samples with TDS Sections and Deducted Amounts
                edited.to_excel(writer, sheet_name='Audit_Samples', index=False)
                # Sheet 2: Raw Data (Full Uploaded Ledger)
                db.to_excel(writer, sheet_name='Full_Raw_Ledger', index=False)
                # Sheet 3: Party-wise TDS Applicability Summary
                tds_party = db.groupby(['Party Name', 'TDS Section']).agg({'Amount': 'sum', 'TDS Deducted Amount': 'sum'}).reset_index()
                tds_party.to_excel(writer, sheet_name='TDS_Applicability_Partywise', index=False)

            st.download_button(
                label="💾 Export Professional Audit Report (Excel)",
                data=output.getvalue(),
                file_name="Audit_Report_FY26.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Upload your transaction ledger to begin the audit.")
