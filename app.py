import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import holidays
import io
import numpy as np

# --- 1. CONFIGURATION & COMPLIANCE MASTER ---
# Thresholds and Rates updated for FY 2025-26
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

# --- 2. SESSION STATE MANAGEMENT ---
if 'db' not in st.session_state: st.session_state.db = None
if 'audit_samples' not in st.session_state: st.session_state.audit_samples = None

# --- 3. CORE PROCESSING ENGINE ---
def process_ledger(df, materiality):
    # Numeric Normalization
    numeric_cols = ['Amount', 'CGST', 'SGST', 'IGST', 'Total Value']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    df = df.dropna(subset=['Invoice Date'])

    # TDS Calculation Logic
    def calculate_tds(row):
        sec = str(row.get('TDS Section', ''))
        if sec in TDS_MASTER:
            return row['Amount'] * TDS_MASTER[sec]['rate']
        return 0.0

    def get_flags(row):
        f = []
        if row['Invoice Date'].weekday() == 6: f.append("Sunday")
        if row['Invoice Date'] in ind_holidays: f.append("Holiday")
        if row['Amount'] >= materiality: f.append("Material Item")
        return " | ".join(f) if f else "Routine"

    df['TDS Deducted Amount'] = df.apply(calculate_tds, axis=1)
    df['Risk Flags'] = df.apply(get_flags, axis=1)
    df['Vouching Status'] = "Pending"
    return df

# --- 4. SIDEBAR & TEMPLATE ---
with st.sidebar:
    st.title("🛡️ Audit Control Center")
    materiality = st.number_input("Performance Materiality", value=100000)
    
    st.subheader("📥 Master Template")
    tpl_buf = io.BytesIO()
    pd.DataFrame({
        'Invoice Date': ['2026-03-31'], 'Invoice No': ['V-001'], 'Party Name': ['Reliance Ind'],
        'TDS Section': ['194Q'], 'Amount': [6000000], 'CGST': [0], 'SGST': [0],
        'IGST': [1080000], 'Total Value': [7080000]
    }).to_csv(tpl_buf, index=False)
    st.download_button("Download Mandatory Excel Template", tpl_buf.getvalue(), "audit_template.csv")
    
    uploaded_file = st.file_uploader("Upload Ledger", type=['csv', 'xlsx'])

# --- 5. MAIN APPLICATION UI ---
if uploaded_file:
    if st.session_state.db is None or st.sidebar.button("Reload Ledger Data"):
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        st.session_state.db = process_ledger(raw, materiality)

    db = st.session_state.db

    # --- DASHBOARD METRICS ---
    st.title("📊 Ledger Intelligence Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ledger Records", len(db))
    m2.metric("Total Book Value", f"₹{db['Amount'].sum():,.0f}")
    m3.metric("Total TDS Potential", f"₹{db['TDS Deducted Amount'].sum():,.0f}")
    m4.metric("High Risk Flags", len(db[db['Risk Flags'] != "Routine"]))

    tab1, tab2, tab3 = st.tabs(["📉 Population Analytics", "🎯 Sampling Console", "📋 Multi-Sheet Workpaper"])

    with tab1:
        st.subheader("TDS Applicability & Section Analysis")
        # Party-wise Section Breakdown
        party_sec = db.groupby(['Party Name', 'TDS Section'])['Amount'].sum().reset_index()
        fig_party = px.bar(party_sec, x='Party Name', y='Amount', color='TDS Section', title="Party-wise TDS Exposure")
        st.plotly_chart(fig_party, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(db, values='TDS Deducted Amount', names='TDS Section', title="TDS Value by Section", hole=0.4))
        with c2:
            st.plotly_chart(px.sunburst(db, path=['TDS Section', 'Risk Flags'], values='Amount', title="Section Risk Matrix"))

    with tab2:
        st.header("Strategic Sampling Logic")
        # List of all requested methods
        methods = st.multiselect("Select Sampling Methodology", [
            "Simple Random", "Systematic", "Stratified (TDS Section)", "Monetary Unit Sampling (MUS)", 
            "Haphazard", "Judgmental", "Block Sampling", "PPS Sampling"
        ])
        coverage = st.slider("Cumulative Selection Rate (%)", 5, 100, 25)

        if st.button("🚀 Execute Audit Engine"):
            # Start with Materiality items (Key Items)
            results = db[db['Risk Flags'].str.contains("Material")].copy()
            results['Selection Basis'] = "Key Item (100% Test)"
            
            pool = db[~db.index.isin(results.index)]
            needed = int(len(db) * (coverage / 100))
            
            if methods:
                per_m = max(1, needed // len(methods))
                for m in methods:
                    if m == "Simple Random": s = pool.sample(n=min(len(pool), per_m))
                    elif m == "Monetary Unit Sampling (MUS)": s = pool.nlargest(per_m, 'Amount')
                    elif m == "Stratified (TDS Section)": s = pool.groupby('TDS Section').apply(lambda x: x.sample(frac=0.1))
                    else: s = pool.sample(n=min(len(pool), per_m))
                    
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
                    "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Verified", "Pending", "Query", "Mismatched"]),
                    "Selection Basis": st.column_config.TextColumn("Methodology", width="medium"),
                },
                disabled=["Invoice Date", "Amount", "TDS Section", "Party Name", "Selection Basis"],
                hide_index=True, use_container_width=True
            )
            st.session_state.audit_samples = edited

            # MULTI-SHEET EXCEL EXPORT
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                edited.to_excel(writer, sheet_name='Audit_Samples', index=False)
                db.to_excel(writer, sheet_name='Full_Raw_Ledger', index=False)
                # Applicability Sheet
                app_sheet = db.groupby('TDS Section').agg({'Amount': 'sum', 'TDS Deducted Amount': 'sum', 'Invoice No': 'count'}).reset_index()
                app_sheet.to_excel(writer, sheet_name='TDS_Applicability_Summary', index=False)

            st.download_button(
                label="💾 Export Professional Audit Report (Excel)",
                data=output.getvalue(),
                file_name="Audit_Report_FY26.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Please upload your client's transaction ledger to unlock the dashboard.")
