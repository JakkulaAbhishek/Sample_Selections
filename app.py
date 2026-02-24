import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import holidays
import io
import numpy as np
import random

# --- 1. GLOBAL CONFIGURATION ---
TDS_MASTER = {
    "192": "Salary", "194A": "Interest", "194C": "Contractors", "194H": "Commission",
    "194I": "Rent", "194J": "Professional", "194Q": "Goods Purchase", "194R": "Perquisites"
}
ind_holidays = holidays.India(years=[2025, 2026])

st.set_page_config(page_title="Audit Master Suite", layout="wide")

# --- 2. SESSION STATE (The "Lock" for your data) ---
if 'db' not in st.session_state: st.session_state.db = None
if 'audit_samples' not in st.session_state: st.session_state.audit_samples = None

# --- 3. DATA CLEANING ENGINE ---
def clean_and_enrich(df, materiality):
    # Clean Numbers
    cols = ['Amount', 'CGST', 'SGST', 'IGST', 'Total Value']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # Clean Dates
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    df = df.dropna(subset=['Invoice Date'])
    
    # Risk Flags
    def get_flags(row):
        f = []
        if row['Invoice Date'].weekday() == 6: f.append("Sunday")
        if row['Invoice Date'] in ind_holidays: f.append("Holiday")
        if row['Amount'] >= materiality: f.append("Key Item")
        return " | ".join(f) if f else "Routine"

    df['Risk Flags'] = df.apply(get_flags, axis=1)
    df['Vouching Status'] = "Pending"
    return df

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Audit Master")
    materiality = st.number_input("Performance Materiality", value=100000)
    
    # Mandatory Template
    tpl_buf = io.BytesIO()
    pd.DataFrame({
        'Invoice Date': ['2026-03-31'], 'Invoice No': ['V-001'], 'Party Name': ['Client X'],
        'TDS Section': ['194J'], 'Amount': [150000], 'CGST': [13500], 'SGST': [13500],
        'IGST': [0], 'Total Value': [177000]
    }).to_csv(tpl_buf, index=False)
    st.download_button("📥 Download Mandatory Template", tpl_buf.getvalue(), "audit_template.csv")
    
    uploaded_file = st.file_uploader("Upload Ledger", type=['csv', 'xlsx'])

# --- 5. MAIN LOGIC ---
if uploaded_file:
    if st.session_state.db is None or st.sidebar.button("Reload Uploaded Data"):
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        st.session_state.db = clean_and_enrich(raw, materiality)

    db = st.session_state.db

    # --- DASHBOARD ---
    st.title("📊 Audit Intelligence Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ledger Count", f"{len(db):,}")
    m2.metric("Total Value", f"₹{db['Amount'].sum():,.0f}")
    m3.metric("Material Risks", len(db[db['Risk Flags'] != "Routine"]))
    m4.metric("Avg. Tran Size", f"₹{db['Amount'].mean():,.0f}")

    tab1, tab2, tab3 = st.tabs(["📉 Population Analytics", "🎯 Master Sampling Console", "📋 Digital Workpaper"])

    with tab1:
        st.subheader("TDS Applicability by Party-wise")
        party_tds = db.groupby(['Party Name', 'TDS Section'])['Amount'].sum().reset_index()
        fig_party = px.bar(party_tds, x='Party Name', y='Amount', color='TDS Section', title="Party-wise Section Exposure")
        st.plotly_chart(fig_party, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Section-wise Distribution**")
            st.plotly_chart(px.pie(db, values='Amount', names='TDS Section', hole=0.4))
        with c2:
            st.write("**Risk profile by Section**")
            st.plotly_chart(px.sunburst(db, path=['TDS Section', 'Risk Flags'], values='Amount'))

    with tab2:
        st.header("Select Audit Methodology")
        methods = st.multiselect("Sampling Methods", [
            "Simple Random", "Systematic", "Stratified", "Monetary Unit Sampling (MUS)", 
            "Haphazard", "Judgmental", "Block Sampling", "PPS Sampling"
        ])
        target_pct = st.slider("Target Coverage (%)", 5, 100, 25)

        if st.button("🚀 Run Sampling Engine"):
            # 1. Start with Key Items
            results = db[db['Risk Flags'].str.contains("Key Item")].copy()
            results['Selection Method'] = "Key Item (100% Testing)"
            
            pool = db[~db.index.isin(results.index)]
            sample_size = int(len(db) * (target_pct / 100))
            
            if methods:
                per_method = sample_size // len(methods)
                for m in methods:
                    if m == "Simple Random":
                        s = pool.sample(n=min(len(pool), per_method))
                    elif m == "Monetary Unit Sampling (MUS)":
                        s = pool.nlargest(per_method, 'Amount') # Simplified MUS
                    else:
                        s = pool.sample(n=min(len(pool), per_method))
                    
                    s['Selection Method'] = m
                    results = pd.concat([results, s])
                    pool = pool[~pool.index.isin(s.index)]
            
            st.session_state.audit_samples = results.drop_duplicates()
            st.success(f"Successfully selected {len(st.session_state.audit_samples)} samples.")

    with tab3:
        if st.session_state.audit_samples is not None:
            st.subheader("Digital Audit Working Paper")
            edited = st.data_editor(
                st.session_state.audit_samples,
                column_config={
                    "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Verified", "Pending", "Query", "Exception"]),
                    "Selection Method": st.column_config.TextColumn("Selection Justification", width="medium"),
                },
                disabled=["Invoice Date", "Amount", "TDS Section", "Party Name", "Risk Flags", "Selection Method"],
                hide_index=True, use_container_width=True
            )
            st.session_state.audit_samples = edited
            
            # TDS SECTION BREAKDOWN IN OUTPUT PREVIEW
            st.write("### Breakdown of TDS Sections in Sample")
            section_summary = edited.groupby('TDS Section').agg({'Amount': 'sum', 'Invoice No': 'count'}).reset_index()
            st.table(section_summary)

            csv = edited.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Export Verified Master Workpaper", csv, "Master_Audit_Report.csv", "text/csv")
else:
    st.info("Please upload your audit ledger to begin.")
