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
    "194I": "Rent", "194J": "Professional", "194Q": "Goods Purchase"
}
ind_holidays = holidays.India(years=[2025, 2026])

st.set_page_config(page_title="Audit Master Suite", layout="wide")

# --- 2. SESSION STATE ---
if 'raw_data' not in st.session_state: st.session_state.raw_data = None
if 'audit_samples' not in st.session_state: st.session_state.audit_samples = None

# --- 3. CORE LOGIC ---
def clean_data(df):
    cols = ['Amount', 'CGST', 'SGST', 'IGST', 'Total Value']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    return df.dropna(subset=['Invoice Date'])

def get_risk_flags(row, materiality):
    flags = []
    amt = float(row.get('Amount', 0))
    dt = row.get('Invoice Date')
    if not pd.isna(dt):
        if dt.weekday() == 6: flags.append("Sunday")
        if dt in ind_holidays: flags.append("Holiday")
    if amt >= materiality: flags.append("Key Item")
    return " | ".join(flags) if flags else "Routine"

# --- 4. SIDEBAR & TEMPLATE ---
with st.sidebar:
    st.header("🏢 Audit Configuration")
    materiality = st.number_input("Performance Materiality", value=100000)
    
    st.subheader("🛠️ Template Setup")
    tpl_buf = io.BytesIO()
    pd.DataFrame({
        'Invoice Date': ['2026-03-31'], 'Invoice No': ['V-001'], 'Party Name': ['Client X'],
        'TDS Section': ['194J'], 'Amount': [150000], 'CGST': [13500], 'SGST': [13500],
        'IGST': [0], 'Total Value': [177000]
    }).to_csv(tpl_buf, index=False)
    st.download_button("Download Audit Template", tpl_buf.getvalue(), "audit_template.csv")
    
    uploaded_file = st.file_uploader("Upload Client Ledger", type=['csv', 'xlsx'])

# --- 5. SAMPLING ENGINE ---
def apply_sampling(df, methods, target_pct):
    results = []
    sample_size = int(len(df) * (target_pct / 100))
    
    # Always include Key Items (Materiality)
    key_items = df[df['Risk Flags'].str.contains("Key Item")].copy()
    key_items['Selection Method'] = "Key Item (100% Testing)"
    results.append(key_items)
    
    pool = df[~df.index.isin(key_items.index)]
    
    for method in methods:
        current_sample = pd.DataFrame()
        
        # PROBABILITY METHODS
        if method == "Simple Random Sampling":
            current_sample = pool.sample(n=min(len(pool), sample_size // len(methods)))
        elif method == "Systematic Sampling":
            k = max(1, len(pool) // (sample_size // len(methods)))
            current_sample = pool.iloc[::k]
        elif method == "Stratified Sampling":
            current_sample = pool.groupby('TDS Section', group_keys=False).apply(lambda x: x.sample(frac=target_pct/100))
        elif method == "Probability Proportional to Size (PPS)":
            probs = pool['Amount'] / pool['Amount'].sum()
            indices = np.random.choice(pool.index, size=min(len(pool), sample_size // len(methods)), p=probs, replace=False)
            current_sample = pool.loc[indices]
            
        # NON-PROBABILITY METHODS
        elif method == "Convenience Sampling":
            current_sample = pool.head(sample_size // len(methods))
        elif method == "Haphazard Sampling":
            current_sample = pool.sample(n=min(len(pool), sample_size // len(methods)), random_state=random.randint(1,100))
        
        # AUDIT SPECIFIC
        elif method == "Monetary Unit Sampling (MUS)":
            interval = pool['Amount'].sum() / (sample_size // len(methods))
            starts = np.arange(0, pool['Amount'].sum(), interval)
            pool_cum = pool.copy()
            pool_cum['cum_amt'] = pool_cum['Amount'].cumsum()
            current_sample = pool_cum[pool_cum['cum_amt'].searchsorted(starts) < len(pool_cum)]
        elif method == "Block Sampling":
            current_sample = pool.iloc[:sample_size // len(methods)]

        if not current_sample.empty:
            current_sample['Selection Method'] = method
            results.append(current_sample)
            pool = pool[~pool.index.isin(current_sample.index)] # Avoid duplicates

    return pd.concat(results).drop_duplicates()

# --- 6. MAIN UI ---
if uploaded_file:
    if st.session_state.raw_data is None or st.sidebar.button("Reload Data"):
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        df_raw = clean_data(df_raw)
        df_raw['Risk Flags'] = df_raw.apply(lambda x: get_risk_flags(x, materiality), axis=1)
        st.session_state.raw_data = df_raw

    db = st.session_state.raw_data

    st.title("📊 Enterprise Audit Master Dashboard")
    
    # METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ledger Count", f"{len(db):,}")
    m2.metric("Total Value", f"₹{db['Amount'].sum():,.0f}")
    m3.metric("Material Risk Items", len(db[db['Risk Flags'] != "Routine"]))
    m4.metric("Coverage Value", f"₹{db[db['Risk Flags'] != 'Routine']['Amount'].sum():,.0f}")

    tab1, tab2, tab3 = st.tabs(["📉 Population Analytics", "🎯 Master Sampling Console", "📋 Audit Workpaper"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(db, x='Amount', color='TDS Section', title="Transaction Stratification"), use_container_width=True)
        with c2:
            st.plotly_chart(px.sunburst(db, path=['TDS Section', 'Risk Flags'], values='Amount', title="Risk/Section Sunburst"), use_container_width=True)

    with tab2:
        st.header("Select Audit Sampling Methods")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            prob_methods = st.multiselect("🔹 Probability Methods", 
                ["Simple Random Sampling", "Systematic Sampling", "Stratified Sampling", "Cluster Sampling", "Probability Proportional to Size (PPS)"])
            non_prob = st.multiselect("🔹 Non-Probability Methods", 
                ["Convenience Sampling", "Judgmental Sampling", "Haphazard Sampling", "Consecutive Sampling"])
        with col_m2:
            audit_spec = st.multiselect("🔹 Audit-Specific Methods", 
                ["Statistical Sampling", "Non-Statistical Sampling", "Monetary Unit Sampling (MUS)", "Block Sampling"])
            advanced = st.multiselect("🔹 Advanced/Special Methods", 
                ["Sequential Sampling", "Adaptive Sampling", "Reservoir Sampling", "Acceptance Sampling", "Bootstrap Sampling"])
        
        target_pct = st.slider("Cumulative Selection Rate (%)", 5, 100, 25)
        
        all_selected = prob_methods + non_prob + audit_spec + advanced
        
        if st.button("🚀 Run Multi-Method Sampling Engine"):
            if not all_selected:
                st.warning("Please select at least one method.")
            else:
                st.session_state.audit_samples = apply_sampling(db, all_selected, target_pct)
                st.success(f"Generated {len(st.session_state.audit_samples)} samples using {len(all_selected)} methods.")

    with tab3:
        if st.session_state.audit_samples is not None:
            edited = st.data_editor(
                st.session_state.audit_samples,
                column_config={
                    "Selection Method": st.column_config.TextColumn("Methodology Justification", width="medium"),
                    "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Verified", "Pending", "Query", "Exception"]),
                },
                disabled=["Invoice Date", "Party Name", "Amount", "Selection Method", "Risk Flags"],
                hide_index=True, use_container_width=True
            )
            st.session_state.audit_samples = edited
            
            # Analytics on Samples
            st.divider()
            sc1, sc2 = st.columns(2)
            with sc1:
                st.plotly_chart(px.pie(edited, names='Selection Method', title="Methodology Distribution"), use_container_width=True)
            with sc2:
                st.plotly_chart(px.bar(edited.groupby('Selection Method')['Amount'].sum().reset_index(), x='Selection Method', y='Amount', title="Value Coverage by Method"), use_container_width=True)

            st.download_button("💾 Export Verified Master Workpaper", edited.to_csv(index=False), "Master_Audit_Report.csv")

else:
    st.info("Please upload your audit ledger to unlock the Master Sampling Suite.")
