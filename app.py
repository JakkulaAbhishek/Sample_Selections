import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import holidays
import io
import numpy as np

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="CA Audit Intelligence Suite", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f8f9fa; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MASTER DATA ---
TDS_MASTER = {
    "192": "Salary", "194A": "Interest", "194C": "Contractors", "194H": "Commission",
    "194I": "Rent", "194J": "Professional", "194Q": "Goods Purchase", "194R": "Perquisites"
}
ind_holidays = holidays.India(years=[2025, 2026])

# --- 3. SESSION STATE ---
if 'raw_data' not in st.session_state: st.session_state.raw_data = None
if 'audit_samples' not in st.session_state: st.session_state.audit_samples = None

# --- 4. CORE UTILITIES ---
def get_fiscal_quarter(date):
    if pd.isna(date): return "N/A"
    m = date.month
    return f"Q{((m-4)%12)//3 + 1} (FY25-26)"

def risk_engine(row, materiality):
    flags = []
    amt = float(row.get('Amount', 0))
    dt = pd.to_datetime(row.get('Invoice Date'), errors='coerce')
    if not pd.isna(dt):
        if dt.weekday() == 6: flags.append("Sunday")
        if dt in ind_holidays: flags.append("Holiday")
    if amt >= materiality: flags.append("Key Item")
    if amt > 0 and amt % 1000 == 0: flags.append("Round Sum")
    return " | ".join(flags) if flags else "Routine"

# --- 5. SIDEBAR: DATA INPUT & TEMPLATE ---
with st.sidebar:
    st.title("🛡️ Audit Master")
    materiality = st.number_input("Performance Materiality", value=100000)
    
    st.subheader("📥 1. Prepare Data")
    # Generate Professional Template
    tpl_buf = io.BytesIO()
    pd.DataFrame({
        'Invoice Date': ['2026-03-31', '2026-01-26'], 'Invoice No': ['INV01', 'INV02'],
        'Party Name': ['Client A', 'Vendor B'], 'TDS Section': ['194J', '194C'],
        'Amount': [100000, 45000], 'CGST': [9000, 4050], 'SGST': [9000, 4050],
        'IGST': [0, 0], 'Total Value': [118000, 53100]
    }).to_csv(tpl_buf, index=False)
    st.download_button("Download Mandatory Template", tpl_buf.getvalue(), "audit_template.csv")
    
    uploaded_file = st.file_uploader("Upload Ledger", type=['csv', 'xlsx'])

# --- 6. DATA ENGINE ---
if uploaded_file:
    if st.session_state.raw_data is None or st.sidebar.button("Refresh Ledger"):
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        # Force Numeric & Cleaning
        cols = ['Amount', 'CGST', 'SGST', 'IGST', 'Total Value']
        for c in cols: df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        df = df.dropna(subset=['Invoice Date'])
        # Intelligence
        df['Quarter'] = df['Invoice Date'].apply(get_fiscal_quarter)
        df['Risk Flags'] = df.apply(lambda x: risk_engine(x, materiality), axis=1)
        df['Vouching Status'] = "Pending"
        st.session_state.raw_data = df

    db = st.session_state.raw_data

    # --- 7. DASHBOARD ---
    st.title("📊 Ledger Intelligence & Risk Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Population", f"{len(db):,}")
    m2.metric("Total Book Value", f"₹{db['Amount'].sum():,.0f}")
    m3.metric("High Risk Items", len(db[db['Risk Flags'] != "Routine"]))
    m4.metric("Active TDS Sections", db['TDS Section'].nunique())

    tab1, tab2, tab3 = st.tabs(["📉 Population Analysis", "🎯 Sampling Console", "📋 Digital Workpaper"])

    with tab1:
        st.subheader("Full Ledger Analytics")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(db, x='Amount', color='Quarter', nbins=50, title="Transaction Size Distribution"), use_container_width=True)
        with c2:
            st.plotly_chart(px.sunburst(db, path=['Quarter', 'TDS Section'], values='Amount', title="Expenditure Breakdown"), use_container_width=True)
        
        st.subheader("Top Parties by Value")
        st.plotly_chart(px.bar(db.groupby('Party Name')['Amount'].sum().nlargest(10).reset_index(), x='Party Name', y='Amount', color='Amount'), use_container_width=True)

    with tab2:
        st.subheader("Smart Selection Configuration")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            method = st.selectbox("Sampling Method", ["Stratified (Risk-Based)", "Random Statistical", "Monetary Unit Sampling (MUS)"])
        with col_s2:
            target_pct = st.slider("Population Coverage %", 5, 100, 25)
        with col_s3:
            include_risk = st.checkbox("Prioritize High-Risk Flags", value=True)

        if st.button("🚀 Execute Audit Sampling"):
            # Logic: Always separate Key Items (Materiality)
            key_items = db[db['Risk Flags'].str.contains("Key Item")].copy()
            key_items['Selection Logic'] = "Key Item (100% Testing)"
            
            remaining = db[~db.index.isin(key_items.index)]
            
            if method == "Stratified (Risk-Based)":
                risk_items = remaining[remaining['Risk Flags'] != "Routine"].copy()
                risk_items['Selection Logic'] = "Stratified (Risk Flag)"
                others = remaining[remaining['Risk Flags'] == "Routine"]
                size = int(len(others) * (target_pct/100))
                randoms = others.sample(n=min(len(others), size)).copy()
                randoms['Selection Logic'] = f"Statistical Random ({target_pct}%)"
                st.session_state.audit_samples = pd.concat([key_items, risk_items, randoms])
            
            elif method == "Random Statistical":
                size = int(len(remaining) * (target_pct/100))
                randoms = remaining.sample(n=min(len(remaining), size)).copy()
                randoms['Selection Logic'] = "Random Statistical Selection"
                st.session_state.audit_samples = pd.concat([key_items, randoms])

            elif method == "Monetary Unit Sampling (MUS)":
                # Simplified MUS logic for CA usage
                remaining['CumSum'] = remaining['Amount'].cumsum()
                interval = remaining['Amount'].sum() / (len(remaining) * (target_pct/100))
                selection = np.arange(0, remaining['Amount'].sum(), interval)
                mus_samples = remaining[remaining['CumSum'].searchsorted(selection, side='right') < len(remaining)].copy()
                mus_samples['Selection Logic'] = "Monetary Unit Sampling (MUS)"
                st.session_state.audit_samples = pd.concat([key_items, mus_samples])
            
            st.success(f"Audit Samples Generated: {len(st.session_state.audit_samples)} items selected.")

    with tab3:
        if st.session_state.audit_samples is not None:
            st.subheader("Digital Audit Working Paper")
            # This incorporates the Raw Data + Selected Samples
            edited = st.data_editor(
                st.session_state.audit_samples,
                column_config={
                    "Selection Logic": st.column_config.TextColumn("Why Selected?", width="medium"),
                    "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Verified", "Pending", "Query", "Exception"]),
                    "Risk Flags": st.column_config.TextColumn("Risk Alerts", width="medium"),
                },
                disabled=["Invoice Date", "Party Name", "Amount", "Quarter", "Selection Logic", "Risk Flags"],
                hide_index=True, use_container_width=True
            )
            st.session_state.audit_samples = edited
            
            st.divider()
            # Final Charts for the output file
            st.write("### Sample Representation Analysis")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.plotly_chart(px.pie(edited, names='Selection Logic', title="Selection Methodology Breakdown"), use_container_width=True)
            with sc2:
                st.plotly_chart(px.bar(edited.groupby('TDS Section')['Amount'].count().reset_index(), x='TDS Section', y='Amount', title="Sample Count per TDS Section"), use_container_width=True)

            st.download_button("💾 Export Verified Audit File", edited.to_csv(index=False), "Audit_Working_Paper.csv")
else:
    st.info("Upload your Excel/CSV ledger to unlock the Audit Intelligence suite.")
