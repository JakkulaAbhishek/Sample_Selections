import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import holidays
import io
import numpy as np

# --- 1. CONFIGURATION & TAX MASTERS ---
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

# --- 2. DATA UTILITIES ---
def get_fiscal_quarter(date):
    if pd.isna(date): return "Invalid Date"
    month = date.month
    if month in [4, 5, 6]: return "Q1 (Apr-Jun)"
    if month in [7, 8, 9]: return "Q2 (Jul-Sep)"
    if month in [10, 11, 12]: return "Q3 (Oct-Dec)"
    return "Q4 (Jan-Mar)"

def get_audit_flags(row, materiality):
    reasons = []
    amt = float(row.get('Amount', 0))
    dt = pd.to_datetime(row.get('Invoice Date'), errors='coerce')
    if not pd.isna(dt):
        if dt.weekday() == 6: reasons.append("Sunday Posting")
        if dt in ind_holidays: reasons.append("Holiday")
    if amt >= materiality: reasons.append("Above Materiality")
    if amt > 0 and amt % 1000 == 0: reasons.append("Round Sum Case")
    sec = str(row.get('TDS Section', ''))
    if sec in TDS_MASTER and amt >= TDS_MASTER[sec]['limit']:
        reasons.append(f"TDS Threshold {sec}")
    return " | ".join(reasons) if reasons else "Routine"

# --- 3. UI SETUP ---
st.set_page_config(page_title="Audit Analytics Suite", layout="wide")

if 'audit_data' not in st.session_state:
    st.session_state.audit_data = None
if 'final_samples' not in st.session_state:
    st.session_state.final_samples = None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Audit Control Panel")
    materiality = st.number_input("Materiality Limit (INR)", value=100000, step=10000)
    sample_pct = st.slider("Sampling Rate (%)", 5, 100, 25)
    uploaded_file = st.file_uploader("Upload Ledger", type=['csv', 'xlsx'])
    
    if st.button("🔄 Reset Workpaper"):
        st.session_state.audit_data = None
        st.session_state.final_samples = None
        st.rerun()

# --- 5. DATA ENGINE ---
if uploaded_file:
    if st.session_state.audit_data is None or st.session_state.get('last_file') != uploaded_file.name:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        
        # Cleaning
        df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        df = df.dropna(subset=['Invoice Date'])
        
        # Audit Context
        df['Quarter'] = df['Invoice Date'].apply(get_fiscal_quarter)
        df['Audit Flags'] = df.apply(lambda x: get_audit_flags(x, materiality), axis=1)
        
        # GST Columns
        for c in ['CGST', 'SGST', 'IGST']:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        df['Vouching Status'] = "Pending"
        df['Error Amount'] = 0.0  # For Materiality Projection
        df['Auditor Remarks'] = ""
        
        st.session_state.audit_data = df
        st.session_state.last_file = uploaded_file.name

    df = st.session_state.audit_data

    # --- 6. DASHBOARD & TRENDS ---
    st.title("📊 Executive Audit Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ledger Count", f"{len(df):,}")
    col2.metric("Total Value", f"₹{df['Amount'].sum():,.0f}")
    col3.metric("Materiality", f"₹{materiality:,.0f}")
    
    tab1, tab2, tab3 = st.tabs(["📈 Analysis", "📝 Vouching", "🧮 Materiality Projection"])
    
    with tab1:
        st.plotly_chart(px.bar(df.groupby('Quarter')['Amount'].sum().reset_index(), x='Quarter', y='Amount', title="Quarterly Exposure"), use_container_width=True)
        
    with tab2:
        st.write("### Smart Sampling Engine")
        party_totals = df.groupby('Party Name')['Amount'].transform('sum')
        df['Party Scale'] = np.where(party_totals > (materiality * 5), "Large Party", "Routine")
        
        if st.button("🚀 Generate Stratified Workpaper"):
            key_items = df[(df['Audit Flags'] != "Routine") | (df['Party Scale'] == "Large Party")].copy()
            key_items['Selection Reason'] = "Key Item (Risk/Volume)"
            
            remaining = df[~df.index.isin(key_items.index)]
            sample_count = int(len(remaining) * (sample_pct / 100))
            random_samples = remaining.sample(n=min(len(remaining), sample_count)).copy()
            random_samples['Selection Reason'] = f"Statistical Sample ({sample_pct}%)"
            
            st.session_state.final_samples = pd.concat([key_items, random_samples])

        if st.session_state.final_samples is not None:
            edited_samples = st.data_editor(
                st.session_state.final_samples,
                column_config={
                    "Vouching Status": st.column_config.SelectboxColumn("Status", options=["Pending", "Verified", "Query", "Error Found"]),
                    "Error Amount": st.column_config.NumberColumn("Misstatement Amt", help="Enter the actual error amount found in this invoice"),
                },
                disabled=["Invoice Date", "Party Name", "Amount", "Quarter", "Audit Flags", "Selection Reason", "Party Scale"],
                hide_index=True, use_container_width=True
            )
            st.session_state.final_samples = edited_samples
            st.download_button("💾 Export Workpaper", edited_samples.to_csv(index=False), "Audit_Workpaper.csv")

    with tab3:
        if st.session_state.final_samples is not None:
            st.header("🧮 Error Projection (SA 530)")
            
            # Calculations
            total_population_val = df['Amount'].sum()
            sample_val = st.session_state.final_samples['Amount'].sum()
            errors_found = st.session_state.final_samples['Error Amount'].sum()
            
            # Projection Formula: (Error in Sample / Sample Value) * Total Population Value
            error_rate = (errors_found / sample_val) if sample_val > 0 else 0
            projected_error = error_rate * total_population_val
            
            c1, c2 = st.columns(2)
            c1.metric("Actual Errors Found", f"₹{errors_found:,.2f}")
            c2.metric("Projected Total Error", f"₹{projected_error:,.2f}", 
                      delta="EXCEEDS MATERIALITY" if projected_error > materiality else "Within Limits",
                      delta_color="inverse")
            
            # Projection Chart
            proj_df = pd.DataFrame({
                'Category': ['Materiality Threshold', 'Projected Misstatement', 'Actual Error Found'],
                'Value': [materiality, projected_error, errors_found]
            })
            fig_proj = px.bar(proj_df, x='Category', y='Value', color='Category', 
                             color_discrete_map={'Materiality Threshold': '#333', 'Projected Misstatement': 'red', 'Actual Error Found': 'orange'})
            st.plotly_chart(fig_proj, use_container_width=True)
            
            if projected_error > materiality:
                st.error("⚠️ PROJECTION ALERT: The projected misstatement exceeds your materiality threshold. You may need to increase your sample size or perform additional procedures.")
            else:
                st.success("✅ PROJECTION SAFE: The projected misstatement is below the materiality threshold.")
