import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from io import BytesIO
from datetime import datetime, timedelta
import calendar
import warnings
import base64
warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Ultra-Audit Pro | by Jakkula Abhishek",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS WITH ULTRA-STYLISH DESIGN ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Global Styles with Cyberpunk Theme */
    .main {
        background: linear-gradient(135deg, #0a0f1e 0%, #1a1f35 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Animated Gradient Header */
    .cyber-header {
        background: linear-gradient(270deg, #00ff87, #60efff, #0061ff, #ff00ff);
        background-size: 300% 300%;
        animation: gradientShift 10s ease infinite;
        padding: 2.5rem;
        border-radius: 30px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,255,135,0.3);
        border: 2px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px 0 rgba(0,255,135,0.3);
        border: 1px solid #00ff87;
    }
    
    /* Neon Text Effects */
    .neon-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        text-shadow: 0 0 10px #00ff87, 0 0 20px #00ff87, 0 0 40px #00ff87;
        color: white;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { text-shadow: 0 0 10px #00ff87, 0 0 20px #00ff87; }
        50% { text-shadow: 0 0 20px #00ff87, 0 0 40px #00ff87, 0 0 60px #00ff87; }
        100% { text-shadow: 0 0 10px #00ff87, 0 0 20px #00ff87; }
    }
    
    /* Developer Signature */
    .developer-signature {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00ff87, #60efff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: right;
        padding: 10px;
        border-right: 3px solid #00ff87;
        animation: slideIn 1s ease;
    }
    
    @keyframes slideIn {
        from { transform: translateX(100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Metric Cards with Glow */
    .metric-card-ultra {
        background: rgba(0, 255, 135, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid #00ff87;
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0,255,135,0.3);
    }
    
    .metric-card-ultra:hover {
        transform: scale(1.05);
        box-shadow: 0 0 40px rgba(0,255,135,0.6);
    }
    
    /* Cyberpunk Buttons */
    .cyber-button {
        background: transparent;
        border: 2px solid #00ff87;
        color: #00ff87;
        padding: 12px 30px;
        border-radius: 10px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .cyber-button:hover {
        background: #00ff87;
        color: #0a0f1e;
        box-shadow: 0 0 30px #00ff87;
    }
    
    .cyber-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .cyber-button:hover::before {
        left: 100%;
    }
    
    /* Party Cards */
    .party-card {
        background: linear-gradient(135deg, rgba(0,255,135,0.1), rgba(96,239,255,0.1));
        border: 1px solid #60efff;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s;
    }
    
    .party-card:hover {
        background: linear-gradient(135deg, rgba(0,255,135,0.3), rgba(96,239,255,0.3));
        transform: translateX(10px);
        border-color: #00ff87;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 5px;
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: white;
        font-family: 'Orbitron', sans-serif;
        border-radius: 10px;
        padding: 10px 25px;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: #00ff87 !important;
        color: #0a0f1e !important;
        font-weight: 700;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        background: #0a0f1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00ff87, #60efff);
        border-radius: 5px;
    }
    
    /* Data Table Styling */
    .dataframe {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 15px !important;
        border: 1px solid #00ff87 !important;
    }
    
    /* Animated Background */
    .animated-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 50% 50%, #1a1f35, #0a0f1e);
        z-index: -1;
        animation: backgroundPulse 10s infinite;
    }
    
    @keyframes backgroundPulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
</style>

<div class="animated-bg"></div>
""", unsafe_allow_html=True)

# --- 3. DEVELOPER ATTRIBUTION ---
st.markdown("""
<div class="developer-signature">
    ⚡ Developed by: JAKKULA ABHISHEK | 📧 jakkulaabhishek5@gmail.com ⚡
</div>
""", unsafe_allow_html=True)

# --- 4. CYBERPUNK HEADER ---
st.markdown("""
<div class="cyber-header">
    <h1 style="font-family: 'Orbitron', sans-serif; font-size: 4rem; margin:0; color: white; text-align: center;">
        ⚡ ULTRA-AUDIT PRO ⚡
    </h1>
    <p style="font-family: 'Orbitron', sans-serif; font-size: 1.2rem; text-align: center; color: rgba(255,255,255,0.9); margin-top: 10px;">
        Next-Gen AI-Powered Audit Intelligence | Materiality Analysis | TDS Compliance
    </p>
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 30px;">
        <span style="background: rgba(255,255,255,0.1); padding: 10px 25px; border-radius: 30px; border: 1px solid #00ff87;">
            🔥 CRITICAL
        </span>
        <span style="background: rgba(255,255,255,0.1); padding: 10px 25px; border-radius: 30px; border: 1px solid #60efff;">
            ⚡ HIGH
        </span>
        <span style="background: rgba(255,255,255,0.1); padding: 10px 25px; border-radius: 30px; border: 1px solid #0061ff;">
            💫 MEDIUM
        </span>
        <span style="background: rgba(255,255,255,0.1); padding: 10px 25px; border-radius: 30px; border: 1px solid #ff00ff;">
            🌟 LOW
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. DATA PROCESSING ENGINE WITH FORMULAS ---
class DataProcessor:
    """Advanced data processing with AI-powered column detection"""
    
    @staticmethod
    def clean_numeric(series):
        """Intelligent numeric cleaning"""
        if series.dtype == 'object':
            series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
            series = series.replace('', '0')
            series = series.replace('nan', '0')
            series = series.replace('None', '0')
        return pd.to_numeric(series, errors='coerce').fillna(0)
    
    @staticmethod
    def detect_columns(df):
        """AI-powered column detection"""
        column_patterns = {
            'date': ['date', 'transaction date', 'entry date', 'dt'],
            'party': ['party name', 'party', 'vendor', 'supplier', 'customer', 'name'],
            'invoice': ['invoice no', 'invoice', 'inv no', 'invoice number'],
            'gross': ['gross total', 'gross', 'total amount', 'bill amount'],
            'taxable': ['taxable value', 'taxable', 'taxable amount', 'value'],
            'cgst': ['input cgst', 'cgst', 'central gst'],
            'sgst': ['input sgst', 'sgst', 'state gst'],
            'igst': ['input igst', 'igst', 'integrated gst'],
            'tds': ['tds deducted', 'tds', 'tax deducted'],
            'tds_section': ['tds section', 'section', 'tds_sec']
        }
        
        mapped_columns = {}
        df_columns_lower = {col.lower().strip(): col for col in df.columns}
        
        for std_name, variations in column_patterns.items():
            for var in variations:
                if var in df_columns_lower:
                    mapped_columns[std_name] = df_columns_lower[var]
                    break
        
        return mapped_columns
    
    @staticmethod
    def apply_formulas(df):
        """Apply dynamic formulas to create calculated columns"""
        
        # Formula 1: Total GST = CGST + SGST + IGST
        if all(col in df.columns for col in ['Input CGST', 'Input SGST', 'Input IGST']):
            df['Total GST'] = df['Input CGST'] + df['Input SGST'] + df['Input IGST']
        
        # Formula 2: GST Rate % = (Total GST / Taxable Value) * 100
        if 'Total GST' in df.columns and 'taxable value' in df.columns:
            df['GST Rate %'] = (df['Total GST'] / df['taxable value'].replace(0, np.nan)) * 100
            df['GST Rate %'] = df['GST Rate %'].fillna(0).round(2)
        
        # Formula 3: TDS Rate % = (TDS Deducted / Taxable Value) * 100
        if all(col in df.columns for col in ['TDS deducted', 'taxable value']):
            df['TDS Rate %'] = (df['TDS deducted'] / df['taxable value'].replace(0, np.nan)) * 100
            df['TDS Rate %'] = df['TDS Rate %'].fillna(0).round(2)
        
        # Formula 4: TDS Shortfall = Max(0, Required TDS - Actual TDS)
        tds_rates = {'194C': 1, '194J': 10, '194I': 10, '194H': 5, '194Q': 0.1}
        df['Required TDS'] = df.apply(
            lambda row: (row['taxable value'] * tds_rates.get(str(row['TDS Section']).upper(), 1) / 100)
            if pd.notna(row.get('TDS Section')) else 0, axis=1
        )
        df['TDS Shortfall'] = np.maximum(0, df['Required TDS'] - df['TDS deducted'])
        
        # Formula 5: Interest on Shortfall (1.5% per month for 3 months)
        df['Interest Payable'] = df['TDS Shortfall'] * 0.015 * 3
        
        # Formula 6: Net Payable = Taxable Value + Total GST - TDS deducted
        if 'Total GST' in df.columns:
            df['Net Payable'] = df['taxable value'] + df['Total GST'] - df['TDS deducted']
        
        # Formula 7: Compliance Status
        df['Compliance Status'] = df.apply(
            lambda row: '✅ Compliant' if row['TDS Shortfall'] == 0 
            else '⚠️ Shortfall' if row['TDS Shortfall'] > 0 
            else '❌ Not Deducted', axis=1
        )
        
        return df

# --- 6. MATERIALITY ENGINE ---
class MaterialityEngine:
    def __init__(self, threshold=5.0):
        self.threshold = threshold
    
    def calculate(self, df):
        if 'taxable value' not in df.columns:
            return df
        
        total = df['taxable value'].sum()
        materiality_amount = total * (self.threshold / 100)
        
        # Materiality Score
        df['Materiality Score'] = df['taxable value'] / materiality_amount if materiality_amount > 0 else 0
        
        # Materiality Level
        conditions = [
            df['Materiality Score'] >= 0.5,
            df['Materiality Score'] >= 0.2,
            df['Materiality Score'] >= 0.1,
            df['Materiality Score'] >= 0.05,
            df['Materiality Score'] < 0.05
        ]
        levels = ['🔥 CRITICAL', '⚡ HIGH', '💫 MEDIUM', '🌟 LOW', '📦 IMMATERIAL']
        df['Materiality Level'] = np.select(conditions, levels, default='📦 IMMATERIAL')
        
        # Audit Priority
        priority_map = {
            '🔥 CRITICAL': 1,
            '⚡ HIGH': 2,
            '💫 MEDIUM': 3,
            '🌟 LOW': 4,
            '📦 IMMATERIAL': 5
        }
        df['Audit Priority'] = df['Materiality Level'].map(priority_map)
        
        return df, total, materiality_amount

# --- 7. SAMPLING ENGINE ---
class SamplingEngine:
    @staticmethod
    def percentage_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        return df.sample(n=min(n, len(df)))
    
    @staticmethod
    def materiality_sampling(df, percentage):
        samples = []
        for level in ['🔥 CRITICAL', '⚡ HIGH', '💫 MEDIUM', '🌟 LOW']:
            level_df = df[df['Materiality Level'] == level]
            if len(level_df) > 0:
                if level == '🔥 CRITICAL':
                    pct = min(percentage * 2, 100)
                elif level == '⚡ HIGH':
                    pct = percentage
                else:
                    pct = percentage * 0.5
                n = max(1, int(len(level_df) * (pct / 100)))
                samples.append(level_df.sample(n=min(n, len(level_df))))
        
        return pd.concat(samples) if samples else df.sample(n=max(1, int(len(df) * percentage/100)))

# --- 8. PARTY-WISE DASHBOARD ---
def create_party_dashboard(df):
    """Create comprehensive party-wise analysis"""
    
    # Party-wise Aggregation
    party_stats = df.groupby('Party name').agg({
        'taxable value': ['sum', 'count', 'mean'],
        'TDS deducted': 'sum',
        'Required TDS': 'sum',
        'TDS Shortfall': 'sum',
        'Interest Payable': 'sum',
        'Total GST': 'sum' if 'Total GST' in df.columns else 'sum'
    }).round(2)
    
    party_stats.columns = ['Total Value', 'Transactions', 'Avg Value', 
                          'TDS Paid', 'TDS Required', 'TDS Shortfall', 
                          'Interest', 'Total GST']
    
    party_stats['Compliance %'] = (party_stats['TDS Paid'] / party_stats['TDS Required'] * 100).fillna(100)
    party_stats['Risk Score'] = 100 - party_stats['Compliance %']
    
    return party_stats.sort_values('Total Value', ascending=False)

# --- 9. SAMPLE EXCEL GENERATOR ---
@st.cache_data
def generate_sample_data():
    """Generate sample data with formulas"""
    
    np.random.seed(42)
    
    parties = [
        "Precision Engineering Works", "Vijayalakshmi Electricals", "Geeta Steel Traders",
        "Roots Multiclient Ltd", "Shri Aajii Industrial", "FM Engineers", "B Anjaiash",
        "K Engineers", "ACE CNC TECHNOLOGIES", "B-SON Electricals", "Nehwari Engineering",
        "Hindusthan Metals", "Tech Solutions Inc", "City Hospital", "Royal Properties"
    ]
    
    sections = ['194C', '194J', '194I', '194H', '194Q']
    
    data = []
    start_date = datetime(2023, 4, 1)
    
    for i in range(100):
        date = start_date + timedelta(days=np.random.randint(0, 365))
        party = np.random.choice(parties)
        section = np.random.choice(sections, p=[0.5, 0.2, 0.1, 0.1, 0.1])
        
        if section == '194C':
            taxable = np.random.uniform(5000, 500000)
        elif section == '194J':
            taxable = np.random.uniform(25000, 300000)
        else:
            taxable = np.random.uniform(10000, 200000)
        
        gross = taxable * 1.18  # 18% GST
        cgst = gross * 0.09 if np.random.random() > 0.3 else 0
        sgst = gross * 0.09 if cgst > 0 else 0
        igst = gross * 0.18 if cgst == 0 else 0
        
        tds_rate = {'194C': 0.01, '194J': 0.10, '194I': 0.10, '194H': 0.05, '194Q': 0.001}[section]
        tds_deducted = taxable * tds_rate * np.random.choice([0, 0.5, 0.8, 1.0], p=[0.1, 0.1, 0.2, 0.6])
        
        data.append([
            date.strftime('%d-%m-%Y'),
            party,
            f"INV-{2024000+i}",
            round(gross, 2),
            round(taxable, 2),
            round(cgst, 2),
            round(sgst, 2),
            round(igst, 2),
            round(tds_deducted, 2),
            section
        ])
    
    df = pd.DataFrame(data, columns=[
        'Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value',
        'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted', 'TDS Section'
    ])
    
    # Apply formulas
    processor = DataProcessor()
    df = processor.apply_formulas(df)
    
    return df

# --- 10. MAIN APPLICATION ---
def main():
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="background: rgba(0,255,135,0.1); padding: 20px; border-radius: 20px; border: 1px solid #00ff87;">
            <h3 style="color: #00ff87; font-family: 'Orbitron', sans-serif;">⚡ CONTROL PANEL</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample Data Download
        sample_df = generate_sample_data()
        sample_excel = BytesIO()
        with pd.ExcelWriter(sample_excel, engine='xlsxwriter') as writer:
            sample_df.to_excel(writer, sheet_name='Sample Data', index=False)
        
        st.download_button(
            label="📥 DOWNLOAD SAMPLE DATA",
            data=sample_excel.getvalue(),
            file_name="Ultra_Audit_Sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Parameters
        materiality_threshold = st.slider("🎯 Materiality Threshold %", 0.1, 10.0, 5.0, 0.1)
        sample_percentage = st.slider("📊 Sample Selection %", 1, 100, 20)
        sampling_method = st.selectbox("🎲 Sampling Method", ["Percentage Based", "Materiality Weighted"])
        interest_months = st.number_input("💰 Interest Months", 1, 12, 3)
    
    # File Upload
    st.markdown("### 📤 UPLOAD LEDGER FILE")
    uploaded_file = st.file_uploader("", type=['xlsx', 'csv'], label_visibility="collapsed")
    
    if uploaded_file:
        try:
            # Load Data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Process Data
            processor = DataProcessor()
            column_mapping = processor.detect_columns(df)
            
            # Rename columns
            rename_map = {}
            for std, orig in column_mapping.items():
                if std == 'taxable': rename_map[orig] = 'taxable value'
                elif std == 'tds': rename_map[orig] = 'TDS deducted'
                elif std == 'tds_section': rename_map[orig] = 'TDS Section'
                elif std == 'party': rename_map[orig] = 'Party name'
                elif std == 'invoice': rename_map[orig] = 'Invoice no'
                elif std == 'gross': rename_map[orig] = 'Gross Total'
                elif std == 'cgst': rename_map[orig] = 'Input CGST'
                elif std == 'sgst': rename_map[orig] = 'Input SGST'
                elif std == 'igst': rename_map[orig] = 'Input IGST'
            
            df = df.rename(columns=rename_map)
            
            # Clean numeric columns
            for col in ['Gross Total', 'taxable value', 'TDS deducted', 
                       'Input CGST', 'Input SGST', 'Input IGST']:
                if col in df.columns:
                    df[col] = processor.clean_numeric(df[col])
                else:
                    df[col] = 0
            
            # Apply formulas
            df = processor.apply_formulas(df)
            
            # Calculate Materiality
            materiality_engine = MaterialityEngine(materiality_threshold)
            df, total_value, materiality_amount = materiality_engine.calculate(df)
            
            # Apply Sampling
            sampling_engine = SamplingEngine()
            if sampling_method == "Materiality Weighted":
                sample_df = sampling_engine.materiality_sampling(df, sample_percentage)
            else:
                sample_df = sampling_engine.percentage_sampling(df, sample_percentage)
            
            # Party Dashboard
            party_stats = create_party_dashboard(df)
            
            # ===== DASHBOARD =====
            
            # Key Metrics
            st.markdown("### 📊 REAL-TIME METRICS")
            cols = st.columns(5)
            
            metrics = [
                ("💰 Total Value", f"₹{total_value:,.0f}", "#00ff87"),
                ("📦 Transactions", f"{len(df):,}", "#60efff"),
                ("🔥 Critical Items", f"{len(df[df['Materiality Level']=='🔥 CRITICAL'])}", "#ff00ff"),
                ("🎯 Sample Size", f"{len(sample_df)}", "#0061ff"),
                ("⚠️ TDS Shortfall", f"₹{df['TDS Shortfall'].sum():,.0f}", "#ff4444")
            ]
            
            for i, (label, value, color) in enumerate(metrics):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card-ultra" style="border-color: {color};">
                        <h4 style="color: {color}; margin:0;">{label}</h4>
                        <h2 style="color: white; margin:10px 0;">{value}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎯 PARTY ANALYSIS", "📈 MATERIALITY VIEW", "🔍 SAMPLE DETAILS", 
                "💰 TDS COMPLIANCE", "📊 TREND ANALYSIS"
            ])
            
            with tab1:
                st.markdown("### 🏢 PARTY-WISE ANALYSIS")
                
                # Party Selector
                selected_party = st.selectbox("Select Party", ['All'] + list(party_stats.index[:20]))
                
                if selected_party != 'All':
                    party_data = df[df['Party name'] == selected_party]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Party Summary
                        st.markdown(f"""
                        <div class="party-card">
                            <h3 style="color: #00ff87;">{selected_party}</h3>
                            <p>Total Value: ₹{party_data['taxable value'].sum():,.0f}</p>
                            <p>Transactions: {len(party_data)}</p>
                            <p>TDS Paid: ₹{party_data['TDS deducted'].sum():,.0f}</p>
                            <p>TDS Shortfall: ₹{party_data['TDS Shortfall'].sum():,.0f}</p>
                            <p>Compliance: {(party_data['TDS deducted'].sum()/party_data['Required TDS'].sum()*100):.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Party Transactions Chart
                        fig = px.bar(party_data, x='Date', y='taxable value', 
                                   title=f"{selected_party} - Transaction Trend",
                                   color='Materiality Level',
                                   color_discrete_map={
                                       '🔥 CRITICAL': '#ff00ff',
                                       '⚡ HIGH': '#00ff87',
                                       '💫 MEDIUM': '#60efff',
                                       '🌟 LOW': '#0061ff'
                                   })
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Party Transactions Table
                    st.dataframe(party_data, use_container_width=True)
                
                else:
                    # Party Stats Table
                    st.dataframe(party_stats.style.background_gradient(
                        subset=['Risk Score'], cmap='RdYlGn_r'
                    ).format({
                        'Total Value': '₹{:,.0f}',
                        'TDS Paid': '₹{:,.0f}',
                        'TDS Shortfall': '₹{:,.0f}',
                        'Compliance %': '{:.1f}%',
                        'Risk Score': '{:.1f}'
                    }), use_container_width=True)
                    
                    # Top Parties Chart
                    top_parties = party_stats.head(10).reset_index()
                    fig = px.bar(top_parties, x='Party name', y='Total Value',
                               title="Top 10 Parties by Value",
                               color='Risk Score', color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Materiality Distribution
                    mat_dist = df['Materiality Level'].value_counts().reset_index()
                    mat_dist.columns = ['Level', 'Count']
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=mat_dist['Level'],
                        values=mat_dist['Count'],
                        hole=0.4,
                        marker_colors=['#ff00ff', '#00ff87', '#60efff', '#0061ff', '#95a5a6']
                    )])
                    fig.update_layout(title="Materiality Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Value by Materiality
                    val_dist = df.groupby('Materiality Level')['taxable value'].sum().reset_index()
                    
                    fig = px.bar(val_dist, x='Materiality Level', y='taxable value',
                               color='Materiality Level',
                               color_discrete_map={
                                   '🔥 CRITICAL': '#ff00ff',
                                   '⚡ HIGH': '#00ff87',
                                   '💫 MEDIUM': '#60efff',
                                   '🌟 LOW': '#0061ff'
                               })
                    fig.update_layout(title="Value by Materiality")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("### 🔍 SAMPLE DETAILS")
                st.dataframe(sample_df, use_container_width=True)
                
                # Sample Composition
                sample_composition = sample_df['Materiality Level'].value_counts()
                fig = px.pie(values=sample_composition.values, names=sample_composition.index,
                           title="Sample Composition", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    # TDS Shortfall by Party
                    shortfall_by_party = df.groupby('Party name')['TDS Shortfall'].sum().nlargest(10).reset_index()
                    fig = px.bar(shortfall_by_party, x='Party name', y='TDS Shortfall',
                               title="Top 10 TDS Shortfalls", color='TDS Shortfall')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # TDS Compliance Rate
                    compliance_by_section = df.groupby('TDS Section').agg({
                        'TDS deducted': 'sum',
                        'Required TDS': 'sum'
                    }).reset_index()
                    compliance_by_section['Compliance %'] = (compliance_by_section['TDS deducted'] / 
                                                            compliance_by_section['Required TDS'] * 100)
                    
                    fig = px.bar(compliance_by_section, x='TDS Section', y='Compliance %',
                               title="TDS Compliance by Section", color='Compliance %',
                               color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    
                    # Monthly Trend
                    monthly = df.groupby(df['Date'].dt.to_period('M'))['taxable value'].sum().reset_index()
                    monthly['Date'] = monthly['Date'].astype(str)
                    
                    fig = px.line(monthly, x='Date', y='taxable value',
                                title="Monthly Transaction Trend", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Export
            st.markdown("### 📥 EXPORT REPORT")
            if st.button("⚡ GENERATE ULTRA REPORT", use_container_width=True):
                with st.spinner("Generating..."):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='Complete Data', index=False)
                        sample_df.to_excel(writer, sheet_name='Sample', index=False)
                        party_stats.to_excel(writer, sheet_name='Party Analysis')
                    
                    st.download_button(
                        label="📥 DOWNLOAD EXCEL REPORT",
                        data=output.getvalue(),
                        file_name=f"Ultra_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    else:
        # Welcome Screen
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2 style="color: #00ff87; font-family: 'Orbitron', sans-serif;">🚀 READY FOR NEXT-GEN AUDIT</h2>
            <p style="color: white; font-size: 1.2rem;">Upload your ledger file to begin</p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 30px;">
                <div class="glass-card">🎯 5-Level Materiality</div>
                <div class="glass-card">📊 Dynamic Formulas</div>
                <div class="glass-card">💰 TDS Intelligence</div>
                <div class="glass-card">📈 Party Analytics</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
