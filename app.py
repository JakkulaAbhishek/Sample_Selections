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
warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Ultra-Audit Pro | by Jakkula Abhishek",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ULTRA-STYLISH CSS ---
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
        padding: 2rem;
        border-radius: 30px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,255,135,0.3);
        border: 2px solid rgba(255,255,255,0.1);
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
    
    /* Neon Text */
    .neon-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        text-shadow: 0 0 10px #00ff87, 0 0 20px #00ff87;
        color: white;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { text-shadow: 0 0 10px #00ff87, 0 0 20px #00ff87; }
        50% { text-shadow: 0 0 20px #00ff87, 0 0 40px #00ff87; }
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
    
    /* Metric Cards */
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
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: white;
        font-family: 'Orbitron', sans-serif;
        border-radius: 10px;
        padding: 10px 25px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #00ff87 !important;
        color: #0a0f1e !important;
        font-weight: 700;
    }
</style>

<div class="developer-signature">
    ⚡ Developed by: JAKKULA ABHISHEK | 📧 jakkulaabhishek5@gmail.com ⚡
</div>
""", unsafe_allow_html=True)

# --- 3. CYBERPUNK HEADER ---
st.markdown("""
<div class="cyber-header">
    <h1 style="font-family: 'Orbitron', sans-serif; font-size: 3.5rem; margin:0; color: white; text-align: center;">
        ⚡ ULTRA-AUDIT PRO ⚡
    </h1>
    <p style="font-family: 'Orbitron', sans-serif; font-size: 1.2rem; text-align: center; color: rgba(255,255,255,0.9); margin-top: 10px;">
        Next-Gen AI-Powered Audit Intelligence | Materiality Analysis | TDS Compliance
    </p>
</div>
""", unsafe_allow_html=True)

# --- 4. SAMPLE DATA GENERATOR (EXACT COLUMNS) ---
@st.cache_data
def generate_sample_data():
    """Generate sample data with exact columns specified"""
    
    sample_data = [
        # Date, Party name, Invoice no, Gross Total, taxable value, Input CGST, Input SGST, Input IGST, TDS deducted, TDS Section
        ["01-04-2023", "Precision Engineering Works", "PEW/001/23-24", 135405.00, 114750.00, 10327.50, 10327.50, 0.00, 1147.50, "194C"],
        ["05-04-2023", "Vijayalakshmi Electricals", "ST/23-24/468", 78479.44, 66508.00, 5985.72, 5985.72, 0.00, 665.08, "194C"],
        ["10-04-2023", "Geeta Steel Traders", "533", 25250.14, 21322.16, 1963.99, 1963.99, 0.00, 213.22, "194C"],
        ["15-04-2023", "Roots Multiclient Ltd", "6112303938", 10664.84, 9038.00, 0.00, 0.00, 1626.84, 90.38, "194C"],
        ["20-04-2023", "Shri Aajii Industrial", "SAI/787/23-24", 67021.01, 5605.00, 512.55, 512.55, 0.00, 56.05, "194C"],
        ["25-04-2023", "FM Engineers", "5034", 5310.00, 4500.00, 405.00, 405.00, 0.00, 45.00, "194C"],
        ["01-05-2023", "B Anjaiash", "158", 9600.00, 9600.00, 864.00, 864.00, 0.00, 96.00, "194C"],
        ["05-05-2023", "K Engineers", "GST-23-24/142", 21760.00, 17000.00, 1530.00, 1530.00, 0.00, 170.00, "194C"],
        ["10-05-2023", "ACE CNC TECHNOLOGIES", "ACT/TI/022-23-24", 25960.00, 22000.00, 1980.00, 1980.00, 0.00, 220.00, "194C"],
        ["15-05-2023", "B-SON Electricals", "207", 15008.42, 12719.00, 1144.71, 1144.71, 0.00, 127.19, "194C"],
        ["20-05-2023", "Nehwari Engineering", "11/2023-24", 8850.00, 7500.00, 675.00, 675.00, 0.00, 75.00, "194C"],
        ["25-05-2023", "Hindusthan Metals", "HM/23-24/198", 92040.00, 78000.00, 7020.00, 7020.00, 0.00, 780.00, "194C"],
        ["01-06-2023", "City Hospital", "CH/2023/001", 295000.00, 250000.00, 22500.00, 22500.00, 0.00, 25000.00, "194J"],
        ["05-06-2023", "Dr. Mehta's Clinic", "DMC/06/23", 59000.00, 50000.00, 4500.00, 4500.00, 0.00, 5000.00, "194J"],
        ["10-06-2023", "Legal Eagles Associates", "LEA/234", 118000.00, 100000.00, 9000.00, 9000.00, 0.00, 10000.00, "194J"],
        ["15-06-2023", "Royal Properties", "RP/JUNE/01", 295000.00, 250000.00, 22500.00, 22500.00, 0.00, 25000.00, "194I"],
        ["20-06-2023", "Godrej Properties", "GP/2023/456", 590000.00, 500000.00, 45000.00, 45000.00, 0.00, 50000.00, "194I"],
        ["25-06-2023", "Marketing Gurus", "MG/COM/07", 59000.00, 50000.00, 4500.00, 4500.00, 0.00, 2500.00, "194H"],
        ["01-07-2023", "Steel Suppliers Ltd", "SSL/07/001", 2360000.00, 2000000.00, 0.00, 0.00, 360000.00, 2000.00, "194Q"],
        ["05-07-2023", "Cement Corporation", "CC/07/089", 1180000.00, 1000000.00, 0.00, 0.00, 180000.00, 1000.00, "194Q"]
    ]
    
    df = pd.DataFrame(sample_data, columns=[
        'Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value',
        'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted', 'TDS Section'
    ])
    
    return df

# --- 5. DATA PROCESSING ENGINE ---
class DataProcessor:
    @staticmethod
    def clean_numeric(series):
        """Clean numeric values"""
        if series.dtype == 'object':
            series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
            series = series.replace('', '0')
        return pd.to_numeric(series, errors='coerce').fillna(0)
    
    @staticmethod
    def apply_formulas(df):
        """Apply dynamic formulas"""
        
        # Formula 1: Total GST
        if all(col in df.columns for col in ['Input CGST', 'Input SGST', 'Input IGST']):
            df['Total GST'] = df['Input CGST'] + df['Input SGST'] + df['Input IGST']
        
        # Formula 2: GST Rate %
        if 'Total GST' in df.columns and 'taxable value' in df.columns:
            df['GST Rate %'] = (df['Total GST'] / df['taxable value'].replace(0, np.nan)) * 100
            df['GST Rate %'] = df['GST Rate %'].fillna(0).round(2)
        
        # Formula 3: TDS Rate %
        tds_rates = {
            '194C': 1.0, '194J': 10.0, '194I': 10.0, 
            '194H': 5.0, '194Q': 0.1, '194IA': 1.0,
            '194IB': 5.0, '194M': 5.0
        }
        
        df['Std TDS Rate %'] = df['TDS Section'].map(lambda x: tds_rates.get(str(x).upper(), 1.0))
        df['Applied TDS Rate %'] = (df['TDS deducted'] / df['taxable value'].replace(0, np.nan)) * 100
        df['Applied TDS Rate %'] = df['Applied TDS Rate %'].fillna(0).round(2)
        
        # Formula 4: Required TDS
        df['Required TDS'] = (df['taxable value'] * df['Std TDS Rate %'] / 100).round(2)
        
        # Formula 5: TDS Shortfall
        df['TDS Shortfall'] = np.maximum(0, df['Required TDS'] - df['TDS deducted']).round(2)
        
        # Formula 6: Interest Payable (1.5% per month for 3 months)
        df['Interest Payable'] = (df['TDS Shortfall'] * 0.015 * 3).round(2)
        
        # Formula 7: Net Payable
        if 'Total GST' in df.columns:
            df['Net Payable'] = (df['taxable value'] + df['Total GST'] - df['TDS deducted']).round(2)
        
        # Formula 8: Compliance Status
        df['Compliance Status'] = df.apply(
            lambda row: '✅ FULLY COMPLIANT' if row['TDS Shortfall'] == 0 
            else '⚠️ SHORTFALL' if row['TDS Shortfall'] > 0 
            else '❌ NOT DEDUCTED', axis=1
        )
        
        # Formula 9: TDS Compliance %
        df['TDS Compliance %'] = df.apply(
            lambda row: round((row['TDS deducted'] / row['Required TDS'] * 100), 2) 
            if row['Required TDS'] > 0 else 100, axis=1
        )
        
        return df

# --- 6. MATERIALITY ENGINE ---
class MaterialityEngine:
    def __init__(self, threshold=5.0):
        self.threshold = threshold
    
    def calculate(self, df):
        if 'taxable value' not in df.columns:
            return df, 0, 0
        
        total = df['taxable value'].sum()
        materiality_amount = total * (self.threshold / 100)
        
        # Materiality Score
        df['Materiality Score'] = (df['taxable value'] / materiality_amount).round(2) if materiality_amount > 0 else 0
        
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

# --- 7. COMPREHENSIVE SAMPLING ENGINE (ALL METHODS) ---
class SamplingEngine:
    """Complete sampling engine with ALL methods"""
    
    @staticmethod
    def simple_random_sampling(df, percentage):
        """Method 1: Simple Random Sampling"""
        n = max(1, int(len(df) * (percentage / 100)))
        return df.sample(n=min(n, len(df)), random_state=42)
    
    @staticmethod
    def systematic_sampling(df, percentage):
        """Method 2: Systematic Sampling"""
        n = max(1, int(len(df) * (percentage / 100)))
        if len(df) <= n:
            return df
        step = len(df) // n
        start = np.random.randint(0, step)
        indices = range(start, len(df), step)
        return df.iloc[indices].head(n)
    
    @staticmethod
    def stratified_sampling(df, percentage, strata_col='Materiality Level'):
        """Method 3: Stratified Sampling"""
        n = max(1, int(len(df) * (percentage / 100)))
        samples = []
        
        for stratum in df[strata_col].unique():
            stratum_df = df[df[strata_col] == stratum]
            stratum_n = max(1, int(n * len(stratum_df) / len(df)))
            samples.append(stratum_df.sample(n=min(stratum_n, len(stratum_df)), random_state=42))
        
        return pd.concat(samples) if samples else df.sample(n=n)
    
    @staticmethod
    def cluster_sampling(df, percentage, n_clusters=5):
        """Method 4: Cluster Sampling"""
        from sklearn.cluster import KMeans
        
        if 'taxable value' not in df.columns or len(df) < n_clusters:
            return SamplingEngine.simple_random_sampling(df, percentage)
        
        # Create clusters based on value
        X = df[['taxable value']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
        
        # Select random clusters
        n = max(1, int(len(df) * (percentage / 100)))
        n_clusters_to_select = max(1, int(n_clusters * percentage / 100))
        selected_clusters = np.random.choice(df['Cluster'].unique(), n_clusters_to_select, replace=False)
        
        return df[df['Cluster'].isin(selected_clusters)]
    
    @staticmethod
    def multistage_sampling(df, percentage):
        """Method 5: Multistage Sampling"""
        # Stage 1: Cluster by party
        parties = df['Party name'].unique()
        n_parties = max(1, int(len(parties) * (percentage / 100)))
        selected_parties = np.random.choice(parties, n_parties, replace=False)
        
        # Stage 2: Sample within selected parties
        samples = []
        for party in selected_parties:
            party_df = df[df['Party name'] == party]
            party_n = max(1, int(len(party_df) * (percentage / 100)))
            samples.append(party_df.sample(n=min(party_n, len(party_df)), random_state=42))
        
        return pd.concat(samples) if samples else df.sample(n=max(1, int(len(df) * percentage/100)))
    
    @staticmethod
    def pps_sampling(df, percentage):
        """Method 6: Probability Proportional to Size"""
        if 'taxable value' not in df.columns:
            return SamplingEngine.simple_random_sampling(df, percentage)
        
        n = max(1, int(len(df) * (percentage / 100)))
        # Probability proportional to taxable value
        probs = df['taxable value'] / df['taxable value'].sum()
        return df.sample(n=min(n, len(df)), weights=probs, random_state=42)
    
    @staticmethod
    def convenience_sampling(df, percentage):
        """Method 7: Convenience Sampling (first N records)"""
        n = max(1, int(len(df) * (percentage / 100)))
        return df.head(n)
    
    @staticmethod
    def judgmental_sampling(df, percentage):
        """Method 8: Judgmental Sampling (top values)"""
        n = max(1, int(len(df) * (percentage / 100)))
        return df.nlargest(n, 'taxable value')
    
    @staticmethod
    def quota_sampling(df, percentage, quota_col='Materiality Level'):
        """Method 9: Quota Sampling"""
        n = max(1, int(len(df) * (percentage / 100)))
        samples = []
        
        for stratum in df[quota_col].unique():
            stratum_df = df[df[quota_col] == stratum]
            # Fixed quota for each stratum
            quota = max(1, int(n / len(df[quota_col].unique())))
            samples.append(stratum_df.head(quota))
        
        return pd.concat(samples).head(n)
    
    @staticmethod
    def snowball_sampling(df, percentage):
        """Method 10: Snowball Sampling (linked by party)"""
        n = max(1, int(len(df) * (percentage / 100)))
        
        # Start with random seed
        seed_idx = np.random.randint(0, len(df))
        seed_party = df.iloc[seed_idx]['Party name']
        
        # Get all transactions of that party and connected parties
        sample_df = df[df['Party name'] == seed_party]
        
        # Add more parties until we reach desired size
        while len(sample_df) < n:
            remaining = df[~df['Party name'].isin(sample_df['Party name'].unique())]
            if len(remaining) == 0:
                break
            next_party = remaining['Party name'].iloc[0]
            sample_df = pd.concat([sample_df, df[df['Party name'] == next_party]])
        
        return sample_df.head(n)
    
    @staticmethod
    def mus_sampling(df, percentage):
        """Method 11: Monetary Unit Sampling"""
        if 'taxable value' not in df.columns or df['taxable value'].sum() == 0:
            return SamplingEngine.simple_random_sampling(df, percentage)
        
        n = max(1, int(len(df) * (percentage / 100)))
        total = df['taxable value'].sum()
        interval = total / n
        
        # Sort by value and calculate cumulative
        df_sorted = df.sort_values('taxable value', ascending=False).reset_index(drop=True)
        df_sorted['cumulative'] = df_sorted['taxable value'].cumsum()
        
        samples = []
        current = interval
        
        for _, row in df_sorted.iterrows():
            if row['cumulative'] >= current and len(samples) < n:
                samples.append(row)
                current += interval
        
        return pd.DataFrame(samples) if samples else df.sample(n=n)
    
    @staticmethod
    def block_sampling(df, percentage):
        """Method 12: Block Sampling"""
        n = max(1, int(len(df) * (percentage / 100)))
        
        # Create blocks by month if date exists
        if 'Date' in df.columns:
            df['Month'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce').dt.month
            block_col = 'Month'
        else:
            # Create artificial blocks
            df['Block'] = pd.qcut(df['taxable value'], q=5, labels=['B1', 'B2', 'B3', 'B4', 'B5'])
            block_col = 'Block'
        
        # Select one complete block
        blocks = df[block_col].unique()
        selected_block = np.random.choice(blocks)
        block_df = df[df[block_col] == selected_block]
        
        if len(block_df) >= n:
            return block_df.head(n)
        else:
            # Supplement with another block
            remaining_blocks = [b for b in blocks if b != selected_block]
            if remaining_blocks:
                second_block = np.random.choice(remaining_blocks)
                second_df = df[df[block_col] == second_block]
                return pd.concat([block_df, second_df]).head(n)
        
        return df.head(n)
    
    @staticmethod
    def adaptive_sampling(df, percentage):
        """Method 13: Adaptive Sampling"""
        n = max(1, int(len(df) * (percentage / 100)))
        
        # Start with high-value items
        high_value = df[df['taxable value'] > df['taxable value'].quantile(0.75)]
        
        if len(high_value) >= n:
            return high_value.head(n)
        
        # Add medium value items
        remaining_n = n - len(high_value)
        medium_value = df[(df['taxable value'] <= df['taxable value'].quantile(0.75)) & 
                          (df['taxable value'] > df['taxable value'].quantile(0.5))]
        
        sample = pd.concat([high_value, medium_value.head(remaining_n)])
        
        if len(sample) < n:
            # Add random low value items
            low_value = df[df['taxable value'] <= df['taxable value'].quantile(0.5)]
            remaining = n - len(sample)
            sample = pd.concat([sample, low_value.sample(n=min(remaining, len(low_value)), random_state=42)])
        
        return sample
    
    @staticmethod
    def sequential_sampling(df, percentage):
        """Method 14: Sequential Sampling"""
        n = max(1, int(len(df) * (percentage / 100)))
        
        # Sort by date if available
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date')
        else:
            df_sorted = df.sort_values('taxable value', ascending=False)
        
        # Take first N in sequence
        return df_sorted.head(n)

# --- 8. PARTY-WISE DASHBOARD ---
def create_party_dashboard(df):
    """Create party-wise analysis"""
    
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
    
    party_stats['TDS Compliance %'] = (party_stats['TDS Paid'] / party_stats['TDS Required'] * 100).fillna(100).round(2)
    party_stats['Risk Score'] = (100 - party_stats['TDS Compliance %']).round(2)
    
    return party_stats.sort_values('Total Value', ascending=False)

# --- 9. MAIN APPLICATION ---
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
            label="📥 DOWNLOAD SAMPLE EXCEL",
            data=sample_excel.getvalue(),
            file_name="Ultra_Audit_Sample_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Parameters
        materiality_threshold = st.slider("🎯 Materiality Threshold %", 0.1, 10.0, 5.0, 0.1)
        sample_percentage = st.slider("📊 Sample Selection %", 1, 100, 20)
        
        # ALL 14 SAMPLING METHODS
        sampling_method = st.selectbox(
            "🎲 SELECT SAMPLING METHOD",
            [
                "1. Simple Random Sampling",
                "2. Systematic Sampling",
                "3. Stratified Sampling",
                "4. Cluster Sampling",
                "5. Multistage Sampling",
                "6. PPS Sampling",
                "7. Convenience Sampling",
                "8. Judgmental Sampling",
                "9. Quota Sampling",
                "10. Snowball Sampling",
                "11. MUS Sampling",
                "12. Block Sampling",
                "13. Adaptive Sampling",
                "14. Sequential Sampling"
            ]
        )
        
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
            
            # Ensure columns are correct
            expected_cols = ['Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value',
                           'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted', 'TDS Section']
            
            for col in expected_cols:
                if col not in df.columns:
                    st.warning(f"Column '{col}' not found. Please ensure your file has all required columns.")
            
            # Clean numeric columns
            processor = DataProcessor()
            numeric_cols = ['Gross Total', 'taxable value', 'TDS deducted', 
                          'Input CGST', 'Input SGST', 'Input IGST']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = processor.clean_numeric(df[col])
            
            # Apply formulas
            df = processor.apply_formulas(df)
            
            # Calculate Materiality
            materiality_engine = MaterialityEngine(materiality_threshold)
            df, total_value, materiality_amount = materiality_engine.calculate(df)
            
            # Apply selected sampling method
            sampling_engine = SamplingEngine()
            method_map = {
                "1. Simple Random Sampling": sampling_engine.simple_random_sampling,
                "2. Systematic Sampling": sampling_engine.systematic_sampling,
                "3. Stratified Sampling": lambda d, p: sampling_engine.stratified_sampling(d, p, 'Materiality Level'),
                "4. Cluster Sampling": sampling_engine.cluster_sampling,
                "5. Multistage Sampling": sampling_engine.multistage_sampling,
                "6. PPS Sampling": sampling_engine.pps_sampling,
                "7. Convenience Sampling": sampling_engine.convenience_sampling,
                "8. Judgmental Sampling": sampling_engine.judgmental_sampling,
                "9. Quota Sampling": lambda d, p: sampling_engine.quota_sampling(d, p, 'Materiality Level'),
                "10. Snowball Sampling": sampling_engine.snowball_sampling,
                "11. MUS Sampling": sampling_engine.mus_sampling,
                "12. Block Sampling": sampling_engine.block_sampling,
                "13. Adaptive Sampling": sampling_engine.adaptive_sampling,
                "14. Sequential Sampling": sampling_engine.sequential_sampling
            }
            
            sample_df = method_map[sampling_method](df, sample_percentage)
            
            # Party Dashboard
            party_stats = create_party_dashboard(df)
            
            # ===== DASHBOARD =====
            
            # Key Metrics
            st.markdown("### 📊 REAL-TIME METRICS")
            cols = st.columns(5)
            
            metrics = [
                ("💰 TOTAL VALUE", f"₹{total_value:,.0f}"),
                ("📦 TRANSACTIONS", f"{len(df)}"),
                ("🔥 CRITICAL", f"{len(df[df['Materiality Level']=='🔥 CRITICAL'])}"),
                ("🎯 SAMPLE SIZE", f"{len(sample_df)} ({sample_percentage}%)"),
                ("⚠️ SHORTFALL", f"₹{df['TDS Shortfall'].sum():,.0f}")
            ]
            
            for i, (label, value) in enumerate(metrics):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card-ultra">
                        <h4 style="color: #00ff87; margin:0;">{label}</h4>
                        <h2 style="color: white; margin:10px 0;">{value}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎯 PARTY ANALYSIS", "📈 MATERIALITY", "🔍 SAMPLE DETAILS", 
                "💰 TDS COMPLIANCE", "📊 FORMULA VIEW"
            ])
            
            with tab1:
                st.markdown("### 🏢 PARTY-WISE ANALYSIS")
                
                # Party Selector
                selected_party = st.selectbox("Select Party", ['All'] + list(party_stats.index[:20]))
                
                if selected_party != 'All':
                    party_data = df[df['Party name'] == selected_party]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        total_tds = party_data['TDS deducted'].sum()
                        required_tds = party_data['Required TDS'].sum()
                        compliance = (total_tds / required_tds * 100) if required_tds > 0 else 100
                        
                        st.markdown(f"""
                        <div class="party-card">
                            <h3 style="color: #00ff87;">{selected_party}</h3>
                            <p>📊 Total Value: ₹{party_data['taxable value'].sum():,.0f}</p>
                            <p>📦 Transactions: {len(party_data)}</p>
                            <p>💰 TDS Paid: ₹{total_tds:,.0f}</p>
                            <p>⚠️ TDS Shortfall: ₹{party_data['TDS Shortfall'].sum():,.0f}</p>
                            <p>✅ Compliance: {compliance:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Party transactions chart
                        fig = px.bar(party_data, x='Date', y='taxable value', 
                                   title=f"{selected_party} - Transactions",
                                   color='Materiality Level')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Party transactions table
                    st.dataframe(party_data, use_container_width=True)
                
                else:
                    # Party stats table
                    st.dataframe(party_stats.style.background_gradient(
                        subset=['Risk Score'], cmap='RdYlGn_r'
                    ).format({
                        'Total Value': '₹{:,.0f}',
                        'TDS Paid': '₹{:,.0f}',
                        'TDS Shortfall': '₹{:,.0f}',
                        'TDS Compliance %': '{:.1f}%',
                        'Risk Score': '{:.1f}'
                    }), use_container_width=True)
                    
                    # Top parties chart
                    top_10 = party_stats.head(10).reset_index()
                    fig = px.bar(top_10, x='Party name', y='Total Value',
                               title="Top 10 Parties by Value",
                               color='Risk Score', color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Materiality distribution
                    mat_dist = df['Materiality Level'].value_counts().reset_index()
                    mat_dist.columns = ['Level', 'Count']
                    
                    colors = {
                        '🔥 CRITICAL': '#ff00ff',
                        '⚡ HIGH': '#00ff87',
                        '💫 MEDIUM': '#60efff',
                        '🌟 LOW': '#0061ff',
                        '📦 IMMATERIAL': '#95a5a6'
                    }
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=mat_dist['Level'],
                        values=mat_dist['Count'],
                        hole=0.4,
                        marker_colors=[colors.get(l, '#00ff87') for l in mat_dist['Level']]
                    )])
                    fig.update_layout(title="Materiality Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Value by materiality
                    val_dist = df.groupby('Materiality Level')['taxable value'].sum().reset_index()
                    
                    fig = px.bar(val_dist, x='Materiality Level', y='taxable value',
                               color='Materiality Level',
                               color_discrete_map=colors,
                               title="Value by Materiality Level")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown(f"### 🔍 SAMPLE DETAILS - Method: {sampling_method}")
                st.info(f"Sample Size: {len(sample_df)} transactions ({sample_percentage}% of population)")
                
                # Show sample composition
                sample_composition = sample_df['Materiality Level'].value_counts()
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig = px.pie(values=sample_composition.values, names=sample_composition.index,
                               title="Sample Composition", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Sample stats
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4>Sample Statistics</h4>
                        <p>Total Value: ₹{sample_df['taxable value'].sum():,.0f}</p>
                        <p>Avg Value: ₹{sample_df['taxable value'].mean():,.0f}</p>
                        <p>TDS Shortfall: ₹{sample_df['TDS Shortfall'].sum():,.0f}</p>
                        <p>Critical Items: {len(sample_df[sample_df['Materiality Level']=='🔥 CRITICAL'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Sample data table
                st.dataframe(sample_df, use_container_width=True)
            
            with tab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    # TDS shortfall by party
                    shortfall_by_party = df.groupby('Party name')['TDS Shortfall'].sum().nlargest(10).reset_index()
                    fig = px.bar(shortfall_by_party, x='Party name', y='TDS Shortfall',
                               title="Top 10 TDS Shortfalls", color='TDS Shortfall')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Compliance by section
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
                
                # TDS summary table
                tds_summary = df.groupby('TDS Section').agg({
                    'taxable value': 'sum',
                    'TDS deducted': 'sum',
                    'Required TDS': 'sum',
                    'TDS Shortfall': 'sum',
                    'Interest Payable': 'sum'
                }).round(2)
                
                st.dataframe(tds_summary.style.format('₹{:,.0f}'), use_container_width=True)
            
            with tab5:
                st.markdown("### 📊 FORMULA CALCULATIONS VIEW")
                st.markdown("""
                <div class="glass-card">
                    <h4>Dynamic Formulas Applied:</h4>
                    <ul>
                        <li>✅ Total GST = CGST + SGST + IGST</li>
                        <li>✅ GST Rate % = (Total GST / Taxable Value) × 100</li>
                        <li>✅ Standard TDS Rate % = Based on TDS Section</li>
                        <li>✅ Applied TDS Rate % = (TDS Deducted / Taxable Value) × 100</li>
                        <li>✅ Required TDS = Taxable Value × Standard TDS Rate %</li>
                        <li>✅ TDS Shortfall = Max(0, Required TDS - Actual TDS)</li>
                        <li>✅ Interest Payable = Shortfall × 1.5% × 3 months</li>
                        <li>✅ Net Payable = Taxable Value + GST - TDS</li>
                        <li>✅ TDS Compliance % = (Actual TDS / Required TDS) × 100</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Show columns with formulas
                formula_cols = ['Party name', 'Invoice no', 'taxable value', 'TDS Section',
                              'Std TDS Rate %', 'Applied TDS Rate %', 'Required TDS',
                              'TDS deducted', 'TDS Shortfall', 'Interest Payable',
                              'TDS Compliance %', 'Compliance Status', 'Materiality Level']
                
                formula_cols = [col for col in formula_cols if col in df.columns]
                st.dataframe(df[formula_cols].head(20), use_container_width=True)
            
            # Export
            st.markdown("### 📥 EXPORT REPORT")
            if st.button("⚡ GENERATE COMPLETE REPORT", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='Complete Data', index=False)
                        sample_df.to_excel(writer, sheet_name=f'Sample ({sampling_method[:20]})', index=False)
                        party_stats.to_excel(writer, sheet_name='Party Analysis')
                        
                        # Add formula sheet
                        formula_summary = pd.DataFrame({
                            'Formula': [
                                'Total GST', 'GST Rate %', 'Standard TDS Rate %', 'Applied TDS Rate %',
                                'Required TDS', 'TDS Shortfall', 'Interest Payable', 'Net Payable',
                                'TDS Compliance %', 'Materiality Score'
                            ],
                            'Calculation': [
                                'CGST + SGST + IGST',
                                '(Total GST / Taxable Value) × 100',
                                'Based on TDS Section (194C:1%, 194J:10%, 194I:10%, 194H:5%, 194Q:0.1%)',
                                '(TDS Deducted / Taxable Value) × 100',
                                'Taxable Value × Standard TDS Rate %',
                                'MAX(0, Required TDS - Actual TDS)',
                                'Shortfall × 1.5% × 3 months',
                                'Taxable Value + Total GST - TDS Deducted',
                                '(Actual TDS / Required TDS) × 100',
                                'Taxable Value / (Total × Materiality Threshold %)'
                            ]
                        })
                        formula_summary.to_excel(writer, sheet_name='Formula Reference', index=False)
                    
                    st.download_button(
                        label="📥 DOWNLOAD EXCEL REPORT",
                        data=output.getvalue(),
                        file_name=f"Ultra_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome Screen
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2 style="color: #00ff87; font-family: 'Orbitron', sans-serif;">🚀 ULTRA-AUDIT PRO READY</h2>
            <p style="color: white; font-size: 1.2rem;">Upload your ledger file with these exact columns:</p>
            <div style="background: rgba(0,255,135,0.1); padding: 20px; border-radius: 15px; margin: 20px auto; max-width: 800px;">
                <code style="color: #00ff87; font-size: 1rem;">
                    Date | Party name | Invoice no | Gross Total | taxable value | Input CGST | Input SGST | Input IGST | TDS deducted | TDS Section
                </code>
            </div>
            <p style="color: #60efff;">Click the button in sidebar to download sample Excel with exact format</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
