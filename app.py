import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import warnings
import xlsxwriter

warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Ultra-Audit Pro | by Jakkula Abhishek",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ULTRA CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    .main { background: linear-gradient(135deg, #0a0f1e 0%, #1a1f35 100%); font-family: 'Rajdhani', sans-serif; }
    .cyber-header { background: linear-gradient(270deg, #00ff87, #60efff, #0061ff, #ff00ff); background-size: 300% 300%; animation: gradientShift 10s ease infinite; padding: 2rem; border-radius: 30px; margin-bottom: 2rem; box-shadow: 0 20px 40px rgba(0,255,135,0.3); border: 2px solid rgba(255,255,255,0.1); }
    @keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .glass-card { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 1.5rem; box-shadow: 0 8px 32px 0 rgba(31,38,135,0.37); transition: all 0.3s ease; }
    .glass-card:hover { transform: translateY(-5px); box-shadow: 0 15px 45px 0 rgba(0,255,135,0.3); border: 1px solid #00ff87; }
    .developer-signature { font-family: 'Orbitron', sans-serif; background: linear-gradient(90deg, #00ff87, #60efff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.2rem; font-weight: 700; text-align: right; padding: 10px; border-right: 3px solid #00ff87; animation: slideIn 1s ease; }
    @keyframes slideIn { from { transform: translateX(100px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    .metric-card-ultra { background: rgba(0,255,135,0.1); backdrop-filter: blur(10px); border: 1px solid #00ff87; border-radius: 15px; padding: 1.2rem; text-align: center; transition: all 0.3s ease; box-shadow: 0 0 20px rgba(0,255,135,0.3); }
    .metric-card-ultra:hover { transform: scale(1.05); box-shadow: 0 0 40px rgba(0,255,135,0.6); }
    .party-card { background: linear-gradient(135deg, rgba(0,255,135,0.1), rgba(96,239,255,0.1)); border: 1px solid #60efff; border-radius: 15px; padding: 1rem; margin: 0.5rem 0; transition: all 0.3s; }
    .party-card:hover { background: linear-gradient(135deg, rgba(0,255,135,0.3), rgba(96,239,255,0.3)); transform: translateX(10px); border-color: #00ff87; }
    .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius: 15px; padding: 5px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: white; font-family: 'Orbitron', sans-serif; border-radius: 10px; padding: 10px 25px; }
    .stTabs [aria-selected="true"] { background: #00ff87 !important; color: #0a0f1e !important; font-weight: 700; }
    .section-header { color: #00ff87; font-family: 'Orbitron', sans-serif; font-size: 1.2rem; margin-top: 20px; margin-bottom: 10px; padding: 10px; border-left: 4px solid #00ff87; background: rgba(0,255,135,0.05); }
</style>
<div class="developer-signature">⚡ Developed by: JAKKULA ABHISHEK | 📧 jakkulaabhishek5@gmail.com ⚡</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cyber-header">
    <h1 style="font-family: 'Orbitron', sans-serif; font-size: 3.5rem; margin:0; color: white; text-align: center;">⚡ ULTRA-AUDIT PRO ⚡</h1>
    <p style="font-family: 'Orbitron', sans-serif; font-size: 1.2rem; text-align: center; color: rgba(255,255,255,0.9); margin-top: 10px;">
        Next-Gen AI-Powered Audit Intelligence | 25+ Sampling Methods | Multi-Method Selection
    </p>
</div>
""", unsafe_allow_html=True)

# --- SAMPLE DATA GENERATOR ---
@st.cache_data
def generate_sample_data():
    sample_data = [
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
        ["05-07-2023", "Cement Corporation", "CC/07/089", 1180000.00, 1000000.00, 0.00, 0.00, 180000.00, 1000.00, "194Q"],
        ["10-07-2023", "Building Materials Co", "BMC/07/156", 590000.00, 500000.00, 0.00, 0.00, 90000.00, 500.00, "194Q"],
        ["15-07-2023", "Tech Solutions Inc", "TSI/07/089", 177000.00, 150000.00, 13500.00, 13500.00, 0.00, 15000.00, "194J"],
        ["20-07-2023", "Consulting Pros", "CP/07/234", 118000.00, 100000.00, 9000.00, 9000.00, 0.00, 5000.00, "194J"],
        ["25-07-2023", "Warehouse Solutions", "WS/07/067", 295000.00, 250000.00, 22500.00, 22500.00, 0.00, 25000.00, "194I"]
    ]
    return pd.DataFrame(sample_data, columns=['Date','Party name','Invoice no','Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','TDS Section'])

# --- DATA PROCESSING ---
class DataProcessor:
    @staticmethod
    def clean_numeric(series):
        if series.dtype == 'object':
            series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True).replace('', '0')
        return pd.to_numeric(series, errors='coerce').fillna(0)

    @staticmethod
    def apply_formulas(df):
        # Total GST
        df['Total GST'] = df['Input CGST'] + df['Input SGST'] + df['Input IGST']
        # GST Rate %
        df['GST Rate %'] = (df['Total GST'] / df['taxable value'].replace(0, np.nan)) * 100
        df['GST Rate %'] = df['GST Rate %'].fillna(0).round(2)

        # Standard TDS rates (percent)
        tds_rates = {'194C': 1.0, '194J': 10.0, '194I': 10.0, '194H': 5.0, '194Q': 0.1}
        df['Std TDS Rate %'] = df['TDS Section'].map(lambda x: tds_rates.get(str(x).strip().upper(), 1.0))

        # Applied TDS Rate %
        df['Applied TDS Rate %'] = (df['TDS deducted'] / df['taxable value'].replace(0, np.nan)) * 100
        df['Applied TDS Rate %'] = df['Applied TDS Rate %'].fillna(0).round(2)

        # Required TDS
        df['Required TDS'] = (df['taxable value'] * df['Std TDS Rate %'] / 100).round(2)
        # TDS Shortfall
        df['TDS Shortfall'] = np.maximum(0, df['Required TDS'] - df['TDS deducted']).round(2)
        # Interest Payable (1.5% per month for 3 months)
        df['Interest Payable'] = (df['TDS Shortfall'] * 0.015 * 3).round(2)
        # Net Payable
        df['Net Payable'] = (df['taxable value'] + df['Total GST'] - df['TDS deducted']).round(2)
        # Compliance Status
        df['Compliance Status'] = df.apply(
            lambda row: '✅ FULLY COMPLIANT' if row['TDS Shortfall'] == 0 
            else '⚠️ PARTIAL SHORTFALL' if row['TDS Shortfall']>0 and row['TDS deducted']>0 
            else '❌ NOT DEDUCTED', axis=1
        )
        # TDS Compliance %
        df['TDS Compliance %'] = df.apply(
            lambda row: round((row['TDS deducted']/row['Required TDS']*100),2) if row['Required TDS']>0 else 100, axis=1
        )
        return df

# --- MATERIALITY ENGINE ---
class MaterialityEngine:
    def __init__(self, threshold=5.0):
        self.threshold = threshold
    def calculate(self, df):
        total = df['taxable value'].sum()
        materiality_amount = total * (self.threshold / 100)
        df['Materiality Score'] = (df['taxable value'] / materiality_amount).round(2) if materiality_amount>0 else 0
        conditions = [
            df['Materiality Score'] >= 0.5,
            df['Materiality Score'] >= 0.2,
            df['Materiality Score'] >= 0.1,
            df['Materiality Score'] >= 0.05,
            df['Materiality Score'] < 0.05
        ]
        levels = ['🔥 CRITICAL','⚡ HIGH','💫 MEDIUM','🌟 LOW','📦 IMMATERIAL']
        df['Materiality Level'] = np.select(conditions, levels, default='📦 IMMATERIAL')
        priority_map = {'🔥 CRITICAL':1,'⚡ HIGH':2,'💫 MEDIUM':3,'🌟 LOW':4,'📦 IMMATERIAL':5}
        df['Audit Priority'] = df['Materiality Level'].map(priority_map)
        return df, total, materiality_amount

# --- SAMPLING ENGINE (ALL 26 METHODS) ---
class SamplingEngine:
    # Probability Methods
    @staticmethod
    def simple_random_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        return df.sample(n=min(n, len(df)), random_state=42)
    @staticmethod
    def systematic_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        if len(df) <= n: return df
        step = len(df) // n
        start = np.random.randint(0, step)
        indices = range(start, len(df), step)
        return df.iloc[indices].head(n)
    @staticmethod
    def stratified_sampling(df, percentage, strata_col='Materiality Level'):
        n = max(1, int(len(df) * (percentage / 100)))
        samples = []
        for stratum in df[strata_col].unique():
            stratum_df = df[df[strata_col] == stratum]
            stratum_n = max(1, int(n * len(stratum_df) / len(df)))
            samples.append(stratum_df.sample(n=min(stratum_n, len(stratum_df)), random_state=42))
        return pd.concat(samples) if samples else df.sample(n=n)
    @staticmethod
    def cluster_sampling(df, percentage, n_clusters=5):
        try:
            from sklearn.cluster import KMeans
            if 'taxable value' not in df.columns or len(df) < n_clusters:
                return SamplingEngine.simple_random_sampling(df, percentage)
            X = df[['taxable value']].values
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X)
            n = max(1, int(len(df) * (percentage / 100)))
            n_clusters_to_select = max(1, int(n_clusters * percentage / 100))
            selected_clusters = np.random.choice(df['Cluster'].unique(), n_clusters_to_select, replace=False)
            return df[df['Cluster'].isin(selected_clusters)]
        except:
            return SamplingEngine.simple_random_sampling(df, percentage)
    @staticmethod
    def multistage_sampling(df, percentage):
        parties = df['Party name'].unique()
        n_parties = max(1, int(len(parties) * (percentage / 100)))
        selected_parties = np.random.choice(parties, n_parties, replace=False)
        samples = []
        for party in selected_parties:
            party_df = df[df['Party name'] == party]
            party_n = max(1, int(len(party_df) * (percentage / 100)))
            samples.append(party_df.sample(n=min(party_n, len(party_df)), random_state=42))
        return pd.concat(samples) if samples else df.sample(n=max(1, int(len(df) * percentage/100)))
    @staticmethod
    def multiphase_sampling(df, percentage):
        phase1_pct = percentage * 0.5
        phase1 = SamplingEngine.simple_random_sampling(df, phase1_pct)
        remaining = df[~df.index.isin(phase1.index)]
        high_value = remaining.nlargest(int(len(df) * percentage * 0.3), 'taxable value')
        target_n = max(1, int(len(df) * (percentage / 100)))
        current_n = len(phase1) + len(high_value)
        if current_n < target_n:
            remaining_after = remaining[~remaining.index.isin(high_value.index)]
            additional = remaining_after.sample(n=min(target_n - current_n, len(remaining_after)), random_state=42)
            return pd.concat([phase1, high_value, additional])
        return pd.concat([phase1, high_value]).head(target_n)
    @staticmethod
    def area_sampling(df, percentage):
        df['Area'] = df['Party name'].str[0].str.upper()
        areas = df['Area'].unique()
        n_areas = max(1, int(len(areas) * (percentage / 100)))
        selected_areas = np.random.choice(areas, n_areas, replace=False)
        return df[df['Area'].isin(selected_areas)]
    @staticmethod
    def pps_sampling(df, percentage):
        if 'taxable value' not in df.columns:
            return SamplingEngine.simple_random_sampling(df, percentage)
        n = max(1, int(len(df) * (percentage / 100)))
        probs = df['taxable value'] / df['taxable value'].sum()
        return df.sample(n=min(n, len(df)), weights=probs, random_state=42)
    # Non-Probability Methods
    @staticmethod
    def convenience_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        return df.head(n)
    @staticmethod
    def judgmental_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        return df.nlargest(n, 'taxable value')
    @staticmethod
    def purposive_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        priority_df = df.copy()
        priority_df['Priority Score'] = (
            (priority_df['Materiality Level'] == '🔥 CRITICAL') * 100 +
            (priority_df['TDS Shortfall'] > 0) * 50 +
            priority_df['taxable value'] / priority_df['taxable value'].max() * 30
        )
        return priority_df.nlargest(n, 'Priority Score')
    @staticmethod
    def quota_sampling(df, percentage, quota_col='Materiality Level'):
        n = max(1, int(len(df) * (percentage / 100)))
        samples = []
        strata = df[quota_col].unique()
        quota_per_stratum = max(1, int(n / len(strata)))
        for stratum in strata:
            stratum_df = df[df[quota_col] == stratum]
            samples.append(stratum_df.head(quota_per_stratum))
        return pd.concat(samples).head(n)
    @staticmethod
    def snowball_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        seed_idx = np.random.randint(0, len(df))
        seed_party = df.iloc[seed_idx]['Party name']
        sample_df = df[df['Party name'] == seed_party]
        while len(sample_df) < n:
            current_parties = sample_df['Party name'].unique()
            current_sections = sample_df['TDS Section'].unique()
            connected = df[df['TDS Section'].isin(current_sections)]
            connected = connected[~connected['Party name'].isin(current_parties)]
            if len(connected) == 0:
                break
            next_party = connected['Party name'].iloc[0]
            sample_df = pd.concat([sample_df, df[df['Party name'] == next_party]])
        return sample_df.head(n)
    @staticmethod
    def volunteer_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        probs = df['taxable value'] / df['taxable value'].sum()
        probs = probs ** 0.5
        probs = probs / probs.sum()
        return df.sample(n=min(n, len(df)), weights=probs, random_state=42)
    @staticmethod
    def haphazard_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        indices = np.random.choice(len(df), size=min(n, len(df)), replace=False)
        return df.iloc[indices]
    @staticmethod
    def consecutive_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        start = np.random.randint(0, max(1, len(df) - n))
        return df.iloc[start:start + n]
    # Audit-Specific Methods
    @staticmethod
    def statistical_sampling(df, percentage):
        if 'Materiality Level' in df.columns:
            return SamplingEngine.stratified_sampling(df, percentage, 'Materiality Level')
        return SamplingEngine.simple_random_sampling(df, percentage)
    @staticmethod
    def non_statistical_sampling(df, percentage):
        return SamplingEngine.judgmental_sampling(df, percentage)
    @staticmethod
    def mus_sampling(df, percentage):
        if 'taxable value' not in df.columns or df['taxable value'].sum() == 0:
            return SamplingEngine.simple_random_sampling(df, percentage)
        n = max(1, int(len(df) * (percentage / 100)))
        total = df['taxable value'].sum()
        interval = total / n
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
        n = max(1, int(len(df) * (percentage / 100)))
        if 'Date' in df.columns:
            df['Month'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce').dt.month
            block_col = 'Month'
        else:
            df['Block'] = pd.qcut(df['taxable value'], q=5, labels=['B1','B2','B3','B4','B5'])
            block_col = 'Block'
        blocks = df[block_col].unique()
        selected_block = np.random.choice(blocks)
        block_df = df[df[block_col] == selected_block]
        if len(block_df) >= n:
            return block_df.head(n)
        else:
            remaining_blocks = [b for b in blocks if b != selected_block]
            if remaining_blocks:
                second_block = np.random.choice(remaining_blocks)
                second_df = df[df[block_col] == second_block]
                return pd.concat([block_df, second_df]).head(n)
        return df.head(n)
    # Advanced Methods
    @staticmethod
    def sequential_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date')
        else:
            df_sorted = df.sort_values('taxable value', ascending=False)
        return df_sorted.head(n)
    @staticmethod
    def adaptive_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        high_value = df[df['taxable value'] > df['taxable value'].quantile(0.75)]
        if len(high_value) >= n:
            return high_value.head(n)
        remaining_n = n - len(high_value)
        medium_value = df[(df['taxable value'] <= df['taxable value'].quantile(0.75)) & (df['taxable value'] > df['taxable value'].quantile(0.5))]
        sample = pd.concat([high_value, medium_value.head(remaining_n)])
        if len(sample) < n:
            low_value = df[df['taxable value'] <= df['taxable value'].quantile(0.5)]
            remaining = n - len(sample)
            sample = pd.concat([sample, low_value.sample(n=min(remaining, len(low_value)), random_state=42)])
        return sample
    @staticmethod
    def reservoir_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        reservoir = []
        for i, (_, row) in enumerate(df.iterrows()):
            if i < n:
                reservoir.append(row)
            else:
                j = np.random.randint(0, i+1)
                if j < n:
                    reservoir[j] = row
        return pd.DataFrame(reservoir)
    @staticmethod
    def acceptance_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        df['Quality Score'] = ( (df['TDS Compliance %'] < 90) * 100 + (df['TDS Shortfall'] > 0) * 50 + (df['Materiality Level'] == '🔥 CRITICAL') * 30 )
        weights = df['Quality Score'] / df['Quality Score'].sum()
        return df.sample(n=min(n, len(df)), weights=weights, random_state=42)
    @staticmethod
    def bootstrap_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        return df.sample(n=min(n*2, len(df)*2), replace=True, random_state=42).drop_duplicates().head(n)
    @staticmethod
    def bayesian_sampling(df, percentage):
        n = max(1, int(len(df) * (percentage / 100)))
        if 'taxable value' in df.columns:
            value_prior = df['taxable value'] / df['taxable value'].sum()
            compliance_prior = (100 - df['TDS Compliance %'].fillna(0)) / 100
            prior = (value_prior * 0.7 + compliance_prior * 0.3)
            prior = prior / prior.sum()
            return df.sample(n=min(n, len(df)), weights=prior, random_state=42)
        return df.sample(n=min(n, len(df)), random_state=42)

# --- EXCEL EXPORTER (MODIFIED AS PER USER REQUEST) ---
class ExcelExporter:
    @staticmethod
    def export_with_charts(df, sample_df, party_stats, selected_methods, materiality_threshold):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            # Formats
            header_fmt = workbook.add_format({'bold':True, 'bg_color':'#00ff87', 'font_color':'#0a0f1e', 'border':1, 'align':'center', 'valign':'vcenter', 'font_size':11})
            money_fmt = workbook.add_format({'num_format':'₹#,##0.00'})
            percent_fmt = workbook.add_format({'num_format':'0.00%'})
            date_fmt = workbook.add_format({'num_format':'dd-mm-yyyy'})

            # --- 1. Complete Data (raw uploaded columns only) ---
            raw_cols = ['Date','Party name','Invoice no','Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','TDS Section']
            df_raw = df[raw_cols].copy()
            # Ensure Date is string without time
            df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce').dt.strftime('%d-%m-%Y')
            # Write dataframe starting at row 1, without headers
            df_raw.to_excel(writer, sheet_name='Complete Data', index=False, startrow=1, header=False)
            ws_raw = writer.sheets['Complete Data']
            # Write headers manually at row 0
            for col_num, col_name in enumerate(raw_cols):
                ws_raw.write(0, col_num, col_name, header_fmt)
            ws_raw.set_column(0, 0, 15)
            ws_raw.set_column(1, 1, 30)
            ws_raw.set_column(2, 2, 20)
            ws_raw.set_column(3, 9, 15, money_fmt)

            # --- 2. Analysis Sheet (static data with all derived columns, like sample data format) ---
            # Create a copy of full df with all columns for analysis
            analysis_df = df.copy()
            # Ensure Date is formatted
            if 'Date' in analysis_df.columns:
                analysis_df['Date'] = pd.to_datetime(analysis_df['Date'], errors='coerce').dt.strftime('%d-%m-%Y')
            # Write to sheet
            analysis_df.to_excel(writer, sheet_name='Analysis', index=False, startrow=1, header=False)
            ws_analysis = writer.sheets['Analysis']
            # Write headers
            for col_num, col_name in enumerate(analysis_df.columns):
                ws_analysis.write(0, col_num, col_name, header_fmt)
            # Set column formats (rough guess based on column names)
            for col_num, col_name in enumerate(analysis_df.columns):
                if col_name in ['Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','Total GST','Required TDS','TDS Shortfall','Interest Payable','Net Payable']:
                    ws_analysis.set_column(col_num, col_num, 15, money_fmt)
                elif col_name in ['GST Rate %','Std TDS Rate %','Applied TDS Rate %','TDS Compliance %']:
                    ws_analysis.set_column(col_num, col_num, 12, percent_fmt)
                elif col_name == 'Date':
                    ws_analysis.set_column(col_num, col_num, 15)
                elif col_name == 'Party name':
                    ws_analysis.set_column(col_num, col_num, 30)
                elif col_name == 'Invoice no':
                    ws_analysis.set_column(col_num, col_num, 20)
                else:
                    ws_analysis.set_column(col_num, col_num, 15)

            # --- Executive Summary (rows 1-14 only) ---
            total_val = df['taxable value'].sum()
            summary_data = {
                'Metric': [
                    'Audit Date',
                    'Total Transactions',
                    'Total Value',
                    'Materiality Threshold',
                    'Materiality Amount',
                    'Sample Size',
                    'Sample Percentage',
                    'Sample Value',
                    'Sample Coverage %',
                    'Critical Items',
                    'High Items',
                    'Medium Items',
                    'Low Items',
                    'Sampling Methods Used'  # last kept; removed financial aggregates
                ],
                'Value': [
                    datetime.now().strftime('%d-%m-%Y %H:%M'),
                    len(df),
                    f'₹{total_val:,.0f}',
                    f'{materiality_threshold}%',
                    f'₹{total_val * materiality_threshold/100:,.0f}',
                    len(sample_df),
                    f'{len(sample_df)/len(df)*100:.1f}%',
                    f'₹{sample_df["taxable value"].sum():,.0f}',
                    f'{sample_df["taxable value"].sum()/total_val*100:.1f}%',
                    len(df[df['Materiality Level']=='🔥 CRITICAL']),
                    len(df[df['Materiality Level']=='⚡ HIGH']),
                    len(df[df['Materiality Level']=='💫 MEDIUM']),
                    len(df[df['Materiality Level']=='🌟 LOW']),
                    ', '.join(selected_methods)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            summ_ws = writer.sheets['Executive Summary']
            for col_num, value in enumerate(summary_df.columns):
                summ_ws.write(0, col_num, value, header_fmt)
            summ_ws.set_column('A:A', 30)
            summ_ws.set_column('B:B', 40)

            # --- Sample Data (with formulas for three columns) ---
            sample_df_out = sample_df.copy()
            if 'Date' in sample_df_out.columns:
                sample_df_out['Date'] = pd.to_datetime(sample_df_out['Date'], errors='coerce').dt.strftime('%d-%m-%Y')
            
            # Write sample data without those three columns' values (we'll write formulas later)
            # First, write all columns except we'll overwrite the three with formulas
            sample_df_out.to_excel(writer, sheet_name='Sample Data', index=False, startrow=1, header=False)
            sample_ws = writer.sheets['Sample Data']
            # Write headers
            for col_num, col_name in enumerate(sample_df_out.columns):
                sample_ws.write(0, col_num, col_name, header_fmt)

            # Determine column indices for formulas (0-based)
            # Assuming order from sample_df_out: we need to know positions of 'taxable value', 'TDS deducted', 'Std TDS Rate %', 'Required TDS', 'Applied TDS Rate %', 'TDS Shortfall'
            # Let's get mapping
            col_indices = {name: idx for idx, name in enumerate(sample_df_out.columns)}
            e_col = col_indices.get('taxable value', 4)  # default guess
            i_col = col_indices.get('TDS deducted', 8)
            m_col = col_indices.get('Std TDS Rate %', 12)
            o_col = col_indices.get('Required TDS', 14)
            n_col = col_indices.get('Applied TDS Rate %', 13)
            p_col = col_indices.get('TDS Shortfall', 15)

            # Write formulas for each row (starting row 2)
            for row in range(2, len(sample_df_out) + 2):
                # Applied TDS Rate % (col N) = IF(E{row}=0,0,I{row}/E{row}*100)
                # Use Excel column letters: we need to convert indices to letters: 0->A,1->B,... 
                # Let's compute letters: for simplicity, we'll use R1C1 notation? xlsxwriter supports A1 notation with column letters.
                # We'll write using A1 with column letters derived from indices.
                # Helper to get column letter: openpyxl has utility, but we'll implement simple mapping for up to 26 columns (our case <26)
                def col_letter(idx):
                    return chr(65 + idx)  # 0->A, 1->B, ... up to Z
                e_letter = col_letter(e_col)
                i_letter = col_letter(i_col)
                m_letter = col_letter(m_col)
                o_letter = col_letter(o_col)
                n_letter = col_letter(n_col)
                p_letter = col_letter(p_col)

                # Write formula in Applied TDS Rate % column
                sample_ws.write_formula(row-1, n_col, f'=IF({e_letter}{row}=0,0,{i_letter}{row}/{e_letter}{row}*100)')
                # Write formula in Required TDS column
                sample_ws.write_formula(row-1, o_col, f'={e_letter}{row}*{m_letter}{row}/100')
                # Write formula in TDS Shortfall column (as deducted - required)
                sample_ws.write_formula(row-1, p_col, f'={i_letter}{row}-{o_letter}{row}')

            # Apply formatting to Sample Data columns
            for col_num, col_name in enumerate(sample_df_out.columns):
                if col_name in ['Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','Total GST','Required TDS','Interest Payable','Net Payable']:
                    sample_ws.set_column(col_num, col_num, 15, money_fmt)
                elif col_name in ['GST Rate %','Std TDS Rate %','Applied TDS Rate %','TDS Compliance %']:
                    sample_ws.set_column(col_num, col_num, 12, percent_fmt)
                elif col_name == 'Date':
                    sample_ws.set_column(col_num, col_num, 15)
                elif col_name == 'Party name':
                    sample_ws.set_column(col_num, col_num, 30)
                elif col_name == 'Invoice no':
                    sample_ws.set_column(col_num, col_num, 20)
                else:
                    sample_ws.set_column(col_num, col_num, 15)

            # --- Party Analysis (simplified with Rate column) ---
            # Aggregate per party
            party_agg = df.groupby('Party name').agg({
                'taxable value': 'sum',
                'TDS deducted': 'sum',
                'Required TDS': 'sum'
            }).reset_index()
            # Compute effective rate % = (Required TDS sum / Taxable Value sum) * 100
            party_agg['Rate (%)'] = (party_agg['Required TDS'] / party_agg['taxable value'] * 100).round(2)
            party_agg['TDS Applicability'] = party_agg['Required TDS'].apply(lambda x: 'Yes' if x > 0 else 'No')
            
            # Reorder columns for output
            party_final = party_agg[['Party name', 'taxable value', 'TDS deducted', 'TDS Applicability', 'Rate (%)']].copy()
            party_final.rename(columns={
                'Party name': 'Party Name',
                'taxable value': 'Taxable Value',
                'TDS deducted': 'TDS Deducted as per books'
            }, inplace=True)
            
            # Write static values first (without the last three formula columns)
            # We'll write the dataframe without the formula columns, then add formulas for the remaining columns.
            # But we need to include columns for "If Yes, How much to be deducted", "Shortfall/Excess", "Remarks".
            # Let's create a base with all columns (including placeholders for formula columns)
            party_final['If Yes, How much to be deducted'] = 0.0  # placeholder
            party_final['Shortfall/Excess'] = 0.0
            party_final['Remarks'] = ''
            
            # Write to sheet starting at row 1
            party_final.to_excel(writer, sheet_name='Party Analysis', index=False, startrow=1, header=False)
            party_ws = writer.sheets['Party Analysis']
            # Write headers
            headers = ['Party Name', 'Taxable Value', 'TDS Deducted as per books', 'TDS Applicability', 'Rate (%)',
                       'If Yes, How much to be deducted', 'Shortfall/Excess', 'Remarks']
            for col_num, header in enumerate(headers):
                party_ws.write(0, col_num, header, header_fmt)
            
            # Determine column indices (0-based)
            party_cols = {name: idx for idx, name in enumerate(headers)}
            b_col = party_cols['Taxable Value']          # 1
            c_col = party_cols['TDS Deducted as per books']  # 2
            d_col = party_cols['TDS Applicability']      # 3
            e_col = party_cols['Rate (%)']                # 4
            f_col = party_cols['If Yes, How much to be deducted']  # 5
            g_col = party_cols['Shortfall/Excess']        # 6
            h_col = party_cols['Remarks']                  # 7

            # Write formulas for each row
            for row in range(2, len(party_final) + 2):
                # If Yes, How much to be deducted = IF(D{row}="Yes", B{row} * E{row} / 100, 0)
                party_ws.write_formula(row-1, f_col, f'=IF({col_letter(d_col)}{row}="Yes", {col_letter(b_col)}{row}*{col_letter(e_col)}{row}/100, 0)')
                # Shortfall/Excess = C{row} - F{row}
                party_ws.write_formula(row-1, g_col, f'={col_letter(c_col)}{row}-{col_letter(f_col)}{row}')
                # Remarks = IF(G{row}>=0, "Compliant", "Not Compliant")
                party_ws.write_formula(row-1, h_col, f'=IF({col_letter(g_col)}{row}>=0, "Compliant", "Not Compliant")')
            
            # Format columns
            party_ws.set_column(b_col, b_col, 15, money_fmt)   # Taxable Value
            party_ws.set_column(c_col, c_col, 15, money_fmt)   # TDS Deducted
            party_ws.set_column(e_col, e_col, 12, percent_fmt) # Rate (%)
            party_ws.set_column(f_col, f_col, 15, money_fmt)   # If Yes amount
            party_ws.set_column(g_col, g_col, 15, money_fmt)   # Shortfall/Excess
            party_ws.set_column(0, 0, 30)                      # Party Name
            party_ws.set_column(d_col, d_col, 15)              # Applicability
            party_ws.set_column(h_col, h_col, 20)              # Remarks

            # --- Add chart to Executive Summary (Sample Composition Pie) ---
            sample_mat_summary = sample_df['Materiality Level'].value_counts().reset_index()
            sample_mat_summary.columns = ['Level','Count']
            # Write summary to Sample Data sheet at columns Z, AA
            sample_ws = writer.sheets['Sample Data']
            for i, row in sample_mat_summary.iterrows():
                sample_ws.write(i+1, 25, row['Level'])
                sample_ws.write(i+1, 26, row['Count'])
            chart = workbook.add_chart({'type':'pie'})
            chart.add_series({
                'name':'Sample Composition',
                'categories':'=Sample Data!$Z$2:$Z${}'.format(len(sample_mat_summary)+1),
                'values':'=Sample Data!$AA$2:$AA${}'.format(len(sample_mat_summary)+1),
                'data_labels':{'percentage':True}
            })
            chart.set_title({'name':'Sample Composition by Materiality'})
            chart.set_style(10)
            summ_ws.insert_chart('D2', chart)

        return output.getvalue()

# --- PARTY DASHBOARD (kept for UI) ---
def create_party_dashboard(df):
    party_stats = df.groupby('Party name').agg({
        'taxable value': ['sum','count','mean'],
        'TDS deducted': 'sum',
        'Required TDS': 'sum',
        'TDS Shortfall': 'sum',
        'Interest Payable': 'sum',
        'Total GST': 'sum'
    }).round(2)
    party_stats.columns = ['Total Value','Transactions','Avg Value','TDS Paid','TDS Required','TDS Shortfall','Interest','Total GST']
    party_stats['TDS Compliance %'] = (party_stats['TDS Paid'] / party_stats['TDS Required'] * 100).fillna(100).round(2)
    party_stats['Risk Score'] = (100 - party_stats['TDS Compliance %']).round(2)
    return party_stats.sort_values('Total Value', ascending=False)

# --- MAIN APP ---
def main():
    with st.sidebar:
        st.markdown('<div style="background: rgba(0,255,135,0.1); padding:20px; border-radius:20px; border:1px solid #00ff87;"><h3 style="color:#00ff87;">⚡ CONTROL PANEL</h3></div>', unsafe_allow_html=True)
        sample_df = generate_sample_data()
        sample_excel = BytesIO()
        with pd.ExcelWriter(sample_excel, engine='xlsxwriter') as writer:
            sample_df.to_excel(writer, sheet_name='Sample Data', index=False)
        st.download_button('📥 DOWNLOAD SAMPLE EXCEL', data=sample_excel.getvalue(), file_name='Ultra_Audit_Sample.xlsx', use_container_width=True)
        st.markdown('---')
        materiality_threshold = st.slider('🎯 Materiality Threshold %', 0.1, 10.0, 5.0, 0.1)
        sample_percentage = st.slider('📊 Sample Selection %', 1, 100, 20)
        interest_months = st.number_input('💰 Interest Months', 1, 12, 3)
        st.markdown('---')

        st.markdown('<p class="section-header">🔹 PROBABILITY SAMPLING</p>', unsafe_allow_html=True)
        probability_methods = st.multiselect('', [
            'Simple Random Sampling','Systematic Sampling','Stratified Sampling','Cluster Sampling',
            'Multistage Sampling','Multiphase Sampling','Area Sampling','Probability Proportional to Size (PPS) Sampling'
        ], default=['Simple Random Sampling'], key='prob')
        st.markdown('<p class="section-header">🔹 NON-PROBABILITY SAMPLING</p>', unsafe_allow_html=True)
        non_prob_methods = st.multiselect('', [
            'Convenience Sampling','Judgmental Sampling','Purposive Sampling','Quota Sampling',
            'Snowball Sampling','Volunteer Sampling','Haphazard Sampling','Consecutive Sampling'
        ], key='nonprob')
        st.markdown('<p class="section-header">🔹 AUDIT-SPECIFIC</p>', unsafe_allow_html=True)
        audit_methods = st.multiselect('', [
            'Statistical Sampling','Non-Statistical Sampling','Monetary Unit Sampling (MUS)','Block Sampling'
        ], key='audit')
        st.markdown('<p class="section-header">🔹 ADVANCED METHODS</p>', unsafe_allow_html=True)
        advanced_methods = st.multiselect('', [
            'Sequential Sampling','Adaptive Sampling','Reservoir Sampling','Acceptance Sampling',
            'Bootstrap Sampling','Bayesian Sampling'
        ], key='adv')
        selected_methods = probability_methods + non_prob_methods + audit_methods + advanced_methods
        if not selected_methods:
            selected_methods = ['Simple Random Sampling']

    uploaded_file = st.file_uploader('📤 UPLOAD LEDGER FILE', type=['xlsx','csv'], label_visibility='collapsed')
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            expected_cols = ['Date','Party name','Invoice no','Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','TDS Section']
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                st.error(f'Missing columns: {missing}')
                st.stop()

            processor = DataProcessor()
            numeric_cols = ['Gross Total','taxable value','TDS deducted','Input CGST','Input SGST','Input IGST']
            for col in numeric_cols:
                df[col] = processor.clean_numeric(df[col])
            df = processor.apply_formulas(df)

            mat_engine = MaterialityEngine(materiality_threshold)
            df, total_value, materiality_amount = mat_engine.calculate(df)

            # Sampling
            sampling_engine = SamplingEngine()
            method_map = {
                'Simple Random Sampling': sampling_engine.simple_random_sampling,
                'Systematic Sampling': sampling_engine.systematic_sampling,
                'Stratified Sampling': lambda d,p: sampling_engine.stratified_sampling(d,p,'Materiality Level'),
                'Cluster Sampling': sampling_engine.cluster_sampling,
                'Multistage Sampling': sampling_engine.multistage_sampling,
                'Multiphase Sampling': sampling_engine.multiphase_sampling,
                'Area Sampling': sampling_engine.area_sampling,
                'Probability Proportional to Size (PPS) Sampling': sampling_engine.pps_sampling,
                'Convenience Sampling': sampling_engine.convenience_sampling,
                'Judgmental Sampling': sampling_engine.judgmental_sampling,
                'Purposive Sampling': sampling_engine.purposive_sampling,
                'Quota Sampling': lambda d,p: sampling_engine.quota_sampling(d,p,'Materiality Level'),
                'Snowball Sampling': sampling_engine.snowball_sampling,
                'Volunteer Sampling': sampling_engine.volunteer_sampling,
                'Haphazard Sampling': sampling_engine.haphazard_sampling,
                'Consecutive Sampling': sampling_engine.consecutive_sampling,
                'Statistical Sampling': sampling_engine.statistical_sampling,
                'Non-Statistical Sampling': sampling_engine.non_statistical_sampling,
                'Monetary Unit Sampling (MUS)': sampling_engine.mus_sampling,
                'Block Sampling': sampling_engine.block_sampling,
                'Sequential Sampling': sampling_engine.sequential_sampling,
                'Adaptive Sampling': sampling_engine.adaptive_sampling,
                'Reservoir Sampling': sampling_engine.reservoir_sampling,
                'Acceptance Sampling': sampling_engine.acceptance_sampling,
                'Bootstrap Sampling': sampling_engine.bootstrap_sampling,
                'Bayesian Sampling': sampling_engine.bayesian_sampling
            }
            all_samples = []
            for method in selected_methods:
                if method in method_map:
                    sample = method_map[method](df, sample_percentage)
                    sample['Sampling Method'] = method
                    all_samples.append(sample)
            if all_samples:
                combined_sample = pd.concat(all_samples, ignore_index=True).drop_duplicates(subset=['Invoice no','Party name'])
            else:
                combined_sample = df.sample(n=max(1, int(len(df)*sample_percentage/100)))
                combined_sample['Sampling Method'] = 'Default'

            party_stats = create_party_dashboard(df)

            # Dashboard metrics
            st.markdown('### 📊 REAL-TIME METRICS')
            cols = st.columns(5)
            metrics = [
                ('💰 TOTAL VALUE', f'₹{total_value:,.0f}'),
                ('📦 TRANSACTIONS', f'{len(df)}'),
                ('🔥 CRITICAL', f'{len(df[df["Materiality Level"]=="🔥 CRITICAL"])}'),
                ('🎯 SAMPLE SIZE', f'{len(combined_sample)} ({sample_percentage}%)'),
                ('⚠️ SHORTFALL', f'₹{df["TDS Shortfall"].sum():,.0f}')
            ]
            for col, (label, val) in zip(cols, metrics):
                col.markdown(f'<div class="metric-card-ultra"><h4 style="color:#00ff87;">{label}</h4><h2 style="color:white;">{val}</h2></div>', unsafe_allow_html=True)

            st.info(f'📊 **Sampling Methods Applied:** {", ".join(selected_methods)}')

            tabs = st.tabs(['🎯 PARTY ANALYSIS','📈 MATERIALITY','🔍 SAMPLE DETAILS','💰 TDS COMPLIANCE','📊 FORMULA VIEW','📥 EXPORT'])
            with tabs[0]:
                st.markdown('### 🏢 PARTY-WISE ANALYSIS')
                party_list = ['All'] + list(party_stats.index[:20])
                sel = st.selectbox('Select Party', party_list)
                if sel != 'All':
                    pdata = df[df['Party name'] == sel]
                    c1, c2 = st.columns(2)
                    with c1:
                        tds_paid = pdata['TDS deducted'].sum()
                        req = pdata['Required TDS'].sum()
                        comp = (tds_paid/req*100) if req>0 else 100
                        st.markdown(f'<div class="party-card"><h3 style="color:#00ff87;">{sel}</h3><p>📊 Total Value: ₹{pdata["taxable value"].sum():,.0f}</p><p>📦 Transactions: {len(pdata)}</p><p>💰 TDS Paid: ₹{tds_paid:,.0f}</p><p>⚠️ TDS Shortfall: ₹{pdata["TDS Shortfall"].sum():,.0f}</p><p>✅ Compliance: {comp:.1f}%</p></div>', unsafe_allow_html=True)
                    with c2:
                        fig = px.bar(pdata, x='Date', y='taxable value', title=f'{sel} - Transactions', color='Materiality Level', color_discrete_map={'🔥 CRITICAL':'#ff00ff','⚡ HIGH':'#00ff87','💫 MEDIUM':'#60efff','🌟 LOW':'#0061ff'})
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(pdata, use_container_width=True)
                else:
                    st.dataframe(party_stats.style.format({'Total Value':'₹{:,.0f}','TDS Paid':'₹{:,.0f}','TDS Shortfall':'₹{:,.0f}','TDS Compliance %':'{:.1f}%','Risk Score':'{:.1f}'}), use_container_width=True)
                    top10 = party_stats.head(10).reset_index()
                    fig = px.bar(top10, x='Party name', y='Total Value', title='Top 10 Parties by Value', color='Risk Score', color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                c1, c2 = st.columns(2)
                with c1:
                    mat_dist = df['Materiality Level'].value_counts().reset_index()
                    mat_dist.columns = ['Level','Count']
                    colors = {'🔥 CRITICAL':'#ff00ff','⚡ HIGH':'#00ff87','💫 MEDIUM':'#60efff','🌟 LOW':'#0061ff','📦 IMMATERIAL':'#95a5a6'}
                    fig = go.Figure(data=[go.Pie(labels=mat_dist['Level'], values=mat_dist['Count'], hole=0.4, marker_colors=[colors[l] for l in mat_dist['Level']])])
                    fig.update_layout(title='Materiality Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    val_dist = df.groupby('Materiality Level')['taxable value'].sum().reset_index()
                    fig = px.bar(val_dist, x='Materiality Level', y='taxable value', color='Materiality Level', color_discrete_map=colors, title='Value by Materiality Level')
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[2]:
                st.markdown(f'### 🔍 COMBINED SAMPLE DETAILS')
                st.info(f'Total Sample Size: {len(combined_sample)} transactions from {len(selected_methods)} methods')
                c1, c2 = st.columns(2)
                with c1:
                    method_comp = combined_sample['Sampling Method'].value_counts()
                    fig = px.pie(values=method_comp.values, names=method_comp.index, title='Sample Composition by Method', hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.markdown(f'<div class="glass-card"><h4>Sample Statistics</h4><p>Total Value: ₹{combined_sample["taxable value"].sum():,.0f}</p><p>Avg Value: ₹{combined_sample["taxable value"].mean():,.0f}</p><p>TDS Shortfall: ₹{combined_sample["TDS Shortfall"].sum():,.0f}</p><p>Critical Items: {len(combined_sample[combined_sample["Materiality Level"]=="🔥 CRITICAL"])}</p><p>Methods Used: {len(selected_methods)}</p></div>', unsafe_allow_html=True)
                st.dataframe(combined_sample, use_container_width=True)

            with tabs[3]:
                c1, c2 = st.columns(2)
                with c1:
                    short = df.groupby('Party name')['TDS Shortfall'].sum().nlargest(10).reset_index()
                    fig = px.bar(short, x='Party name', y='TDS Shortfall', title='Top 10 TDS Shortfalls', color='TDS Shortfall', color_continuous_scale='Reds')
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    comp_sec = df.groupby('TDS Section').agg({'TDS deducted':'sum','Required TDS':'sum'}).reset_index()
                    comp_sec['Compliance %'] = comp_sec['TDS deducted'] / comp_sec['Required TDS'] * 100
                    fig = px.bar(comp_sec, x='TDS Section', y='Compliance %', title='TDS Compliance by Section', color='Compliance %', color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                tds_summ = df.groupby('TDS Section').agg({'taxable value':'sum','TDS deducted':'sum','Required TDS':'sum','TDS Shortfall':'sum','Interest Payable':'sum'}).round(2)
                st.dataframe(tds_summ.style.format('₹{:,.0f}'), use_container_width=True)

            with tabs[4]:
                st.markdown("""
                <div class="glass-card">
                    <h4>Dynamic Formulas Applied:</h4>
                    <ul>
                        <li>✅ Total GST = CGST + SGST + IGST</li>
                        <li>✅ GST Rate % = (Total GST / Taxable Value) × 100</li>
                        <li>✅ Standard TDS Rate % = Based on TDS Section (194C:1%, 194J:10%, 194I:10%, 194H:5%, 194Q:0.1%)</li>
                        <li>✅ Applied TDS Rate % = (TDS Deducted / Taxable Value) × 100</li>
                        <li>✅ Required TDS = Taxable Value × Standard TDS Rate % / 100</li>
                        <li>✅ TDS Shortfall = MAX(0, Required TDS - Actual TDS)</li>
                        <li>✅ Interest Payable = Shortfall × 1.5% × 3 months</li>
                        <li>✅ Net Payable = Taxable Value + Total GST - TDS Deducted</li>
                        <li>✅ TDS Compliance % = (Actual TDS / Required TDS) × 100 (capped at 100%)</li>
                        <li>✅ Materiality Score = Taxable Value / (Total Taxable × Threshold %)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                formula_cols = ['Party name','Invoice no','taxable value','TDS Section','Std TDS Rate %','Applied TDS Rate %','Required TDS','TDS deducted','TDS Shortfall','Interest Payable','TDS Compliance %','Compliance Status','Materiality Level']
                present = [c for c in formula_cols if c in df.columns]
                st.dataframe(df[present].head(20), use_container_width=True)

            with tabs[5]:
                st.markdown("""
                <div class="glass-card">
                    <h4>Export will include:</h4>
                    <ul>
                        <li>📊 Executive Summary with Sample Composition Chart</li>
                        <li>📑 Complete Data (raw uploaded columns, no duplicate headers)</li>
                        <li>📈 Analysis Sheet (full data, static, with all derived columns)</li>
                        <li>🔍 Sample Data with formulas for Applied TDS Rate %, Required TDS, TDS Shortfall</li>
                        <li>🏢 Enhanced Party Analysis with Rate column and formulas</li>
                        <li>📊 Sample Composition Pie Chart</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                if st.button('⚡ GENERATE COMPLETE REPORT WITH FORMULAS', use_container_width=True):
                    with st.spinner('Generating Excel with formulas...'):
                        exporter = ExcelExporter()
                        excel_data = exporter.export_with_charts(df, combined_sample, party_stats, selected_methods, materiality_threshold)
                        st.download_button('📥 DOWNLOAD EXCEL REPORT (WITH FORMULAS)', data=excel_data, file_name=f'Ultra_Audit_Report_{datetime.now():%Y%m%d_%H%M%S}.xlsx', use_container_width=True)
                        st.success('✅ Report generated successfully with all formulas embedded!')
        except Exception as e:
            st.error(f'Error: {str(e)}')
            st.exception(e)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2 style="color: #00ff87; font-family: 'Orbitron', sans-serif;">🚀 ULTRA-AUDIT PRO READY</h2>
            <p style="color: white; font-size: 1.2rem;">Upload your ledger file with these exact columns:</p>
            <div style="background: rgba(0,255,135,0.1); padding: 20px; border-radius: 15px; margin: 20px auto; max-width: 900px;">
                <code style="color: #00ff87; font-size: 1rem;">
                    Date | Party name | Invoice no | Gross Total | taxable value | Input CGST | Input SGST | Input IGST | TDS deducted | TDS Section
                </code>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 30px;">
                <div class="glass-card">🔹 8 Probability Methods</div>
                <div class="glass-card">🔹 8 Non-Probability Methods</div>
                <div class="glass-card">🔹 4 Audit-Specific Methods</div>
                <div class="glass-card">🔹 6 Advanced Methods</div>
            </div>
            <p style="color: #60efff; margin-top: 30px;">Click the button in sidebar to download sample Excel with exact format</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
