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
    page_title="Samples Made Easy | by Jakkula Abhishek",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ULTRA CSS (unchanged) ---
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
        Next-Gen AI-Powered Audit Intelligence | 25+ Sampling Methods | Multi-Method Selection | TDS Check with Limits
    </p>
</div>
""", unsafe_allow_html=True)

# --- SAMPLE DATA GENERATOR ---
@st.cache_data
def generate_sample_data():
    sample_data = [
        ["01-04-2023", "AAAA", "PEW/001/23-24", 135405.00, 114750.00, 10327.50, 10327.50, 0.00, 1147.50, "194C"],
        ["05-04-2023", "BBBB", "ST/23-24/468", 78479.44, 66508.00, 5985.72, 5985.72, 0.00, 665.08, "194C"],
        ["10-04-2023", "CCCC", "533", 25250.14, 21322.16, 1963.99, 1963.99, 0.00, 213.22, "194C"],
        ["15-04-2023", "DDDD", "6112303938", 10664.84, 9038.00, 0.00, 0.00, 1626.84, 90.38, "194C"],
        ["20-04-2023", "EEEE", "SAI/787/23-24", 67021.01, 5605.00, 512.55, 512.55, 0.00, 56.05, "194C"],
    ]
    return pd.DataFrame(sample_data, columns=['Date','Party name','Invoice no','Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','TDS Section'])

# --- TDS RATES WITH LIMITS (used for calculations and Excel export) ---
TDS_RATES_DATA = [
    ['Section', 'Explanation', 'Rate', 'Limit'],
    ['192A', 'Premature withdrawal of Employees Provident Fund (before 5 years service)', 0.10, 0],
    ['193', 'Interest on securities such as debentures, bonds, government securities', 0.10, 0],
    ['194', 'Dividend paid by domestic company to resident shareholder', 0.10, 0],
    ['194A', 'Interest other than securities (bank interest, FD interest, loan interest)', 0.10, 0],
    ['194B', 'Winnings from lottery, crossword puzzles, game shows', 0.30, 0],
    ['194BA', 'Net winnings from online games', 0.30, 0],
    ['194BB', 'Winnings from horse races', 0.30, 0],
    ['194C', 'Payment to individual or HUF contractor for carrying out any work contract', 0.01, 30000],
    ['194C', 'Payment to firm/company/LLP/others for carrying out any work contract', 0.02, 30000],
    ['194C', 'Payment to transport contractor owning ≤10 goods carriages and furnishing PAN declaration', 0.00, 0],
    ['194D', 'Insurance commission paid to insurance agents', 0.02, 0],
    ['194DA', 'Taxable portion of life insurance policy maturity proceeds', 0.02, 0],
    ['194EE', 'Payment from National Savings Scheme deposits', 0.10, 0],
    ['194G', 'Commission on sale of lottery tickets', 0.02, 0],
    ['194H', 'Commission or brokerage (excluding insurance commission)', 0.02, 0],
    ['194I', 'Rent for plant, machinery, land, building, furniture', 0.02, 15000],  # per month proxy
    ['194IA', 'Purchase of immovable property (other than agricultural land)', 0.10, 0],
    ['194IB', 'Rent exceeding ₹50,000 per month paid by individual/HUF not liable for audit', 0.02, 50000],
    ['194J(a)', 'Fees for technical services (non-professional services)', 0.02, 30000],
    ['194J(b)', 'Fees for professional services such as legal, medical, CA, consultancy', 0.10, 30000],
    ['194LA', 'Compensation on compulsory acquisition of immovable property', 0.10, 0],
    ['194M', 'Contract payment/commission/professional fees by individual/HUF not liable to audit', 0.02, 0],
    ['194N', 'Cash withdrawal exceeding specified limits from bank/cooperative bank/post office', 0.02, 0],
    ['194O', 'E-commerce operator facilitating sale of goods/services through digital platform', 0.001, 0],
    ['194P', 'Specified senior citizen aged 75+ (specified bank responsible for tax computation)', 0.00, 0],
    ['194Q', 'Purchase of goods exceeding specified turnover and threshold limit', 0.001, 0],
    ['194R', 'Benefit or perquisite arising from business or profession', 0.10, 0],
    ['194S', 'Transfer of virtual digital assets (crypto etc.)', 0.01, 0],
    ['194T', 'Payment of salary, remuneration, commission, bonus or interest to partner by firm', 0.10, 0]
]

# Build a dictionary for quick lookup in Python
tds_rate_dict = {}
tds_limit_dict = {}
for row in TDS_RATES_DATA[1:]:
    section = row[0]
    rate = row[2]
    limit = row[3]
    tds_rate_dict[section] = rate
    tds_limit_dict[section] = limit

# --- DATA PROCESSING (with TDS limit check) ---
class DataProcessor:
    @staticmethod
    def clean_numeric(series):
        if series.dtype == 'object':
            series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True).replace('', '0')
        return pd.to_numeric(series, errors='coerce').fillna(0)

    @staticmethod
    def apply_formulas(df, interest_months=3):
        # Total GST
        df['Total GST'] = df['Input CGST'] + df['Input SGST'] + df['Input IGST']
        # GST Rate %
        df['GST Rate %'] = (df['Total GST'] / df['taxable value'].replace(0, np.nan)) * 100
        df['GST Rate %'] = df['GST Rate %'].fillna(0).round(2)

        # Standard TDS rates and limits from dictionary
        df['Std TDS Rate %'] = df['TDS Section'].map(lambda x: tds_rate_dict.get(str(x).strip().upper(), 0.01) * 100)  # store as percentage
        df['TDS Limit'] = df['TDS Section'].map(lambda x: tds_limit_dict.get(str(x).strip().upper(), 0))

        # TDS Applicable flag (True if taxable value > limit)
        df['TDS Applicable'] = df['taxable value'] > df['TDS Limit']

        # Applied TDS Rate %
        df['Applied TDS Rate %'] = (df['TDS deducted'] / df['taxable value'].replace(0, np.nan)) * 100
        df['Applied TDS Rate %'] = df['Applied TDS Rate %'].fillna(0).round(2)

        # Required TDS: if applicable, taxable value * rate / 100; else 0
        df['Required TDS'] = np.where(df['TDS Applicable'], 
                                      (df['taxable value'] * df['Std TDS Rate %'] / 100).round(2), 
                                      0.0)

        # TDS Shortfall: actual - required (can be negative if excess deducted)
        df['TDS Shortfall'] = (df['TDS deducted'] - df['Required TDS']).round(2)

        # Interest Payable (1.5% per month for given months, only if shortfall positive)
        df['Interest Payable'] = np.maximum(0, df['TDS Shortfall']) * 0.015 * interest_months
        df['Interest Payable'] = df['Interest Payable'].round(2)

        # Net Payable
        df['Net Payable'] = (df['taxable value'] + df['Total GST'] - df['TDS deducted']).round(2)

        # Compliance Status
        conditions = [
            (df['TDS Shortfall'] == 0),
            (df['TDS Shortfall'] > 0) & (df['TDS deducted'] > 0),
            (df['TDS deducted'] == 0) & (df['TDS Applicable'])
        ]
        choices = ['✅ FULLY COMPLIANT', '⚠️ PARTIAL SHORTFALL', '❌ NOT DEDUCTED']
        df['Compliance Status'] = np.select(conditions, choices, default='✅ FULLY COMPLIANT')

        # TDS Compliance % (capped at 100% for display, but can be >100% if excess deducted)
        df['TDS Compliance %'] = np.where(df['Required TDS'] > 0,
                                          (df['TDS deducted'] / df['Required TDS'] * 100).clip(upper=100).round(2),
                                          100.0)
        return df

# --- MATERIALITY ENGINE (unchanged) ---
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

# --- SAMPLING ENGINE (unchanged) ---
class SamplingEngine:
    # ... (all methods remain exactly as before) ...
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

# --- UPDATED EXCEL EXPORTER (with TDS limits, Party Analysis changes, and new chart) ---
class ExcelExporter:
    @staticmethod
    def export_with_charts(df, sample_df, party_stats, selected_methods, materiality_threshold, interest_months=3):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            # Formats
            header_fmt = workbook.add_format({'bold':True, 'bg_color':'#00ff87', 'font_color':'#0a0f1e', 'border':1, 'align':'center', 'valign':'vcenter', 'font_size':11})
            money_fmt = workbook.add_format({'num_format':'₹#,##0.00'})
            percent_fmt = workbook.add_format({'num_format':'0.00%'})
            comma_fmt = workbook.add_format({'num_format':'#,##0'})
            comma2_fmt = workbook.add_format({'num_format':'#,##0.00'})
            date_fmt = workbook.add_format({'num_format':'dd-mm-yyyy'})

            # --- 0. TDS Rates Sheet (with Limit column) ---
            tds_rates_df = pd.DataFrame(TDS_RATES_DATA[1:], columns=TDS_RATES_DATA[0])
            tds_rates_df.to_excel(writer, sheet_name='TDS Rates', index=False, startrow=1, header=False)
            tds_ws = writer.sheets['TDS Rates']
            for col_num, col_name in enumerate(tds_rates_df.columns):
                tds_ws.write(0, col_num, col_name, header_fmt)
            tds_ws.set_column('A:A', 25)   # Section
            tds_ws.set_column('B:B', 60)   # Explanation
            tds_ws.set_column('C:C', 15, percent_fmt)   # Rate
            tds_ws.set_column('D:D', 15, comma_fmt)     # Limit

            # --- 1. Complete Data (raw uploaded columns only) ---
            raw_cols = ['Date','Party name','Invoice no','Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','TDS Section']
            df_raw = df[raw_cols].copy()
            df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce').dt.strftime('%d-%m-%Y')
            df_raw.to_excel(writer, sheet_name='Complete Data', index=False, startrow=2, header=False)
            ws_raw = writer.sheets['Complete Data']
            for col_num, col_name in enumerate(raw_cols):
                ws_raw.write(1, col_num, col_name, header_fmt)
            for col_num, col_name in enumerate(raw_cols):
                if col_name in ['Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted']:
                    col_letter = chr(65 + col_num)
                    formula = f'=SUM({col_letter}3:{col_letter}50000)'
                    ws_raw.write(0, col_num, formula, money_fmt)
            ws_raw.set_column(0, 0, 15)   # Date
            ws_raw.set_column(1, 1, 30)   # Party name
            ws_raw.set_column(2, 2, 20)   # Invoice no
            ws_raw.set_column(3, 9, 15, money_fmt)

            # --- 2. Executive Summary (updated with new chart at A20:B29) ---
            ws_summ = workbook.add_worksheet('Executive Summary')
            ws_summ.write(0, 0, 'Metric', header_fmt)
            ws_summ.write(0, 1, 'Value', header_fmt)

            labels = [
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
                'Sampling Methods Used'
            ]
            for i, label in enumerate(labels):
                ws_summ.write(i+1, 0, label)

            ws_summ.write(1, 1, datetime.now().strftime('%d-%m-%Y %H:%M'))
            ws_summ.write_formula(2, 1, '=COUNTA(\'Complete Data\'!A:A)-2', comma_fmt)
            ws_summ.write_formula(3, 1, '=SUM(\'Complete Data\'!E:E)', money_fmt)
            ws_summ.write(4, 1, materiality_threshold / 100, percent_fmt)
            ws_summ.write_formula(5, 1, '=B4*B5', money_fmt)
            ws_summ.write_formula(6, 1, '=COUNTA(\'Sample Data\'!A:A)-2', comma_fmt)
            ws_summ.write_formula(7, 1, '=B7/B3', percent_fmt)
            ws_summ.write_formula(8, 1, '=SUM(\'Sample Data\'!E:E)', money_fmt)
            ws_summ.write_formula(9, 1, '=B9/B4', percent_fmt)
            ws_summ.write(10, 1, len(df[df['Materiality Level']=='🔥 CRITICAL']), comma_fmt)
            ws_summ.write(11, 1, len(df[df['Materiality Level']=='⚡ HIGH']), comma_fmt)
            ws_summ.write(12, 1, len(df[df['Materiality Level']=='💫 MEDIUM']), comma_fmt)
            ws_summ.write(13, 1, len(df[df['Materiality Level']=='🌟 LOW']), comma_fmt)
            ws_summ.write(14, 1, ', '.join(selected_methods))
            ws_summ.write(16, 0, 'Chart Explanation:')
            ws_summ.write(16, 1, 'Bar chart shows top 10 parties by TDS deducted (from Analysis sheet).')

            # Top 10 parties by TDS deducted (from Analysis sheet, where TDS deducted reflects applicability)
            # We'll compute using Python and write static values at A20:B29
            # Use the full df (which has TDS Applicable flag) to get TDS deducted per party
            party_tds = df.groupby('Party name')['TDS deducted'].sum().nlargest(10).reset_index()
            ws_summ.write(19, 0, 'Party Name', header_fmt)
            ws_summ.write(19, 1, 'TDS Deducted', header_fmt)
            for i, row in party_tds.iterrows():
                ws_summ.write(20 + i, 0, row['Party name'])
                ws_summ.write(20 + i, 1, row['TDS deducted'], money_fmt)

            # Create bar chart referencing A20:B29
            chart = workbook.add_chart({'type':'bar'})
            chart.add_series({
                'name':'TDS Deducted',
                'categories':'=Executive Summary!$A$21:$A${}'.format(20+len(party_tds)),
                'values':'=Executive Summary!$B$21:$B${}'.format(20+len(party_tds)),
            })
            chart.set_title({'name':'Top 10 Parties by TDS Deducted'})
            chart.set_x_axis({'name':'Party'})
            chart.set_y_axis({'name':'TDS Deducted (₹)'})
            chart.set_style(11)
            ws_summ.insert_chart('D2', chart)

            ws_summ.set_column('A:A', 30)
            ws_summ.set_column('B:B', 40)

            # --- 3. Sample Data (with formulas incorporating TDS limit) ---
            sample_df_out = sample_df.copy()
            if 'Date' in sample_df_out.columns:
                sample_df_out['Date'] = pd.to_datetime(sample_df_out['Date'], errors='coerce').dt.strftime('%d-%m-%Y')
            sample_df_out.to_excel(writer, sheet_name='Sample Data', index=False, startrow=2, header=False)
            sample_ws = writer.sheets['Sample Data']
            for col_num, col_name in enumerate(sample_df_out.columns):
                sample_ws.write(1, col_num, col_name, header_fmt)

            # Write subtotals at row 0
            numeric_cols_sample = ['Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted',
                                    'Total GST','Required TDS','Interest Payable','Net Payable','TDS Shortfall']
            for col_num, col_name in enumerate(sample_df_out.columns):
                if col_name in numeric_cols_sample:
                    col_letter = chr(65 + col_num)
                    formula = f'=SUM({col_letter}3:{col_letter}50000)'
                    sample_ws.write(0, col_num, formula, money_fmt)

            # Map column indices
            col_indices = {name: idx for idx, name in enumerate(sample_df_out.columns)}
            taxable_col = col_indices.get('taxable value', 4)
            tds_deducted_col = col_indices.get('TDS deducted', 8)
            tds_section_col = col_indices.get('TDS Section', 9)
            cgst_col = col_indices.get('Input CGST', 5)
            sgst_col = col_indices.get('Input SGST', 6)
            igst_col = col_indices.get('Input IGST', 7)
            total_gst_col = col_indices.get('Total GST', None)
            std_tds_rate_col = col_indices.get('Std TDS Rate %', None)
            applied_tds_rate_col = col_indices.get('Applied TDS Rate %', None)
            required_tds_col = col_indices.get('Required TDS', None)
            tds_shortfall_col = col_indices.get('TDS Shortfall', None)
            tds_compliance_col = col_indices.get('TDS Compliance %', None)

            # Write formulas for data rows (starting at row 3, index 2)
            for row in range(2, len(sample_df_out) + 2):
                # Total GST
                if total_gst_col is not None:
                    total_gst_formula = f'=SUM({chr(65+cgst_col)}{row+1}:{chr(65+igst_col)}{row+1})'
                    sample_ws.write_formula(row, total_gst_col, total_gst_formula, money_fmt)

                # GST Rate %
                if total_gst_col is not None and taxable_col is not None:
                    gst_rate_idx = col_indices.get('GST Rate %', None)
                    if gst_rate_idx is not None:
                        sample_ws.write_formula(row, gst_rate_idx, f'={chr(65+total_gst_col)}{row+1}/{chr(65+taxable_col)}{row+1}', percent_fmt)

                # Std TDS Rate % (just rate from TDS Rates)
                if std_tds_rate_col is not None:
                    std_formula = f"=IFERROR(VLOOKUP({chr(65+tds_section_col)}{row+1},'TDS Rates'!$A$2:$C$100,3,FALSE),\"Please Enter TDS Section rate as per TDS rates sheet\")"
                    sample_ws.write_formula(row, std_tds_rate_col, std_formula, percent_fmt)

                # Applied TDS Rate %
                if applied_tds_rate_col is not None:
                    applied_formula = f'={chr(65+tds_deducted_col)}{row+1}/{chr(65+taxable_col)}{row+1}'
                    sample_ws.write_formula(row, applied_tds_rate_col, applied_formula, percent_fmt)

                # Required TDS: =IF(E3 > VLOOKUP(J3, 'TDS Rates'!$A$2:$D$100,4,FALSE), E3 * VLOOKUP(J3, 'TDS Rates'!$A$2:$C$100,3,FALSE), 0)
                if required_tds_col is not None:
                    limit_vlookup = f"VLOOKUP({chr(65+tds_section_col)}{row+1},'TDS Rates'!$A$2:$D$100,4,FALSE)"
                    rate_vlookup = f"VLOOKUP({chr(65+tds_section_col)}{row+1},'TDS Rates'!$A$2:$C$100,3,FALSE)"
                    required_formula = f"=IF({chr(65+taxable_col)}{row+1}>{limit_vlookup}, {chr(65+taxable_col)}{row+1}*{rate_vlookup}, 0)"
                    sample_ws.write_formula(row, required_tds_col, required_formula, money_fmt)

                # TDS Shortfall
                if tds_shortfall_col is not None and required_tds_col is not None:
                    shortfall_formula = f'={chr(65+tds_deducted_col)}{row+1}-{chr(65+required_tds_col)}{row+1}'
                    sample_ws.write_formula(row, tds_shortfall_col, shortfall_formula, money_fmt)

                # TDS Compliance %
                if tds_compliance_col is not None and required_tds_col is not None:
                    compliance_formula = f'=IF({chr(65+required_tds_col)}{row+1}=0,1,{chr(65+tds_deducted_col)}{row+1}/{chr(65+required_tds_col)}{row+1})'
                    sample_ws.write_formula(row, tds_compliance_col, compliance_formula, percent_fmt)

            # Apply formatting to Sample Data columns
            sample_ws.set_column(0, 0, 15)   # Date
            sample_ws.set_column(1, 1, 30)   # Party name
            sample_ws.set_column(2, 2, 20)   # Invoice no
            for col_num, col_name in enumerate(sample_df_out.columns):
                if col_name in ['Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','Total GST','Required TDS','Interest Payable','Net Payable','TDS Shortfall']:
                    sample_ws.set_column(col_num, col_num, 15, money_fmt)
                elif col_name in ['GST Rate %','Std TDS Rate %','Applied TDS Rate %','TDS Compliance %']:
                    sample_ws.set_column(col_num, col_num, 12, percent_fmt)
                else:
                    sample_ws.set_column(col_num, col_num, 15)

            # --- 4. Analysis Sheet (same formulas as Sample Data) ---
            analysis_df = df.copy()
            if 'Date' in analysis_df.columns:
                analysis_df['Date'] = pd.to_datetime(analysis_df['Date'], errors='coerce').dt.strftime('%d-%m-%Y')
            analysis_df.to_excel(writer, sheet_name='Analysis', index=False, startrow=2, header=False)
            analysis_ws = writer.sheets['Analysis']
            for col_num, col_name in enumerate(analysis_df.columns):
                analysis_ws.write(1, col_num, col_name, header_fmt)

            for col_num, col_name in enumerate(analysis_df.columns):
                if col_name in numeric_cols_sample:
                    col_letter = chr(65 + col_num)
                    formula = f'=SUM({col_letter}3:{col_letter}50000)'
                    analysis_ws.write(0, col_num, formula, money_fmt)

            a_col_indices = {name: idx for idx, name in enumerate(analysis_df.columns)}
            a_taxable = a_col_indices.get('taxable value', 4)
            a_tds_deducted = a_col_indices.get('TDS deducted', 8)
            a_tds_section = a_col_indices.get('TDS Section', 9)
            a_cgst = a_col_indices.get('Input CGST', 5)
            a_sgst = a_col_indices.get('Input SGST', 6)
            a_igst = a_col_indices.get('Input IGST', 7)
            a_total_gst = a_col_indices.get('Total GST', None)
            a_gst_rate = a_col_indices.get('GST Rate %', None)
            a_std_rate = a_col_indices.get('Std TDS Rate %', None)
            a_applied_rate = a_col_indices.get('Applied TDS Rate %', None)
            a_required = a_col_indices.get('Required TDS', None)
            a_shortfall = a_col_indices.get('TDS Shortfall', None)
            a_compliance = a_col_indices.get('TDS Compliance %', None)

            for row in range(2, len(analysis_df) + 2):
                if a_total_gst is not None:
                    analysis_ws.write_formula(row, a_total_gst,
                        f'=SUM({chr(65+a_cgst)}{row+1}:{chr(65+a_igst)}{row+1})', money_fmt)
                if a_gst_rate is not None and a_total_gst is not None:
                    analysis_ws.write_formula(row, a_gst_rate,
                        f'={chr(65+a_total_gst)}{row+1}/{chr(65+a_taxable)}{row+1}', percent_fmt)
                if a_std_rate is not None:
                    std_formula = f"=IFERROR(VLOOKUP({chr(65+a_tds_section)}{row+1},'TDS Rates'!$A$2:$C$100,3,FALSE),\"Please Enter TDS Section rate as per TDS rates sheet\")"
                    analysis_ws.write_formula(row, a_std_rate, std_formula, percent_fmt)
                if a_applied_rate is not None:
                    analysis_ws.write_formula(row, a_applied_rate,
                        f'={chr(65+a_tds_deducted)}{row+1}/{chr(65+a_taxable)}{row+1}', percent_fmt)
                if a_required is not None:
                    limit_vlookup = f"VLOOKUP({chr(65+a_tds_section)}{row+1},'TDS Rates'!$A$2:$D$100,4,FALSE)"
                    rate_vlookup = f"VLOOKUP({chr(65+a_tds_section)}{row+1},'TDS Rates'!$A$2:$C$100,3,FALSE)"
                    required_formula = f"=IF({chr(65+a_taxable)}{row+1}>{limit_vlookup}, {chr(65+a_taxable)}{row+1}*{rate_vlookup}, 0)"
                    analysis_ws.write_formula(row, a_required, required_formula, money_fmt)
                if a_shortfall is not None and a_required is not None:
                    analysis_ws.write_formula(row, a_shortfall,
                        f'={chr(65+a_tds_deducted)}{row+1}-{chr(65+a_required)}{row+1}', money_fmt)
                if a_compliance is not None and a_required is not None:
                    analysis_ws.write_formula(row, a_compliance,
                        f'=IF({chr(65+a_required)}{row+1}=0,1,{chr(65+a_tds_deducted)}{row+1}/{chr(65+a_required)}{row+1})', percent_fmt)

            analysis_ws.set_column(0, 0, 15)
            analysis_ws.set_column(1, 1, 30)
            analysis_ws.set_column(2, 2, 20)
            for col_num, col_name in enumerate(analysis_df.columns):
                if col_name in ['Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','Total GST','Required TDS','Interest Payable','Net Payable','TDS Shortfall']:
                    analysis_ws.set_column(col_num, col_num, 15, money_fmt)
                elif col_name in ['GST Rate %','Std TDS Rate %','Applied TDS Rate %','TDS Compliance %']:
                    analysis_ws.set_column(col_num, col_num, 12, percent_fmt)
                else:
                    analysis_ws.set_column(col_num, col_num, 15)

            # --- 5. Party Analysis (updated: new column "Applicable Limit", formulas reference Analysis sheet) ---
            # Compute per-party mode TDS section and its limit
            party_mode_section = df.groupby('Party name')['TDS Section'].agg(lambda x: x.mode()[0] if not x.mode().empty else '194C').to_dict()
            party_limit = {party: tds_limit_dict.get(section, 0) for party, section in party_mode_section.items()}

            # Build party summary using Analysis sheet aggregates (but we'll write static values for some columns)
            # We'll create a DataFrame with Party Name, Taxable Value (total), Applicable Limit, TDS Deducted, Required TDS, etc.
            party_agg = df.groupby('Party name').agg({
                'taxable value': 'sum',
                'TDS deducted': 'sum',
                'Required TDS': 'sum'
            }).reset_index()
            party_agg['Applicable Limit'] = party_agg['Party name'].map(party_limit)
            party_agg['Rate (decimal)'] = (party_agg['Required TDS'] / party_agg['taxable value']).fillna(0)
            party_agg['TDS Applicability'] = party_agg['Required TDS'].apply(lambda x: 'Yes' if x > 0 else 'No')
            party_agg['If Yes, How much to be deducted'] = 0.0  # will be formula
            party_agg['Shortfall/Excess'] = 0.0                 # will be formula
            party_agg['Remarks'] = ''

            party_final = party_agg[['Party name', 'taxable value', 'Applicable Limit', 'TDS deducted', 'TDS Applicability', 'Rate (decimal)',
                                      'If Yes, How much to be deducted', 'Shortfall/Excess', 'Remarks']].copy()
            party_final.rename(columns={
                'Party name': 'Party Name',
                'taxable value': 'Taxable Value',
                'TDS deducted': 'TDS Deducted'
            }, inplace=True)

            party_final.to_excel(writer, sheet_name='Party Analysis', index=False, startrow=2, header=False)
            party_ws = writer.sheets['Party Analysis']
            headers = ['Party Name', 'Taxable Value', 'Applicable Limit', 'TDS Deducted', 'TDS Applicability', 'Rate (decimal)',
                       'If Yes, How much to be deducted', 'Shortfall/Excess', 'Remarks']
            for col_num, header in enumerate(headers):
                party_ws.write(1, col_num, header, header_fmt)

            # Subtotal row at row 0
            party_ws.write(0, 0, 'Total')
            # Sum formulas for numeric columns (Taxable Value col B, Applicable Limit col C, TDS Deducted col D, If Yes col G, Shortfall col H)
            numeric_cols_party = [1, 2, 3, 6, 7]  # 0-based indices
            for col_num in numeric_cols_party:
                col_letter = chr(65 + col_num)
                formula = f'=SUM({col_letter}3:{col_letter}50000)'
                party_ws.write(0, col_num, formula, money_fmt)

            # Write formulas for data rows (starting row 3)
            # We'll use SUMIF on Analysis sheet for Taxable Value, TDS Deducted, Required TDS.
            # Get column letters for Analysis sheet: assume Taxable Value is col E, TDS Deducted col I, Required TDS col ? (let's find index)
            # We need to know the column index of 'Required TDS' in Analysis sheet. We can get from a_col_indices.
            req_tds_col_idx = a_col_indices.get('Required TDS', None)
            if req_tds_col_idx is None:
                # fallback
                req_tds_col_idx = a_col_indices.get('Required TDS', 17)  # approximate

            for row in range(2, len(party_final) + 2):
                party_name_cell = f'A{row+1}'
                # Taxable Value from Analysis sheet
                party_ws.write_formula(row, 1, f'=SUMIF(\'Analysis\'!B:B, {party_name_cell}, \'Analysis\'!E:E)', money_fmt)
                # Applicable Limit is static (already written from Python)
                # TDS Deducted from Analysis sheet
                party_ws.write_formula(row, 3, f'=SUMIF(\'Analysis\'!B:B, {party_name_cell}, \'Analysis\'!I:I)', money_fmt)
                # TDS Applicability: =IF(SUMIF('Analysis'!B:B, A3, 'Analysis'!Required_TDS_col) > 0, "Yes", "No")
                req_tds_col_letter = chr(65 + req_tds_col_idx)
                party_ws.write_formula(row, 4, f'=IF(SUMIF(\'Analysis\'!B:B, {party_name_cell}, \'Analysis\'!{req_tds_col_letter}:{req_tds_col_letter}) > 0, "Yes", "No")')
                # Rate (decimal): =SUMIF('Analysis'!B:B, A3, 'Analysis'!Required_TDS_col) / SUMIF('Analysis'!B:B, A3, 'Analysis'!E:E)
                party_ws.write_formula(row, 5, f'=IFERROR(SUMIF(\'Analysis\'!B:B, {party_name_cell}, \'Analysis\'!{req_tds_col_letter}:{req_tds_col_letter}) / SUMIF(\'Analysis\'!B:B, {party_name_cell}, \'Analysis\'!E:E), 0)', percent_fmt)
                # If Yes, How much to be deducted: =SUMIF('Analysis'!B:B, A3, 'Analysis'!Required_TDS_col)
                party_ws.write_formula(row, 6, f'=SUMIF(\'Analysis\'!B:B, {party_name_cell}, \'Analysis\'!{req_tds_col_letter}:{req_tds_col_letter})', money_fmt)
                # Shortfall/Excess: =SUMIF('Analysis'!B:B, A3, 'Analysis'!I:I) - SUMIF('Analysis'!B:B, A3, 'Analysis'!Required_TDS_col)
                party_ws.write_formula(row, 7, f'=SUMIF(\'Analysis\'!B:B, {party_name_cell}, \'Analysis\'!I:I) - SUMIF(\'Analysis\'!B:B, {party_name_cell}, \'Analysis\'!{req_tds_col_letter}:{req_tds_col_letter})', money_fmt)
                # Remarks: =IF(G3>=0, "Compliant", "Not Compliant")  // G3 is If Yes col
                party_ws.write_formula(row, 8, f'=IF(G{row+1}>=0, "Compliant", "Not Compliant")')

            party_ws.set_column(0, 0, 30)   # Party Name
            party_ws.set_column(1, 1, 15, money_fmt)   # Taxable Value
            party_ws.set_column(2, 2, 15, comma_fmt)   # Applicable Limit
            party_ws.set_column(3, 3, 15, money_fmt)   # TDS Deducted
            party_ws.set_column(4, 4, 15)              # TDS Applicability
            party_ws.set_column(5, 5, 12, percent_fmt) # Rate
            party_ws.set_column(6, 6, 15, money_fmt)   # If Yes
            party_ws.set_column(7, 7, 15, money_fmt)   # Shortfall/Excess
            party_ws.set_column(8, 8, 20)              # Remarks

            # --- Add pie chart to Executive Summary (unchanged) ---
            sample_mat_summary = sample_df['Materiality Level'].value_counts().reset_index()
            sample_mat_summary.columns = ['Level','Count']
            chart_start_row = len(sample_df_out) + 5
            for i, row in sample_mat_summary.iterrows():
                sample_ws.write(chart_start_row + i, 25, row['Level'])
                sample_ws.write(chart_start_row + i, 26, row['Count'])
            pie_chart = workbook.add_chart({'type':'pie'})
            pie_chart.add_series({
                'name':'Sample Composition',
                'categories':'=Sample Data!$Z${}:$Z${}'.format(chart_start_row+1, chart_start_row+len(sample_mat_summary)),
                'values':'=Sample Data!$AA${}:$AA${}'.format(chart_start_row+1, chart_start_row+len(sample_mat_summary)),
                'data_labels':{'percentage':True}
            })
            pie_chart.set_title({'name':'Sample Composition by Materiality'})
            pie_chart.set_style(10)
            ws_summ.insert_chart('P2', pie_chart)

        return output.getvalue()

# --- PARTY DASHBOARD (updated to use new TDS applicability) ---
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

# --- MAIN APP (updated to pass interest_months to DataProcessor) ---
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
            df = processor.apply_formulas(df, interest_months)

            mat_engine = MaterialityEngine(materiality_threshold)
            df, total_value, materiality_amount = mat_engine.calculate(df)

            # Sampling (unchanged)
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
                ('🎯 SAMPLE SIZE', f'{len(combined_sample)} ({sample_percentage}%)')
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
                    <h4>Dynamic Formulas Applied (with TDS Limits):</h4>
                    <ul>
                        <li>✅ Total GST = CGST + SGST + IGST</li>
                        <li>✅ GST Rate % = (Total GST / Taxable Value) × 100</li>
                        <li>✅ TDS Applicable = Taxable Value > Limit (from TDS Rates sheet)</li>
                        <li>✅ Standard TDS Rate % = VLOOKUP from TDS Rates sheet</li>
                        <li>✅ Required TDS = IF(TDS Applicable, Taxable Value × Rate, 0)</li>
                        <li>✅ TDS Shortfall = Actual TDS - Required TDS</li>
                        <li>✅ Interest Payable = Max(0, Shortfall) × 1.5% × months</li>
                        <li>✅ TDS Compliance % = (Actual / Required) × 100 (capped at 100%)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                formula_cols = ['Party name','Invoice no','taxable value','TDS Section','TDS Limit','TDS Applicable','Std TDS Rate %','Applied TDS Rate %','Required TDS','TDS deducted','TDS Shortfall','Interest Payable','TDS Compliance %','Compliance Status','Materiality Level']
                present = [c for c in formula_cols if c in df.columns]
                st.dataframe(df[present].head(20), use_container_width=True)

            with tabs[5]:
                st.markdown("""
                <div class="glass-card">
                    <h4>Export will include:</h4>
                    <ul>
                        <li>📊 Executive Summary with Top 10 Parties by TDS Deducted chart (A20:B29)</li>
                        <li>📑 TDS Rates sheet with Limit column</li>
                        <li>📑 Complete Data (raw uploaded columns, no duplicate headers)</li>
                        <li>🔍 Sample Data with formulas incorporating TDS limit checks</li>
                        <li>📊 Analysis Sheet (full data with same formulas)</li>
                        <li>🏢 Party Analysis with Applicable Limit column and formulas referencing Analysis sheet</li>
                        <li>➕ Subtotals row on all data sheets</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                if st.button('⚡ GENERATE COMPLETE REPORT WITH FORMULAS', use_container_width=True):
                    with st.spinner('Generating Excel with formulas...'):
                        exporter = ExcelExporter()
                        excel_data = exporter.export_with_charts(df, combined_sample, party_stats, selected_methods, materiality_threshold, interest_months)
                        st.download_button('📥 DOWNLOAD EXCEL REPORT (WITH FORMULAS)', data=excel_data, file_name=f'Ultra_Audit_Report_{datetime.now():%Y%m%d_%H%M%S}.xlsx', use_container_width=True)
                        st.success('✅ Report generated successfully with TDS limits integrated!')
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
