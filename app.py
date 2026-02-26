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
                    'Sampling Methods Used'
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

            # --- Sample Data (with formulas) ---
            sample_df_out = sample_df.copy()
            if 'Date' in sample_df_out.columns:
                sample_df_out['Date'] = pd.to_datetime(sample_df_out['Date'], errors='coerce').dt.strftime('%d-%m-%Y')
            
            # Write sample data without formula columns (we'll write formulas later)
            sample_df_out.to_excel(writer, sheet_name='Sample Data', index=False, startrow=1, header=False)
            sample_ws = writer.sheets['Sample Data']
            # Write headers
            for col_num, col_name in enumerate(sample_df_out.columns):
                sample_ws.write(0, col_num, col_name, header_fmt)

            # Helper to get column letter
            def col_letter(idx):
                return chr(65 + idx)

            # Map column names to indices
            col_indices = {name: idx for idx, name in enumerate(sample_df_out.columns)}
            # Expected columns (adjust if names differ)
            taxable_col = col_indices.get('taxable value', 4)
            tds_deducted_col = col_indices.get('TDS deducted', 8)
            tds_section_col = col_indices.get('TDS Section', 9)
            total_gst_col = col_indices.get('Total GST', None)
            std_tds_rate_col = col_indices.get('Std TDS Rate %', None)
            applied_tds_rate_col = col_indices.get('Applied TDS Rate %', None)
            required_tds_col = col_indices.get('Required TDS', None)
            tds_shortfall_col = col_indices.get('TDS Shortfall', None)
            tds_compliance_col = col_indices.get('TDS Compliance %', None)

            # Write formulas for each row (starting row 2)
            for row in range(2, len(sample_df_out) + 2):
                # GST Rate % (if column exists) = Total GST / taxable value
                if total_gst_col is not None:
                    gst_rate_formula = f'={col_letter(total_gst_col)}{row}/{col_letter(taxable_col)}{row}'
                    sample_ws.write_formula(row-1, total_gst_col, gst_rate_formula)
                
                # Std TDS Rate % (decimal) based on TDS Section
                if std_tds_rate_col is not None:
                    std_formula = ('=IF(' + col_letter(tds_section_col) + str(row) + '="194C",0.01,' +
                                   'IF(' + col_letter(tds_section_col) + str(row) + '="194J",0.10,' +
                                   'IF(' + col_letter(tds_section_col) + str(row) + '="194I",0.10,' +
                                   'IF(' + col_letter(tds_section_col) + str(row) + '="194H",0.05,' +
                                   'IF(' + col_letter(tds_section_col) + str(row) + '="194Q",0.001,0.01)))))')
                    sample_ws.write_formula(row-1, std_tds_rate_col, std_formula)
                
                # Applied TDS Rate % = TDS deducted / taxable value
                if applied_tds_rate_col is not None:
                    applied_formula = f'={col_letter(tds_deducted_col)}{row}/{col_letter(taxable_col)}{row}'
                    sample_ws.write_formula(row-1, applied_tds_rate_col, applied_formula)
                
                # Required TDS = taxable value * Std TDS Rate % (using Std TDS Rate % column)
                if required_tds_col is not None and std_tds_rate_col is not None:
                    required_formula = f'={col_letter(taxable_col)}{row}*{col_letter(std_tds_rate_col)}{row}'
                    sample_ws.write_formula(row-1, required_tds_col, required_formula)
                
                # TDS Shortfall = Required TDS - TDS deducted (but can be negative, we use MAX(0, ...)? User wants deducted - required? They said "TDS Shortfall I2-O2" which is deducted - required, giving negative if shortfall. We'll keep as is, but note that earlier we had MAX(0, required - deducted). User now wants deducted - required, which could be negative. We'll follow user: I2-O2 (TDS deducted - Required TDS). That shows excess as positive, shortfall as negative.
                if tds_shortfall_col is not None and required_tds_col is not None:
                    shortfall_formula = f'={col_letter(tds_deducted_col)}{row}-{col_letter(required_tds_col)}{row}'
                    sample_ws.write_formula(row-1, tds_shortfall_col, shortfall_formula)
                
                # TDS Compliance % = TDS deducted / Required TDS
                if tds_compliance_col is not None and required_tds_col is not None:
                    compliance_formula = f'=IF({col_letter(required_tds_col)}{row}=0,1,{col_letter(tds_deducted_col)}{row}/{col_letter(required_tds_col)}{row})'
                    sample_ws.write_formula(row-1, tds_compliance_col, compliance_formula)

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

            # --- Party Analysis (with SUMIF for Taxable Value and TDS Deducted) ---
            # Get unique parties and aggregated data
            party_agg = df.groupby('Party name').agg({
                'taxable value': 'sum',
                'TDS deducted': 'sum',
                'Required TDS': 'sum'
            }).reset_index()
            # Compute effective rate as decimal (Required TDS / Taxable Value)
            party_agg['Rate'] = (party_agg['Required TDS'] / party_agg['taxable value']).fillna(0)
            party_agg['TDS Applicability'] = party_agg['Required TDS'].apply(lambda x: 'Yes' if x > 0 else 'No')
            # Compute derived columns
            party_agg['If Yes, How much to be deducted'] = party_agg.apply(
                lambda row: row['taxable value'] * row['Rate'] if row['TDS Applicability'] == 'Yes' else 0, axis=1
            )
            party_agg['Shortfall/Excess'] = party_agg['TDS deducted'] - party_agg['If Yes, How much to be deducted']
            party_agg['Remarks'] = party_agg['Shortfall/Excess'].apply(lambda x: 'Compliant' if x >= 0 else 'Not Compliant')
            
            # Prepare final dataframe for writing static values (except B and C which will be formulas)
            party_final = party_agg[['Party name', 'taxable value', 'TDS deducted', 'TDS Applicability', 'Rate',
                                      'If Yes, How much to be deducted', 'Shortfall/Excess', 'Remarks']].copy()
            party_final.rename(columns={
                'Party name': 'Party Name',
                'taxable value': 'Taxable Value (static)',
                'TDS deducted': 'TDS Deducted (static)',
                'Rate': 'Rate (decimal)'
            }, inplace=True)
            
            # Write static data starting at row 1 (without headers)
            party_final.to_excel(writer, sheet_name='Party Analysis', index=False, startrow=1, header=False)
            party_ws = writer.sheets['Party Analysis']
            # Write headers
            headers = ['Party Name', 'Taxable Value', 'TDS Deducted as per books', 'TDS Applicability', 'Rate (decimal)',
                       'If Yes, How much to be deducted', 'Shortfall/Excess', 'Remarks']
            for col_num, header in enumerate(headers):
                party_ws.write(0, col_num, header, header_fmt)
            
            # Now overwrite columns B and C with SUMIF formulas
            # Column indices: 0 = Party Name, 1 = Taxable Value, 2 = TDS Deducted, ...
            for row in range(2, len(party_final) + 2):
                party_name_cell = f'A{row}'
                # Taxable Value (col B) = SUMIF('Complete Data'!B:B, A2, 'Complete Data'!E:E)
                sumif_taxable = f'=SUMIF(\'Complete Data\'!B:B, {party_name_cell}, \'Complete Data\'!E:E)'
                party_ws.write_formula(row-1, 1, sumif_taxable)
                # TDS Deducted (col C) = SUMIF('Complete Data'!B:B, A2, 'Complete Data'!I:I)
                sumif_deducted = f'=SUMIF(\'Complete Data\'!B:B, {party_name_cell}, \'Complete Data\'!I:I)'
                party_ws.write_formula(row-1, 2, sumif_deducted)
            
            # Format columns
            party_ws.set_column(1, 1, 15, money_fmt)   # Taxable Value
            party_ws.set_column(2, 2, 15, money_fmt)   # TDS Deducted
            party_ws.set_column(4, 4, 12, percent_fmt) # Rate (decimal) as percentage? Actually it's decimal, format as 0.00% would show 1% as 1.00%, which is wrong. Use general number.
            party_ws.set_column(4, 4, 12)              # Rate as decimal, no special format
            party_ws.set_column(5, 5, 15, money_fmt)   # If Yes amount
            party_ws.set_column(6, 6, 15, money_fmt)   # Shortfall/Excess
            party_ws.set_column(0, 0, 30)              # Party Name
            party_ws.set_column(3, 3, 15)              # TDS Applicability
            party_ws.set_column(7, 7, 20)              # Remarks

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
                        <li>🔍 Sample Data with formulas for GST Rate %, Std TDS Rate %, Applied TDS Rate %, Required TDS, TDS Shortfall, TDS Compliance %</li>
                        <li>🏢 Party Analysis with SUMIF for Taxable Value and TDS Deducted, plus derived columns</li>
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
