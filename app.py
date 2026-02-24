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

# --- 1. ADVANCED UI CONFIGURATION ---
st.set_page_config(
    page_title="Ultra-Audit Pro | Enterprise Edition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM CUSTOM CSS ---
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Header */
    .premium-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    /* Materiality Badges */
    .badge-high {
        background: linear-gradient(135deg, #f43b47 0%, #453a94 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    
    .badge-medium {
        background: linear-gradient(135deg, #f9d423 0%, #f83600 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    
    .badge-low {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    
    /* Progress Bar */
    .progress-container {
        background: #e0e0e0;
        border-radius: 10px;
        height: 10px;
        margin: 10px 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        height: 10px;
        transition: width 0.5s ease;
    }
    
    /* Custom Button */
    .custom-button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 12px 30px;
        border-radius: 25px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    
    .custom-button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Info Box */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background: #333;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ADVANCED DATA PROCESSING ENGINE ---
class DataProcessor:
    """Advanced data processing with AI-powered column detection"""
    
    @staticmethod
    def clean_numeric(series):
        """Intelligent numeric cleaning"""
        if series.dtype == 'object':
            # Remove all non-numeric characters except decimal and minus
            series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
            series = series.replace('', '0')
            series = series.replace('nan', '0')
            series = series.replace('None', '0')
        return pd.to_numeric(series, errors='coerce').fillna(0)
    
    @staticmethod
    def detect_columns(df):
        """AI-powered column detection with fuzzy matching"""
        
        # Comprehensive column mapping dictionary
        column_patterns = {
            'date': ['date', 'transaction date', 'entry date', 'dt', 'trans date', 'posting date', 'voucher date'],
            'party': ['party name', 'party', 'vendor', 'supplier', 'customer', 'name', 'party_name', 'creditor', 'debtor', 'account'],
            'invoice': ['invoice no', 'invoice', 'inv no', 'invoice number', 'inv_no', 'voucher no', 'bill no', 'reference', 'doc no'],
            'gross': ['gross total', 'gross', 'total amount', 'bill amount', 'invoice amount', 'amount', 'total', 'value'],
            'taxable': ['taxable value', 'taxable', 'taxable amount', 'value', 'txbl value', 'assessable value', 'net amount'],
            'cgst': ['input cgst', 'cgst', 'central gst', 'cgst amount'],
            'sgst': ['input sgst', 'sgst', 'state gst', 'sgst amount'],
            'igst': ['input igst', 'igst', 'integrated gst', 'igst amount'],
            'tds_deducted': ['tds deducted', 'tds', 'tax deducted', 'tds amount', 'tax', 'tds paid'],
            'tds_section': ['tds section', 'section', 'tds_sec', 'tax section', 'code']
        }
        
        mapped_columns = {}
        df_columns_lower = {col.lower().strip(): col for col in df.columns}
        
        for standard_name, variations in column_patterns.items():
            for var in variations:
                # Exact match
                if var in df_columns_lower:
                    mapped_columns[standard_name] = df_columns_lower[var]
                    break
                
                # Partial match (for cases like "TDS Section" vs "TDS_Section")
                for col_lower, col_original in df_columns_lower.items():
                    if var.replace(' ', '') in col_lower.replace(' ', ''):
                        mapped_columns[standard_name] = col_original
                        break
        
        return mapped_columns

# --- 4. MATERIALITY ANALYSIS ENGINE ---
class MaterialityEngine:
    """Advanced materiality calculation and classification"""
    
    def __init__(self, threshold_percent=5.0):
        self.threshold_percent = threshold_percent
        self.materiality_levels = {
            'Critical': {'color': '#f43b47', 'weight': 3, 'threshold': 0.5},
            'High': {'color': '#f9d423', 'weight': 2, 'threshold': 0.2},
            'Medium': {'color': '#00b09b', 'weight': 1.5, 'threshold': 0.1},
            'Low': {'color': '#667eea', 'weight': 1, 'threshold': 0.05},
            'Immaterial': {'color': '#95a5a6', 'weight': 0.5, 'threshold': 0}
        }
    
    def calculate(self, df, value_column):
        """Calculate materiality and classify transactions"""
        
        if value_column not in df.columns:
            return df, 0, 0, value_column
        
        total_value = df[value_column].sum()
        materiality_amount = total_value * (self.threshold_percent / 100)
        
        # Calculate materiality score for each transaction
        df['materiality_score'] = df[value_column] / materiality_amount
        df['materiality_level'] = 'Immaterial'
        df['materiality_weight'] = 0.5
        df['audit_priority'] = 'Low'
        
        # Classify based on materiality score
        conditions = [
            (df['materiality_score'] >= 0.5),
            (df['materiality_score'] >= 0.2),
            (df['materiality_score'] >= 0.1),
            (df['materiality_score'] >= 0.05),
            (df['materiality_score'] < 0.05)
        ]
        
        levels = ['Critical', 'High', 'Medium', 'Low', 'Immaterial']
        weights = [3, 2, 1.5, 1, 0.5]
        priorities = ['Critical', 'High', 'Medium', 'Low', 'Low']
        
        df['materiality_level'] = np.select(conditions, levels, default='Immaterial')
        df['materiality_weight'] = np.select(conditions, weights, default=0.5)
        df['audit_priority'] = np.select(conditions, priorities, default='Low')
        
        return df, total_value, materiality_amount, value_column

# --- 5. INTELLIGENT SAMPLING ENGINE ---
class SamplingEngine:
    """Advanced sampling with multiple methodologies"""
    
    @staticmethod
    def systematic_sampling(df, n, value_column):
        """Systematic sampling with random start"""
        if len(df) <= n:
            return df
        step = len(df) // n
        start = np.random.randint(0, step)
        indices = range(start, len(df), step)
        return df.iloc[indices].head(n)
    
    @staticmethod
    def stratified_sampling(df, n, strata_column, value_column):
        """Stratified sampling maintaining proportions"""
        samples = []
        for stratum in df[strata_column].unique():
            stratum_df = df[df[strata_column] == stratum]
            stratum_n = max(1, int(n * len(stratum_df) / len(df)))
            samples.append(stratum_df.sample(n=min(stratum_n, len(stratum_df))))
        return pd.concat(samples)
    
    @staticmethod
    def mus_sampling(df, n, value_column):
        """Monetary Unit Sampling"""
        if value_column not in df.columns:
            return df.sample(n=min(n, len(df)))
        
        # Calculate sampling interval
        total_value = df[value_column].sum()
        interval = total_value / n
        
        # Select samples based on cumulative value
        df_sorted = df.sort_values(value_column, ascending=False)
        df_sorted['cumulative'] = df_sorted[value_column].cumsum()
        
        samples = []
        current_sample = interval
        for _, row in df_sorted.iterrows():
            if row['cumulative'] >= current_sample and len(samples) < n:
                samples.append(row)
                current_sample += interval
        
        return pd.DataFrame(samples) if samples else df.sample(n=min(n, len(df)))
    
    @staticmethod
    def materiality_weighted_sampling(df, n, value_column):
        """Weighted sampling based on materiality"""
        if 'materiality_weight' not in df.columns:
            return df.sample(n=min(n, len(df)))
        
        # Calculate sampling probabilities based on materiality weight
        weights = df['materiality_weight'] / df['materiality_weight'].sum()
        return df.sample(n=min(n, len(df)), weights=weights)

# --- 6. TDS COMPLIANCE ENGINE ---
class TDSComplianceEngine:
    """Comprehensive TDS calculation and compliance checking"""
    
    def __init__(self):
        self.tds_rates = {
            # Section: (rate, threshold, description)
            '194C': (0.01, 30000, 'Contractors - Single Contract'),
            '194C-2': (0.01, 100000, 'Contractors - Aggregate'),
            '194J': (0.10, 30000, 'Professional Services'),
            '194I': (0.10, 180000, 'Rent - Plant/Machinery'),
            '194I-2': (0.10, 180000, 'Rent - Land/Building'),
            '194H': (0.05, 15000, 'Commission/Brokerage'),
            '194Q': (0.001, 5000000, 'TDS on Purchase of Goods'),
            '194IA': (0.01, 5000000, 'TDS on Property Purchase'),
            '194IB': (0.05, 50000, 'TDS on Rent (Individual)'),
            '194M': (0.05, 5000000, 'TDS on Contract/Commission'),
            '194N': (0.02, 10000000, 'Cash Withdrawal Limit'),
        }
        
        self.penalty_rates = {
            'interest_1_5': 0.015,  # 1.5% per month
            'interest_1': 0.01,      # 1% per month
            'penalty_200': 200,       # Section 271H
            'penalty_100': 100        # Section 271C
        }
    
    def calculate_tds(self, df):
        """Calculate TDS requirements and identify shortfalls"""
        
        results = []
        
        for _, row in df.iterrows():
            section = str(row.get('TDS Section', 'NA')).strip().upper()
            taxable = row.get('taxable value', 0)
            tds_deducted = row.get('TDS deducted', 0)
            
            # Get applicable rate
            rate_info = self.tds_rates.get(section, (0.01, 0, 'Default Rate'))
            tds_rate = rate_info[0]
            threshold = rate_info[1]
            
            # Calculate required TDS
            if taxable > threshold:
                tds_required = taxable * tds_rate
            else:
                tds_required = 0
            
            # Calculate shortfall
            shortfall = max(0, tds_required - tds_deducted)
            
            # Calculate interest (assuming 3 months delay)
            interest = shortfall * self.penalty_rates['interest_1_5'] * 3
            
            # Risk score (0-100)
            if tds_required > 0:
                compliance_ratio = tds_deducted / tds_required
                risk_score = max(0, 100 - (compliance_ratio * 100))
            else:
                risk_score = 0
            
            results.append({
                'tds_required': tds_required,
                'tds_shortfall': shortfall,
                'interest_payable': interest,
                'compliance_ratio': (tds_deducted / tds_required * 100) if tds_required > 0 else 100,
                'risk_score': risk_score,
                'penalty_applicable': shortfall > 0
            })
        
        return pd.DataFrame(results)

# --- 7. REPORT GENERATOR ---
class ReportGenerator:
    """Professional report generation with multiple formats"""
    
    @staticmethod
    def generate_excel_report(df, sample_df, tds_summary, materiality_df):
        """Generate comprehensive Excel report"""
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Premium formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter',
                'font_size': 11
            })
            
            money_format = workbook.add_format({'num_format': '₹#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            date_format = workbook.add_format({'num_format': 'dd-mm-yyyy'})
            
            # Executive Summary Sheet
            summary_data = {
                'Metric': [
                    'Total Transactions',
                    'Total Value',
                    'Sample Size',
                    'Sample Coverage',
                    'High Materiality Items',
                    'TDS Shortfall',
                    'Interest Payable'
                ],
                'Value': [
                    len(df),
                    df['taxable value'].sum() if 'taxable value' in df.columns else 0,
                    len(sample_df),
                    (sample_df['taxable value'].sum() / df['taxable value'].sum() * 100) if 'taxable value' in df.columns else 0,
                    len(df[df['materiality_level'] == 'Critical']) if 'materiality_level' in df.columns else 0,
                    tds_summary['tds_shortfall'].sum() if 'tds_shortfall' in tds_summary.columns else 0,
                    tds_summary['interest_payable'].sum() if 'interest_payable' in tds_summary.columns else 0
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Format sheets
            sheets = {
                'Materiality_Analysis': materiality_df,
                'Sample_Selection': sample_df,
                'TDS_Compliance': tds_summary,
                'Raw_Data': df
            }
            
            for sheet_name, data in sheets.items():
                if not data.empty:
                    data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
                    worksheet = writer.sheets[sheet_name]
                    
                    # Write headers
                    for col_num, col_name in enumerate(data.columns):
                        worksheet.write(0, col_num, col_name, header_format)
                    
                    # Auto-adjust columns
                    for col_num, col_name in enumerate(data.columns):
                        max_len = max(
                            data[col_name].astype(str).map(len).max() if not data[col_name].empty else 0,
                            len(col_name)
                        ) + 2
                        worksheet.set_column(col_num, col_num, min(max_len, 50))
                    
                    # Add totals row for numeric columns
                    total_row = len(data) + 2
                    worksheet.write(total_row, 0, 'TOTAL', header_format)
                    
                    for col_num, col_name in enumerate(data.columns):
                        if data[col_name].dtype in ['float64', 'int64']:
                            worksheet.write_formula(
                                total_row, col_num,
                                f'=SUM({chr(65+col_num)}3:{chr(65+col_num)}{total_row-1})',
                                money_format
                            )
        
        return output.getvalue()

# --- 8. MAIN APPLICATION ---
def main():
    """Main application with enhanced UI/UX"""
    
    # Premium Header
    st.markdown("""
    <div class="premium-header">
        <h1 style="margin:0; font-size:2.5rem;">🔍 Ultra-Audit Pro</h1>
        <p style="margin:10px 0 0 0; opacity:0.9;">Enterprise Edition | AI-Powered Materiality Analysis & TDS Compliance</p>
        <div style="margin-top:20px; display:flex; gap:10px;">
            <span class="badge-high">Critical</span>
            <span class="badge-medium">High</span>
            <span class="badge-low">Medium</span>
            <span style="background:#667eea; color:white; padding:5px 15px; border-radius:20px;">Low</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Audit Configuration")
        
        # Materiality Settings
        with st.expander("🎯 Materiality Parameters", expanded=True):
            materiality_threshold = st.slider(
                "Materiality Threshold (%)",
                min_value=0.1,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="Percentage of total value for materiality"
            )
            
            risk_appetite = st.select_slider(
                "Risk Appetite",
                options=['Conservative', 'Moderate', 'Aggressive'],
                value='Moderate',
                help="Higher risk appetite = lower sample size"
            )
        
        # Sampling Configuration
        with st.expander("📊 Sampling Strategy", expanded=True):
            sampling_method = st.selectbox(
                "Primary Sampling Method",
                [
                    "Materiality-Weighted Sampling",
                    "Monetary Unit Sampling (MUS)",
                    "Stratified Sampling",
                    "Systematic Sampling",
                    "Simple Random Sampling",
                    "Judgmental Sampling"
                ]
            )
            
            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=80,
                max_value=99,
                value=95,
                step=1
            )
            
            margin_of_error = st.slider(
                "Margin of Error (%)",
                min_value=1,
                max_value=10,
                value=5,
                step=1
            )
            
            # Calculate sample size based on parameters
            population_size = 1000  # Placeholder
            z_score = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}
            z = z_score[confidence_level]
            p = 0.5  # Maximum variability
            e = margin_of_error / 100
            
            sample_size = int((z**2 * p * (1-p)) / e**2)
            sample_size = min(sample_size, population_size)
            
            st.info(f"📈 Recommended Sample Size: {sample_size}")
        
        # TDS Configuration
        with st.expander("💰 TDS Settings", expanded=True):
            interest_months = st.number_input(
                "Interest Calculation (Months)",
                min_value=1,
                max_value=12,
                value=3
            )
            
            include_penalty = st.checkbox("Include Penalty Calculations", value=True)
        
        # Export Settings
        with st.expander("📤 Export Options", expanded=False):
            export_format = st.radio(
                "Export Format",
                ["Excel (Detailed)", "CSV (Quick)", "Both"]
            )
    
    # File Upload Section
    st.markdown("### 📤 Upload Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Drop your ledger file here",
            type=['xlsx', 'csv', 'xls'],
            help="Supported formats: Excel (.xlsx, .xls) or CSV"
        )
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Expected Format</h4>
            <p style="font-size:12px; margin:5px 0;">Date | Party Name | Invoice No | Gross Total | Taxable Value | CGST | SGST | IGST | TDS Deducted | TDS Section</p>
            <p style="font-size:11px; color:#666;">The system auto-detects column names</p>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Show preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
                st.caption(f"Total Rows: {len(df)} | Total Columns: {len(df.columns)}")
            
            # Process data
            with st.spinner("🔄 Processing data with AI engine..."):
                
                # Detect columns
                processor = DataProcessor()
                column_mapping = processor.detect_columns(df)
                
                # Rename columns for consistency
                rename_map = {}
                for std_name, orig_name in column_mapping.items():
                    if std_name == 'taxable':
                        rename_map[orig_name] = 'taxable value'
                    elif std_name == 'tds_deducted':
                        rename_map[orig_name] = 'TDS deducted'
                    elif std_name == 'tds_section':
                        rename_map[orig_name] = 'TDS Section'
                    elif std_name == 'party':
                        rename_map[orig_name] = 'Party name'
                    elif std_name == 'invoice':
                        rename_map[orig_name] = 'Invoice no'
                    elif std_name == 'gross':
                        rename_map[orig_name] = 'Gross Total'
                
                df = df.rename(columns=rename_map)
                
                # Clean numeric columns
                numeric_cols = ['Gross Total', 'taxable value', 'TDS deducted', 
                              'Input CGST', 'Input SGST', 'Input IGST']
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = processor.clean_numeric(df[col])
                    else:
                        df[col] = 0
                
                # Ensure required columns exist
                if 'TDS Section' not in df.columns:
                    df['TDS Section'] = 'NA'
                
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df['Month'] = df['Date'].dt.month
                    df['Year'] = df['Date'].dt.year
                    df['Quarter'] = df['Date'].dt.quarter
                
                # Calculate materiality
                materiality_engine = MaterialityEngine(materiality_threshold)
                df, total_value, materiality_amount, value_col = materiality_engine.calculate(df, 'taxable value')
                
                st.session_state.data_processed = True
            
            # Premium Metrics Dashboard
            st.markdown("### 📈 Real-Time Audit Intelligence")
            
            # Key Metrics Row
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color:#666; margin:0;">Total Population</h4>
                    <h2 style="color:#667eea; margin:10px 0;">{}</h2>
                    <p style="color:#999; margin:0;">Transactions</p>
                </div>
                """.format(f"{len(df):,}"), unsafe_allow_html=True)
            
            with metric_cols[1]:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color:#666; margin:0;">Total Value</h4>
                    <h2 style="color:#667eea; margin:10px 0;">₹{}</h2>
                    <p style="color:#999; margin:0;">Taxable Amount</p>
                </div>
                """.format(f"{total_value:,.0f}"), unsafe_allow_html=True)
            
            with metric_cols[2]:
                critical_count = len(df[df['materiality_level'] == 'Critical'])
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color:#666; margin:0;">Critical Items</h4>
                    <h2 style="color:#f43b47; margin:10px 0;">{}</h2>
                    <p style="color:#999; margin:0;">High Risk</p>
                </div>
                """.format(critical_count), unsafe_allow_html=True)
            
            with metric_cols[3]:
                sample_size_calc = min(sample_size, len(df))
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color:#666; margin:0;">Target Sample</h4>
                    <h2 style="color:#00b09b; margin:10px 0;">{}</h2>
                    <p style="color:#999; margin:0;">@{:.1f}% Coverage</p>
                </div>
                """.format(sample_size_calc, (sample_size_calc/len(df)*100)), unsafe_allow_html=True)
            
            with metric_cols[4]:
                tds_shortfall = df['TDS deducted'].sum() * 0.1  # Placeholder
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color:#666; margin:0;">TDS Shortfall</h4>
                    <h2 style="color:#f43b47; margin:10px 0;">₹{}</h2>
                    <p style="color:#999; margin:0;">Estimated</p>
                </div>
                """.format(f"{tds_shortfall:,.0f}"), unsafe_allow_html=True)
            
            # Materiality Distribution
            st.markdown("### 🎯 Materiality Analysis")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Materiality Distribution Chart
                materiality_dist = df['materiality_level'].value_counts().reset_index()
                materiality_dist.columns = ['Level', 'Count']
                
                colors = {
                    'Critical': '#f43b47',
                    'High': '#f9d423',
                    'Medium': '#00b09b',
                    'Low': '#667eea',
                    'Immaterial': '#95a5a6'
                }
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=materiality_dist['Level'],
                        values=materiality_dist['Count'],
                        marker_colors=[colors.get(level, '#667eea') for level in materiality_dist['Level']],
                        hole=0.4,
                        textinfo='label+percent',
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Transaction Distribution by Materiality",
                    showlegend=False,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Value Distribution by Materiality
                value_by_materiality = df.groupby('materiality_level')['taxable value'].sum().reset_index()
                
                fig = px.bar(
                    value_by_materiality,
                    x='materiality_level',
                    y='taxable value',
                    color='materiality_level',
                    color_discrete_map=colors,
                    title="Value Distribution by Materiality Level",
                    labels={'taxable value': 'Amount (₹)', 'materiality_level': 'Materiality Level'}
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Sampling Execution
            st.markdown("### 🎲 Intelligent Sampling Results")
            
            # Apply selected sampling method
            sampling_engine = SamplingEngine()
            
            if sampling_method == "Materiality-Weighted Sampling":
                sample_df = sampling_engine.materiality_weighted_sampling(df, sample_size_calc, 'taxable value')
            elif sampling_method == "Monetary Unit Sampling (MUS)":
                sample_df = sampling_engine.mus_sampling(df, sample_size_calc, 'taxable value')
            elif sampling_method == "Stratified Sampling":
                sample_df = sampling_engine.stratified_sampling(df, sample_size_calc, 'materiality_level', 'taxable value')
            elif sampling_method == "Systematic Sampling":
                sample_df = sampling_engine.systematic_sampling(df, sample_size_calc, 'taxable value')
            else:
                sample_df = df.sample(n=min(sample_size_calc, len(df)))
            
            # Calculate TDS compliance
            tds_engine = TDSComplianceEngine()
            tds_results = tds_engine.calculate_tds(sample_df)
            sample_df = pd.concat([sample_df.reset_index(drop=True), tds_results], axis=1)
            
            # Sample Results Dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sample_value = sample_df['taxable value'].sum() if 'taxable value' in sample_df.columns else 0
                st.metric(
                    "Sample Value",
                    f"₹{sample_value:,.0f}",
                    f"{(sample_value/total_value*100):.1f}% of population"
                )
            
            with col2:
                critical_in_sample = len(sample_df[sample_df['materiality_level'] == 'Critical'])
                st.metric(
                    "Critical Items in Sample",
                    critical_in_sample,
                    f"{(critical_in_sample/len(sample_df)*100):.1f}% of sample"
                )
            
            with col3:
                avg_risk = sample_df['risk_score'].mean() if 'risk_score' in sample_df.columns else 0
                st.metric(
                    "Average Risk Score",
                    f"{avg_risk:.1f}",
                    delta=None
                )
            
            with col4:
                total_shortfall = sample_df['tds_shortfall'].sum() if 'tds_shortfall' in sample_df.columns else 0
                st.metric(
                    "TDS Shortfall",
                    f"₹{total_shortfall:,.0f}",
                    "Projected"
                )
            
            # Detailed Sample View
            with st.expander("🔍 View Sample Details", expanded=True):
                display_cols = ['Party name', 'Invoice no', 'taxable value', 'TDS Section', 
                              'materiality_level', 'tds_shortfall', 'risk_score']
                display_cols = [col for col in display_cols if col in sample_df.columns]
                
                # Style the dataframe
                def color_risk(val):
                    if val > 75:
                        return 'background-color: #f43b47; color: white'
                    elif val > 50:
                        return 'background-color: #f9d423; color: black'
                    elif val > 25:
                        return 'background-color: #00b09b; color: white'
                    else:
                        return 'background-color: #667eea; color: white'
                
                styled_df = sample_df[display_cols].style.applymap(
                    color_risk, subset=['risk_score'] if 'risk_score' in display_cols else []
                ).format({
                    'taxable value': '₹{:,.2f}',
                    'tds_shortfall': '₹{:,.2f}',
                    'risk_score': '{:.1f}'
                })
                
                st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Advanced Analytics
            st.markdown("### 📊 Advanced Audit Analytics")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Risk Analysis",
                "💰 TDS Compliance",
                "📈 Trend Analysis",
                "📑 Audit Report"
            ])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk Distribution
                    risk_dist = sample_df['risk_score'].value_counts(bins=5).reset_index()
                    risk_dist.columns = ['Risk Range', 'Count']
                    
                    fig = px.pie(
                        risk_dist,
                        values='Count',
                        names='Risk Range',
                        title="Risk Distribution in Sample",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Materiality vs Risk
                    fig = px.scatter(
                        sample_df,
                        x='taxable value',
                        y='risk_score',
                        color='materiality_level',
                        size='materiality_weight',
                        title="Materiality vs Risk Analysis",
                        labels={'taxable value': 'Transaction Value (₹)', 'risk_score': 'Risk Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # TDS Shortfall by Section
                    shortfall_by_section = sample_df.groupby('TDS Section')['tds_shortfall'].sum().reset_index()
                    
                    fig = px.bar(
                        shortfall_by_section,
                        x='TDS Section',
                        y='tds_shortfall',
                        title="TDS Shortfall by Section",
                        color='tds_shortfall',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Compliance Ratio
                    compliance_data = sample_df[['Party name', 'compliance_ratio']].head(10)
                    
                    fig = px.bar(
                        compliance_data,
                        x='Party name',
                        y='compliance_ratio',
                        title="Top 10 Parties - TDS Compliance Ratio",
                        color='compliance_ratio',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if 'Date' in df.columns and 'Month' in df.columns:
                    # Monthly trends
                    monthly_data = df.groupby(['Year', 'Month'])['taxable value'].sum().reset_index()
                    monthly_data['Period'] = monthly_data['Month'].astype(str) + '-' + monthly_data['Year'].astype(str)
                    
                    fig = px.line(
                        monthly_data,
                        x='Period',
                        y='taxable value',
                        title="Monthly Transaction Value Trend",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Heatmap
                    pivot_data = df.pivot_table(
                        values='taxable value',
                        index=df['Date'].dt.day_name(),
                        columns=df['Date'].dt.month,
                        aggfunc='sum',
                        fill_value=0
                    )
                    
                    fig = px.imshow(
                        pivot_data,
                        title="Transaction Value Heatmap (Day vs Month)",
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.markdown("### 📑 Audit Summary Report")
                
                # Generate report content
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="info-box">
                        <h4>Executive Summary</h4>
                        <ul>
                            <li><strong>Audit Period:</strong> {}</li>
                            <li><strong>Population Size:</strong> {} transactions</li>
                            <li><strong>Sample Size:</strong> {} transactions</li>
                            <li><strong>Confidence Level:</strong> {}%</li>
                            <li><strong>Margin of Error:</strong> ±{}%</li>
                        </ul>
                    </div>
                    """.format(
                        f"{df['Date'].min().strftime('%d-%b-%Y') if 'Date' in df.columns else 'N/A'} to {df['Date'].max().strftime('%d-%b-%Y') if 'Date' in df.columns else 'N/A'}",
                        len(df),
                        len(sample_df),
                        confidence_level,
                        margin_of_error
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="info-box">
                        <h4>Key Findings</h4>
                        <ul>
                            <li><strong>Critical Items Found:</strong> {}</li>
                            <li><strong>Total TDS Shortfall:</strong> ₹{:,.0f}</li>
                            <li><strong>Interest Payable:</strong> ₹{:,.0f}</li>
                            <li><strong>High Risk Parties:</strong> {}</li>
                        </ul>
                    </div>
                    """.format(
                        critical_count,
                        sample_df['tds_shortfall'].sum() if 'tds_shortfall' in sample_df.columns else 0,
                        sample_df['interest_payable'].sum() if 'interest_payable' in sample_df.columns else 0,
                        len(sample_df[sample_df['risk_score'] > 75]) if 'risk_score' in sample_df.columns else 0
                    ), unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("#### 🎯 Audit Recommendations")
                
                rec_cols = st.columns(3)
                
                with rec_cols[0]:
                    st.markdown("""
                    <div class="info-box">
                        <h4 style="color:#f43b47;">⚠️ Critical Issues</h4>
                        <ul style="font-size:12px;">
                            <li>Review all {} critical items immediately</li>
                            <li>Verify TDS deductions for high-value contracts</li>
                            <li>Check Section 194C compliance</li>
                        </ul>
                    </div>
                    """.format(critical_count), unsafe_allow_html=True)
                
                with rec_cols[1]:
                    st.markdown("""
                    <div class="info-box">
                        <h4 style="color:#f9d423;">⚡ High Priority</h4>
                        <ul style="font-size:12px;">
                            <li>Investigate {} medium-risk parties</li>
                            <li>Validate TDS certificates</li>
                            <li>Review late payment interest</li>
                        </ul>
                    </div>
                    """.format(len(sample_df[sample_df['risk_score'].between(50, 75)])), unsafe_allow_html=True)
                
                with rec_cols[2]:
                    st.markdown("""
                    <div class="info-box">
                        <h4 style="color:#00b09b;">📋 Routine Checks</h4>
                        <ul style="font-size:12px;">
                            <li>Reconcile TDS with 26AS</li>
                            <li>Update TDS master data</li>
                            <li>Review vendor contracts</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Export Section
            st.markdown("### 📥 Export Results")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📊 Generate Excel Report", use_container_width=True):
                    with st.spinner("Generating comprehensive report..."):
                        report_gen = ReportGenerator()
                        excel_data = report_gen.generate_excel_report(df, sample_df, sample_df, df)
                        
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_data,
                            file_name=f"Ultra_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
            
            with export_col2:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    st.info("PDF generation coming soon!")
            
            with export_col3:
                if st.button("📧 Email Report", use_container_width=True):
                    st.info("Email functionality coming soon!")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome Screen
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h1 style="color: #667eea;">👋 Welcome to Ultra-Audit Pro Enterprise</h1>
            <p style="color: #666; font-size: 18px; margin: 20px 0;">
                The most advanced AI-powered audit sampling and TDS compliance platform
            </p>
            
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 40px; flex-wrap: wrap;">
                <div class="info-box" style="width: 250px;">
                    <h3 style="color:#667eea;">🎯 AI Materiality</h3>
                    <p>Intelligent transaction classification with 5-level materiality scoring</p>
                </div>
                
                <div class="info-box" style="width: 250px;">
                    <h3 style="color:#667eea;">📊 Smart Sampling</h3>
                    <p>6 advanced sampling methods with confidence-based calculations</p>
                </div>
                
                <div class="info-box" style="width: 250px;">
                    <h3 style="color:#667eea;">💰 TDS Engine</h3>
                    <p>Automated compliance checking with interest & penalty calculation</p>
                </div>
                
                <div class="info-box" style="width: 250px;">
                    <h3 style="color:#667eea;">📈 Analytics</h3>
                    <p>Real-time dashboards with predictive risk scoring</p>
                </div>
            </div>
            
            <div style="margin-top: 50px; background: white; padding: 30px; border-radius: 20px;">
                <h3>✨ Key Features</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;">
                    <div>✓ 5-Level Materiality</div>
                    <div>✓ MUS Sampling</div>
                    <div>✓ TDS Shortfall Analysis</div>
                    <div>✓ Risk Scoring</div>
                    <div>✓ Trend Analysis</div>
                    <div>✓ Executive Reports</div>
                    <div>✓ Penalty Calculator</div>
                    <div>✓ Audit Trail</div>
                    <div>✓ Export to Excel</div>
                </div>
            </div>
            
            <div style="margin-top: 40px; padding: 20px; background: linear-gradient(135deg, #667eea20, #764ba220); border-radius: 10px;">
                <p style="color: #666;">📁 Upload your ledger file to begin the intelligent audit process</p>
                <p style="font-size: 12px; color: #999;">Supports Excel, CSV | Auto-detects column names | No data stored</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
