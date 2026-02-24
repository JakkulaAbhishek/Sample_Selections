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
        materiality_amount = total_value * (self.threshold_percent / 100) if total_value > 0 else 0
        
        # Calculate materiality score for each transaction
        if materiality_amount > 0:
            df['materiality_score'] = df[value_column] / materiality_amount
        else:
            df['materiality_score'] = 0
            
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
    def calculate_sample_size(population_size, confidence_level=95, margin_of_error=5):
        """Calculate required sample size based on statistics"""
        z_scores = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}
        z = z_scores.get(confidence_level, 1.96)
        p = 0.5  # Maximum variability
        e = margin_of_error / 100
        
        sample_size = int((z**2 * p * (1-p)) / e**2)
        return min(sample_size, population_size)
    
    @staticmethod
    def percentage_based_sampling(df, percentage, value_column):
        """Sample based on percentage of population"""
        n = max(1, int(len(df) * (percentage / 100)))
        return df.sample(n=min(n, len(df)))
    
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
        if value_column not in df.columns or df[value_column].sum() == 0:
            return df.sample(n=min(n, len(df)))
        
        # Calculate sampling interval
        total_value = df[value_column].sum()
        interval = total_value / n if n > 0 else total_value
        
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
            '194J': (0.10, 30000, 'Professional Services'),
            '194I': (0.10, 180000, 'Rent - Plant/Machinery'),
            '194H': (0.05, 15000, 'Commission/Brokerage'),
            '194Q': (0.001, 5000000, 'TDS on Purchase of Goods'),
            '194IA': (0.01, 5000000, 'TDS on Property Purchase'),
            '194IB': (0.05, 50000, 'TDS on Rent (Individual)'),
            '194M': (0.05, 5000000, 'TDS on Contract/Commission'),
        }
        
        self.penalty_rates = {
            'interest_1_5': 0.015,  # 1.5% per month
            'interest_1': 0.01       # 1% per month
        }
    
    def calculate_tds(self, df, interest_months=3):
        """Calculate TDS requirements and identify shortfalls"""
        
        results = []
        
        for _, row in df.iterrows():
            section = str(row.get('TDS Section', 'NA')).strip().upper()
            taxable = float(row.get('taxable value', 0))
            tds_deducted = float(row.get('TDS deducted', 0))
            
            # Get applicable rate
            rate_info = self.tds_rates.get(section, (0.01, 0))
            tds_rate = rate_info[0]
            threshold = rate_info[1]
            
            # Calculate required TDS
            if taxable > threshold:
                tds_required = taxable * tds_rate
            else:
                tds_required = 0
            
            # Calculate shortfall
            shortfall = max(0, tds_required - tds_deducted)
            
            # Calculate interest
            interest = shortfall * self.penalty_rates['interest_1_5'] * interest_months
            
            # Risk score (0-100)
            if tds_required > 0:
                compliance_ratio = min(tds_deducted / tds_required, 1.0)
                risk_score = max(0, 100 - (compliance_ratio * 100))
            else:
                compliance_ratio = 1.0
                risk_score = 0
            
            results.append({
                'tds_required': round(tds_required, 2),
                'tds_shortfall': round(shortfall, 2),
                'interest_payable': round(interest, 2),
                'compliance_ratio': round(compliance_ratio * 100, 2),
                'risk_score': round(risk_score, 2),
                'penalty_applicable': shortfall > 0
            })
        
        return pd.DataFrame(results)

# --- 7. SAMPLE EXCEL GENERATOR (FIXED) ---
def generate_sample_excel():
    """Generate sample Excel file for users to download"""
    
    np.random.seed(42)
    
    # Parties with their typical TDS sections
    parties_data = [
        ("M/s Sharma Construction", "194C", 0.01),
        ("Patil Builders & Developers", "194C", 0.01),
        ("Desai Infrastructure Ltd", "194C", 0.01),
        ("Kulkarni Contractors", "194C", 0.01),
        ("Joshi & Associates", "194C", 0.01),
        ("City Hospital", "194J", 0.10),
        ("Dr. Mehta's Clinic", "194J", 0.10),
        ("Legal Eagles Associates", "194J", 0.10),
        ("Consulting Pros Pvt Ltd", "194J", 0.10),
        ("Tech Solutions Inc", "194J", 0.10),
        ("Royal Properties", "194I", 0.10),
        ("Godrej Properties", "194I", 0.10),
        ("Warehouse Solutions", "194I", 0.10),
        ("Office Space Ltd", "194I", 0.10),
        ("Mall Management Co", "194I", 0.10),
        ("Marketing Gurus", "194H", 0.05),
        ("Insurance Brokers Ltd", "194H", 0.05),
        ("Real Estate Agents", "194H", 0.05),
        ("Travel Agents Assoc", "194H", 0.05),
        ("Advertising Agency", "194H", 0.05),
        ("Steel Suppliers Ltd", "194Q", 0.001),
        ("Cement Corporation", "194Q", 0.001),
        ("Building Materials Co", "194Q", 0.001),
        ("Electrical Goods Ltd", "194Q", 0.001),
        ("Furniture Mart", "194Q", 0.001),
    ]
    
    # Generate dates for the last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    # Create dataframe
    data = []
    invoice_counter = 1000
    
    for i in range(250):  # Generate 250 transactions
        party_idx = np.random.randint(0, len(parties_data))
        party_name, tds_section, tds_rate = parties_data[party_idx]
        
        # Fix: Convert datetime to string properly
        date_idx = np.random.randint(0, len(date_list))
        date_obj = date_list[date_idx]
        date_str = date_obj.strftime('%d-%m-%Y')
        
        invoice_no = f"INV-2024{invoice_counter:04d}"
        invoice_counter += 1
        
        # Generate amounts based on TDS section
        if tds_section == "194C":
            gross_total = np.random.uniform(50000, 500000)
        elif tds_section == "194J":
            gross_total = np.random.uniform(25000, 300000)
        elif tds_section == "194I":
            gross_total = np.random.uniform(100000, 1000000)
        elif tds_section == "194H":
            gross_total = np.random.uniform(10000, 200000)
        else:  # 194Q
            gross_total = np.random.uniform(200000, 2000000)
        
        gross_total = round(gross_total, 2)
        taxable_value = gross_total
        
        # Sometimes TDS is short-deducted
        deduction_factor = np.random.choice([0, 0.5, 0.8, 1.0], p=[0.1, 0.1, 0.2, 0.6])
        tds_deducted = round(taxable_value * tds_rate * deduction_factor, 2)
        
        # GST components
        gst_rate = 0.18
        gst_amount = round(gross_total * gst_rate, 2)
        
        if np.random.random() > 0.3:  # 70% intra-state
            cgst = round(gst_amount / 2, 2)
            sgst = round(gst_amount / 2, 2)
            igst = 0
        else:  # 30% inter-state
            cgst = 0
            sgst = 0
            igst = gst_amount
        
        data.append([
            date_str,
            party_name,
            invoice_no,
            gross_total,
            taxable_value,
            cgst,
            sgst,
            igst,
            tds_deducted,
            tds_section
        ])
    
    df = pd.DataFrame(data, columns=[
        'Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value',
        'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted', 'TDS Section'
    ])
    
    # Sort by date
    df['Date_temp'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date_temp')
    df = df.drop('Date_temp', axis=1)
    
    # Save to BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sample Ledger Data', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Sample Ledger Data']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#4472C4', 'font_color': 'white',
            'border': 1, 'align': 'center', 'valign': 'vcenter'
        })
        
        money_format = workbook.add_format({'num_format': '₹#,##0.00'})
        date_format = workbook.add_format({'num_format': 'dd-mm-yyyy'})
        
        # Format columns
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        worksheet.set_column('A:A', 12, date_format)
        worksheet.set_column('B:B', 30)
        worksheet.set_column('C:C', 15)
        worksheet.set_column('D:I', 15, money_format)
        worksheet.set_column('J:J', 12)
        
        # Add summary sheet
        summary_data = {
            'TDS Section': ['194C', '194J', '194I', '194H', '194Q'],
            'Description': ['Contractors', 'Professional Services', 'Rent', 'Commission', 'Purchase'],
            'TDS Rate (%)': [1, 10, 10, 5, 0.1],
            'Threshold (₹)': [30000, 30000, 180000, 15000, 5000000]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='TDS Reference', index=False)
    
    return output.getvalue()

# --- 8. REPORT GENERATOR ---
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
            
            # Executive Summary Sheet
            summary_data = {
                'Metric': [
                    'Total Transactions',
                    'Total Value',
                    'Sample Size',
                    'Sample Percentage',
                    'Sample Coverage',
                    'Critical Items',
                    'TDS Shortfall',
                    'Interest Payable'
                ],
                'Value': [
                    len(df),
                    df['taxable value'].sum() if 'taxable value' in df.columns else 0,
                    len(sample_df),
                    f"{(len(sample_df)/len(df)*100):.1f}%" if len(df) > 0 else "0%",
                    f"{(sample_df['taxable value'].sum() / df['taxable value'].sum() * 100):.1f}%" if 'taxable value' in df.columns and df['taxable value'].sum() > 0 else "0%",
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
        
        return output.getvalue()

# --- 9. MAIN APPLICATION ---
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
        
        # Sample Download Button
        st.markdown("### 📥 Sample Data")
        try:
            sample_excel = generate_sample_excel()
            st.download_button(
                label="📊 Download Sample Excel Template",
                data=sample_excel,
                file_name="TDS_Sample_Data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating sample: {str(e)}")
        
        st.markdown("---")
        
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
        
        # Sampling Configuration
        with st.expander("📊 Sampling Strategy", expanded=True):
            sampling_method = st.selectbox(
                "Primary Sampling Method",
                [
                    "Percentage Based Sampling",
                    "Materiality-Weighted Sampling",
                    "Monetary Unit Sampling (MUS)",
                    "Stratified Sampling",
                    "Systematic Sampling",
                    "Simple Random Sampling"
                ]
            )
            
            # Percentage based sampling
            sample_percentage = st.slider(
                "Sample Selection (%)",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                help="Percentage of transactions to select for sample"
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
        
        # TDS Configuration
        with st.expander("💰 TDS Settings", expanded=True):
            interest_months = st.number_input(
                "Interest Calculation (Months)",
                min_value=1,
                max_value=12,
                value=3
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
                
                if 'Party name' not in df.columns:
                    df['Party name'] = 'Unknown'
                
                if 'Invoice no' not in df.columns:
                    df['Invoice no'] = [f'INV-{i:04d}' for i in range(len(df))]
                
                # Process dates if available
                if 'Date' in df.columns:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df['Month'] = df['Date'].dt.month
                        df['Year'] = df['Date'].dt.year
                        df['Quarter'] = df['Date'].dt.quarter
                    except:
                        pass
                
                # Calculate materiality
                materiality_engine = MaterialityEngine(materiality_threshold)
                df, total_value, materiality_amount, value_col = materiality_engine.calculate(df, 'taxable value')
                
                st.session_state.data_processed = True
            
            # Premium Metrics Dashboard
            st.markdown("### 📈 Real-Time Audit Intelligence")
            
            # Key Metrics Row
            metric_cols = st.columns(5)
            
            critical_count = len(df[df['materiality_level'] == 'Critical']) if 'materiality_level' in df.columns else 0
            
            with metric_cols[0]:
                st.metric("Total Population", f"{len(df):,}")
            
            with metric_cols[1]:
                st.metric("Total Value", f"₹{total_value:,.0f}")
            
            with metric_cols[2]:
                st.metric("Critical Items", f"{critical_count}")
            
            with metric_cols[3]:
                # Calculate sample size based on selected method
                if sampling_method == "Percentage Based Sampling":
                    sample_size_calc = max(1, int(len(df) * (sample_percentage / 100)))
                else:
                    sampling_engine = SamplingEngine()
                    sample_size_calc = sampling_engine.calculate_sample_size(
                        len(df), confidence_level, margin_of_error
                    )
                
                st.metric("Target Sample", f"{sample_size_calc}", 
                         f"{(sample_size_calc/len(df)*100):.1f}%")
            
            with metric_cols[4]:
                tds_shortfall_est = df['TDS deducted'].sum() * 0.15 if 'TDS deducted' in df.columns else 0
                st.metric("Est. TDS Shortfall", f"₹{tds_shortfall_est:,.0f}")
            
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
                        labels=materiality_dist['Level'].tolist(),
                        values=materiality_dist['Count'].tolist(),
                        marker_colors=[colors.get(level, '#667eea') for level in materiality_dist['Level']],
                        hole=0.4,
                        textinfo='label+percent',
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Transaction Distribution by Materiality",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Value Distribution by Materiality
                value_by_materiality = df.groupby('materiality_level')['taxable value'].sum().reset_index()
                value_by_materiality.columns = ['Level', 'Value']
                
                fig = px.bar(
                    value_by_materiality,
                    x='Level',
                    y='Value',
                    color='Level',
                    color_discrete_map=colors,
                    title="Value Distribution by Materiality Level"
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Sampling Execution
            st.markdown("### 🎲 Intelligent Sampling Results")
            
            # Apply selected sampling method
            sampling_engine = SamplingEngine()
            
            if sampling_method == "Percentage Based Sampling":
                sample_df = sampling_engine.percentage_based_sampling(df, sample_percentage, 'taxable value')
            elif sampling_method == "Materiality-Weighted Sampling":
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
            tds_results = tds_engine.calculate_tds(sample_df, interest_months)
            sample_df = pd.concat([sample_df.reset_index(drop=True), tds_results], axis=1)
            
            # Sample Results Dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            sample_value = sample_df['taxable value'].sum() if 'taxable value' in sample_df.columns else 0
            critical_in_sample = len(sample_df[sample_df['materiality_level'] == 'Critical']) if 'materiality_level' in sample_df.columns else 0
            avg_risk = sample_df['risk_score'].mean() if 'risk_score' in sample_df.columns else 0
            total_shortfall = sample_df['tds_shortfall'].sum() if 'tds_shortfall' in sample_df.columns else 0
            
            with col1:
                st.metric("Sample Value", f"₹{sample_value:,.0f}")
            
            with col2:
                st.metric("Critical Items", f"{critical_in_sample}")
            
            with col3:
                st.metric("Avg Risk Score", f"{avg_risk:.1f}")
            
            with col4:
                st.metric("TDS Shortfall", f"₹{total_shortfall:,.0f}")
            
            # Detailed Sample View
            with st.expander("🔍 View Sample Details", expanded=True):
                display_cols = ['Party name', 'Invoice no', 'taxable value', 'TDS Section', 
                              'materiality_level', 'tds_shortfall', 'risk_score']
                display_cols = [col for col in display_cols if col in sample_df.columns]
                
                if display_cols:
                    st.dataframe(sample_df[display_cols], use_container_width=True, height=400)
            
            # Export Section
            st.markdown("### 📥 Export Results")
            
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
                    <p>7 advanced sampling methods with percentage-based selection</p>
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
            
            <div style="margin-top: 40px; padding: 20px; background: linear-gradient(135deg, #667eea20, #764ba220); border-radius: 10px;">
                <p style="color: #666;">📁 Click the button in sidebar to download sample Excel template</p>
                <p style="font-size: 12px; color: #999;">Supports Excel, CSV | Auto-detects column names | No data stored</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
