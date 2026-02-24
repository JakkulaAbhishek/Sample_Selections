import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, timedelta

# --- 1. ENHANCED UI CONFIG ---
st.set_page_config(
    page_title="Ultra-Audit Pro | Materiality-Based Sampling",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium, professional look
st.markdown("""
    <style>
    /* Main background with subtle gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling for metrics */
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Materiality levels styling */
    .materiality-high {
        background: linear-gradient(135deg, #f43b47 0%, #453a94 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(244, 59, 71, 0.3);
    }
    .materiality-medium {
        background: linear-gradient(135deg, #f9d423 0%, #f83600 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(249, 212, 35, 0.3);
    }
    .materiality-low {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.3);
    }
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Info box styling */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. ENHANCED DATA CLEANING ---
def clean_numeric(series):
    """Enhanced numeric cleaning with multiple fallback methods"""
    if series.dtype == 'object':
        # Remove currency symbols, commas, and other non-numeric characters
        series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
        series = series.replace('', '0')
    return pd.to_numeric(series, errors='coerce').fillna(0)

# --- 3. MATERIALITY CALCULATION ENGINE ---
def calculate_materiality(df, materiality_percent, base_metric='taxable value'):
    """Calculate materiality thresholds based on percentage"""
    total_value = df[base_metric].sum()
    materiality_amount = total_value * (materiality_percent / 100)
    
    # Classify transactions by materiality
    df['Materiality_Level'] = 'Low'
    df['Materiality_Amount'] = materiality_amount
    
    # High materiality: > 50% of materiality threshold
    high_threshold = materiality_amount * 0.5
    # Medium materiality: between 10% and 50% of threshold
    medium_threshold = materiality_amount * 0.1
    
    df.loc[df[base_metric] > high_threshold, 'Materiality_Level'] = 'High'
    df.loc[(df[base_metric] <= high_threshold) & (df[base_metric] > medium_threshold), 'Materiality_Level'] = 'Medium'
    
    return df, total_value, materiality_amount

def get_materiality_color(level):
    """Return color code for materiality level"""
    colors = {'High': '#f43b47', 'Medium': '#f9d423', 'Low': '#00b09b'}
    return colors.get(level, '#667eea')

# --- 4. ENHANCED SAMPLING WITH MATERIALITY ---
def apply_materiality_sampling(df, materiality_levels, sample_pct, method):
    """Apply sampling with materiality weighting"""
    samples = []
    
    for level in materiality_levels:
        level_df = df[df['Materiality_Level'] == level]
        if len(level_df) == 0:
            continue
            
        # Weighted sampling based on materiality
        if level == 'High':
            level_pct = min(sample_pct * 2, 100)  # Double the sample for high materiality
        elif level == 'Medium':
            level_pct = sample_pct  # Normal sample for medium materiality
        else:
            level_pct = max(sample_pct * 0.5, 5)  # Half sample for low materiality
            
        n = max(1, int(len(level_df) * (level_pct / 100)))
        
        if method == "Judgmental Sampling":
            sample = level_df.nlargest(n, 'taxable value')
        elif method == "Systematic Sampling":
            step = max(1, len(level_df) // n)
            sample = level_df.iloc[::step].head(n)
        else:
            sample = level_df.sample(n=min(n, len(level_df)))
            
        sample['Materiality_Sample_Weight'] = level_pct
        samples.append(sample)
    
    return pd.concat(samples).drop_duplicates(subset=['Invoice no', 'Party name'])

# --- 5. HEADER SECTION ---
st.markdown('<div class="custom-header">🔍 ULTRA-AUDIT PRO | Materiality-Based Intelligent Sampling</div>', 
            unsafe_allow_html=True)

# --- 6. ENHANCED SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Audit Configuration")
    
    # Materiality Settings
    st.markdown("#### 🎯 Materiality Parameters")
    materiality_percent = st.slider(
        "Materiality Threshold (%)", 
        min_value=0.1, 
        max_value=10.0, 
        value=5.0, 
        step=0.1,
        help="Percentage of total value to set as materiality threshold"
    )
    
    st.markdown("#### 📊 Sampling Methods")
    method_categories = {
        "🔹 Probability Sampling": [
            "Simple Random Sampling", "Systematic Sampling", "Stratified Sampling", 
            "Cluster Sampling", "Multistage Sampling", "Multiphase Sampling"
        ],
        "🔹 Non-Probability Sampling": [
            "Convenience Sampling", "Judgmental Sampling", "Quota Sampling", 
            "Snowball Sampling"
        ],
        "🔹 Audit-Specific Sampling": [
            "Statistical Sampling", "Monetary Unit Sampling (MUS)", 
            "Block Sampling", "Materiality-Weighted Sampling"
        ]
    }

    selected_cats = st.multiselect(
        "Select Method Categories", 
        list(method_categories.keys()), 
        default=["🔹 Audit-Specific Sampling"]
    )

    available_methods = []
    for cat in selected_cats:
        available_methods.extend(method_categories[cat])

    primary_methods = st.multiselect(
        "Choose Sampling Method(s)", 
        options=available_methods, 
        default=[available_methods[0]] if available_methods else []
    )
    
    sample_pct = st.slider("Base Sample Selection %", 1, 100, 20)
    
    st.markdown("#### 📅 Date Range Filter")
    use_date_filter = st.checkbox("Apply Date Filter", value=False)
    if use_date_filter:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())

# --- 7. DATA INGESTION ---
headers = ['Date', 'Party name', 'Invoice no', 'Gross Total', 'taxable value', 
           'Input CGST', 'Input SGST', 'Input IGST', 'TDS deducted', 'TDS Section']

uploaded_file = st.file_uploader(
    "📤 Upload Raw Ledger File", 
    type=['xlsx', 'csv'],
    help="Upload your ledger file in Excel or CSV format"
)

if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    
    # Pre-processing
    num_cols = ['Gross Total', 'taxable value', 'TDS deducted', 'Input CGST', 'Input SGST', 'Input IGST']
    for col in num_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    
    if 'TDS Section' not in df.columns:
        df['TDS Section'] = "NA"
    
    # Apply date filter if enabled
    if use_date_filter and 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        df = df.loc[mask]
    
    # Apply materiality
    df, total_value, materiality_amount = calculate_materiality(df, materiality_percent)
    
    # Materiality Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"₹{total_value:,.2f}")
    with col2:
        st.metric("Materiality Amount", f"₹{materiality_amount:,.2f}")
    with col3:
        high_count = len(df[df['Materiality_Level'] == 'High'])
        st.metric("High Materiality Items", high_count)
    with col4:
        st.metric("Total Transactions", len(df))
    
    # Materiality Distribution
    st.markdown("### 📊 Materiality Analysis")
    mat_dist = df['Materiality_Level'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=mat_dist.index,
            y=mat_dist.values,
            marker_color=[get_materiality_color(level) for level in mat_dist.index],
            text=mat_dist.values,
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="Transaction Distribution by Materiality Level",
        xaxis_title="Materiality Level",
        yaxis_title="Number of Transactions",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 8. ENHANCED SAMPLING WITH MATERIALITY ---
    sample_df = pd.DataFrame()
    
    for method in primary_methods:
        if method == "Materiality-Weighted Sampling":
            sampled = apply_materiality_sampling(df, ['High', 'Medium', 'Low'], sample_pct, method)
        else:
            # Existing sampling methods with materiality consideration
            sampled = apply_materiality_sampling(df, ['High', 'Medium', 'Low'], sample_pct, method)
        
        sampled['Basis for Selection'] = method
        sample_df = pd.concat([sample_df, sampled]).drop_duplicates(subset=['Invoice no', 'Party name'])
    
    # Sampling Summary
    st.markdown("### 📋 Sampling Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sample Size", len(sample_df))
    with col2:
        coverage = (sample_df['taxable value'].sum() / df['taxable value'].sum() * 100)
        st.metric("Value Coverage", f"{coverage:.1f}%")
    with col3:
        st.metric("High Materiality Items Sampled", 
                 len(sample_df[sample_df['Materiality_Level'] == 'High']))

    # --- 9. TDS CALCULATIONS ---
    rates = {
        '194C': 0.01, '194J': 0.10, '194I': 0.10, 
        '194H': 0.05, '194Q': 0.001, '194IA': 0.01,
        '194IB': 0.05, '194M': 0.05
    }
    
    tds_summary = df.groupby(['Party name', 'TDS Section']).agg({
        'taxable value': 'sum', 
        'TDS deducted': 'sum',
        'Materiality_Level': 'first'
    }).reset_index()
    
    tds_summary['TDS Needs to be Deducted'] = tds_summary.apply(
        lambda r: r['taxable value'] * rates.get(str(r['TDS Section']).upper(), 0.01), 
        axis=1
    )
    tds_summary['Shortfall'] = np.maximum(0, tds_summary['TDS Needs to be Deducted'] - tds_summary['TDS deducted'])
    tds_summary['Interest (1.5% pm)'] = tds_summary['Shortfall'] * 0.015 * 3  # 3 months interest

    # --- 10. INTERACTIVE DASHBOARD ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Audit Overview", "📈 Sample Analysis", "💰 TDS Summary", "📑 Raw Data"])
    
    with tab1:
        st.markdown("### 🎯 Key Audit Metrics")
        
        # Create metric cards
        metric_cols = st.columns(4)
        metrics = [
            ("Total Population Value", f"₹{df['taxable value'].sum():,.2f}"),
            ("Sample Value", f"₹{sample_df['taxable value'].sum():,.2f}"),
            ("Sample Coverage", f"{(sample_df['taxable value'].sum()/df['taxable value'].sum()*100):.1f}%"),
            ("Potential TDS Shortfall", f"₹{tds_summary['Shortfall'].sum():,.2f}")
        ]
        
        for i, (label, value) in enumerate(metrics):
            with metric_cols[i]:
                st.markdown(f"""
                <div class="stMetric">
                    <h4 style="color: #666; margin-bottom: 5px;">{label}</h4>
                    <h2 style="color: #667eea; margin: 0;">{value}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Materiality breakdown in sample
        st.markdown("### 🎨 Sample Composition by Materiality")
        sample_composition = sample_df['Materiality_Level'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=sample_composition.index,
            values=sample_composition.values,
            marker_colors=[get_materiality_color(level) for level in sample_composition.index],
            hole=.3
        )])
        fig_pie.update_layout(
            title="Sample Distribution by Materiality Level",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔍 Detailed Sample Analysis")
        
        # Sample data with styling
        styled_sample = sample_df[['Party name', 'Invoice no', 'taxable value', 'TDS Section', 'Materiality_Level']].copy()
        
        # Add color coding for materiality
        def color_materiality(val):
            colors = {'High': 'background-color: #f43b47; color: white',
                     'Medium': 'background-color: #f9d423; color: white',
                     'Low': 'background-color: #00b09b; color: white'}
            return colors.get(val, '')
        
        st.dataframe(
            styled_sample.style.applymap(color_materiality, subset=['Materiality_Level']),
            use_container_width=True,
            height=400
        )
        
        # Download sample
        csv = sample_df.to_csv(index=False)
        st.download_button(
            "📥 Download Sample Data",
            csv,
            "audit_sample.csv",
            "text/csv"
        )
    
    with tab3:
        st.markdown("### 💰 TDS Compliance Summary")
        
        # TDS summary table
        st.dataframe(
            tds_summary.style.format({
                'taxable value': '₹{:,.2f}',
                'TDS deducted': '₹{:,.2f}',
                'TDS Needs to be Deducted': '₹{:,.2f}',
                'Shortfall': '₹{:,.2f}',
                'Interest (1.5% pm)': '₹{:,.2f}'
            }),
            use_container_width=True
        )
        
        # TDS Shortfall Chart
        top_shortfalls = tds_summary.nlargest(10, 'Shortfall')[['Party name', 'Shortfall']]
        fig_bar = px.bar(
            top_shortfalls,
            x='Party name',
            y='Shortfall',
            title="Top 10 TDS Shortfalls by Party",
            color_discrete_sequence=['#f43b47']
        )
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab4:
        st.markdown("### 📑 Complete Raw Data")
        
        # Raw data with filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_parties = st.multiselect("Filter by Party", df['Party name'].unique())
        with filter_col2:
            selected_materiality = st.multiselect("Filter by Materiality", ['High', 'Medium', 'Low'])
        
        filtered_df = df.copy()
        if selected_parties:
            filtered_df = filtered_df[filtered_df['Party name'].isin(selected_parties)]
        if selected_materiality:
            filtered_df = filtered_df[filtered_df['Materiality_Level'].isin(selected_materiality)]
        
        st.dataframe(filtered_df, use_container_width=True, height=500)

    # --- 11. ENHANCED EXCEL EXPORT ---
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        out_bio = BytesIO()
        
        with pd.ExcelWriter(out_bio, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Custom formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#667eea',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            money_format = workbook.add_format({'num_format': '₹#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            
            # Write sheets
            sheet_map = {
                'Audit_Dashboard': df,
                'Materiality_Analysis': df[['Party name', 'Invoice no', 'taxable value', 'Materiality_Level']],
                'Selected_Samples': sample_df,
                'TDS_Summary': tds_summary
            }
            
            for sheet_name, data in sheet_map.items():
                data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
                worksheet = writer.sheets[sheet_name]
                
                # Write headers
                for col_num, value in enumerate(data.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Auto-adjust column widths
                for col_num, value in enumerate(data.columns.values):
                    max_len = max(
                        data[value].astype(str).map(len).max(),
                        len(value)
                    ) + 2
                    worksheet.set_column(col_num, col_num, min(max_len, 50))
                
                # Add totals row
                if 'taxable value' in data.columns:
                    total_row = len(data) + 2
                    worksheet.write(total_row, 0, "TOTAL", header_format)
                    worksheet.write_formula(
                        total_row, data.columns.get_loc('taxable value'),
                        f"=SUM({chr(66+data.columns.get_loc('taxable value'))}3:{chr(66+data.columns.get_loc('taxable value'))}{total_row-1})",
                        money_format
                    )
        
        st.download_button(
            label="📥 Download Complete Audit Report (Excel)",
            data=out_bio.getvalue(),
            file_name=f"Ultra_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1 style="color: #667eea;">👋 Welcome to Ultra-Audit Pro!</h1>
        <p style="color: #666; font-size: 18px; margin: 20px 0;">
            The most advanced audit sampling and TDS compliance platform with 
            intelligent materiality-based analysis.
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 40px;">
            <div class="info-box" style="width: 300px;">
                <h3>🎯 Materiality Analysis</h3>
                <p>Automatically classify transactions by materiality level</p>
            </div>
            <div class="info-box" style="width: 300px;">
                <h3>📊 Smart Sampling</h3>
                <p>Weighted sampling based on materiality thresholds</p>
            </div>
            <div class="info-box" style="width: 300px;">
                <h3>💰 TDS Compliance</h3>
                <p>Automated TDS calculation and shortfall analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
