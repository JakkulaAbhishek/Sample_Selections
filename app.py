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
        series = series.replace('nan', '0')
    return pd.to_numeric(series, errors='coerce').fillna(0)

# --- 3. COLUMN NAME DETECTION AND MAPPING ---
def detect_and_map_columns(df):
    """Automatically detect and map column names with flexibility"""
    
    # Common column name variations
    column_mappings = {
        'date': ['date', 'transaction date', 'entry date', 'dt', 'trans date'],
        'party': ['party name', 'party', 'vendor', 'supplier', 'customer', 'name', 'party_name'],
        'invoice': ['invoice no', 'invoice', 'inv no', 'invoice number', 'inv_no', 'voucher no'],
        'gross': ['gross total', 'gross', 'total amount', 'bill amount', 'invoice amount'],
        'taxable': ['taxable value', 'taxable', 'taxable amount', 'value', 'txbl value'],
        'cgst': ['input cgst', 'cgst', 'central gst'],
        'sgst': ['input sgst', 'sgst', 'state gst'],
        'igst': ['input igst', 'igst', 'integrated gst'],
        'tds_deducted': ['tds deducted', 'tds', 'tax deducted', 'tds amount'],
        'tds_section': ['tds section', 'section', 'tds_sec']
    }
    
    mapped_columns = {}
    df_columns_lower = {col.lower().strip(): col for col in df.columns}
    
    for standard_name, variations in column_mappings.items():
        for var in variations:
            if var in df_columns_lower:
                mapped_columns[standard_name] = df_columns_lower[var]
                break
    
    return mapped_columns

# --- 4. MATERIALITY CALCULATION ENGINE ---
def calculate_materiality(df, materiality_percent, value_column='taxable value'):
    """Calculate materiality thresholds based on percentage"""
    if value_column not in df.columns:
        # Try to find a suitable value column
        possible_value_cols = ['taxable value', 'taxable', 'value', 'amount', 'gross total']
        for col in possible_value_cols:
            if col in df.columns:
                value_column = col
                break
        else:
            # If no value column found, use the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    value_column = numeric_cols[0]
                else:
                    st.error("No numeric column found for materiality calculation!")
                    return df, 0, 0
    
    total_value = df[value_column].sum()
    materiality_amount = total_value * (materiality_percent / 100)
    
    # Classify transactions by materiality
    df['Materiality_Level'] = 'Low'
    df['Materiality_Amount'] = materiality_amount
    df['Value_For_Materiality'] = df[value_column]
    
    # High materiality: > 50% of materiality threshold
    high_threshold = materiality_amount * 0.5
    # Medium materiality: between 10% and 50% of threshold
    medium_threshold = materiality_amount * 0.1
    
    df.loc[df[value_column] > high_threshold, 'Materiality_Level'] = 'High'
    df.loc[(df[value_column] <= high_threshold) & (df[value_column] > medium_threshold), 'Materiality_Level'] = 'Medium'
    
    return df, total_value, materiality_amount, value_column

def get_materiality_color(level):
    """Return color code for materiality level"""
    colors = {'High': '#f43b47', 'Medium': '#f9d423', 'Low': '#00b09b'}
    return colors.get(level, '#667eea')

# --- 5. ENHANCED SAMPLING WITH MATERIALITY ---
def apply_materiality_sampling(df, materiality_levels, sample_pct, method, value_column):
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
            sample = level_df.nlargest(n, value_column)
        elif method == "Systematic Sampling":
            step = max(1, len(level_df) // n)
            sample = level_df.iloc[::step].head(n)
        else:
            sample = level_df.sample(n=min(n, len(level_df)))
            
        sample['Materiality_Sample_Weight'] = level_pct
        samples.append(sample)
    
    return pd.concat(samples).drop_duplicates(subset=['Invoice no', 'Party name'] if all(col in df.columns for col in ['Invoice no', 'Party name']) else None)

# --- 6. HEADER SECTION ---
st.markdown('<div class="custom-header">🔍 ULTRA-AUDIT PRO | Materiality-Based Intelligent Sampling</div>', 
            unsafe_allow_html=True)

# --- 7. ENHANCED SIDEBAR ---
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

# --- 8. DATA INGESTION WITH FLEXIBLE COLUMN MAPPING ---
st.markdown("### 📤 Upload Your Data")

uploaded_file = st.file_uploader(
    "Upload Raw Ledger File", 
    type=['xlsx', 'csv'],
    help="Upload your ledger file in Excel or CSV format"
)

if uploaded_file:
    # Load data with error handling
    try:
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
        
        # Show original column names
        with st.expander("📋 View Original Column Names"):
            st.write(list(df.columns))
        
        # Detect and map columns
        column_mapping = detect_and_map_columns(df)
        
        # Show detected mapping
        if column_mapping:
            st.info("✅ Column mapping detected successfully!")
            with st.expander("🔍 View Column Mapping"):
                mapping_df = pd.DataFrame([
                    {"Standard Field": k, "Original Column": v} 
                    for k, v in column_mapping.items()
                ])
                st.dataframe(mapping_df)
        
        # Rename columns for consistency (if mapping exists)
        rename_dict = {}
        for std_name, orig_name in column_mapping.items():
            if std_name == 'taxable':
                rename_dict[orig_name] = 'taxable value'
            elif std_name == 'tds_deducted':
                rename_dict[orig_name] = 'TDS deducted'
            elif std_name == 'tds_section':
                rename_dict[orig_name] = 'TDS Section'
            elif std_name == 'party':
                rename_dict[orig_name] = 'Party name'
            elif std_name == 'invoice':
                rename_dict[orig_name] = 'Invoice no'
            elif std_name == 'gross':
                rename_dict[orig_name] = 'Gross Total'
            elif std_name == 'cgst':
                rename_dict[orig_name] = 'Input CGST'
            elif std_name == 'sgst':
                rename_dict[orig_name] = 'Input SGST'
            elif std_name == 'igst':
                rename_dict[orig_name] = 'Input IGST'
        
        df = df.rename(columns=rename_dict)
        
        # Pre-processing numeric columns
        expected_numeric_cols = ['Gross Total', 'taxable value', 'TDS deducted', 
                                'Input CGST', 'Input SGST', 'Input IGST']
        
        for col in expected_numeric_cols:
            if col in df.columns:
                df[col] = clean_numeric(df[col])
            else:
                # Create missing numeric columns with zeros
                df[col] = 0
        
        if 'TDS Section' not in df.columns:
            df['TDS Section'] = "NA"
        
        # Apply date filter if enabled
        if use_date_filter and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
            df = df.loc[mask]
            st.info(f"Date filter applied: {len(df)} rows remaining")
        
        # Apply materiality
        df, total_value, materiality_amount, value_col = calculate_materiality(df, materiality_percent)
        
        if total_value == 0:
            st.warning("⚠️ Total value is zero. Please check your data.")
        else:
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

            # --- 9. ENHANCED SAMPLING WITH MATERIALITY ---
            sample_df = pd.DataFrame()
            
            for method in primary_methods:
                if method == "Materiality-Weighted Sampling":
                    sampled = apply_materiality_sampling(df, ['High', 'Medium', 'Low'], sample_pct, method, value_col)
                else:
                    sampled = apply_materiality_sampling(df, ['High', 'Medium', 'Low'], sample_pct, method, value_col)
                
                if not sampled.empty:
                    sampled['Basis for Selection'] = method
                    sample_df = pd.concat([sample_df, sampled]).drop_duplicates(
                        subset=['Invoice no', 'Party name'] if all(col in sampled.columns for col in ['Invoice no', 'Party name']) else None
                    )
            
            if not sample_df.empty:
                # Sampling Summary
                st.markdown("### 📋 Sampling Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sample Size", len(sample_df))
                with col2:
                    sample_value = sample_df[value_col].sum() if value_col in sample_df.columns else 0
                    coverage = (sample_value / total_value * 100) if total_value > 0 else 0
                    st.metric("Value Coverage", f"{coverage:.1f}%")
                with col3:
                    st.metric("High Materiality Items Sampled", 
                            len(sample_df[sample_df['Materiality_Level'] == 'High']))

                # --- 10. TDS CALCULATIONS ---
                rates = {
                    '194C': 0.01, '194J': 0.10, '194I': 0.10, 
                    '194H': 0.05, '194Q': 0.001, '194IA': 0.01,
                    '194IB': 0.05, '194M': 0.05
                }
                
                tds_summary = df.groupby(['Party name', 'TDS Section']).agg({
                    'taxable value': 'sum', 
                    'TDS deducted': 'sum',
                }).reset_index() if all(col in df.columns for col in ['Party name', 'TDS Section']) else pd.DataFrame()
                
                if not tds_summary.empty:
                    tds_summary['TDS Needs to be Deducted'] = tds_summary.apply(
                        lambda r: r['taxable value'] * rates.get(str(r['TDS Section']).upper(), 0.01), 
                        axis=1
                    )
                    tds_summary['Shortfall'] = np.maximum(0, tds_summary['TDS Needs to be Deducted'] - tds_summary['TDS deducted'])
                    tds_summary['Interest (1.5% pm)'] = tds_summary['Shortfall'] * 0.015 * 3

                # --- 11. INTERACTIVE DASHBOARD ---
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Audit Overview", "📈 Sample Analysis", "💰 TDS Summary", "📑 Raw Data"])
                
                with tab1:
                    st.markdown("### 🎯 Key Audit Metrics")
                    
                    # Create metric cards
                    metric_cols = st.columns(4)
                    sample_value = sample_df[value_col].sum() if value_col in sample_df.columns else 0
                    tds_shortfall = tds_summary['Shortfall'].sum() if not tds_summary.empty else 0
                    
                    metrics = [
                        ("Total Population Value", f"₹{total_value:,.2f}"),
                        ("Sample Value", f"₹{sample_value:,.2f}"),
                        ("Sample Coverage", f"{(sample_value/total_value*100):.1f}%"),
                        ("Potential TDS Shortfall", f"₹{tds_shortfall:,.2f}")
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
                    display_cols = ['Party name', 'Invoice no', value_col, 'TDS Section', 'Materiality_Level']
                    display_cols = [col for col in display_cols if col in sample_df.columns]
                    
                    styled_sample = sample_df[display_cols].copy()
                    
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
                    if not tds_summary.empty:
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
                        if not top_shortfalls.empty:
                            fig_bar = px.bar(
                                top_shortfalls,
                                x='Party name',
                                y='Shortfall',
                                title="Top 10 TDS Shortfalls by Party",
                                color_discrete_sequence=['#f43b47']
                            )
                            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("TDS summary not available - required columns missing")
                
                with tab4:
                    st.markdown("### 📑 Complete Raw Data")
                    
                    # Raw data with filters
                    filter_col1, filter_col2 = st.columns(2)
                    with filter_col1:
                        if 'Party name' in df.columns:
                            selected_parties = st.multiselect("Filter by Party", df['Party name'].unique())
                        else:
                            selected_parties = []
                    with filter_col2:
                        selected_materiality = st.multiselect("Filter by Materiality", ['High', 'Medium', 'Low'])
                    
                    filtered_df = df.copy()
                    if selected_parties and 'Party name' in df.columns:
                        filtered_df = filtered_df[filtered_df['Party name'].isin(selected_parties)]
                    if selected_materiality:
                        filtered_df = filtered_df[filtered_df['Materiality_Level'].isin(selected_materiality)]
                    
                    st.dataframe(filtered_df, use_container_width=True, height=500)

                # --- 12. ENHANCED EXCEL EXPORT ---
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    out_bio = BytesIO()
                    
                    try:
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
                                'Materiality_Analysis': df[['Party name', 'Invoice no', value_col, 'Materiality_Level']] if all(col in df.columns for col in ['Party name', 'Invoice no']) else df,
                                'Selected_Samples': sample_df,
                                'TDS_Summary': tds_summary if not tds_summary.empty else pd.DataFrame({'Message': ['No TDS data available']})
                            }
                            
                            for sheet_name, data in sheet_map.items():
                                if not data.empty:
                                    data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
                                    worksheet = writer.sheets[sheet_name]
                                    
                                    # Write headers
                                    for col_num, col_name in enumerate(data.columns):
                                        worksheet.write(0, col_num, col_name, header_format)
                                    
                                    # Auto-adjust column widths
                                    for col_num, col_name in enumerate(data.columns):
                                        max_len = max(
                                            data[col_name].astype(str).map(len).max() if not data[col_name].empty else 0,
                                            len(col_name)
                                        ) + 2
                                        worksheet.set_column(col_num, col_num, min(max_len, 50))
                            
                            st.success("✅ Excel report generated successfully!")
                            st.download_button(
                                label="📥 Download Complete Audit Report (Excel)",
                                data=out_bio.getvalue(),
                                file_name=f"Ultra_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    except Exception as e:
                        st.error(f"Error generating Excel report: {str(e)}")
                        # Fallback to CSV download
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV Backup",
                            csv_data,
                            "audit_backup.csv",
                            "text/csv"
                        )
            else:
                st.warning("No samples generated. Please check your sampling methods and data.")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your file format matches the expected structure.")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1 style="color: #667eea;">👋 Welcome to Ultra-Audit Pro!</h1>
        <p style="color: #666; font-size: 18px; margin: 20px 0;">
            The most advanced audit sampling and TDS compliance platform with 
            intelligent materiality-based analysis.
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 40px; flex-wrap: wrap;">
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
        <div style="margin-top: 40px; padding: 20px; background: #f0f2f6; border-radius: 10px;">
            <h3>📋 Expected Column Format:</h3>
            <p>Date, Party name, Invoice no, Gross Total, taxable value, Input CGST, Input SGST, Input IGST, TDS deducted, TDS Section</p>
            <p style="color: #666; font-size: 14px;">The system will automatically detect variations of these column names!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
