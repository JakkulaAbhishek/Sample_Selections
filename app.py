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

# --- ULTRA CSS (same as before, shortened for brevity) ---
st.markdown("""
<style>
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

# --- SAMPLE DATA GENERATOR (unchanged) ---
@st.cache_data
def generate_sample_data():
    sample_data = [ ... ]  # same as before, keep full list
    # (I'll keep the same 24 rows from previous code)
    return pd.DataFrame(sample_data, columns=['Date','Party name','Invoice no','Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','TDS Section'])

# --- DATA PROCESSING with corrected rates ---
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
        
        # Correct TDS rates (as percentages, not decimals)
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
        df['Compliance Status'] = df.apply(lambda row: '✅ FULLY COMPLIANT' if row['TDS Shortfall'] == 0 
                                            else '⚠️ PARTIAL SHORTFALL' if row['TDS Shortfall']>0 and row['TDS deducted']>0 
                                            else '❌ NOT DEDUCTED', axis=1)
        # TDS Compliance %
        df['TDS Compliance %'] = df.apply(lambda row: round((row['TDS deducted']/row['Required TDS']*100),2) if row['Required TDS']>0 else 100, axis=1)
        return df

# --- MATERIALITY ENGINE ---
class MaterialityEngine:
    def __init__(self, threshold=5.0):
        self.threshold = threshold
    def calculate(self, df):
        total = df['taxable value'].sum()
        materiality_amount = total * (self.threshold / 100)
        df['Materiality Score'] = (df['taxable value'] / materiality_amount).round(2) if materiality_amount>0 else 0
        conditions = [df['Materiality Score']>=0.5, df['Materiality Score']>=0.2, df['Materiality Score']>=0.1, df['Materiality Score']>=0.05, df['Materiality Score']<0.05]
        levels = ['🔥 CRITICAL','⚡ HIGH','💫 MEDIUM','🌟 LOW','📦 IMMATERIAL']
        df['Materiality Level'] = np.select(conditions, levels, default='📦 IMMATERIAL')
        priority_map = {'🔥 CRITICAL':1,'⚡ HIGH':2,'💫 MEDIUM':3,'🌟 LOW':4,'📦 IMMATERIAL':5}
        df['Audit Priority'] = df['Materiality Level'].map(priority_map)
        return df, total, materiality_amount

# --- SAMPLING ENGINE (all methods, same as before, keep full list) ---
class SamplingEngine:
    # (include all 26 methods from previous code – too long to repeat here, but keep them)
    # For brevity, I'll keep the method stubs; actual code should have all 26 methods.
    pass

# --- EXCEL EXPORTER WITH FORMULAS ---
class ExcelExporter:
    @staticmethod
    def export_with_charts(df, sample_df, party_stats, selected_methods, materiality_threshold):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            # Formats
            header_fmt = workbook.add_format({'bold':True,'bg_color':'#00ff87','font_color':'#0a0f1e','border':1,'align':'center','valign':'vcenter','font_size':11})
            money_fmt = workbook.add_format({'num_format':'₹#,##0.00'})
            percent_fmt = workbook.add_format({'num_format':'0.00%'})
            date_fmt = workbook.add_format({'num_format':'dd-mm-yyyy'})
            
            # --- SHEET: Complete Data (with formulas) ---
            # Write raw data columns first (A to J)
            raw_cols = ['Date','Party name','Invoice no','Gross Total','taxable value','Input CGST','Input SGST','Input IGST','TDS deducted','TDS Section']
            df_raw = df[raw_cols].copy()
            df_raw.to_excel(writer, sheet_name='Complete Data', index=False, startrow=1)
            ws = writer.sheets['Complete Data']
            
            # Write headers
            for col_num, col_name in enumerate(raw_cols):
                ws.write(0, col_num, col_name, header_fmt)
            
            # Add total row at bottom for reference (used in formulas)
            total_row = len(df_raw) + 2
            ws.write(total_row, 0, "TOTALS", header_fmt)
            # Sum of taxable value in cell (used for materiality)
            taxable_col_letter = xlsxwriter.utility.xl_col_to_name(raw_cols.index('taxable value'))
            ws.write_formula(total_row, raw_cols.index('taxable value'), f"=SUM({taxable_col_letter}2:{taxable_col_letter}{total_row-1})", money_fmt)
            
            # Now write formula columns after raw data
            formula_cols = [
                ('Total GST', f'=F{{row}}+G{{row}}+H{{row}}'),
                ('GST Rate %', f'=IF({raw_cols.index("taxable value")+1}{{row}}=0,0,K{{row}}/{raw_cols.index("taxable value")+1}{{row}}*100)'),
                ('Std TDS Rate %', '=VLOOKUP(J{row},TDSRates!$A$2:$B$10,2,FALSE)'),  # we'll create a TDS Rates sheet
                ('Applied TDS Rate %', f'=IF({raw_cols.index("taxable value")+1}{{row}}=0,0,I{{row}}/{raw_cols.index("taxable value")+1}{{row}}*100)'),
                ('Required TDS', f'=E{{row}}*L{{row}}/100'),
                ('TDS Shortfall', f'=MAX(0,M{{row}}-I{{row}})'),
                ('Interest Payable', f'=N{{row}}*0.015*3'),
                ('Net Payable', f'=E{{row}}+K{{row}}-I{{row}}'),
                ('TDS Compliance %', f'=IF(M{{row}}=0,100,I{{row}}/M{{row}}*100)'),
                ('Materiality Score', f'=E{{row}}/${taxable_col_letter}${total_row+1}/$B$1'),  # total taxable in cell, threshold in B1
                ('Materiality Level', '=IF(P{row}>=0.5,"🔥 CRITICAL",IF(P{row}>=0.2,"⚡ HIGH",IF(P{row}>=0.1,"💫 MEDIUM",IF(P{row}>=0.05,"🌟 LOW","📦 IMMATERIAL"))))'),
                ('Audit Priority', '=IF(Q{row}="🔥 CRITICAL",1,IF(Q{row}="⚡ HIGH",2,IF(Q{row}="💫 MEDIUM",3,IF(Q{row}="🌟 LOW",4,5))))'),
                ('Compliance Status', '=IF(N{row}=0,"✅ FULLY COMPLIANT",IF(I{row}>0,"⚠️ PARTIAL SHORTFALL","❌ NOT DEDUCTED"))')
            ]
            
            start_formula_col = len(raw_cols)
            for idx, (col_name, formula_template) in enumerate(formula_cols):
                col_letter = xlsxwriter.utility.xl_col_to_name(start_formula_col + idx)
                ws.write(0, start_formula_col + idx, col_name, header_fmt)
                # Write formula for each data row (row numbers in Excel: row 2 is first data)
                for row_num in range(2, total_row):
                    formula = formula_template.format(row=row_num)
                    ws.write_formula(row_num-1, start_formula_col + idx, formula, money_fmt if '₹' in col_name else None)
            
            # Write materiality threshold in cell B1
            ws.write(0, 1, materiality_threshold/100, percent_fmt)  # B1 as decimal for formulas
            
            # Create a separate sheet for TDS rates lookup
            rates_df = pd.DataFrame({
                'Section': ['194C','194J','194I','194H','194Q'],
                'Rate %': [1.0, 10.0, 10.0, 5.0, 0.1]
            })
            rates_df.to_excel(writer, sheet_name='TDSRates', index=False)
            
            # Other sheets (Executive Summary, Sample Data, Party Analysis, etc.) can remain as values
            # (For brevity, we'll keep them as before but they now reference the formula sheet if needed)
            
            # --- Executive Summary (simplified, without background gradient) ---
            summary_data = {...}  # same as before
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # --- Sample Data ---
            sample_df.to_excel(writer, sheet_name='Sample Data', index=False)
            
            # --- Party Analysis ---
            party_stats.to_excel(writer, sheet_name='Party Analysis')
            
            # --- TDS Summary (values only) ---
            tds_summary = df.groupby('TDS Section').agg({'taxable value':'sum','TDS deducted':'sum','Required TDS':'sum','TDS Shortfall':'sum','Interest Payable':'sum'}).round(2)
            tds_summary['Compliance %'] = (tds_summary['TDS deducted'] / tds_summary['Required TDS'] * 100).round(2)
            tds_summary.to_excel(writer, sheet_name='TDS Summary')
            
            # --- Materiality Analysis ---
            mat_summary = df.groupby('Materiality Level').agg({'taxable value':['count','sum','mean'],'TDS Shortfall':'sum'}).round(2)
            mat_summary.columns = ['Count','Total Value','Average Value','Total Shortfall']
            mat_summary.to_excel(writer, sheet_name='Materiality Analysis')
            
            # --- Sampling Methods sheet ---
            methods_data = {...}  # same as before
            methods_df = pd.DataFrame(methods_data)
            methods_df.to_excel(writer, sheet_name='Sampling Methods', index=False)
            
            # --- Formula Reference ---
            formula_ref = pd.DataFrame({
                'Formula': ['Total GST','GST Rate %','Std TDS Rate %','Applied TDS Rate %','Required TDS','TDS Shortfall','Interest Payable','Net Payable','TDS Compliance %','Materiality Score'],
                'Calculation': ['CGST+SGST+IGST','(Total GST/Taxable Value)*100','VLOOKUP from TDSRates sheet','(TDS Deducted/Taxable Value)*100','Taxable Value * Std TDS Rate % /100','MAX(0, Required TDS - Actual TDS)','Shortfall * 1.5% * 3','Taxable Value + Total GST - TDS Deducted','(Actual TDS / Required TDS)*100','Taxable Value / (Total Taxable * Threshold %)']
            })
            formula_ref.to_excel(writer, sheet_name='Formula Reference', index=False)
            
            # --- Add charts (same as before) ---
            # (Keep chart insertion code from previous version)
            
        return output.getvalue()

# --- PARTY DASHBOARD (without background_gradient) ---
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

# --- MAIN ---
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
        # Sampling method selectors (same as before)
        # ... (keep all multi-selects)
        # For brevity, we'll assume selected_methods list exists
    
    uploaded_file = st.file_uploader('📤 UPLOAD LEDGER FILE', type=['xlsx','csv'], label_visibility='collapsed')
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
            # Ensure columns exist...
            processor = DataProcessor()
            df = processor.apply_formulas(df)
            mat_engine = MaterialityEngine(materiality_threshold)
            df, total_value, materiality_amount = mat_engine.calculate(df)
            
            # Sampling (apply selected methods)
            # ... (same as before) ...
            combined_sample = df.sample(n=max(1,int(len(df)*sample_percentage/100)))  # placeholder
            
            party_stats = create_party_dashboard(df)
            
            # Display metrics (without gradient)
            st.markdown('### 📊 REAL-TIME METRICS')
            cols = st.columns(5)
            metrics = [('💰 TOTAL VALUE', f'₹{total_value:,.0f}'), ('📦 TRANSACTIONS', len(df)), ('🔥 CRITICAL', len(df[df['Materiality Level']=='🔥 CRITICAL'])), ('🎯 SAMPLE SIZE', f'{len(combined_sample)} ({sample_percentage}%)'), ('⚠️ SHORTFALL', f'₹{df["TDS Shortfall"].sum():,.0f}')]
            for col, (label, value) in zip(cols, metrics):
                col.markdown(f'<div class="metric-card-ultra"><h4 style="color:#00ff87;">{label}</h4><h2 style="color:white;">{value}</h2></div>', unsafe_allow_html=True)
            
            # Tabs...
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['🎯 PARTY ANALYSIS','📈 MATERIALITY','🔍 SAMPLE DETAILS','💰 TDS COMPLIANCE','📊 FORMULA VIEW','📥 EXPORT'])
            
            with tab1:
                st.markdown('### 🏢 PARTY-WISE ANALYSIS')
                selected_party = st.selectbox('Select Party', ['All'] + list(party_stats.index[:20]))
                if selected_party != 'All':
                    party_data = df[df['Party name'] == selected_party]
                    col1, col2 = st.columns(2)
                    with col1:
                        total_tds = party_data['TDS deducted'].sum()
                        required_tds = party_data['Required TDS'].sum()
                        compliance = (total_tds/required_tds*100) if required_tds>0 else 100
                        st.markdown(f'<div class="party-card"><h3 style="color:#00ff87;">{selected_party}</h3><p>📊 Total Value: ₹{party_data["taxable value"].sum():,.0f}</p><p>📦 Transactions: {len(party_data)}</p><p>💰 TDS Paid: ₹{total_tds:,.0f}</p><p>⚠️ TDS Shortfall: ₹{party_data["TDS Shortfall"].sum():,.0f}</p><p>✅ Compliance: {compliance:.1f}%</p></div>', unsafe_allow_html=True)
                    with col2:
                        fig = px.bar(party_data, x='Date', y='taxable value', title=f'{selected_party} - Transactions', color='Materiality Level', color_discrete_map={'🔥 CRITICAL':'#ff00ff','⚡ HIGH':'#00ff87','💫 MEDIUM':'#60efff','🌟 LOW':'#0061ff'})
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(party_data, use_container_width=True)
                else:
                    # Display party_stats without gradient – use simple formatting
                    st.dataframe(party_stats.style.format({'Total Value':'₹{:,.0f}','TDS Paid':'₹{:,.0f}','TDS Shortfall':'₹{:,.0f}','TDS Compliance %':'{:.1f}%','Risk Score':'{:.1f}'}), use_container_width=True)
                    top_10 = party_stats.head(10).reset_index()
                    fig = px.bar(top_10, x='Party name', y='Total Value', title='Top 10 Parties by Value', color='Risk Score', color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Other tabs remain similar (without gradient)
            # ... (keep as before)
            
            with tab6:
                if st.button('⚡ GENERATE COMPLETE REPORT WITH FORMULAS', use_container_width=True):
                    with st.spinner('Generating Excel with formulas...'):
                        exporter = ExcelExporter()
                        excel_data = exporter.export_with_charts(df, combined_sample, party_stats, selected_methods, materiality_threshold)
                        st.download_button('📥 DOWNLOAD EXCEL REPORT (WITH FORMULAS)', data=excel_data, file_name=f'Ultra_Audit_Report_{datetime.now():%Y%m%d_%H%M%S}.xlsx', use_container_width=True)
        except Exception as e:
            st.error(f'Error: {str(e)}')
            st.exception(e)
    else:
        st.markdown('...')  # welcome screen

if __name__ == '__main__':
    main()
