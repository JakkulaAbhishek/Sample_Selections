# =============================================================================
# ULTRA AUDIT INTELLIGENCE v2.1
# -----------------------------------------------------------------------------
# Enterprise‑grade Sample + TDS Check Platform
# 30+ Sampling Methods | Dynamic TDS Engine | Materiality Analysis | Export
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import warnings
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import xlsxwriter

# Optional sklearn – fallback gracefully
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    """Central configuration for the audit tool."""
    DATE_FORMAT = "%d-%m-%Y"
    SAMPLE_SEED = 42
    INTEREST_RATE_PER_MONTH = 0.015  # 1.5% per month
    MATERIALITY_DEFAULT = 500000
    SAMPLE_PERCENT_DEFAULT = 20

    # TDS Rate Table: Section -> (Rate %, Limit, Category specific note)
    TDS_RATES = {
        '192': ('Slab rates', 'Basic exemption limit', 'Salary'),
        '192A': (10, 50000, 'Premature EPF withdrawal'),
        '193': (10, 10000, 'Interest on securities'),
        '194': (10, 10000, 'Dividends'),
        '194A': (10, 50000, 'Interest (Bank/Post Office)'),
        '194B': (30, 10000, 'Winnings – Lottery/Puzzle'),
        '194BA': (30, 0, 'Online gaming winnings'),
        '194BB': (30, 10000, 'Horse races'),
        '194C': ('dynamic', 100000, 'Contractor – rate depends on PAN/GSTIN'),
        '194D': ('dynamic', 20000, 'Insurance commission – 2% for Ind/HUF, 10% for others'),
        '194DA': (2, 100000, 'Life insurance policy'),
        '194EE': (10, 2500, 'NSS deposits'),
        '194G': (2, 20000, 'Lottery commission'),
        '194H': (2, 20000, 'Commission/Brokerage'),
        '194I': ('dynamic', 600000, 'Rent – 2% for P&M, 10% for land/building'),
        '194IB': (2, 600000, 'Rent (Ind/HUF not under 194I)'),
        '194J(a)': (2, 50000, 'Tech services/Royalty/Call Centre'),
        '194J(b)': (10, 50000, 'Professional services'),
        '194LA': (10, 500000, 'Enhanced compensation (property)'),
        '194M': (2, 5000000, 'Contracts/Professional fees'),
        '194N': ('dynamic', 2000000, 'Cash withdrawal – 2% normal, 5% non-filer'),
        '194O': (0.10, 500000, 'E‑commerce participants'),
        '194P': ('Slab rates', 'Basic exemption', 'Specified Senior Citizen'),
        '194Q': (0.10, 5000000, 'Purchase of goods'),
        '194R': (10, 20000, 'Benefits/perquisites'),
        '194S': ('dynamic', 10000, 'Virtual digital assets – 1% for normal, 1% for specified'),
        '194T': (10, 20000, 'Payment to partner of firm')
    }

    # Mapping for 194C rate based on entity type (4th char of PAN / 6th of GSTIN)
    ENTITY_RATE_194C = {
        'P': 1,   # Individual
        'F': 2,   # Firm
        'C': 2,   # Company
        'H': 1,   # HUF
        'A': 2,   # AOP
        'T': 2,   # Trust
        'L': 2    # LLP
    }

    # 194D rate based on entity type
    ENTITY_RATE_194D = {
        'P': 2,   # Individual/HUF → 2%
        'F': 10,  # Others → 10%
        'C': 10,
        'H': 2,
        'A': 10,
        'T': 10,
        'L': 10
    }

    # 194I rate – default 10% (can be overridden in UI)
    RATE_194I_DEFAULT = 10

    # 194N rate – default 2%
    RATE_194N_DEFAULT = 2


# =============================================================================
# DATA MODELS
# =============================================================================
@dataclass
class AuditRecord:
    """Represents a single transaction after processing."""
    date: str
    party: str
    gstin: str
    invoice: str
    gross: float
    taxable: float
    cgst: float
    sgst: float
    igst: float
    tds_deducted: float
    section: str
    total_gst: float = 0.0
    gst_rate: float = 0.0
    party_section_total: float = 0.0
    tds_applicable: bool = False
    std_tds_rate: float = 0.0
    applied_tds_rate: float = 0.0
    required_tds: float = 0.0
    tds_shortfall: float = 0.0
    interest: float = 0.0
    compliance_status: str = ""
    compliance_pct: float = 100.0
    materiality_level: str = ""
    audit_priority: int = 5
    sampling_method: str = ""

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def safe_float(val: Any) -> float:
    """Convert to float, handling commas and currency symbols."""
    if pd.isna(val) or val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in ('na', 'n/a', '-'):
        return 0.0
    cleaned = re.sub(r'[₹,\sRs\.INR]', '', s)
    try:
        return float(cleaned)
    except:
        return 0.0


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string with fallback."""
    if pd.isna(date_str) or not str(date_str).strip():
        return None
    for fmt in (Config.DATE_FORMAT, '%d/%m/%Y', '%Y-%m-%d', '%d-%b-%Y'):
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except:
            continue
    return None


def get_entity_type(identifier: str) -> str:
    """Extract entity type from PAN (4th char) or GSTIN (6th char)."""
    if not identifier or len(identifier) < 6:
        return 'P'  # default Individual
    ident = str(identifier).strip().upper()
    if len(ident) == 10:
        return ident[3]  # PAN
    elif len(ident) == 15:
        return ident[5]  # GSTIN
    return 'P'


def get_tds_rate(section: str, identifier: str = '') -> float:
    """
    Return the TDS rate (percentage) for a given section and entity identifier.
    Handles dynamic sections: 194C, 194D, 194I, 194N, 194S.
    """
    section = str(section).strip().upper()
    if section not in Config.TDS_RATES:
        return 0.0
    rate_info = Config.TDS_RATES[section]
    rate = rate_info[0]

    # Dynamic rate handling
    if section == '194C':
        entity = get_entity_type(identifier)
        return Config.ENTITY_RATE_194C.get(entity, 2)
    elif section == '194D':
        entity = get_entity_type(identifier)
        return Config.ENTITY_RATE_194D.get(entity, 10)
    elif section == '194I':
        # For simplicity, we use 10% (could be enhanced with asset type selection)
        return 10.0
    elif section == '194N':
        return 2.0  # default; could be 5% if non-filer
    elif section == '194S':
        return 1.0  # flat 1%
    else:
        try:
            return float(rate)
        except:
            return 0.0


def get_tds_limit(section: str) -> float:
    """Return the cumulative limit for a section."""
    section = str(section).strip().upper()
    if section not in Config.TDS_RATES:
        return 0.0
    limit = Config.TDS_RATES[section][1]
    try:
        return float(limit)
    except:
        return 0.0


# =============================================================================
# DATA PROCESSOR
# =============================================================================
class DataProcessor:
    """Cleans, computes, and enriches the transaction data."""

    @staticmethod
    def load_data(file) -> pd.DataFrame:
        """Load CSV or Excel with column mapping."""
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        # Normalise column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        expected = ['date', 'party_name', 'gst_number', 'invoice_no',
                    'gross_total', 'taxable_value', 'input_cgst', 'input_sgst',
                    'input_igst', 'tds_deducted', 'tds_section']
        # Map possible variations
        mapping = {
            'date': 'date',
            'party name': 'party_name',
            'party_name': 'party_name',
            'gst number': 'gst_number',
            'gst_number': 'gst_number',
            'invoice no': 'invoice_no',
            'invoice_no': 'invoice_no',
            'gross total': 'gross_total',
            'gross_total': 'gross_total',
            'taxable value': 'taxable_value',
            'taxable_value': 'taxable_value',
            'input cgst': 'input_cgst',
            'input_cgst': 'input_cgst',
            'input sgst': 'input_sgst',
            'input_sgst': 'input_sgst',
            'input igst': 'input_igst',
            'input_igst': 'input_igst',
            'tds deducted': 'tds_deducted',
            'tds_deducted': 'tds_deducted',
            'tds section': 'tds_section',
            'tds_section': 'tds_section'
        }
        for old, new in mapping.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        # Ensure all required columns exist
        required = ['date', 'party_name', 'gst_number', 'invoice_no',
                    'gross_total', 'taxable_value', 'input_cgst', 'input_sgst',
                    'input_igst', 'tds_deducted', 'tds_section']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Please check your file format.")
        return df[required]

    @staticmethod
    def process(df: pd.DataFrame, interest_months: int = 3) -> pd.DataFrame:
        """Apply all calculations and enrichments."""
        df = df.copy()
        # Ensure numeric types
        num_cols = ['gross_total', 'taxable_value', 'input_cgst', 'input_sgst',
                    'input_igst', 'tds_deducted']
        for col in num_cols:
            df[col] = df[col].apply(safe_float)

        # Basic derived fields
        df['total_gst'] = df['input_cgst'] + df['input_sgst'] + df['input_igst']
        df['gst_rate'] = np.where(df['taxable_value'] > 0,
                                  (df['total_gst'] / df['taxable_value']) * 100, 0)

        # Party + Section total
        df['party_section'] = df['party_name'] + '||' + df['tds_section'].astype(str)
        df['party_section_total'] = df.groupby('party_section')['taxable_value'].transform('sum')

        # TDS applicability based on cumulative limit
        df['tds_limit'] = df['tds_section'].apply(get_tds_limit)

        def is_tds_applicable(row):
            section = str(row['tds_section']).strip().upper()
            if section == '194C':
                # Special rule: if total < 100k, then per invoice > 30k triggers TDS
                if row['party_section_total'] < 100000:
                    return row['taxable_value'] > 30000
                else:
                    return True
            else:
                return row['party_section_total'] > row['tds_limit']

        df['tds_applicable'] = df.apply(is_tds_applicable, axis=1)

        # Standard TDS rate (based on identifier)
        df['std_tds_rate'] = df.apply(
            lambda r: get_tds_rate(r['tds_section'], r['gst_number']), axis=1
        )

        # Required TDS
        df['required_tds'] = np.where(
            df['tds_applicable'],
            (df['taxable_value'] * df['std_tds_rate'] / 100).round(2),
            0.0
        )

        # Applied rate
        df['applied_tds_rate'] = np.where(
            df['taxable_value'] > 0,
            (df['tds_deducted'] / df['taxable_value']) * 100,
            0.0
        ).round(2)

        # Shortfall
        df['tds_shortfall'] = (df['tds_deducted'] - df['required_tds']).round(2)

        # Interest
        df['interest'] = np.maximum(0, df['tds_shortfall']) * Config.INTEREST_RATE_PER_MONTH * interest_months
        df['interest'] = df['interest'].round(2)

        # Compliance status
        conditions = [
            df['tds_shortfall'] == 0,
            (df['tds_shortfall'] > 0) & (df['tds_deducted'] > 0),
            (df['tds_deducted'] == 0) & (df['tds_applicable'])
        ]
        choices = ['✅ FULLY COMPLIANT', '⚠️ PARTIAL SHORTFALL', '❌ NOT DEDUCTED']
        df['compliance_status'] = np.select(conditions, choices, default='✅ FULLY COMPLIANT')

        # Compliance %
        df['compliance_pct'] = np.where(
            df['required_tds'] > 0,
            (df['tds_deducted'] / df['required_tds'] * 100).clip(upper=100).round(2),
            100.0
        )

        # Drop temporary column
        df.drop(columns=['party_section'], inplace=True)
        return df


# =============================================================================
# MATERIALITY ENGINE
# =============================================================================
class MaterialityEngine:
    """Assigns materiality levels based on transaction value relative to threshold."""
    THRESHOLD_PERCENTS = [0.5, 0.2, 0.1, 0.05]  # fractions of materiality amount
    LEVELS = ['🔥 CRITICAL', '⚡ HIGH', '💫 MEDIUM', '🌟 LOW', '📦 IMMATERIAL']

    @classmethod
    def apply(cls, df: pd.DataFrame, materiality_amount: float) -> pd.DataFrame:
        df = df.copy()
        df['materiality_score'] = df['taxable_value'] / materiality_amount if materiality_amount > 0 else 0
        conditions = [
            df['materiality_score'] >= cls.THRESHOLD_PERCENTS[0],
            df['materiality_score'] >= cls.THRESHOLD_PERCENTS[1],
            df['materiality_score'] >= cls.THRESHOLD_PERCENTS[2],
            df['materiality_score'] >= cls.THRESHOLD_PERCENTS[3],
            df['materiality_score'] < cls.THRESHOLD_PERCENTS[3]
        ]
        df['materiality_level'] = np.select(conditions, cls.LEVELS, default=cls.LEVELS[-1])
        priority_map = {level: idx+1 for idx, level in enumerate(cls.LEVELS)}
        df['audit_priority'] = df['materiality_level'].map(priority_map)
        return df


# =============================================================================
# SAMPLING ENGINE – 30+ Methods
# =============================================================================
class SamplingEngine:
    """Factory for all sampling methods. Each method returns a sampled DataFrame."""
    SEED = Config.SAMPLE_SEED
    np.random.seed(SEED)

    @staticmethod
    def simple_random(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        return df.sample(n=min(n, len(df)), random_state=SamplingEngine.SEED)

    @staticmethod
    def systematic(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        if len(df) <= n:
            return df
        step = len(df) // n
        start = np.random.randint(0, step)
        indices = list(range(start, len(df), step))[:n]
        return df.iloc[indices]

    @staticmethod
    def stratified(df: pd.DataFrame, pct: float, strata_col: str = 'materiality_level') -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        samples = []
        for stratum in df[strata_col].unique():
            stratum_df = df[df[strata_col] == stratum]
            stratum_n = max(1, int(n * len(stratum_df) / len(df)))
            samples.append(stratum_df.sample(n=min(stratum_n, len(stratum_df)),
                                            random_state=SamplingEngine.SEED))
        result = pd.concat(samples).drop_duplicates()
        if len(result) < n:
            # fill remaining with random from unsampled
            remaining = df[~df.index.isin(result.index)]
            if not remaining.empty:
                extra = remaining.sample(n=min(n-len(result), len(remaining)),
                                         random_state=SamplingEngine.SEED)
                result = pd.concat([result, extra])
        return result.head(n)

    @staticmethod
    def cluster(df: pd.DataFrame, pct: float, n_clusters: int = 5) -> pd.DataFrame:
        """Cluster sampling – falls back to simple random if sklearn not available."""
        if not SKLEARN_AVAILABLE:
            return SamplingEngine.simple_random(df, pct)
        if 'taxable_value' not in df.columns or len(df) < n_clusters:
            return SamplingEngine.simple_random(df, pct)
        try:
            X = df[['taxable_value']].values
            kmeans = KMeans(n_clusters=n_clusters, random_state=SamplingEngine.SEED, n_init=10)
            df_copy = df.copy()
            df_copy['cluster'] = kmeans.fit_predict(X)
            n = max(1, int(len(df) * pct / 100))
            n_clusters_to_select = max(1, int(n_clusters * pct / 100))
            selected = np.random.choice(df_copy['cluster'].unique(), n_clusters_to_select, replace=False)
            return df_copy[df_copy['cluster'].isin(selected)].drop(columns=['cluster'])
        except:
            return SamplingEngine.simple_random(df, pct)

    @staticmethod
    def multistage(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        parties = df['party_name'].unique()
        n_parties = max(1, int(len(parties) * pct / 100))
        selected = np.random.choice(parties, n_parties, replace=False)
        samples = []
        for party in selected:
            party_df = df[df['party_name'] == party]
            party_n = max(1, int(len(party_df) * pct / 100))
            samples.append(party_df.sample(n=min(party_n, len(party_df)),
                                           random_state=SamplingEngine.SEED))
        return pd.concat(samples).drop_duplicates()

    @staticmethod
    def multiphase(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        phase1 = SamplingEngine.simple_random(df, pct * 0.5)
        remaining = df[~df.index.isin(phase1.index)]
        high_value = remaining.nlargest(int(len(df) * pct * 0.3), 'taxable_value')
        target = max(1, int(len(df) * pct / 100))
        result = pd.concat([phase1, high_value]).drop_duplicates()
        if len(result) < target:
            extra = remaining[~remaining.index.isin(high_value.index)]
            if not extra.empty:
                add = extra.sample(n=min(target-len(result), len(extra)),
                                   random_state=SamplingEngine.SEED)
                result = pd.concat([result, add])
        return result.head(target)

    @staticmethod
    def area(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['area'] = df_copy['party_name'].str[0].str.upper()
        areas = df_copy['area'].unique()
        n_areas = max(1, int(len(areas) * pct / 100))
        selected = np.random.choice(areas, n_areas, replace=False)
        return df_copy[df_copy['area'].isin(selected)].drop(columns=['area'])

    @staticmethod
    def pps(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        if 'taxable_value' not in df.columns or df['taxable_value'].sum() == 0:
            return SamplingEngine.simple_random(df, pct)
        n = max(1, int(len(df) * pct / 100))
        probs = df['taxable_value'] / df['taxable_value'].sum()
        return df.sample(n=min(n, len(df)), weights=probs, random_state=SamplingEngine.SEED)

    @staticmethod
    def convenience(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        return df.head(n)

    @staticmethod
    def judgmental(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        return df.nlargest(n, 'taxable_value')

    @staticmethod
    def purposive(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        df_copy = df.copy()
        df_copy['priority_score'] = (
            (df_copy['materiality_level'] == '🔥 CRITICAL') * 100 +
            (df_copy['tds_shortfall'] > 0) * 50 +
            df_copy['taxable_value'] / df_copy['taxable_value'].max() * 30
        )
        return df_copy.nlargest(n, 'priority_score').drop(columns=['priority_score'])

    @staticmethod
    def quota(df: pd.DataFrame, pct: float, quota_col: str = 'materiality_level') -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        samples = []
        strata = df[quota_col].unique()
        quota_per = max(1, int(n / len(strata)))
        for stratum in strata:
            stratum_df = df[df[quota_col] == stratum]
            samples.append(stratum_df.head(quota_per))
        result = pd.concat(samples).drop_duplicates()
        return result.head(n)

    @staticmethod
    def snowball(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        if len(df) == 0:
            return df
        seed_idx = np.random.randint(0, len(df))
        seed_party = df.iloc[seed_idx]['party_name']
        sample = df[df['party_name'] == seed_party]
        while len(sample) < n:
            current_parties = sample['party_name'].unique()
            current_sections = sample['tds_section'].unique()
            connected = df[df['tds_section'].isin(current_sections)]
            connected = connected[~connected['party_name'].isin(current_parties)]
            if connected.empty:
                break
            next_party = connected['party_name'].iloc[0]
            sample = pd.concat([sample, df[df['party_name'] == next_party]])
        return sample.head(n)

    @staticmethod
    def volunteer(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        probs = df['taxable_value'] / df['taxable_value'].sum() if df['taxable_value'].sum() > 0 else None
        if probs is not None:
            probs = probs ** 0.5
            probs = probs / probs.sum()
        return df.sample(n=min(n, len(df)), weights=probs, random_state=SamplingEngine.SEED)

    @staticmethod
    def haphazard(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        indices = np.random.choice(len(df), size=min(n, len(df)), replace=False)
        return df.iloc[indices]

    @staticmethod
    def consecutive(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        start = np.random.randint(0, max(1, len(df) - n))
        return df.iloc[start:start+n]

    @staticmethod
    def statistical(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        return SamplingEngine.stratified(df, pct)

    @staticmethod
    def non_statistical(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        return SamplingEngine.judgmental(df, pct)

    @staticmethod
    def mus(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        if 'taxable_value' not in df.columns or df['taxable_value'].sum() == 0:
            return SamplingEngine.simple_random(df, pct)
        n = max(1, int(len(df) * pct / 100))
        total = df['taxable_value'].sum()
        interval = total / n
        df_sorted = df.sort_values('taxable_value', ascending=False).reset_index(drop=True)
        df_sorted['cum'] = df_sorted['taxable_value'].cumsum()
        samples = []
        current = interval
        for _, row in df_sorted.iterrows():
            if row['cum'] >= current and len(samples) < n:
                samples.append(row)
                current += interval
        return pd.DataFrame(samples) if samples else df.sample(n=n, random_state=SamplingEngine.SEED)

    @staticmethod
    def block(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        if 'date' in df.columns:
            df_copy = df.copy()
            df_copy['month'] = pd.to_datetime(df_copy['date'], format=Config.DATE_FORMAT, errors='coerce').dt.month
            block_col = 'month'
        else:
            df_copy = df.copy()
            df_copy['block'] = pd.qcut(df['taxable_value'], q=5, labels=['B1','B2','B3','B4','B5'])
            block_col = 'block'
        blocks = df_copy[block_col].unique()
        selected = np.random.choice(blocks)
        block_df = df_copy[df_copy[block_col] == selected]
        if len(block_df) >= n:
            return block_df.head(n).drop(columns=[block_col])
        else:
            remaining = [b for b in blocks if b != selected]
            if remaining:
                second = np.random.choice(remaining)
                second_df = df_copy[df_copy[block_col] == second]
                result = pd.concat([block_df, second_df]).head(n)
                return result.drop(columns=[block_col])
        return df.head(n)

    @staticmethod
    def sequential(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        if 'date' in df.columns:
            df_sorted = df.sort_values('date')
        else:
            df_sorted = df.sort_values('taxable_value', ascending=False)
        return df_sorted.head(n)

    @staticmethod
    def adaptive(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        high = df[df['taxable_value'] > df['taxable_value'].quantile(0.75)]
        if len(high) >= n:
            return high.head(n)
        remaining_n = n - len(high)
        mid = df[(df['taxable_value'] <= df['taxable_value'].quantile(0.75)) &
                 (df['taxable_value'] > df['taxable_value'].quantile(0.5))]
        sample = pd.concat([high, mid.head(remaining_n)])
        if len(sample) < n:
            low = df[df['taxable_value'] <= df['taxable_value'].quantile(0.5)]
            extra = n - len(sample)
            sample = pd.concat([sample, low.sample(n=min(extra, len(low)),
                                                   random_state=SamplingEngine.SEED)])
        return sample

    @staticmethod
    def reservoir(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
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
    def acceptance(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        df_copy = df.copy()
        df_copy['quality'] = ((df_copy['compliance_pct'] < 90) * 100 +
                              (df_copy['tds_shortfall'] > 0) * 50 +
                              (df_copy['materiality_level'] == '🔥 CRITICAL') * 30)
        weights = df_copy['quality'] / df_copy['quality'].sum()
        return df_copy.sample(n=min(n, len(df)), weights=weights,
                              random_state=SamplingEngine.SEED).drop(columns=['quality'])

    @staticmethod
    def bootstrap(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        return df.sample(n=min(n*2, len(df)*2), replace=True,
                         random_state=SamplingEngine.SEED).drop_duplicates().head(n)

    @staticmethod
    def bayesian(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        n = max(1, int(len(df) * pct / 100))
        if 'taxable_value' in df.columns and df['taxable_value'].sum() > 0:
            value_prior = df['taxable_value'] / df['taxable_value'].sum()
            compliance_prior = (100 - df['compliance_pct'].fillna(0)) / 100
            prior = value_prior * 0.7 + compliance_prior * 0.3
            prior = prior / prior.sum()
            return df.sample(n=min(n, len(df)), weights=prior,
                             random_state=SamplingEngine.SEED)
        return df.sample(n=min(n, len(df)), random_state=SamplingEngine.SEED)

    # Map method names to functions
    METHOD_MAP = {
        'Simple Random Sampling': simple_random,
        'Systematic Sampling': systematic,
        'Stratified Sampling': stratified,
        'Cluster Sampling': cluster,
        'Multistage Sampling': multistage,
        'Multiphase Sampling': multiphase,
        'Area Sampling': area,
        'Probability Proportional to Size (PPS) Sampling': pps,
        'Convenience Sampling': convenience,
        'Judgmental Sampling': judgmental,
        'Purposive Sampling': purposive,
        'Quota Sampling': quota,
        'Snowball Sampling': snowball,
        'Volunteer Sampling': volunteer,
        'Haphazard Sampling': haphazard,
        'Consecutive Sampling': consecutive,
        'Statistical Sampling': statistical,
        'Non-Statistical Sampling': non_statistical,
        'Monetary Unit Sampling (MUS)': mus,
        'Block Sampling': block,
        'Sequential Sampling': sequential,
        'Adaptive Sampling': adaptive,
        'Reservoir Sampling': reservoir,
        'Acceptance Sampling': acceptance,
        'Bootstrap Sampling': bootstrap,
        'Bayesian Sampling': bayesian
    }

    @classmethod
    def get_method_names(cls) -> List[str]:
        return list(cls.METHOD_MAP.keys())

    @classmethod
    def apply_method(cls, df: pd.DataFrame, method: str, pct: float) -> pd.DataFrame:
        if method in cls.METHOD_MAP:
            return cls.METHOD_MAP[method](cls, df, pct)
        else:
            return cls.simple_random(df, pct)


# =============================================================================
# SAMPLING DESCRIPTIONS (with examples)
# =============================================================================
SAMPLING_DESCRIPTIONS = {
    'Simple Random Sampling': 'Every item has an equal chance. Example: Assign random numbers to 1,000 invoices and select the 100 smallest.',
    'Systematic Sampling': 'Select every kth item after random start. Example: Start at 5, then pick 15, 25, ... up to 100 items (k=10).',
    'Stratified Sampling': 'Divide into strata (e.g., materiality levels) and sample proportionally from each.',
    'Cluster Sampling': 'Select entire groups (clusters) at random. Example: Pick 3 out of 10 branches and audit all invoices in those branches.',
    'Multistage Sampling': 'First select clusters, then sample within clusters. Example: Pick 5 branches, then 20 invoices per branch.',
    'Multiphase Sampling': 'Collect preliminary data from a large sample, then subsample for detailed review.',
    'Area Sampling': 'Similar to cluster but based on geographic areas (postal codes).',
    'Probability Proportional to Size (PPS) Sampling': 'Higher value items have higher selection probability.',
    'Convenience Sampling': 'Select easiest items (e.g., first 50 invoices).',
    'Judgmental Sampling': 'Auditor selects based on risk (e.g., all high‑value transactions).',
    'Purposive Sampling': 'Select items for a specific purpose (e.g., all critical items).',
    'Quota Sampling': 'Fill predefined quotas per category (e.g., 25 items from each materiality level).',
    'Snowball Sampling': 'Start with a seed, then follow connections (e.g., same party or section).',
    'Volunteer Sampling': 'Items self‑select (e.g., departments submit high‑risk transactions).',
    'Haphazard Sampling': 'Arbitrary selection without a formal random process.',
    'Consecutive Sampling': 'Select a contiguous block (e.g., all invoices from last week).',
    'Statistical Sampling': 'Uses probability theory; results can be projected statistically.',
    'Non-Statistical Sampling': 'Based on auditor judgment; no statistical projection.',
    'Monetary Unit Sampling (MUS)': 'Each monetary unit is a sampling unit; high‑value items more likely.',
    'Block Sampling': 'Select a contiguous block (e.g., all invoices from a specific month).',
    'Sequential Sampling': 'Select items in sequence until a stopping rule is met.',
    'Adaptive Sampling': 'Sampling intensity increases in areas with more errors.',
    'Reservoir Sampling': 'Maintains a random sample without knowing total population size.',
    'Acceptance Sampling': 'Used to accept/reject a population based on sample error rate.',
    'Bootstrap Sampling': 'Resampling with replacement to estimate sampling distribution.',
    'Bayesian Sampling': 'Combines prior information with sample evidence to update probabilities.'
}


# =============================================================================
# EXCEL EXPORTER (Enhanced)
# =============================================================================
class ExcelExporter:
    """Generates a multi‑sheet Excel workbook with formulas and embedded charts."""

    @staticmethod
    def col_letter(idx: int) -> str:
        letter = ''
        while idx >= 0:
            letter = chr(idx % 26 + 65) + letter
            idx = idx // 26 - 1
        return letter

    @classmethod
    def export(cls, df: pd.DataFrame, sample_df: pd.DataFrame,
               party_stats: pd.DataFrame, selected_methods: List[str],
               materiality_amount: float, interest_months: int) -> bytes:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            # Formats
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#1e3a5f',
                                              'font_color': 'white', 'border': 1,
                                              'align': 'center', 'valign': 'vcenter'})
            money_fmt = workbook.add_format({'num_format': '₹#,##0.00'})
            pct_fmt = workbook.add_format({'num_format': '0.00%'})
            comma_fmt = workbook.add_format({'num_format': '#,##0'})

            # --- 1. Summary Sheet with pie chart ---
            ws_summ = workbook.add_worksheet('Summary')
            ws_summ.write(0, 0, 'Metric', header_fmt)
            ws_summ.write(0, 1, 'Value', header_fmt)
            metrics = [
                ('Audit Date', datetime.now().strftime('%d-%m-%Y %H:%M')),
                ('Total Transactions', len(df)),
                ('Total Taxable Value', df['taxable_value'].sum()),
                ('Materiality Amount', materiality_amount),
                ('Sample Size', len(sample_df)),
                ('Sample %', len(sample_df)/len(df)*100 if len(df)>0 else 0),
                ('Sample Value', sample_df['taxable_value'].sum()),
                ('Critical Items', len(df[df['materiality_level']=='🔥 CRITICAL'])),
                ('High Items', len(df[df['materiality_level']=='⚡ HIGH'])),
                ('Medium Items', len(df[df['materiality_level']=='💫 MEDIUM'])),
                ('Low Items', len(df[df['materiality_level']=='🌟 LOW'])),
                ('Sampling Methods', ', '.join(selected_methods))
            ]
            for i, (label, val) in enumerate(metrics, start=1):
                ws_summ.write(i, 0, label)
                if isinstance(val, (int, float)):
                    ws_summ.write(i, 1, val, money_fmt if 'Value' in label or 'Amount' in label else comma_fmt)
                else:
                    ws_summ.write(i, 1, val)
            ws_summ.set_column('A:A', 25)
            ws_summ.set_column('B:B', 30)

            # Pie chart data (sample composition by materiality)
            pie_data = sample_df['materiality_level'].value_counts().reset_index()
            pie_data.columns = ['Level', 'Count']
            if not pie_data.empty:
                start_row = len(metrics) + 3
                for i, row in pie_data.iterrows():
                    ws_summ.write(start_row + i, 3, row['Level'])
                    ws_summ.write(start_row + i, 4, row['Count'])
                pie_chart = workbook.add_chart({'type': 'pie'})
                pie_chart.add_series({
                    'name': 'Sample Composition',
                    'categories': f'=Summary!$D${start_row+1}:$D${start_row+len(pie_data)}',
                    'values': f'=Summary!$E${start_row+1}:$E${start_row+len(pie_data)}',
                    'data_labels': {'percentage': True, 'category': True, 'leader_lines': True}
                })
                pie_chart.set_title({'name': 'Sample Composition by Materiality Level'})
                pie_chart.set_style(10)
                ws_summ.insert_chart('D2', pie_chart)

            # --- 2. TDS Rates Sheet ---
            tds_ws = workbook.add_worksheet('TDS Rates')
            tds_ws.write(0, 0, 'Section', header_fmt)
            tds_ws.write(0, 1, 'Rate (%)', header_fmt)
            tds_ws.write(0, 2, 'Limit (₹)', header_fmt)
            tds_ws.write(0, 3, 'Description', header_fmt)
            for i, (section, (rate, limit, desc)) in enumerate(Config.TDS_RATES.items(), start=1):
                tds_ws.write(i, 0, section)
                tds_ws.write(i, 1, rate if isinstance(rate, (int, float)) else rate)
                tds_ws.write(i, 2, limit if isinstance(limit, (int, float)) else limit)
                tds_ws.write(i, 3, desc)
            tds_ws.set_column('A:A', 15)
            tds_ws.set_column('B:B', 15)
            tds_ws.set_column('C:C', 15, comma_fmt)
            tds_ws.set_column('D:D', 40)

            # --- 3. Sampling Methods Sheet ---
            sm_ws = workbook.add_worksheet('Sampling Methods')
            sm_ws.write(0, 0, 'Method', header_fmt)
            sm_ws.write(0, 1, 'Description & Example', header_fmt)
            for i, (method, desc) in enumerate(SAMPLING_DESCRIPTIONS.items(), start=1):
                sm_ws.write(i, 0, method)
                sm_ws.write(i, 1, desc)
            sm_ws.set_column('A:A', 35)
            sm_ws.set_column('B:B', 80)

            # --- 4. Complete Data (raw) ---
            raw_cols = ['date','party_name','gst_number','invoice_no','gross_total',
                        'taxable_value','input_cgst','input_sgst','input_igst','tds_deducted','tds_section']
            raw_df = df[raw_cols].copy()
            if 'date' in raw_df.columns:
                raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce').dt.strftime(Config.DATE_FORMAT)
            raw_df.to_excel(writer, sheet_name='Complete Data', index=False, startrow=1, header=False)
            raw_ws = writer.sheets['Complete Data']
            for col_num, col_name in enumerate(raw_df.columns):
                raw_ws.write(0, col_num, col_name, header_fmt)
            # Subtotals
            for col_num, col_name in enumerate(raw_df.columns):
                if col_name in ['gross_total','taxable_value','input_cgst','input_sgst','input_igst','tds_deducted']:
                    col_letter = cls.col_letter(col_num)
                    raw_ws.write(len(raw_df)+1, col_num, f'=SUM({col_letter}2:{col_letter}{len(raw_df)+1})', money_fmt)
            raw_ws.set_column('A:A', 15)
            raw_ws.set_column('B:B', 25)
            raw_ws.set_column('C:C', 20)
            raw_ws.set_column('D:D', 20)
            for col in range(4, 11):
                raw_ws.set_column(col, col, 15, money_fmt)

            # --- 5. Sample Data (with formulas) ---
            sample_out = sample_df.copy()
            if 'date' in sample_out.columns:
                sample_out['date'] = pd.to_datetime(sample_out['date'], errors='coerce').dt.strftime(Config.DATE_FORMAT)
            # Add columns needed for formulas
            formula_cols = ['total_gst','gst_rate','party_section_total','tds_applicable','std_tds_rate',
                            'applied_tds_rate','required_tds','tds_shortfall','interest','compliance_status',
                            'compliance_pct','materiality_level','audit_priority','sampling_method']
            for col in formula_cols:
                if col not in sample_out.columns:
                    sample_out[col] = np.nan
            # Reorder to put key columns first
            base_cols = ['date','party_name','gst_number','invoice_no','gross_total',
                         'taxable_value','input_cgst','input_sgst','input_igst','tds_deducted','tds_section']
            other_cols = [c for c in sample_out.columns if c not in base_cols]
            sample_out = sample_out[base_cols + other_cols]
            sample_out.to_excel(writer, sheet_name='Sample Data', index=False, startrow=1, header=False)
            samp_ws = writer.sheets['Sample Data']
            for col_num, col_name in enumerate(sample_out.columns):
                samp_ws.write(0, col_num, col_name, header_fmt)
            # Subtotals for numeric columns
            numeric_cols_sample = ['gross_total','taxable_value','input_cgst','input_sgst','input_igst','tds_deducted',
                                   'total_gst','required_tds','interest','tds_shortfall']
            for col_num, col_name in enumerate(sample_out.columns):
                if col_name in numeric_cols_sample:
                    col_letter = cls.col_letter(col_num)
                    samp_ws.write(len(sample_out)+1, col_num, f'=SUM({col_letter}2:{col_letter}{len(sample_out)+1})', money_fmt)
            # Set column widths
            for col_num, col_name in enumerate(sample_out.columns):
                if col_name in numeric_cols_sample + ['party_section_total']:
                    samp_ws.set_column(col_num, col_num, 15, money_fmt)
                elif col_name in ['gst_rate','std_tds_rate','applied_tds_rate','compliance_pct']:
                    samp_ws.set_column(col_num, col_num, 12, pct_fmt)
                elif col_name == 'date':
                    samp_ws.set_column(col_num, col_num, 15)
                elif col_name == 'party_name':
                    samp_ws.set_column(col_num, col_num, 25)
                elif col_name == 'gst_number':
                    samp_ws.set_column(col_num, col_num, 20)
                elif col_name == 'invoice_no':
                    samp_ws.set_column(col_num, col_num, 20)
                else:
                    samp_ws.set_column(col_num, col_num, 15)

            # --- 6. Analysis Sheet (all transactions with formulas) ---
            analysis_df = df.copy()
            if 'date' in analysis_df.columns:
                analysis_df['date'] = pd.to_datetime(analysis_df['date'], errors='coerce').dt.strftime(Config.DATE_FORMAT)
            analysis_df.to_excel(writer, sheet_name='Analysis', index=False, startrow=1, header=False)
            ana_ws = writer.sheets['Analysis']
            for col_num, col_name in enumerate(analysis_df.columns):
                ana_ws.write(0, col_num, col_name, header_fmt)
            # Subtotals
            for col_num, col_name in enumerate(analysis_df.columns):
                if col_name in numeric_cols_sample:
                    col_letter = cls.col_letter(col_num)
                    ana_ws.write(len(analysis_df)+1, col_num, f'=SUM({col_letter}2:{col_letter}{len(analysis_df)+1})', money_fmt)
            # Column widths (similar to sample)
            for col_num, col_name in enumerate(analysis_df.columns):
                if col_name in numeric_cols_sample + ['party_section_total']:
                    ana_ws.set_column(col_num, col_num, 15, money_fmt)
                elif col_name in ['gst_rate','std_tds_rate','applied_tds_rate','compliance_pct']:
                    ana_ws.set_column(col_num, col_num, 12, pct_fmt)
                elif col_name == 'date':
                    ana_ws.set_column(col_num, col_num, 15)
                elif col_name == 'party_name':
                    ana_ws.set_column(col_num, col_num, 25)
                elif col_name == 'gst_number':
                    ana_ws.set_column(col_num, col_num, 20)
                elif col_name == 'invoice_no':
                    ana_ws.set_column(col_num, col_num, 20)
                else:
                    ana_ws.set_column(col_num, col_num, 15)

            # --- 7. Party Analysis ---
            party_stats.to_excel(writer, sheet_name='Party Analysis', index=True, startrow=1, header=False)
            party_ws = writer.sheets['Party Analysis']
            party_ws.write(0, 0, 'Party Name', header_fmt)
            for col_num, col_name in enumerate(party_stats.columns):
                party_ws.write(0, col_num+1, col_name, header_fmt)
            # Subtotals
            for col_num, col_name in enumerate(party_stats.columns):
                if col_name in ['Total Value','TDS Paid','TDS Required','TDS Shortfall','Interest','Total GST']:
                    col_letter = cls.col_letter(col_num+1)
                    party_ws.write(len(party_stats)+1, col_num+1, f'=SUM({col_letter}2:{col_letter}{len(party_stats)+1})', money_fmt)
            party_ws.set_column('A:A', 25)
            for col_num, col_name in enumerate(party_stats.columns):
                if col_name in ['Total Value','TDS Paid','TDS Required','TDS Shortfall','Interest','Total GST']:
                    party_ws.set_column(col_num+1, col_num+1, 15, money_fmt)
                elif col_name in ['TDS Compliance %','Risk Score']:
                    party_ws.set_column(col_num+1, col_num+1, 12, pct_fmt)
                else:
                    party_ws.set_column(col_num+1, col_num+1, 15)

            # --- 8. 194C Detailed Breakdown (if any) ---
            parties_low = df[(df['tds_section'] == '194C') &
                             (df['party_section_total'] < 100000)]['party_name'].unique()
            if len(parties_low) > 0:
                inv_break = df[(df['tds_section'] == '194C') &
                               (df['party_name'].isin(parties_low)) &
                               (df['taxable_value'] > 30000)]
                if not inv_break.empty:
                    inv_break = inv_break[['date','party_name','gst_number','invoice_no',
                                           'taxable_value','tds_deducted','required_tds',
                                           'tds_section','materiality_level']]
                    inv_break['date'] = pd.to_datetime(inv_break['date'], errors='coerce').dt.strftime(Config.DATE_FORMAT)
                    inv_break.to_excel(writer, sheet_name='194C Breakdown', index=False, startrow=1, header=False)
                    br_ws = writer.sheets['194C Breakdown']
                    for col_num, col_name in enumerate(inv_break.columns):
                        br_ws.write(0, col_num, col_name, header_fmt)
                    br_ws.set_column('A:A', 15)
                    br_ws.set_column('B:B', 25)
                    br_ws.set_column('C:C', 20)
                    br_ws.set_column('D:D', 20)
                    for col in range(4, 8):
                        br_ws.set_column(col, col, 15, money_fmt)

        return output.getvalue()


# =============================================================================
# STREAMLIT UI
# =============================================================================
def render_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(-45deg, #f0f9ff, #e6f0fa, #d9e9f5); }
        .glass { backdrop-filter: blur(12px); background: rgba(255,255,255,0.3); border-radius: 20px; border: 1px solid rgba(255,255,255,0.4); box-shadow: 0 8px 32px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 1.5rem; }
        .header { text-align: center; padding: 2rem; background: rgba(255,255,255,0.4); backdrop-filter: blur(16px); border-radius: 24px; border: 1px solid rgba(255,255,255,0.5); margin-bottom: 2rem; }
        .title { font-weight: 800; font-size: 2.8rem; background: linear-gradient(135deg, #2563eb, #0ea5e9, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .metric-card { background: rgba(0,255,135,0.1); backdrop-filter: blur(10px); border: 1px solid #00ff87; border-radius: 15px; padding: 1rem; text-align: center; transition: 0.3s; }
        .metric-card:hover { transform: scale(1.05); box-shadow: 0 0 30px rgba(0,255,135,0.3); }
        .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius: 15px; padding: 5px; }
        .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 20px; }
        .stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }
        .party-card { background: rgba(0,255,135,0.05); border: 1px solid #60efff; border-radius: 15px; padding: 1rem; margin: 0.5rem 0; transition: 0.3s; }
        .party-card:hover { background: rgba(0,255,135,0.15); transform: translateX(8px); }
    </style>
    """, unsafe_allow_html=True)


def main():
    render_css()

    # Header
    st.markdown("""
    <div class="header">
        <div class="title">⚡ ULTRA AUDIT INTELLIGENCE</div>
        <p style="font-size:1.2rem; color:#1e293b;">Enterprise Sample + TDS Check Platform • 30+ Sampling Methods • Dynamic TDS Engine</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        materiality_abs = st.number_input("Materiality Amount (₹)", min_value=1, value=Config.MATERIALITY_DEFAULT, step=10000)
        sample_pct = st.slider("Sample Percentage", 1, 100, Config.SAMPLE_PERCENT_DEFAULT)
        interest_months = st.number_input("Interest Months", 1, 12, 3)

        st.markdown("---")
        st.markdown("### 🎯 Sampling Methods")
        st.markdown("Select any combination of methods. The final sample will be the union (duplicates removed).")

        prob_methods = st.multiselect("Probability Methods",
                                      ['Simple Random Sampling','Systematic Sampling','Stratified Sampling',
                                       'Cluster Sampling','Multistage Sampling','Multiphase Sampling',
                                       'Area Sampling','Probability Proportional to Size (PPS) Sampling'],
                                      default=['Simple Random Sampling'])
        nonprob_methods = st.multiselect("Non-Probability Methods",
                                         ['Convenience Sampling','Judgmental Sampling','Purposive Sampling',
                                          'Quota Sampling','Snowball Sampling','Volunteer Sampling',
                                          'Haphazard Sampling','Consecutive Sampling'])
        audit_methods = st.multiselect("Audit-Specific",
                                       ['Statistical Sampling','Non-Statistical Sampling',
                                        'Monetary Unit Sampling (MUS)','Block Sampling'])
        adv_methods = st.multiselect("Advanced",
                                     ['Sequential Sampling','Adaptive Sampling','Reservoir Sampling',
                                      'Acceptance Sampling','Bootstrap Sampling','Bayesian Sampling'])

        selected_methods = prob_methods + nonprob_methods + audit_methods + adv_methods
        if not selected_methods:
            selected_methods = ['Simple Random Sampling']

        st.markdown("---")
        st.markdown("### 📥 Sample Data")
        sample_df_download = generate_sample_data()
        sample_excel = BytesIO()
        with pd.ExcelWriter(sample_excel, engine='xlsxwriter') as writer:
            sample_df_download.to_excel(writer, sheet_name='Sample', index=False)
        st.download_button("📥 Download Sample Excel", data=sample_excel.getvalue(),
                           file_name="Audit_Sample.xlsx", use_container_width=True)

    # Upload section
    uploaded_file = st.file_uploader("📤 Upload Ledger File (CSV or Excel)", type=['xlsx','csv'],
                                     label_visibility="collapsed")
    if uploaded_file is None:
        st.info("Please upload a file to begin. The sample file in sidebar shows the required format.")
        return

    # Process data
    try:
        with st.spinner("Processing data..."):
            # Load
            df_raw = DataProcessor.load_data(uploaded_file)
            # Process
            df = DataProcessor.process(df_raw, interest_months)
            # Materiality
            materiality_amount = materiality_abs
            df = MaterialityEngine.apply(df, materiality_amount)

            # Store in session state for persistence
            if 'audit_df' not in st.session_state or st.session_state.get('uploaded_hash') != hash(str(df_raw)):
                st.session_state.audit_df = df
                st.session_state.uploaded_hash = hash(str(df_raw))

        # Show dashboard
        display_dashboard(df, selected_methods, sample_pct, materiality_amount, interest_months)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)


def display_dashboard(df: pd.DataFrame, selected_methods: List[str],
                      sample_pct: int, materiality_amount: float, interest_months: int):
    """Main dashboard with tabs and analytics."""
    # Apply sampling
    all_samples = []
    for method in selected_methods:
        sample = SamplingEngine.apply_method(df, method, sample_pct)
        sample['sampling_method'] = method
        all_samples.append(sample)
    if all_samples:
        sample_df = pd.concat(all_samples, ignore_index=True).drop_duplicates(
            subset=['party_name', 'invoice_no', 'gst_number']
        )
    else:
        sample_df = SamplingEngine.simple_random(df, sample_pct)
        sample_df['sampling_method'] = 'Simple Random'

    # Summary metrics
    total_value = df['taxable_value'].sum()
    sample_value = sample_df['taxable_value'].sum()
    critical_count = len(df[df['materiality_level'] == '🔥 CRITICAL'])
    compliance_rate = (df['compliance_status'] == '✅ FULLY COMPLIANT').mean() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Total Value", f"₹{total_value:,.0f}")
    col2.metric("📦 Transactions", len(df))
    col3.metric("🔥 Critical", critical_count)
    col4.metric("🎯 Sample", f"{len(sample_df)} ({sample_pct}%)")
    col5.metric("✅ Compliance", f"{compliance_rate:.1f}%")

    st.info(f"📊 **Sampling Methods Applied:** {', '.join(selected_methods)}")

    # Tabs
    tabs = st.tabs(["📊 Overview", "🔍 Sample Details", "🏢 Party Analysis", "💰 TDS Compliance",
                    "📋 All Data", "📥 Export"])

    with tabs[0]:
        col_left, col_right = st.columns(2)
        with col_left:
            fig = px.pie(df, names='materiality_level', title='Materiality Distribution',
                         color='materiality_level',
                         color_discrete_map={'🔥 CRITICAL':'#ff00ff','⚡ HIGH':'#00ff87',
                                             '💫 MEDIUM':'#60efff','🌟 LOW':'#0061ff',
                                             '📦 IMMATERIAL':'#95a5a6'})
            st.plotly_chart(fig, use_container_width=True)

            # Party value trend
            party_val = df.groupby('party_name')['taxable_value'].sum().nlargest(10).reset_index()
            fig2 = px.bar(party_val, x='party_name', y='taxable_value', title='Top 10 Parties by Value')
            st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            # Compliance status
            comp_counts = df['compliance_status'].value_counts().reset_index()
            comp_counts.columns = ['Status', 'Count']
            fig3 = px.bar(comp_counts, x='Status', y='Count', color='Status',
                          color_discrete_map={'✅ FULLY COMPLIANT':'#10b981',
                                              '⚠️ PARTIAL SHORTFALL':'#f59e0b',
                                              '❌ NOT DEDUCTED':'#ef4444'})
            st.plotly_chart(fig3, use_container_width=True)

            # TDS shortfall by section
            short = df.groupby('tds_section')['tds_shortfall'].sum().nlargest(10).reset_index()
            fig4 = px.bar(short, x='tds_section', y='tds_shortfall', title='Top 10 TDS Shortfalls by Section')
            st.plotly_chart(fig4, use_container_width=True)

    with tabs[1]:
        st.subheader("🔍 Sampled Transactions")
        st.write(f"Total sample size: {len(sample_df)} rows from {len(selected_methods)} methods")
        st.dataframe(sample_df, use_container_width=True)

        # Method composition
        method_counts = sample_df['sampling_method'].value_counts().reset_index()
        method_counts.columns = ['Method', 'Count']
        fig5 = px.pie(method_counts, values='Count', names='Method', title='Sample Composition by Method')
        st.plotly_chart(fig5, use_container_width=True)

        # Sampling descriptions
        with st.expander("📖 See descriptions of selected methods"):
            for method in selected_methods:
                desc = SAMPLING_DESCRIPTIONS.get(method, "No description available.")
                st.markdown(f"**{method}**: {desc}")

    with tabs[2]:
        st.subheader("🏢 Party-Wise Analysis")
        party_stats = df.groupby('party_name').agg({
            'gst_number': 'first',
            'taxable_value': ['sum', 'count', 'mean'],
            'tds_deducted': 'sum',
            'required_tds': 'sum',
            'tds_shortfall': 'sum',
            'interest': 'sum'
        }).round(2)
        party_stats.columns = ['GST', 'Total Value', 'Transactions', 'Avg Value',
                               'TDS Paid', 'TDS Required', 'TDS Shortfall', 'Interest']
        party_stats['Compliance %'] = (party_stats['TDS Paid'] / party_stats['TDS Required'] * 100).fillna(100).round(2)
        party_stats['Risk Score'] = 100 - party_stats['Compliance %']

        # Select party dropdown
        party_list = ['All'] + list(party_stats.index[:20])
        selected_party = st.selectbox("Select Party", party_list)
        if selected_party != 'All':
            pdata = df[df['party_name'] == selected_party]
            st.markdown(f"""
            <div class="party-card">
                <h3 style="color:#00ff87;">{selected_party}</h3>
                <p>GST: {pdata['gst_number'].iloc[0] if 'gst_number' in pdata else 'N/A'}</p>
                <p>Total Value: ₹{pdata['taxable_value'].sum():,.0f}</p>
                <p>Transactions: {len(pdata)}</p>
                <p>TDS Shortfall: ₹{pdata['tds_shortfall'].sum():,.0f}</p>
                <p>Compliance: {(pdata['tds_deducted'].sum() / pdata['required_tds'].sum() * 100) if pdata['required_tds'].sum() > 0 else 100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(pdata, use_container_width=True)
        else:
            st.dataframe(party_stats.style.format({
                'Total Value': '₹{:,.0f}',
                'TDS Paid': '₹{:,.0f}',
                'TDS Shortfall': '₹{:,.0f}',
                'Compliance %': '{:.1f}%',
                'Risk Score': '{:.1f}'
            }), use_container_width=True)

    with tabs[3]:
        st.subheader("💰 TDS Compliance Details")
        tds_summary = df.groupby('tds_section').agg({
            'taxable_value': 'sum',
            'tds_deducted': 'sum',
            'required_tds': 'sum',
            'tds_shortfall': 'sum',
            'interest': 'sum'
        }).round(2)
        tds_summary['Compliance %'] = (tds_summary['tds_deducted'] / tds_summary['required_tds'] * 100).fillna(100).round(2)
        st.dataframe(tds_summary.style.format('₹{:,.0f}'), use_container_width=True)

        # Interest overview
        interest_total = df['interest'].sum()
        st.metric("Total Interest Payable", f"₹{interest_total:,.0f}")

        # Compliance heatmap (party vs section)
        if len(df) > 0:
            heat_data = df.groupby(['party_name', 'tds_section'])['compliance_pct'].mean().unstack().fillna(100)
            if heat_data.shape[1] > 0 and heat_data.shape[0] > 0:
                fig6 = px.imshow(heat_data, text_auto=True, aspect="auto",
                                  title="Compliance % by Party & Section",
                                  color_continuous_scale='RdYlGn')
                st.plotly_chart(fig6, use_container_width=True)

    with tabs[4]:
        st.subheader("📋 Complete Data")
        st.dataframe(df, use_container_width=True)

    with tabs[5]:
        st.subheader("📥 Export Report")
        st.markdown("""
        The Excel report includes:
        - **Summary** with pie chart
        - **TDS Rates** reference sheet
        - **Sampling Methods** with descriptions
        - **Complete Data** (raw)
        - **Sample Data** with formulas
        - **Analysis** (all transactions with derived columns)
        - **Party Analysis** with subtotals
        - **194C Breakdown** (if applicable)
        """)
        if st.button("⚡ Generate & Download Excel", use_container_width=True):
            with st.spinner("Building Excel report..."):
                party_stats = df.groupby('party_name').agg({
                    'gst_number': 'first',
                    'taxable_value': 'sum',
                    'tds_deducted': 'sum',
                    'required_tds': 'sum',
                    'tds_shortfall': 'sum',
                    'interest': 'sum'
                }).round(2)
                party_stats.columns = ['GST Number', 'Total Value', 'TDS Paid',
                                       'TDS Required', 'TDS Shortfall', 'Interest']
                party_stats['TDS Compliance %'] = (party_stats['TDS Paid'] / party_stats['TDS Required'] * 100).fillna(100).round(2)
                party_stats['Risk Score'] = 100 - party_stats['TDS Compliance %']

                excel_data = ExcelExporter.export(df, sample_df, party_stats,
                                                  selected_methods, materiality_amount, interest_months)
                st.download_button("📥 Download Excel", data=excel_data,
                                   file_name=f"Audit_Report_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                                   use_container_width=True)
                st.success("✅ Report generated!")


@st.cache_data
def generate_sample_data() -> pd.DataFrame:
    """Generate a realistic sample dataset for demonstration."""
    data = {
        'date': ['01-04-2023', '05-04-2023', '10-04-2023', '15-04-2023', '20-04-2023',
                 '25-04-2023', '30-04-2023', '05-05-2023', '10-05-2023', '15-05-2023'],
        'party_name': ['Aarav Enterprises', 'Bharat Traders', 'Chandni Logistics', 'Dewan Constructions',
                       'Eco Solutions', 'Falcon Services', 'Ganpati Industries', 'Himalaya Traders',
                       'Ishaan Tech', 'Jai Hind Corporation'],
        'gst_number': ['27AAAAA1234A1Z', '27BBBBB5678B2Y', '27CCCCC9101C3X', '27DDDDD1213D4W',
                       '27EEEEE1415E5V', '27FFFFF1617F6U', '27GGGGG1819G7T', '27HHHHH2021H8S',
                       '27IIIII2223I9R', '27JJJJJ2425J0Q'],
        'invoice_no': ['PEW/001/23-24', 'ST/23-24/468', '533', '6112303938', 'SAI/787/23-24',
                       'INV/001', 'INV/002', 'INV/003', 'INV/004', 'INV/005'],
        'gross_total': [135405.00, 78479.44, 25250.14, 10664.84, 67021.01,
                        500000.00, 120000.00, 75000.00, 200000.00, 100000.00],
        'taxable_value': [114750.00, 66508.00, 21322.16, 9038.00, 5605.00,
                          450000.00, 108000.00, 67500.00, 180000.00, 90000.00],
        'input_cgst': [10327.50, 5985.72, 1963.99, 0.00, 512.55,
                       40500.00, 9720.00, 6075.00, 16200.00, 8100.00],
        'input_sgst': [10327.50, 5985.72, 1963.99, 0.00, 512.55,
                       40500.00, 9720.00, 6075.00, 16200.00, 8100.00],
        'input_igst': [0.00, 0.00, 0.00, 1626.84, 0.00,
                       0.00, 0.00, 0.00, 0.00, 0.00],
        'tds_deducted': [1147.50, 665.08, 213.22, 90.38, 56.05,
                         4500.00, 2160.00, 1350.00, 3600.00, 900.00],
        'tds_section': ['194C', '194C', '194C', '194C', '194C',
                        '194H', '194J(b)', '194D', '194I', '194Q']
    }
    return pd.DataFrame(data)


if __name__ == '__main__':
    main()
