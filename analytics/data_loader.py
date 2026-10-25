"""
data_loader.py
--------------
Discovers and manages multi-version competition datasets across data/ directories:
e.g. data/v1/2026-10-14_day2_duopoly_competition_details.csv, data/v2/..., etc.

Enables the Streamlit app and simulation engine to automatically list available
dataset versions and extract competition opponents.
"""

import os
import glob
import re
import pandas as pd
from typing import Dict, List, Any, Optional

DATA_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def discover_datasets(base_dir: str = DATA_BASE_DIR) -> Dict[str, Dict[str, Any]]:
    """
    Scans the data directory and subfolders (v1, v2, v3, etc.) for competition CSV files.
    Returns a dictionary of discovered datasets with metadata.
    """
    datasets = {}

    if not os.path.exists(base_dir):
        return datasets

    # Find all CSV files in data/ and subfolders
    pattern = os.path.join(base_dir, '**', '*.csv')
    csv_files = glob.glob(pattern, recursive=True)

    for file_path in csv_files:
        rel_path = os.path.relpath(file_path, base_dir)
        filename = os.path.basename(file_path)
        
        # Determine version label from parent folder name if present
        folder_parts = rel_path.split(os.sep)
        if len(folder_parts) > 1:
            version_label = folder_parts[0]
        else:
            match = re.search(r'(v\d+)', filename, re.IGNORECASE)
            version_label = match.group(1).lower() if match else "v1"

        # Extract date from filename or folder if present
        date_match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2}|\d{2}[-_]\d{2}[-_]\d{4})', filename)
        date_str = date_match.group(1) if date_match else "Historical"

        key = f"{version_label.upper()} ({date_str}) - {filename}"
        
        datasets[key] = {
            "key": key,
            "version": version_label,
            "date": date_str,
            "filename": filename,
            "path": file_path,
            "rel_path": rel_path
        }

    return datasets

def load_competition_file(file_path: str) -> pd.DataFrame:
    """
    Loads and standardizes a competition details CSV file.
    Ensures unique_selling_season_key is present.
    """
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)

    if 'unique_selling_season_key' not in df.columns and 'competition_id' in df.columns:
        df['unique_selling_season_key'] = df.apply(
            lambda r: f"{r['competition_id']}_season_{int(r['selling_season'])}", axis=1
        )

    if 'revenue' not in df.columns and 'demand' in df.columns and 'price' in df.columns:
        df['revenue'] = df['demand'] * df['price']

    return df

def get_grouped_competitors(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups competition runs by competitor name, providing Simulation 1 and Simulation 2.
    """
    if 'competitor_id' not in df.columns or 'competition_id' not in df.columns:
        return {}

    grouped = {}
    for rival, g_rival in df.groupby('competitor_id'):
        sims = []
        # Sort by competition_id for deterministic ordering
        unique_comps = sorted(g_rival['competition_id'].unique())
        for sim_idx, comp_id in enumerate(unique_comps, start=1):
            g_comp = g_rival[g_rival['competition_id'] == comp_id]
            sims.append({
                "sim_num": sim_idx,
                "label": f"Simulation {sim_idx} (ID: {comp_id})",
                "competition_id": str(comp_id),
                "seasons": int(g_comp['selling_season'].nunique()),
                "avg_price": float(g_comp['price_competitor'].mean())
            })
        grouped[rival] = sims
    return grouped

def get_competitors_in_dataset(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Returns the distinct competitors and their competition runs in a dataset.
    """
    if 'competitor_id' not in df.columns or 'competition_id' not in df.columns:
        return []

    competitors = []
    grouped = df.groupby(['competitor_id', 'competition_id'])
    for (rival, comp_id), g in grouped:
        competitors.append({
            "competitor_id": rival,
            "competition_id": str(comp_id),
            "label": f"{rival} (Tournament: {comp_id})",
            "seasons": int(g['selling_season'].nunique()),
            "avg_price": float(g['price_competitor'].mean())
        })

    return competitors

def get_historical_summary(df: pd.DataFrame, competition_id: str) -> Dict[str, Any]:
    """
    Computes summary of what the OLD algorithm actually achieved in the CSV dataset.
    """
    sub = df[df['competition_id'] == competition_id].copy()
    if len(sub) == 0:
        return {}

    rival_name = sub['competitor_id'].iloc[0]
    total_revenue = sub['revenue'].sum()
    total_sales = sub['demand'].sum()
    total_seasons = sub['selling_season'].nunique()
    
    # Season aggregation
    season_revs = sub.groupby('selling_season')['revenue'].sum()
    zero_rev_seasons = int((season_revs == 0).sum())
    
    # Sold out seasons for rival
    sold_out_seasons = sub.groupby('selling_season')['competitor_has_capacity'].apply(
        lambda s: not s.iloc[-1]
    ).sum()

    return {
        "competition_id": competition_id,
        "competitor_id": rival_name,
        "total_revenue": total_revenue,
        "mean_season_revenue": total_revenue / max(1, total_seasons),
        "median_season_revenue": season_revs.median(),
        "total_sales": int(total_sales),
        "mean_season_sales": total_sales / max(1, total_seasons),
        "load_factor_pct": round((total_sales / (total_seasons * 80.0)) * 100.0, 1),
        "our_avg_price": sub['price'].mean(),
        "rival_avg_price": sub['price_competitor'].mean(),
        "zero_rev_seasons": zero_rev_seasons,
        "rival_sold_out_seasons": int(sold_out_seasons),
        "total_seasons": total_seasons
    }
