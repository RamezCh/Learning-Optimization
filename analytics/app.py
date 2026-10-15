"""
app.py
------
Streamlit Studio for the Dynamic Pricing Competition (DPC).
Features:
1. Versioned Dataset Selection (v1, v2, v3...)
2. Opponent selection with Simulation 1 / 2 sub-selection
3. Focused KPI cards: Current duopoly.py, Dataset Baseline, Opponent, and Seat Load Factor
4. Clean plot controls: Opponent curves hidden by default for fast behavior analysis
5. Replicating core notebook visualizations (W2_Tasks & W2_load_data)
6. Automated Unit Test & Data Integrity Verification suite
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess

# Ensure root and analytics are importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(__file__))

import duopoly
from simulation_engine import (
    DuopolySimulator,
    CopycatCompetitor,
    AggressiveDumperCompetitor,
    BoundedCompetitor,
    OldV1Competitor,
    HistoricalCSVCompetitor,
    MarketDemandModel
)
from data_loader import (
    discover_datasets,
    load_competition_file,
    get_grouped_competitors,
    get_historical_summary
)

# Page Configuration - Clean professional setup without emojis
st.set_page_config(
    page_title="DPC Analytics Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refined financial/quantitative terminal theme
st.markdown("""
<style>
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
        margin-top: 10px;
    }
    .kpi-card {
        background-color: var(--secondary-background-color, #f8f9fa);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 8px;
        padding: 14px 18px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 3px;
        letter-spacing: -0.5px;
    }
    .kpi-detail {
        font-size: 12px;
        color: #475569;
    }
    .status-tag {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 2px 7px;
        border-radius: 4px;
    }
    .status-lead {
        background-color: rgba(16, 185, 129, 0.12);
        color: #047857;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .status-trail {
        background-color: rgba(239, 68, 68, 0.12);
        color: #b91c1c;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR: Dataset & Scenario Selection
# -----------------------------------------------------------------------------
st.sidebar.markdown("### DPC Analytics Studio")
st.sidebar.caption("Evaluated Code: **duopoly.py**")
st.sidebar.markdown("---")

# 1. Discover Datasets
datasets = discover_datasets()
if not datasets:
    st.sidebar.error("No competition CSV files found in data/ directory.")
    st.stop()

selected_dataset_key = st.sidebar.selectbox(
    "Dataset Version",
    options=list(datasets.keys()),
    index=0,
    help="Select tournament data files (v1, v2, v3, etc.) located in data/ directory."
)

selected_dataset_info = datasets[selected_dataset_key]
dataset_version_label = selected_dataset_info['version'].upper()
df_dataset = load_competition_file(selected_dataset_info['path'])
grouped_competitors = get_grouped_competitors(df_dataset)

st.sidebar.markdown("---")

# 2. Simulation Mode
sim_mode = st.sidebar.radio(
    "Simulation Mode",
    options=["Historical Tournament Replay", "Synthetic Opponent Benchmark"],
    help="Replay past games against recorded rival prices from the CSV, or test against benchmark bots."
)

competitor_agent = None
rival_display_name = ""
historical_meta = None

if sim_mode == "Historical Tournament Replay":
    # 2-step selection: Opponent Name -> Simulation 1 or 2
    rival_names = sorted(grouped_competitors.keys())
    selected_rival_name = st.sidebar.selectbox("Opponent Name", options=rival_names)
    
    sim_runs = grouped_competitors[selected_rival_name]
    selected_sim_meta = st.sidebar.radio(
        "Tournament Match Selection",
        options=sim_runs,
        format_func=lambda s: s['label']
    )
    
    historical_meta = selected_sim_meta
    rival_display_name = selected_rival_name
    competitor_agent = HistoricalCSVCompetitor(df_dataset, selected_sim_meta['competition_id'])
    st.sidebar.caption(f"Historical Opponent Price: ${selected_sim_meta['avg_price']:.2f} ({selected_sim_meta['seasons']} seasons)")

else:
    dynamic_options = [
        "Copycat Competitor (Mirrors Our Price)",
        "Old Baseline Algorithm (Head-to-Head)",
        "Bounded Mid-Market ($40-$70)",
        "Aggressive Dumper ($18-$38)"
    ]
    selected_dynamic = st.sidebar.selectbox("Opponent Strategy", options=dynamic_options)
    
    if "Copycat" in selected_dynamic:
        undercut_val = st.sidebar.slider("Copycat Undercut ($)", min_value=0.0, max_value=3.0, value=0.0, step=0.5,
                                         help="If 0, mirrors exact price. If 1.0, undercuts our price by $1.00.")
        competitor_agent = CopycatCompetitor(initial_price=65.0, undercut=undercut_val)
        rival_display_name = f"Copycat (-${undercut_val:.2f})" if undercut_val > 0 else "Copycat (Mirror)"
    elif "Old Baseline" in selected_dynamic:
        competitor_agent = OldV1Competitor()
        rival_display_name = "Old Baseline"
    elif "Bounded" in selected_dynamic:
        competitor_agent = BoundedCompetitor()
        rival_display_name = "Bounded ($40-$70)"
    else:
        competitor_agent = AggressiveDumperCompetitor()
        rival_display_name = "Aggressive Dumper ($18-$38)"

n_seasons_to_sim = st.sidebar.slider("Seasons to Simulate", min_value=10, max_value=100, value=100, step=10)

run_button = st.sidebar.button("Run Simulation", type="primary", width="stretch")

# -----------------------------------------------------------------------------
# MAIN APP EXECUTION
# -----------------------------------------------------------------------------
session_key = f"{selected_dataset_key}_{sim_mode}_{rival_display_name}_{getattr(historical_meta, '__getitem__', lambda x: '')('competition_id') if historical_meta else ''}_{n_seasons_to_sim}"

if run_button or 'sim_result' not in st.session_state or st.session_state.get('last_session_key') != session_key:
    with st.spinner("Executing simulation..."):
        market = MarketDemandModel(seed=42)
        simulator = DuopolySimulator(algorithm_func=duopoly.p, market_demand=market)
        res = simulator.run(competitor_agent, n_seasons=n_seasons_to_sim, competitor_name=rival_display_name)
        st.session_state['sim_result'] = res
        st.session_state['last_session_key'] = session_key

sim_result = st.session_state['sim_result']
summary = sim_result.summary()
df_sim = sim_result.df

# -----------------------------------------------------------------------------
# HEADER & CLEAN SCORECARD
# -----------------------------------------------------------------------------
st.subheader(f"Evaluation: Current duopoly.py vs. {rival_display_name}")

# Retrieve historical dataset baseline stats if historical replay
hist_rev_str = "N/A"
hist_rev_val = 0
if historical_meta is not None:
    hist_stats = get_historical_summary(df_dataset, historical_meta['competition_id'])
    hist_rev_val = hist_stats.get('total_revenue', 0)
    hist_rev_str = f"${hist_rev_val:,.0f}"

is_our_win = summary['our_total_revenue'] > summary['rival_total_revenue']
status_tag_class = "status-lead" if is_our_win else "status-trail"
status_tag_text = "VICTORY [TOTAL REVENUE LEAD]" if is_our_win else "TRAIL [RUNNER-UP]"

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-label">Current duopoly.py Revenue</div>
        <div class="kpi-value" style="color: #047857;">
            ${summary['our_total_revenue']:,.0f}
        </div>
        <div class="kpi-detail"><span class="status-tag {status_tag_class}">{status_tag_text}</span> · {summary['total_seasons']} seasons</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Dataset Baseline ({dataset_version_label})</div>
        <div class="kpi-value" style="color: {'#b91c1c' if hist_rev_val == 0 else '#334155'};">
            {hist_rev_str}
        </div>
        <div class="kpi-detail">Recorded in historical CSV file</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Opponent ({rival_display_name})</div>
        <div class="kpi-value" style="color: #334155;">
            ${summary['rival_total_revenue']:,.0f}
        </div>
        <div class="kpi-detail">Difference: <strong>${summary['revenue_diff']:+,.0f}</strong></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Seat Load Factor</div>
        <div class="kpi-value" style="color: #1d4ed8;">
            {summary['our_load_factor_pct']}%
        </div>
        <div class="kpi-detail">{summary['our_mean_sales']:.1f} / 80 seats sold per season</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# NOTEBOOK PLOT REPLICATIONS & VERIFICATION (Tabs)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Capacity Utilization & Target Pacing",
    "Season Revenue Distributions",
    "Period Trajectories & Daily Sales",
    "Raw Period Data & Export",
    "Unit Tests & Data Verification"
])

# -----------------------------------------------------------------------------
# TAB 1: Capacity Utilization Curves with Target Line (replicates W2_Tasks.ipynb)
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("##### Cumulative Capacity Utilization vs. Target Line")
    
    # Calculate cumulative utilization per season
    df_sim['cum_sales'] = df_sim.groupby('season')['our_sales'].cumsum()
    df_sim['cum_rival_sales'] = df_sim.groupby('season')['rival_sales'].cumsum()
    df_sim['utilization'] = df_sim['cum_sales'] / 80.0
    df_sim['rival_utilization'] = df_sim['cum_rival_sales'] / 80.0

    util_pivot = df_sim.pivot(index='period', columns='season', values='utilization')
    
    # Identify top 20% revenue seasons
    season_revs = df_sim.groupby('season')['our_revenue'].sum()
    top_thresh = season_revs.quantile(0.8)
    top_seasons = season_revs[season_revs >= top_thresh].index.tolist()
    top_pivot = util_pivot[top_seasons]

    # Target Pacing Line from duopoly.py
    periods_arr = np.arange(1, 101)
    tau = (periods_arr - 1) / 100.0
    target_utilization = 0.85 * tau + 0.15 * (tau ** 2)

    # Plot Controls: Default behavior shows ONLY OURS
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        show_all_lines = st.checkbox("Show all season trajectories", value=True)
    with col_c2:
        show_top_20 = st.checkbox("Highlight top 20% revenue mean", value=True)
    with col_c3:
        show_rival = st.checkbox("Overlay Opponent Curve", value=False)  # DEFAULT FALSE

    fig_util, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_facecolor('#fafafa')
    ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    if show_all_lines:
        for s in util_pivot.columns:
            ax.plot(periods_arr, util_pivot[s], color='#10b981', alpha=0.08, linewidth=0.8)

    # Plot our mean actual curve
    mean_actual_curve = util_pivot.mean(axis=1)
    ax.plot(periods_arr, mean_actual_curve, color='#047857', linewidth=2.5, label='duopoly.py (Mean Realized)')

    if show_top_20:
        mean_top_curve = top_pivot.mean(axis=1)
        ax.plot(periods_arr, mean_top_curve, color='#065f46', linewidth=2.0, linestyle=':', label='Top 20% Seasons Mean')

    # Target line from duopoly.py
    ax.plot(periods_arr, target_utilization, color='#2563eb', linewidth=2.2, linestyle='-.', label='Target Pacing Line: 80 * [0.85t + 0.15t^2]')

    # Opponent curve shown ONLY if user explicitly checked the box
    if show_rival:
        rival_pivot = df_sim.pivot(index='period', columns='season', values='rival_utilization')
        ax.plot(periods_arr, rival_pivot.mean(axis=1), color='#ef4444', linewidth=2.0, linestyle='--', label=f'Opponent ({rival_display_name})')

    ax.set_title('Capacity Utilization Over 100 Selling Periods', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Selling Period (1 to 100)', fontsize=10)
    ax.set_ylabel('Utilization (Sold / 80)', fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='upper left', frameon=True, fontsize=9)

    st.pyplot(fig_util)
    plt.close()

# -----------------------------------------------------------------------------
# TAB 2: Revenue Distributions Across Seasons (replicates W2_load_data.ipynb)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("##### Season Revenue Distributions")
    
    show_rival_dist = st.checkbox("Overlay Opponent Distribution", value=False)  # DEFAULT FALSE

    season_agg = df_sim.groupby('season').agg({
        'our_revenue': 'sum',
        'rival_revenue': 'sum'
    }).reset_index()

    col_h, col_l = st.columns(2)

    with col_h:
        fig_h, ax_h = plt.subplots(figsize=(6, 3.8))
        ax_h.set_facecolor('#fafafa')
        ax_h.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

        ax_h.hist(season_agg['our_revenue'], bins=14, alpha=0.7, color='#10b981', label='duopoly.py')
        ax_h.axvline(season_agg['our_revenue'].median(), color='#047857', linestyle='--', linewidth=2, label=f"Median (${season_agg['our_revenue'].median():,.0f})")

        if show_rival_dist:
            ax_h.hist(season_agg['rival_revenue'], bins=14, alpha=0.55, color='#ef4444', label=f'Opponent ({rival_display_name})')
            ax_h.axvline(season_agg['rival_revenue'].median(), color='#b91c1c', linestyle='--', linewidth=2, label=f"Rival Med (${season_agg['rival_revenue'].median():,.0f})")

        ax_h.set_title('Revenue per Season Histogram', fontsize=11, fontweight='bold')
        ax_h.set_xlabel('Revenue ($)')
        ax_h.set_ylabel('Season Count')
        ax_h.legend(fontsize=8)
        st.pyplot(fig_h)
        plt.close()

    with col_l:
        fig_l, ax_l = plt.subplots(figsize=(6, 3.8))
        ax_l.set_facecolor('#fafafa')
        ax_l.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

        ax_l.plot(season_agg['season'], season_agg['our_revenue'], color='#10b981', linewidth=1.8, label='duopoly.py')
        if show_rival_dist:
            ax_l.plot(season_agg['season'], season_agg['rival_revenue'], color='#ef4444', linewidth=1.5, linestyle='--', label='Opponent')

        ax_l.set_title('Season-by-Season Revenue (Seasons 1..100)', fontsize=11, fontweight='bold')
        ax_l.set_xlabel('Season')
        ax_l.set_ylabel('Revenue ($)')
        ax_l.legend(fontsize=8)
        st.pyplot(fig_l)
        plt.close()

# -----------------------------------------------------------------------------
# TAB 3: Turn-by-Turn Daily Trajectories (replicates W2_load_data.ipynb daily bars)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("##### Turn-by-Turn Daily Price & Capacity Trajectory")
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_season_num = st.slider("Season Index", min_value=1, max_value=n_seasons_to_sim, value=min(25, n_seasons_to_sim))
    with col_s2:
        show_rival_daily = st.checkbox("Overlay Opponent Daily Actions", value=False)  # DEFAULT FALSE

    season_slice = df_sim[df_sim['season'] == selected_season_num].copy()

    fig_daily, (ax_dp, ax_ds, ax_dc) = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
    fig_daily.patch.set_facecolor('#ffffff')

    p_periods = season_slice['period'].values

    # 1. Price Line
    ax_dp.set_facecolor('#fafafa')
    ax_dp.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    ax_dp.plot(p_periods, season_slice['our_price'], color='#059669', linewidth=2.0, label='duopoly.py Price')
    if show_rival_daily:
        ax_dp.plot(p_periods, season_slice['rival_price'], color='#ef4444', linestyle='--', linewidth=1.5, label=f'Opponent ({rival_display_name})')
    ax_dp.axhline(100.0, color='#64748b', linestyle=':', label='WTP Ceiling ($100)')
    ax_dp.set_ylabel('Price ($)')
    ax_dp.set_title(f'Season {selected_season_num}: Pricing, Sales & Capacity', fontsize=11, fontweight='bold')
    ax_dp.legend(loc='upper right', fontsize=8)

    # 2. Daily Sales Bar Chart
    ax_ds.set_facecolor('#fafafa')
    ax_ds.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    if show_rival_daily:
        ax_ds.bar(p_periods - 0.2, season_slice['our_sales'], width=0.4, color='#059669', label='duopoly.py Sales')
        ax_ds.bar(p_periods + 0.2, season_slice['rival_sales'], width=0.4, color='#ef4444', label='Opponent Sales')
    else:
        ax_ds.bar(p_periods, season_slice['our_sales'], width=0.6, color='#059669', label='duopoly.py Sales')
    ax_ds.set_ylabel('Seats Sold')
    ax_ds.legend(loc='upper left', fontsize=8)

    # 3. Remaining Capacity Line vs Target Line
    ax_dc.set_facecolor('#fafafa')
    ax_dc.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    ax_dc.plot(p_periods, season_slice['our_capacity_remaining'], color='#059669', linewidth=2.0, label='Remaining Capacity')
    if show_rival_daily:
        ax_dc.plot(p_periods, season_slice['rival_capacity_remaining'], color='#ef4444', linestyle='--', linewidth=1.5, label='Opponent Remaining Seats')
    
    tau_p = (p_periods - 1) / 100.0
    target_rem = 80.0 * (1.0 - (0.85 * tau_p + 0.15 * (tau_p ** 2)))
    ax_dc.plot(p_periods, target_rem, color='#2563eb', linestyle='-.', label='Target Remaining Seats')
    
    ax_dc.set_xlabel('Selling Period (1 to 100)')
    ax_dc.set_ylabel('Seats Remaining')
    ax_dc.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig_daily)
    plt.close()

# -----------------------------------------------------------------------------
# TAB 4: Raw Period Data & Export
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("##### Simulation Period-by-Period Dataset")
    st.dataframe(df_sim.head(250), width="stretch")

    csv_export = df_sim.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Simulation CSV",
        data=csv_export,
        file_name=f"simulation_{rival_display_name}_{selected_dataset_info['version']}.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# TAB 5: Automated Unit Tests & Data Integrity Verification
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("##### Automated Verification & Unit Test Suite")
    st.caption("Verifies mathematical accounting, historical dataset ground-truth, contract bounds, and simulation integrity.")

    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        run_tests_btn = st.button("Run Unit Tests (pytest)", type="secondary")

    if run_tests_btn or st.session_state.get('ran_tests'):
        st.session_state['ran_tests'] = True
        with st.spinner("Executing 15 unit tests across test suites..."):
            cmd = [sys.executable, "-m", "pytest", "-v", os.path.join(ROOT_DIR, "tests")]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            output = proc.stdout

        if proc.returncode == 0:
            st.success("All 15 Unit Tests Passed Successfully.")
        else:
            st.error("Some tests failed. See output below.")

        st.code(output, language="text")
    else:
        st.info("Click 'Run Unit Tests (pytest)' above to verify data integrity and algorithm contracts.")
