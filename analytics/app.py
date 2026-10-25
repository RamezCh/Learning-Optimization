"""
app.py
------
Streamlit Analytics Studio for the Dynamic Pricing Competition (DPC).
Evaluates duopoly.py against historical tournaments and benchmark bots.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import time

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

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & MODERN STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DPC Analytics Studio",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-impact, clean quantitative terminal theme with large KPIs
st.markdown("""
<style>
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 16px;
        margin-bottom: 20px;
        margin-top: 8px;
    }
    .kpi-card {
        background-color: var(--secondary-background-color, #ffffff);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #475569;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: -0.8px;
        line-height: 1.1;
    }
    .kpi-detail {
        font-size: 13px;
        color: #334155;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .status-tag {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 2px 8px;
        border-radius: 5px;
    }
    .status-lead {
        background-color: rgba(16, 185, 129, 0.15);
        color: #047857;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .status-trail {
        background-color: rgba(239, 68, 68, 0.12);
        color: #b91c1c;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .metric-subcard {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-subcard .val {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
    }
    .metric-subcard .lbl {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to style Matplotlib figures consistently
def style_figure(fig, ax_list):
    fig.patch.set_facecolor('#ffffff')
    if not isinstance(ax_list, (list, np.ndarray)):
        ax_list = [ax_list]
    for ax in ax_list:
        ax.set_facecolor('#fafbfc')
        ax.grid(True, linestyle='--', alpha=0.55, color='#cbd5e1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#94a3b8')
        ax.spines['bottom'].set_color('#94a3b8')
        ax.tick_params(colors='#475569', labelsize=9)

# -----------------------------------------------------------------------------
# SIDEBAR: Dataset & Scenario Selection
# -----------------------------------------------------------------------------
st.sidebar.markdown("### DPC Analytics Studio")
st.sidebar.markdown("**Algorithm:** `duopoly.py`")
st.sidebar.markdown("---")

# 1. Discover Datasets
datasets = discover_datasets()
if not datasets:
    st.sidebar.error("No competition CSV files found in data/ directory.")
    st.stop()

selected_dataset_key = st.sidebar.selectbox(
    "Dataset Version",
    options=list(datasets.keys()),
    index=0
)

selected_dataset_info = datasets[selected_dataset_key]
dataset_version_label = selected_dataset_info['version'].upper()
df_dataset = load_competition_file(selected_dataset_info['path'])
grouped_competitors = get_grouped_competitors(df_dataset)

st.sidebar.markdown("---")

# 2. Simulation Mode
sim_mode = st.sidebar.radio(
    "Simulation Mode",
    options=["Historical Tournament Replay", "Synthetic Opponent Benchmark"]
)

competitor_agent = None
rival_display_name = ""
historical_meta = None

if sim_mode == "Historical Tournament Replay":
    rival_names = sorted(grouped_competitors.keys())
    selected_rival_name = st.sidebar.selectbox("Opponent Name", options=rival_names)
    
    sim_runs = grouped_competitors[selected_rival_name]
    selected_sim_meta = st.sidebar.radio(
        "Match Selection",
        options=sim_runs,
        format_func=lambda s: s['label']
    )
    
    historical_meta = selected_sim_meta
    rival_display_name = selected_rival_name
    competitor_agent = HistoricalCSVCompetitor(df_dataset, selected_sim_meta['competition_id'])
    st.sidebar.caption(f"Opponent Avg Price: ${selected_sim_meta['avg_price']:.2f}")

else:
    dynamic_options = [
        "Copycat Competitor (Mirrors Our Price)",
        "Old Baseline Algorithm (Head-to-Head)",
        "Bounded Mid-Market ($40-$70)",
        "Aggressive Dumper ($18-$38)"
    ]
    selected_dynamic = st.sidebar.selectbox("Opponent Strategy", options=dynamic_options)
    
    if "Copycat" in selected_dynamic:
        undercut_val = st.sidebar.slider("Copycat Undercut ($)", min_value=0.0, max_value=3.0, value=0.0, step=0.5)
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
# EXECUTIVE KPI SCORECARD (Large, High-Prominence)
# -----------------------------------------------------------------------------
st.subheader(f"duopoly.py vs. {rival_display_name}")

hist_rev_str = "N/A"
hist_rev_val = 0
if historical_meta is not None:
    hist_stats = get_historical_summary(df_dataset, historical_meta['competition_id'])
    hist_rev_val = hist_stats.get('total_revenue', 0)
    hist_rev_str = f"${hist_rev_val:,.0f}"

is_our_win = summary['our_total_revenue'] > summary['rival_total_revenue']
status_tag_class = "status-lead" if is_our_win else "status-trail"
status_tag_text = "VICTORY" if is_our_win else "TRAIL"

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-label">duopoly.py Revenue</div>
        <div class="kpi-value" style="color: #047857;">
            ${summary['our_total_revenue']:,.0f}
        </div>
        <div class="kpi-detail"><span class="status-tag {status_tag_class}">{status_tag_text}</span> · {summary['total_seasons']} seasons</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Dataset Baseline ({dataset_version_label})</div>
        <div class="kpi-value" style="color: {'#b91c1c' if hist_rev_val == 0 else '#1e293b'};">
            {hist_rev_str}
        </div>
        <div class="kpi-detail">Recorded in historical CSV file</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Opponent ({rival_display_name})</div>
        <div class="kpi-value" style="color: #1e293b;">
            ${summary['rival_total_revenue']:,.0f}
        </div>
        <div class="kpi-detail">Lead Margin: <strong style="color: {'#047857' if summary['revenue_diff'] > 0 else '#b91c1c'};">${summary['revenue_diff']:+,.0f}</strong></div>
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
# WORKSPACE TABS
# -----------------------------------------------------------------------------
tab_playback, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⏱️ Day-by-Day Simulation",
    "📈 Capacity Pacing",
    "💰 Season Revenue",
    "📉 Daily Trajectories",
    "📊 Dataset Market Statistics & WTP",
    "📋 Raw Data",
    "🧪 Unit Tests"
])

# -----------------------------------------------------------------------------
# TAB 0: Interactive Day-by-Day Simulation & Playback
# -----------------------------------------------------------------------------
with tab_playback:
    st.markdown("### ⏱️ Day-by-Day Simulation & Playback")
    st.caption("Inspect what happened one day at a time: track price, sales, and capacity for both airlines. Step manually or auto-play with 3–5 second pauses.")

    # State initialization
    if 'sim_day' not in st.session_state:
        st.session_state['sim_day'] = 1
    if 'sim_autoplay' not in st.session_state:
        st.session_state['sim_autoplay'] = False
    if 'sim_selected_season' not in st.session_state:
        st.session_state['sim_selected_season'] = 1

    # Controls Row 1: Season & Auto-Play Settings
    col_s, col_pause_speed, col_ap_btn = st.columns([2, 1, 1.2])
    with col_s:
        chosen_season = st.selectbox(
            "Select Season",
            options=list(range(1, n_seasons_to_sim + 1)),
            index=min(st.session_state['sim_selected_season'] - 1, n_seasons_to_sim - 1),
            key="season_selector"
        )
        if chosen_season != st.session_state['sim_selected_season']:
            st.session_state['sim_selected_season'] = chosen_season
            st.session_state['sim_day'] = 1

    with col_pause_speed:
        pause_sec = st.selectbox(
            "Auto-Play Pause",
            options=[3, 4, 5],
            index=0,
            format_func=lambda s: f"{s} seconds"
        )

    with col_ap_btn:
        st.write("")
        st.write("")
        if st.session_state['sim_autoplay']:
            if st.button("⏸ Pause Auto-Play", type="secondary", use_container_width=True):
                st.session_state['sim_autoplay'] = False
                st.rerun()
        else:
            if st.button("▶ Start Auto-Play", type="primary", use_container_width=True, disabled=(st.session_state['sim_day'] >= 100)):
                st.session_state['sim_autoplay'] = True
                st.rerun()

    # Filter data for chosen season
    season_data = df_sim[df_sim['season'] == st.session_state['sim_selected_season']].sort_values('period')
    if len(season_data) == 0:
        st.warning("No data found for this season.")
    else:
        # Navigation Row 2: Step Buttons & Slider
        col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 4, 1])
        
        with col_nav1:
            if st.button("⏮ Day 1", use_container_width=True, disabled=(st.session_state['sim_day'] <= 1)):
                st.session_state['sim_day'] = 1
                st.session_state['sim_autoplay'] = False
                st.rerun()

        with col_nav2:
            if st.button("◀ Previous", use_container_width=True, disabled=(st.session_state['sim_day'] <= 1)):
                st.session_state['sim_day'] = max(1, st.session_state['sim_day'] - 1)
                st.session_state['sim_autoplay'] = False
                st.rerun()

        with col_nav3:
            new_day = st.slider(
                "Current Selling Period / Day",
                min_value=1,
                max_value=100,
                value=st.session_state['sim_day'],
                key="day_slider"
            )
            if new_day != st.session_state['sim_day']:
                st.session_state['sim_day'] = new_day
                st.session_state['sim_autoplay'] = False
                st.rerun()

        with col_nav4:
            if st.button("Next ▶", use_container_width=True, disabled=(st.session_state['sim_day'] >= 100)):
                st.session_state['sim_day'] = min(100, st.session_state['sim_day'] + 1)
                st.session_state['sim_autoplay'] = False
                st.rerun()

        curr_day = st.session_state['sim_day']
        day_row = season_data[season_data['period'] == curr_day]
        
        if len(day_row) > 0:
            row = day_row.iloc[0]

            # Compute cumulative stats up to current day
            hist_up_to_day = season_data[season_data['period'] <= curr_day]
            our_cum_sales = int(hist_up_to_day['our_sales'].sum())
            rival_cum_sales = int(hist_up_to_day['rival_sales'].sum())
            our_cum_rev = float(hist_up_to_day['our_revenue'].sum())
            rival_cum_rev = float(hist_up_to_day['rival_revenue'].sum())

            diff_p = row['rival_price'] - row['our_price']
            diff_p_str = f"+${diff_p:.2f} (We are cheaper)" if diff_p > 0 else (f"-${abs(diff_p):.2f} (Rival is cheaper)" if diff_p < 0 else "Equal ($0.00)")

            # Day status pill
            rival_stock_status = "SOLD OUT (Monopoly Mode)" if row['rival_capacity_remaining'] == 0 else "IN STOCK"
            rival_tag_color = "#b91c1c" if row['rival_capacity_remaining'] == 0 else "#047857"

            st.markdown(f"""
            <div style="background: #f1f5f9; border-radius: 8px; padding: 10px 16px; margin: 12px 0; display: flex; justify-content: space-between; align-items: center; border-left: 4px solid #2563eb;">
                <div>
                    <span style="font-size: 13px; font-weight: 700; text-transform: uppercase; color: #475569;">Current Timeline:</span>
                    <strong style="font-size: 16px; color: #0f172a; margin-left: 6px;">Season {st.session_state['sim_selected_season']} · Day {curr_day} of 100</strong>
                    <span style="margin-left: 12px; font-size: 13px; color: #64748b;">(Progress: {curr_day}% completed)</span>
                </div>
                <div>
                    <span style="font-size: 12px; font-weight: 700; color: #475569;">Rival Inventory:</span>
                    <span style="font-size: 12px; font-weight: 800; color: {rival_tag_color}; background: #ffffff; padding: 3px 8px; border-radius: 4px; margin-left: 4px; border: 1px solid #cbd5e1;">{rival_stock_status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4-KPI comparison cards: Prices & Demands for BOTH users
            c_our_p, c_our_d, c_riv_p, c_riv_d = st.columns(4)
            with c_our_p:
                st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #059669;">
                    <div class="kpi-label">My Price (duopoly.py)</div>
                    <div class="kpi-value" style="color: #059669;">${row['our_price']:.2f}</div>
                    <div class="kpi-detail">Spread: <strong>{diff_p_str}</strong></div>
                </div>
                """, unsafe_allow_html=True)
            with c_our_d:
                st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #059669;">
                    <div class="kpi-label">My Demand (Sold Today)</div>
                    <div class="kpi-value" style="color: #059669;">{int(row['our_sales'])} <span style="font-size: 18px; color: #64748b;">seats</span></div>
                    <div class="kpi-detail">Today Revenue: <strong>${row['our_revenue']:,.2f}</strong></div>
                </div>
                """, unsafe_allow_html=True)
            with c_riv_p:
                st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #e11d48;">
                    <div class="kpi-label">Opponent Price ({rival_display_name})</div>
                    <div class="kpi-value" style="color: #e11d48;">${row['rival_price']:.2f}</div>
                    <div class="kpi-detail">Status: <strong>{'Active' if row['rival_capacity_remaining'] > 0 else 'Stock Depleted'}</strong></div>
                </div>
                """, unsafe_allow_html=True)
            with c_riv_d:
                st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #e11d48;">
                    <div class="kpi-label">Opponent Demand (Sold Today)</div>
                    <div class="kpi-value" style="color: #e11d48;">{int(row['rival_sales'])} <span style="font-size: 18px; color: #64748b;">seats</span></div>
                    <div class="kpi-detail">Today Revenue: <strong>${row['rival_revenue']:,.2f}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            # Cumulative Standing Row
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 18px; margin: 12px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px;">
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">My Remaining Capacity:</span>
                    <div style="font-size: 18px; font-weight: 800; color: #047857;">{int(row['our_capacity_remaining'])} / 80 seats <span style="font-size: 12px; font-weight: 600; color: #64748b;">({our_cum_sales} sold)</span></div>
                </div>
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">Opponent Remaining Capacity:</span>
                    <div style="font-size: 18px; font-weight: 800; color: #e11d48;">{int(row['rival_capacity_remaining'])} / 80 seats <span style="font-size: 12px; font-weight: 600; color: #64748b;">({rival_cum_sales} sold)</span></div>
                </div>
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">My Cumulative Revenue:</span>
                    <div style="font-size: 18px; font-weight: 800; color: #047857;">${our_cum_rev:,.2f}</div>
                </div>
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">Opponent Cumulative Revenue:</span>
                    <div style="font-size: 18px; font-weight: 800; color: #e11d48;">${rival_cum_rev:,.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Real-time Trajectory Visualizations up to Day t
            fig_sim, (ax_p, ax_d, ax_c) = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, dpi=140)
            style_figure(fig_sim, [ax_p, ax_d, ax_c])

            p_days = hist_up_to_day['period'].values
            p_our_prices = hist_up_to_day['our_price'].values
            p_riv_prices = hist_up_to_day['rival_price'].values
            p_our_sales = hist_up_to_day['our_sales'].values
            p_riv_sales = hist_up_to_day['rival_sales'].values
            p_our_caps = hist_up_to_day['our_capacity_remaining'].values
            p_riv_caps = hist_up_to_day['rival_capacity_remaining'].values

            # 1. Price trajectory
            ax_p.plot(p_days, p_our_prices, color='#059669', linewidth=2.2, label='My Price (duopoly.py)')
            ax_p.plot(p_days, p_riv_prices, color='#ef4444', linewidth=1.8, linestyle='--', label=f'Opponent ({rival_display_name})')
            ax_p.scatter([curr_day], [row['our_price']], color='#059669', s=80, zorder=5, edgecolors='#0f172a', label='Today')
            ax_p.scatter([curr_day], [row['rival_price']], color='#ef4444', s=80, zorder=5, edgecolors='#0f172a')
            ax_p.set_xlim(0.5, 100.5)
            ax_p.set_ylabel('Price ($)', fontsize=9, fontweight='bold')
            ax_p.set_title(f"Real-Time Trajectory: Season {st.session_state['sim_selected_season']} (Day 1 to {curr_day})", fontsize=11, fontweight='bold')
            ax_p.legend(loc='upper right', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')

            # 2. Demand bars
            bar_w = 0.38
            ax_d.bar(p_days - bar_w/2, p_our_sales, width=bar_w, color='#059669', label='My Sales')
            ax_d.bar(p_days + bar_w/2, p_riv_sales, width=bar_w, color='#ef4444', label='Opponent Sales')
            ax_d.set_ylabel('Daily Demand', fontsize=9, fontweight='bold')
            ax_d.legend(loc='upper left', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')

            # 3. Capacity curves
            ax_c.plot(p_days, p_our_caps, color='#059669', linewidth=2.2, label='My Remaining Seats')
            ax_c.plot(p_days, p_riv_caps, color='#ef4444', linewidth=1.8, linestyle='--', label='Opponent Remaining Seats')
            tau_all = (np.arange(1, 101) - 1) / 100.0
            target_rem = 80.0 * (1.0 - (0.85 * tau_all + 0.15 * (tau_all ** 2)))
            ax_c.plot(np.arange(1, 101), target_rem, color='#2563eb', linestyle='-.', label='Pacing Target Curve')
            ax_c.set_xlabel('Selling Period / Day', fontsize=9, fontweight='bold')
            ax_c.set_ylabel('Seats Remaining', fontsize=9, fontweight='bold')
            ax_c.legend(loc='upper right', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')

            plt.tight_layout()
            st.pyplot(fig_sim)
            plt.close()

            # Historical day table
            with st.expander("📋 View Daily Breakdown Table (Up to Today)"):
                table_df = hist_up_to_day[['period', 'our_price', 'rival_price', 'our_sales', 'rival_sales', 'our_revenue', 'rival_revenue', 'our_capacity_remaining', 'rival_capacity_remaining']].copy()
                table_df.columns = ['Day', 'My Price ($)', 'Rival Price ($)', 'My Sold', 'Rival Sold', 'My Rev ($)', 'Rival Rev ($)', 'My Left', 'Rival Left']
                st.dataframe(table_df.sort_values('Day', ascending=False), use_container_width=True, height=240)

            # Auto-play loop handling
            if st.session_state['sim_autoplay']:
                if curr_day < 100:
                    st.info(f"⏳ Auto-Play active: Pausing {pause_sec}s before advancing to Day {curr_day + 1}...")
                    time.sleep(pause_sec)
                    st.session_state['sim_day'] = curr_day + 1
                    st.rerun()
                else:
                    st.session_state['sim_autoplay'] = False
                    st.success("✅ Auto-Play completed! Reached Day 100.")
                    st.rerun()

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# TAB 1: Capacity Utilization Curves with Target Line
# -----------------------------------------------------------------------------
with tab1:
    df_sim['cum_sales'] = df_sim.groupby('season')['our_sales'].cumsum()
    df_sim['cum_rival_sales'] = df_sim.groupby('season')['rival_sales'].cumsum()
    df_sim['utilization'] = df_sim['cum_sales'] / 80.0
    df_sim['rival_utilization'] = df_sim['cum_rival_sales'] / 80.0

    util_pivot = df_sim.pivot(index='period', columns='season', values='utilization')
    
    season_revs = df_sim.groupby('season')['our_revenue'].sum()
    top_thresh = season_revs.quantile(0.8)
    top_seasons = season_revs[season_revs >= top_thresh].index.tolist()
    top_pivot = util_pivot[top_seasons]

    periods_arr = np.arange(1, 101)
    tau = (periods_arr - 1) / 100.0
    target_utilization = 0.85 * tau + 0.15 * (tau ** 2)

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        show_all_lines = st.checkbox("Show all seasons", value=True)
    with col_c2:
        show_top_20 = st.checkbox("Highlight top 20% seasons", value=True)
    with col_c3:
        show_rival = st.checkbox("Overlay Opponent", value=False)

    fig_util, ax = plt.subplots(figsize=(10, 4.6), dpi=140)
    style_figure(fig_util, ax)

    if show_all_lines:
        for s in util_pivot.columns:
            ax.plot(periods_arr, util_pivot[s], color='#10b981', alpha=0.08, linewidth=0.8)

    mean_actual_curve = util_pivot.mean(axis=1)
    ax.plot(periods_arr, mean_actual_curve, color='#047857', linewidth=2.5, label='duopoly.py (Realized Mean)')

    if show_top_20:
        mean_top_curve = top_pivot.mean(axis=1)
        ax.plot(periods_arr, mean_top_curve, color='#065f46', linewidth=2.0, linestyle=':', label='Top 20% Seasons')

    ax.plot(periods_arr, target_utilization, color='#2563eb', linewidth=2.2, linestyle='-.', label='Target Curve: 80 * [0.85t + 0.15t²]')

    if show_rival:
        rival_pivot = df_sim.pivot(index='period', columns='season', values='rival_utilization')
        ax.plot(periods_arr, rival_pivot.mean(axis=1), color='#ef4444', linewidth=2.0, linestyle='--', label=f'Opponent ({rival_display_name})')

    ax.set_title('Capacity Utilization Over 100 Selling Periods', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Selling Period (1 to 100)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Utilization (Sold / 80)', fontsize=10, fontweight='bold')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9)

    st.pyplot(fig_util)
    plt.close()

# -----------------------------------------------------------------------------
# TAB 2: Season Revenue Analytics (Clear, Non-Redundant Views)
# -----------------------------------------------------------------------------
with tab2:
    season_agg = df_sim.groupby('season').agg({
        'our_revenue': 'sum',
        'rival_revenue': 'sum'
    }).reset_index()

    season_agg['our_cum_revenue'] = season_agg['our_revenue'].cumsum()
    season_agg['rival_cum_revenue'] = season_agg['rival_revenue'].cumsum()

    # 1. High-Impact KPI Sub-cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val" style="color: #047857;">${season_agg['our_cum_revenue'].iloc[-1]:,.0f}</div>
            <div class="lbl">Total Cumulative Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val">${season_agg['our_revenue'].mean():,.1f}</div>
            <div class="lbl">Mean Revenue / Season</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val">${season_agg['our_revenue'].median():,.1f}</div>
            <div class="lbl">Median Revenue / Season</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val" style="color: {'#047857' if summary['revenue_diff'] > 0 else '#b91c1c'};">
                {summary['revenue_diff']:+,.0f}
            </div>
            <div class="lbl">Net Revenue Lead</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 2. Focused View Controls (Only 2 Clean, Distinct Options)
    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        rev_view_mode = st.radio(
            "Select Chart View",
            options=["Cumulative Revenue", "Revenue per Season"],
            horizontal=True
        )
    with col_v2:
        show_rival_dist = st.checkbox("Overlay Opponent", value=True)

    # 3. Clean Plots
    if rev_view_mode == "Cumulative Revenue":
        fig_cum, ax_c = plt.subplots(figsize=(11, 4.8), dpi=140)
        style_figure(fig_cum, ax_c)

        ax_c.plot(season_agg['season'], season_agg['our_cum_revenue'], color='#059669', linewidth=2.8, label='duopoly.py')
        if show_rival_dist:
            ax_c.plot(season_agg['season'], season_agg['rival_cum_revenue'], color='#e11d48', linewidth=2.0, linestyle='--', label=f'Opponent ({rival_display_name})')
            ax_c.fill_between(
                season_agg['season'],
                season_agg['rival_cum_revenue'],
                season_agg['our_cum_revenue'],
                where=(season_agg['our_cum_revenue'] >= season_agg['rival_cum_revenue']),
                color='#dcfce7', alpha=0.55, label='Profit Lead'
            )

        ax_c.set_title('Cumulative Revenue Growth Across Seasons', fontsize=12, fontweight='bold', pad=10)
        ax_c.set_xlabel('Season', fontsize=10, fontweight='bold')
        ax_c.set_ylabel('Total Revenue ($)', fontsize=10, fontweight='bold')
        ax_c.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9)
        st.pyplot(fig_cum)
        plt.close()

    else:
        fig_ssn, ax_s = plt.subplots(figsize=(11, 4.8), dpi=140)
        style_figure(fig_ssn, ax_s)

        ax_s.plot(season_agg['season'], season_agg['our_revenue'], color='#10b981', alpha=0.45, linewidth=1.2, label='Single Season Revenue')
        rolling_mean = season_agg['our_revenue'].rolling(window=10, min_periods=1).mean()
        ax_s.plot(season_agg['season'], rolling_mean, color='#047857', linewidth=2.5, label='duopoly.py (10-Season Trend)')
        
        if show_rival_dist:
            ax_s.plot(season_agg['season'], season_agg['rival_revenue'], color='#f43f5e', alpha=0.35, linewidth=1.0, linestyle=':')
            rival_rolling = season_agg['rival_revenue'].rolling(window=10, min_periods=1).mean()
            ax_s.plot(season_agg['season'], rival_rolling, color='#e11d48', linewidth=2.2, linestyle='--', label=f'Opponent Trend ({rival_display_name})')

        ax_s.set_title('Revenue Realized per Season (Seasons 1..N)', fontsize=12, fontweight='bold', pad=10)
        ax_s.set_xlabel('Season', fontsize=10, fontweight='bold')
        ax_s.set_ylabel('Revenue ($)', fontsize=10, fontweight='bold')
        ax_s.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9)
        st.pyplot(fig_ssn)
        plt.close()

# -----------------------------------------------------------------------------
# TAB 3: Turn-by-Turn Daily Trajectories
# -----------------------------------------------------------------------------
with tab3:
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_season_num = st.slider("Season Index", min_value=1, max_value=n_seasons_to_sim, value=min(25, n_seasons_to_sim))
    with col_s2:
        show_rival_daily = st.checkbox("Overlay Opponent Actions", value=False)

    season_slice = df_sim[df_sim['season'] == selected_season_num].copy()

    fig_daily, (ax_dp, ax_ds, ax_dc) = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True, dpi=140)
    style_figure(fig_daily, [ax_dp, ax_ds, ax_dc])

    p_periods = season_slice['period'].values

    # 1. Price Line
    ax_dp.plot(p_periods, season_slice['our_price'], color='#059669', linewidth=2.2, label='duopoly.py Price')
    if show_rival_daily:
        ax_dp.plot(p_periods, season_slice['rival_price'], color='#ef4444', linestyle='--', linewidth=1.5, label=f'Opponent ({rival_display_name})')
    ax_dp.axhline(100.0, color='#64748b', linestyle=':', label='WTP Ceiling ($100)')
    ax_dp.set_ylabel('Price ($)', fontsize=9, fontweight='bold')
    ax_dp.set_title(f'Season {selected_season_num}: Pricing, Sales & Capacity', fontsize=11, fontweight='bold')
    ax_dp.legend(loc='upper right', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')

    # 2. Daily Sales Bar Chart
    if show_rival_daily:
        ax_ds.bar(p_periods - 0.2, season_slice['our_sales'], width=0.4, color='#059669', label='duopoly.py Sales')
        ax_ds.bar(p_periods + 0.2, season_slice['rival_sales'], width=0.4, color='#ef4444', label='Opponent Sales')
    else:
        ax_ds.bar(p_periods, season_slice['our_sales'], width=0.6, color='#059669', label='duopoly.py Sales')
    ax_ds.set_ylabel('Seats Sold', fontsize=9, fontweight='bold')
    ax_ds.legend(loc='upper left', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')

    # 3. Remaining Capacity Line vs Target Line
    ax_dc.plot(p_periods, season_slice['our_capacity_remaining'], color='#059669', linewidth=2.2, label='Remaining Seats')
    if show_rival_daily:
        ax_dc.plot(p_periods, season_slice['rival_capacity_remaining'], color='#ef4444', linestyle='--', linewidth=1.5, label='Opponent Seats')
    
    tau_p = (p_periods - 1) / 100.0
    target_rem = 80.0 * (1.0 - (0.85 * tau_p + 0.15 * (tau_p ** 2)))
    ax_dc.plot(p_periods, target_rem, color='#2563eb', linestyle='-.', label='Target Remaining Seats')
    
    ax_dc.set_xlabel('Selling Period (1 to 100)', fontsize=9, fontweight='bold')
    ax_dc.set_ylabel('Seats Remaining', fontsize=9, fontweight='bold')
    ax_dc.legend(loc='upper right', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')

    plt.tight_layout()
    st.pyplot(fig_daily)
    plt.close()

# -----------------------------------------------------------------------------
# TAB 4: Dataset Market Statistics & Empirical WTP Analysis
# -----------------------------------------------------------------------------
with tab4:
    # Scope Selection
    scope_col1, scope_col2 = st.columns([1, 2])
    with scope_col1:
        stat_scope = st.radio(
            "Scope",
            options=["Entire Dataset", "Selected Competitor Matchup"],
            horizontal=True
        )
    
    if stat_scope == "Entire Dataset":
        df_target_stat = df_dataset.copy()
        scope_title = f"{dataset_version_label} (All {df_target_stat['competition_id'].nunique()} Competitions)"
    else:
        if historical_meta is not None:
            df_target_stat = df_dataset[df_dataset['competition_id'] == historical_meta['competition_id']].copy()
            scope_title = f"{rival_display_name} Matchup ({historical_meta['competition_id']})"
        else:
            df_target_stat = df_dataset.copy()
            scope_title = f"{dataset_version_label} (All Competitions)"

    sold_df = df_target_stat[df_target_stat['demand'] > 0]
    has_sales = len(sold_df) > 0

    min_price_sold = float(sold_df['price'].min()) if has_sales else 0.0
    max_price_sold = float(sold_df['price'].max()) if has_sales else 0.0
    median_price_sold = float(sold_df['price'].median()) if has_sales else 0.0
    weighted_mean_sold = float((sold_df['price'] * sold_df['demand']).sum() / sold_df['demand'].sum()) if has_sales else 0.0

    our_mean_posted = float(df_target_stat['price'].mean())
    our_median_posted = float(df_target_stat['price'].median())
    rival_mean_posted = float(df_target_stat['price_competitor'].mean())
    rival_median_posted = float(df_target_stat['price_competitor'].median())

    total_seats_sold = int(df_target_stat['demand'].sum())
    total_periods = len(df_target_stat)
    total_revenue_recorded = float(df_target_stat['revenue'].sum())

    st.markdown(f"**Target:** {scope_title} · **Total Periods:** {total_periods:,} · **Total Seats Sold:** {total_seats_sold:,}")

    # Stat Cards: Realized Prices Sold (Bigger KPIs)
    st.markdown("##### Realized Purchase Prices (Completed Sales)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val" style="color: #047857;">${min_price_sold:.2f}</div>
            <div class="lbl">Min Price Sold</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val" style="color: #047857;">${max_price_sold:.2f}</div>
            <div class="lbl">Max Price Sold</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val" style="color: #047857;">${weighted_mean_sold:.2f}</div>
            <div class="lbl">Weighted Mean Sold</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val" style="color: #047857;">${median_price_sold:.2f}</div>
            <div class="lbl">Median Price Sold</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Stat Cards: Offered Prices
    st.markdown("##### Offered / Posted Prices (All Periods)")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val">${our_mean_posted:.2f}</div>
            <div class="lbl">Our Mean Posted</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val">${our_median_posted:.2f}</div>
            <div class="lbl">Our Median Posted</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val" style="color: #2563eb;">${rival_mean_posted:.2f}</div>
            <div class="lbl">Rival Mean Posted</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown(f"""
        <div class="metric-subcard">
            <div class="val" style="color: #2563eb;">${rival_median_posted:.2f}</div>
            <div class="lbl">Rival Median Posted</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # WTP Tiers Table (Aligned with Course Ground-Truth Model)
    st.markdown("##### Customer Willingness-to-Pay (WTP) Tiers")
    st.caption("Aligned with official DPC demand model (Slide 6: Low ~0-50, Mid ~0-85, High ~0-150; randomly drawn per simulation).")
    
    bin_edges = [25.0, 55.0, 80.0, 105.0]
    bin_labels = ["Budget (Up to Low: $25-$55)", "Leisure (Up to Mid: $55-$80)", "Business (Up to High: $80-$105)"]
    df_target_stat['wtp_bin'] = pd.cut(df_target_stat['price'], bins=bin_edges, labels=bin_labels, include_lowest=True)

    wtp_summary = []
    for b_label in bin_labels:
        sub_b = df_target_stat[df_target_stat['wtp_bin'] == b_label]
        b_periods = len(sub_b)
        b_sales = sub_b['demand'].sum()
        b_rev = sub_b['revenue'].sum()
        b_conv = (sub_b['demand'] > 0).mean() * 100.0 if b_periods > 0 else 0.0
        b_avg_sold_p = (sub_b[sub_b['demand'] > 0]['price'] * sub_b[sub_b['demand'] > 0]['demand']).sum() / b_sales if b_sales > 0 else 0.0
        rev_share = (b_rev / total_revenue_recorded * 100.0) if total_revenue_recorded > 0 else 0.0
        
        wtp_summary.append({
            "Segment Tier": b_label,
            "Periods Offered": f"{b_periods:,}",
            "Seats Sold": f"{int(b_sales):,}",
            "Conversion Rate": f"{b_conv:.1f}%",
            "Avg Sold Price": f"${b_avg_sold_p:.2f}",
            "Total Revenue": f"${b_rev:,.0f}",
            "Revenue Share": f"{rev_share:.1f}%"
        })

    st.dataframe(pd.DataFrame(wtp_summary), width="stretch")

    # Visual Analytics: Clean Plots (Warning Fixed)
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    fig_wtp, (ax_w1, ax_w2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=140)
    style_figure(fig_wtp, [ax_w1, ax_w2])

    if has_sales:
        unsold_df = df_target_stat[df_target_stat['demand'] == 0]
        ax_w1.hist(sold_df['price'], bins=20, alpha=0.75, color='#10b981', label='Sold Prices')
        ax_w1.hist(unsold_df['price'], bins=20, alpha=0.35, color='#94a3b8', label='Unsold Prices')
        ax_w1.axvline(median_price_sold, color='#047857', linestyle='--', linewidth=2, label=f'Median (${median_price_sold:.1f})')
        ax_w1.set_title('Sold vs. Unsold Offered Prices', fontsize=11, fontweight='bold')
        ax_w1.set_xlabel('Price ($)', fontsize=9, fontweight='bold')
        ax_w1.set_ylabel('Periods', fontsize=9, fontweight='bold')
        ax_w1.legend(loc='upper right', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')

    # Conversion Rate by Price Bucket (Fixed ticklabels warning)
    p_step_bins = np.arange(25, 105, 10)
    p_labels = [f"${p_step_bins[i]}-${p_step_bins[i+1]}" for i in range(len(p_step_bins)-1)]
    df_target_stat['p_bucket'] = pd.cut(df_target_stat['price'], bins=p_step_bins, labels=p_labels, include_lowest=True)
    bucket_conv = df_target_stat.groupby('p_bucket', observed=False).apply(
        lambda g: (g['demand'] > 0).mean() * 100.0 if len(g) > 0 else 0.0
    ).values

    bar_colors = ['#10b981' if v > 40 else '#3b82f6' if v > 15 else '#f59e0b' if v > 5 else '#ef4444' for v in bucket_conv]
    x_indices = np.arange(len(p_labels))
    ax_w2.bar(x_indices, bucket_conv, color=bar_colors, alpha=0.85, edgecolor='#cbd5e1')
    ax_w2.set_title('Conversion Rate (%) by Price Tier', fontsize=11, fontweight='bold')
    ax_w2.set_xlabel('Price Tier ($)', fontsize=9, fontweight='bold')
    ax_w2.set_ylabel('Conversion Rate (%)', fontsize=9, fontweight='bold')
    ax_w2.set_xticks(x_indices)
    ax_w2.set_xticklabels(p_labels, rotation=25, fontsize=8)
    
    for i, v in enumerate(bucket_conv):
        if v > 0:
            ax_w2.text(i, v + 1.2, f'{v:.1f}%', ha='center', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig_wtp)
    plt.close()

# -----------------------------------------------------------------------------
# TAB 5: Raw Period Data & Export
# -----------------------------------------------------------------------------
with tab5:
    st.dataframe(df_sim.head(250), width="stretch")
    csv_export = df_sim.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Simulation CSV",
        data=csv_export,
        file_name=f"simulation_{rival_display_name}_{selected_dataset_info['version']}.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# TAB 6: Automated Unit Tests
# -----------------------------------------------------------------------------
with tab6:
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        run_tests_btn = st.button("Run Unit Tests (pytest)", type="secondary")

    if run_tests_btn or st.session_state.get('ran_tests'):
        st.session_state['ran_tests'] = True
        with st.spinner("Executing 15 unit tests..."):
            cmd = [sys.executable, "-m", "pytest", "-v", os.path.join(ROOT_DIR, "tests")]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            output = proc.stdout

        if proc.returncode == 0:
            st.success("All 15 Unit Tests Passed.")
        else:
            st.error("Test failures detected.")

        st.code(output, language="text")
