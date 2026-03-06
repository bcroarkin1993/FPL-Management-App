# scripts/fpl/price_changes.py
"""Price Changes page — risers, fallers, and transfer activity."""

import pandas as pd
import streamlit as st

from scripts.common.styled_tables import render_styled_table
from scripts.common.utils import get_classic_bootstrap_static

# FPL element_type -> position letter
_POS_LETTER = {1: "G", 2: "D", 3: "M", 4: "F"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _format_price(tenths: int) -> str:
    """Convert FPL price (tenths) to display string, e.g. 100 -> '£10.0m'."""
    return f"£{tenths / 10:.1f}m"


def _format_change(tenths: int) -> str:
    """Signed price change, e.g. 5 -> '+£0.5m', -3 -> '-£0.3m'."""
    sign = "+" if tenths > 0 else ""
    return f"{sign}£{tenths / 10:.1f}m"


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def _build_price_data() -> pd.DataFrame:
    """Build a DataFrame of all players with price/transfer data."""
    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        return pd.DataFrame()

    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    rows = []
    for p in bootstrap.get("elements", []):
        rows.append({
            "Player": p.get("web_name", ""),
            "Team": teams.get(p.get("team"), ""),
            "Pos": _POS_LETTER.get(p.get("element_type"), "?"),
            "now_cost": p.get("now_cost", 0),
            "cost_change_event": p.get("cost_change_event", 0),
            "cost_change_start": p.get("cost_change_start", 0),
            "cost_change_start_fall": p.get("cost_change_start_fall", 0),
            "transfers_in_event": p.get("transfers_in_event", 0),
            "transfers_out_event": p.get("transfers_out_event", 0),
            "transfers_in": p.get("transfers_in", 0),
            "transfers_out": p.get("transfers_out", 0),
            "selected_by_percent": p.get("selected_by_percent", "0"),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# CSS for price cards
# ------------------------------------------------------------------
def _price_card_css() -> str:
    return """
    <style>
    .price-cards-container { display: flex; flex-direction: column; gap: 6px; }
    .price-card {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 14px; border-radius: 8px;
        background: #1a1a2e; border: 1px solid #333;
        color: #e0e0e0;
    }
    .price-card-left { display: flex; flex-direction: column; }
    .price-card-name { font-weight: 700; font-size: 0.92rem; color: #e0e0e0; }
    .price-card-meta { font-size: 0.78rem; color: #9ca3af; margin-top: 2px; }
    .price-card-right { display: flex; flex-direction: column; align-items: flex-end; }
    .price-card-price { font-weight: 700; font-size: 0.92rem; color: #e0e0e0; }
    .price-card-change {
        font-weight: 800; font-size: 0.82rem;
        padding: 2px 8px; border-radius: 10px; margin-top: 2px;
    }
    .price-rise { background: rgba(0,255,135,0.15); color: #00ff87; border-left: 3px solid #00ff87; }
    .price-fall { background: rgba(255,71,87,0.1); color: #ff4757; border-left: 3px solid #ff4757; }
    .change-rise { background: rgba(0,255,135,0.2); color: #00ff87; }
    .change-fall { background: rgba(255,71,87,0.15); color: #ff4757; }
    .no-changes-msg {
        text-align: center; padding: 20px; color: #9ca3af;
        background: #1a1a2e; border-radius: 8px; border: 1px solid #333;
    }
    </style>
    """


# ------------------------------------------------------------------
# Section 1: GW Price Changes
# ------------------------------------------------------------------
def _render_gw_price_changes(df: pd.DataFrame):
    st.subheader("This Gameweek's Price Changes")
    st.markdown(_price_card_css(), unsafe_allow_html=True)

    risers = df[df["cost_change_event"] > 0].sort_values("cost_change_event", ascending=False)
    fallers = df[df["cost_change_event"] < 0].sort_values("cost_change_event", ascending=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Risers**")
        if risers.empty:
            st.markdown('<div class="no-changes-msg">No risers this gameweek yet</div>',
                        unsafe_allow_html=True)
        else:
            cards = '<div class="price-cards-container">'
            for _, row in risers.iterrows():
                cards += (
                    f'<div class="price-card price-rise">'
                    f'<div class="price-card-left">'
                    f'<span class="price-card-name">{row["Player"]}</span>'
                    f'<span class="price-card-meta">{row["Team"]} · {row["Pos"]}</span>'
                    f'</div>'
                    f'<div class="price-card-right">'
                    f'<span class="price-card-price">{_format_price(row["now_cost"])}</span>'
                    f'<span class="price-card-change change-rise">{_format_change(row["cost_change_event"])}</span>'
                    f'</div></div>'
                )
            cards += '</div>'
            st.markdown(cards, unsafe_allow_html=True)

    with col2:
        st.markdown("**Fallers**")
        if fallers.empty:
            st.markdown('<div class="no-changes-msg">No fallers this gameweek yet</div>',
                        unsafe_allow_html=True)
        else:
            cards = '<div class="price-cards-container">'
            for _, row in fallers.iterrows():
                cards += (
                    f'<div class="price-card price-fall">'
                    f'<div class="price-card-left">'
                    f'<span class="price-card-name">{row["Player"]}</span>'
                    f'<span class="price-card-meta">{row["Team"]} · {row["Pos"]}</span>'
                    f'</div>'
                    f'<div class="price-card-right">'
                    f'<span class="price-card-price">{_format_price(row["now_cost"])}</span>'
                    f'<span class="price-card-change change-fall">{_format_change(row["cost_change_event"])}</span>'
                    f'</div></div>'
                )
            cards += '</div>'
            st.markdown(cards, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Section 2: Season Price Movers
# ------------------------------------------------------------------
def _render_season_movers(df: pd.DataFrame):
    st.subheader("Season Price Movers")

    df = df.copy()
    df["start_price"] = df["now_cost"] - df["cost_change_start"]

    top_risers = df[df["cost_change_start"] > 0].nlargest(20, "cost_change_start")
    top_fallers = df[df["cost_change_start"] < 0].nsmallest(20, "cost_change_start")

    col1, col2 = st.columns(2)

    with col1:
        if top_risers.empty:
            st.info("No season risers yet.")
        else:
            display = top_risers[["Player", "Team", "Pos"]].copy()
            display["Start"] = top_risers["start_price"].apply(_format_price)
            display["Current"] = top_risers["now_cost"].apply(_format_price)
            display["Change"] = top_risers["cost_change_start"].apply(_format_change)
            render_styled_table(display, title="Biggest Risers")

    with col2:
        if top_fallers.empty:
            st.info("No season fallers yet.")
        else:
            display = top_fallers[["Player", "Team", "Pos"]].copy()
            display["Start"] = top_fallers["start_price"].apply(_format_price)
            display["Current"] = top_fallers["now_cost"].apply(_format_price)
            display["Change"] = top_fallers["cost_change_start"].apply(_format_change)
            render_styled_table(display, title="Biggest Fallers")


# ------------------------------------------------------------------
# Section 3: Transfer Activity
# ------------------------------------------------------------------
def _render_transfer_activity(df: pd.DataFrame):
    st.subheader("Transfer Activity (Price Pressure)")
    st.caption("Players with the highest net transfers are most likely to see price changes.")

    df = df.copy()
    df["net_transfers"] = df["transfers_in_event"] - df["transfers_out_event"]

    most_in = df.nlargest(20, "net_transfers")
    most_out = df.nsmallest(20, "net_transfers")

    col1, col2 = st.columns(2)

    with col1:
        if most_in.empty:
            st.info("No transfer data available.")
        else:
            display = most_in[["Player", "Team", "Pos"]].copy()
            display["Price"] = most_in["now_cost"].apply(_format_price)
            display["In"] = most_in["transfers_in_event"].apply(lambda x: f"{x:,}")
            display["Out"] = most_in["transfers_out_event"].apply(lambda x: f"{x:,}")
            display["Net"] = most_in["net_transfers"].apply(lambda x: f"{x:+,}")
            display["Own%"] = most_in["selected_by_percent"].astype(str) + "%"
            render_styled_table(display, title="Most Transferred In")

    with col2:
        if most_out.empty:
            st.info("No transfer data available.")
        else:
            display = most_out[["Player", "Team", "Pos"]].copy()
            display["Price"] = most_out["now_cost"].apply(_format_price)
            display["In"] = most_out["transfers_in_event"].apply(lambda x: f"{x:,}")
            display["Out"] = most_out["transfers_out_event"].apply(lambda x: f"{x:,}")
            display["Net"] = most_out["net_transfers"].apply(lambda x: f"{x:+,}")
            display["Own%"] = most_out["selected_by_percent"].astype(str) + "%"
            render_styled_table(display, title="Most Transferred Out")


# ------------------------------------------------------------------
# Main page function
# ------------------------------------------------------------------
def show_price_changes_page():
    st.header("💰 Price Changes")
    df = _build_price_data()
    if df.empty:
        st.warning("Could not load player data. Try again in a bit.")
        return
    _render_gw_price_changes(df)
    st.divider()
    _render_season_movers(df)
    st.divider()
    _render_transfer_activity(df)
