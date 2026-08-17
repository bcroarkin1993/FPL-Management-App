"""
Classic FPL - Initial Squad Optimizer

Builds a season-opening 15-man squad (before GW1) using PuLP ILP, blending:
- Season Score:  literal preseason Rotowire Season Rankings (season-long value)
- Week1 Score:   GW1 Rotowire + FFP blended projection x start likelihood,
                 reusing the app's exact 1GW methodology via compute_player_scores()
- Fixture Score: opening-slate fixture ease (configurable GW horizon)

Unlike Wildcard/Free Hit, this intentionally does NOT use ROS/Transfer/Keep
Score — those are diluted by live-season signals (form, starts, multi-GW
FFP data) that don't exist yet before GW1. See CLAUDE.md's "Transfer Scoring
Model" section for why ROS is a poor fit here.

Captaincy is baked into the optimizer's objective (not just assigned
post-hoc): a season-pedigree-heavy Captain Score is rewarded for whichever
starter the solver designates captain, giving it a real incentive to
*acquire* one standout option rather than several merely-good players.
"""

import pandas as pd
import streamlit as st
from typing import Optional
import config

from scripts.common.error_helpers import show_api_error
from scripts.common.optimization import solve_squad_ilp
from scripts.common.styled_tables import render_styled_table
from scripts.common.utils import (
    get_classic_bootstrap_static,
    get_current_gameweek,
    get_rotowire_player_projections,
    get_fixture_difficulty_grid,
    position_converter,
)
from scripts.common.analytics import (
    compute_player_scores,
    merge_ffp_single_gw_data,
    merge_season_projections,
    positional_percentile,
)
from scripts.common.scraping import get_ffp_projections_data, get_rotowire_season_rankings


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def _format_money(value: float) -> str:
    return f"£{value:.1f}m"


def _score_card(label: str, value: str, accent: str = "#00ff87") -> str:
    return (
        f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
        f'</div>'
    )


def rotowire_gw1_url_selector() -> str:
    """Manual GW1 Rotowire URL input.

    GW1's preseason "best picks" article uses a different slug shape than
    the regular weekly rankings articles, so it isn't auto-discoverable —
    defaults to the pinned config value (re-pin each preseason) and lets
    the user override.
    """
    with st.expander("GW1 Rotowire URL", expanded=False):
        url = st.text_input(
            "GW1 rankings article URL",
            value=config.ROTOWIRE_GW1_URL or "",
            placeholder="https://www.rotowire.com/soccer/article/...",
            help="Rotowire's GW1 'best picks' article isn't auto-discoverable like "
                 "weekly rankings articles — paste/update the URL here each preseason.",
        )
        return url.strip()


def _build_full_player_pool(bootstrap: dict) -> pd.DataFrame:
    """Build the full, unfiltered candidate pool (price/team/position/status).

    Deliberately unfiltered so downstream percentile scoring is computed
    against the true full player universe — injury/availability filtering
    happens afterward, narrowing the *candidate set* without skewing the
    *reference distribution* percentiles are computed against.
    """
    elements = bootstrap.get("elements", [])
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}

    rows = []
    for p in elements:
        team_id = p.get("team")
        team_info = teams.get(team_id, {})
        team_short = team_info.get("short_name", "???")
        position = position_converter(p.get("element_type"))
        full_name = f"{p.get('first_name', '')} {p.get('second_name', '')}".strip()

        rows.append({
            "Player_ID": p.get("id"),
            "Player": full_name or p.get("web_name", "Unknown"),
            "Web_Name": p.get("web_name", "Unknown"),
            "Team": team_short,
            "Position": position,
            "Price": p.get("now_cost", 0) / 10.0,
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
            "news": p.get("news", "") or "",
            "total_points": p.get("total_points", 0),
            "form": float(p.get("form", 0) or 0),
            "AvgFDR": 3.0,  # placeholder — ROS is discarded but compute_player_scores needs the column present
        })

    return pd.DataFrame(rows)


def _apply_eligibility_filters(
    pool: pd.DataFrame,
    exclude_injured: bool = True,
    min_chance_of_playing: int = 75,
) -> pd.DataFrame:
    """Narrow the scored pool down to squad-selection candidates."""
    if not exclude_injured:
        return pool

    is_cheap_fodder = pool["Price"] <= 4.5
    chance = pool["chance_of_playing_next_round"]
    news_flag = pool["news"].str.lower().apply(
        lambda n: any(word in n for word in ["injured", "illness", "suspended", "unavailable", "out"])
    )
    low_chance = chance.notna() & (chance < min_chance_of_playing)

    exclude_mask = (~is_cheap_fodder) & (low_chance | news_flag)
    return pool[~exclude_mask].reset_index(drop=True)


def _compute_scores(
    pool: pd.DataFrame,
    gw1_projections_df: pd.DataFrame,
    season_rankings_df: pd.DataFrame,
    ffp_df: Optional[pd.DataFrame],
    fdr_avg: pd.Series,
    current_gw: int,
    w_season: float,
    w_week1: float,
    w_fixture: float,
) -> pd.DataFrame:
    """Merge Season/Week1/Fixture signals and compute the blended Player
    Score + Captain Score that drive squad selection."""
    result = pool.copy()

    # Season-long value: literal preseason Rotowire Season Rankings, not the
    # ongoing (and pre-season-unreliable) ROS blend.
    result = merge_season_projections(result, season_rankings_df, output_col="SeasonProjection")

    # GW1 projection — reuses merge_season_projections since Rotowire's weekly
    # rankings table has the same Player/Team/Points shape as season rankings.
    result = merge_season_projections(result, gw1_projections_df, output_col="Points")
    result["Points"] = result["Points"].fillna(0)

    # FFP single-GW data (Predicted, Start, LongStart) for the 1GW blend.
    result = merge_ffp_single_gw_data(result, ffp_df)

    # Week1 Score: reuse the app's exact 1GW methodology (0.6 Rotowire + 0.4
    # FFP Predicted, x start likelihood). Discard ROS/Transfer/Keep — those
    # depend on live-season signals (form, starts, multi-GW data) that don't
    # exist pre-season.
    scored = compute_player_scores(result, result, current_gw=current_gw, format_context="classic")
    result["Week1 Score"] = scored["1GW"]
    result["GW1 Proj Pts"] = scored["_effective_proj"]

    # Season Score: positional percentile of the literal season-long ranking.
    result["Season Score"] = positional_percentile(
        result, result, value_col="SeasonProjection", position_col="Position"
    )

    # Fixture Score: opening-slate FDR, inverted (easier = better) and percentile-ranked.
    result["Team_AvgFDR"] = result["Team"].map(fdr_avg).fillna(3.0)
    result["_fixture_ease_raw"] = -result["Team_AvgFDR"]
    result["Fixture Score"] = positional_percentile(
        result, result, value_col="_fixture_ease_raw", position_col="Position"
    )

    result["Player Score"] = (
        w_season * result["Season Score"]
        + w_week1 * result["Week1 Score"]
        + w_fixture * result["Fixture Score"]
    )
    # Captain should be a nailed-on week-to-week producer, not a one-week
    # fixture bet — heavier season weighting than the general Player Score.
    result["Captain Score"] = 0.75 * result["Season Score"] + 0.25 * result["Week1 Score"]

    return result


def _display_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        is_cap = bool(row.get("Is_Captain", False))
        name = f"{row['Player']} (C)" if is_cap else row["Player"]
        rows.append({
            "Player": name,
            "Team": row["Team"],
            "Pos": row["Position"],
            "Price": f"£{row['Price']:.1f}m",
            "Season": row["Season Score"],
            "Week1": row["Week1 Score"],
            "Fixture": row["Fixture Score"],
            "Score": row["Player Score"],
            "GW1 Proj": row["GW1 Proj Pts"],
        })
    return pd.DataFrame(rows)


_SCORE_COL_FORMATS = {
    "Season": "{:.2f}", "Week1": "{:.2f}", "Fixture": "{:.2f}",
    "Score": "{:.2f}", "GW1 Proj": "{:.1f}",
}


# ---------------------------
# MAIN PAGE
# ---------------------------

def show_initial_squad_optimizer_page():
    st.title("🆕 Initial Squad Optimizer")
    st.caption(
        "Build your season-opening 15-man squad by blending preseason season-long "
        "value with Week 1 projections and opening-fixture ease — with captaincy "
        "baked into player selection, not just assigned after the fact."
    )

    current_gw = get_current_gameweek() or 1

    st.markdown("### Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input(
            "Total Budget (£m)", min_value=80.0, max_value=120.0,
            value=100.0, step=0.1, format="%.1f",
        )
    with col2:
        horizon = st.slider(
            "Opening Fixture Horizon (GWs)", min_value=1, max_value=5, value=3,
            help="How many opening gameweeks count toward the Fixture Score",
        )
    with col3:
        formation = st.selectbox(
            "Formation",
            ["auto", "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"],
            index=0,
            help="Auto lets the optimizer choose the best formation",
        )

    with st.expander("Scoring Weights", expanded=False):
        st.caption(
            "How much each signal drives player selection. Premium players tend to "
            "score well on both Season and Week1, so these weights mostly matter for "
            "choosing among similarly-priced budget/rotation options."
        )
        wcol1, wcol2, wcol3 = st.columns(3)
        with wcol1:
            w_season_pct = st.slider("Season-Long Weight (%)", 0, 100, 55)
        with wcol2:
            w_week1_pct = st.slider("Week 1 Weight (%)", 0, 100, 30)
        with wcol3:
            w_fixture_pct = st.slider("Fixture Ease Weight (%)", 0, 100, 15)
        w_total = max(w_season_pct + w_week1_pct + w_fixture_pct, 1)
        w_season = w_season_pct / w_total
        w_week1 = w_week1_pct / w_total
        w_fixture = w_fixture_pct / w_total

        st.markdown("---")
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            bench_weight = st.slider(
                "Bench Weight", min_value=0.0, max_value=0.5, value=0.2, step=0.05,
                help="How much bench players' score counts toward the objective. "
                     "Lower than Wildcard's default — this is a starting point, not a final squad.",
            )
        with bcol2:
            captain_bonus_weight = st.slider(
                "Captain Bonus Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                help="How strongly the optimizer is rewarded for acquiring one standout "
                     "captain-caliber player, rather than just picking a captain afterward.",
            )

        st.markdown("#### Player Filters")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            exclude_injured = st.checkbox("Exclude injured/doubtful players", value=True)
        with fcol2:
            min_chance = st.slider(
                "Min chance of playing (%)", min_value=0, max_value=100, value=75, step=25,
                disabled=not exclude_injured,
            )

    gw1_url = rotowire_gw1_url_selector()

    st.markdown("---")

    if st.button("Build Optimal Squad", type="primary"):
        with st.spinner("Loading player data, projections, and fixture data..."):
            bootstrap = get_classic_bootstrap_static()
            if not bootstrap:
                show_api_error("loading player data for the Initial Squad Optimizer")
                return

            gw1_projections_df = pd.DataFrame()
            if gw1_url:
                try:
                    gw1_projections_df = get_rotowire_player_projections(gw1_url)
                except Exception as e:
                    st.warning(f"Could not load GW1 Rotowire projections: {e}")

            try:
                season_rankings_df = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL)
            except Exception as e:
                st.warning(f"Could not load Rotowire season rankings: {e}")
                season_rankings_df = pd.DataFrame()

            try:
                ffp_df = get_ffp_projections_data()
            except Exception:
                ffp_df = None

            try:
                _, _, fdr_avg = get_fixture_difficulty_grid(weeks=horizon)
            except Exception as e:
                st.warning(f"Could not load fixture difficulty data: {e}")
                fdr_avg = pd.Series(dtype=float)

            full_pool = _build_full_player_pool(bootstrap)
            if full_pool.empty:
                st.error("No players available.")
                return

            scored_pool = _compute_scores(
                full_pool, gw1_projections_df, season_rankings_df, ffp_df, fdr_avg,
                current_gw, w_season, w_week1, w_fixture,
            )
            candidate_pool = _apply_eligibility_filters(
                scored_pool, exclude_injured=exclude_injured, min_chance_of_playing=min_chance
            )

            if candidate_pool.empty:
                st.error("No players available after filtering.")
                return

            has_gw1 = not gw1_projections_df.empty
            has_season = season_rankings_df is not None and not season_rankings_df.empty
            st.info(
                f"Candidate pool: {len(candidate_pool)} players | "
                f"GW1 projections: {'loaded' if has_gw1 else 'unavailable — Week1 Score falls back to FFP/neutral'} | "
                f"Season rankings: {'loaded' if has_season else 'unavailable — Season Score falls back to neutral'}"
            )

        with st.spinner("Running optimization..."):
            squad_df, totals = solve_squad_ilp(
                candidate_pool,
                budget,
                score_col="Player Score",
                formation=formation,
                bench_weight=bench_weight,
                captain_score_col="Captain Score",
                captain_bonus_weight=captain_bonus_weight,
                problem_name="FPL_Initial_Squad_Optimizer",
            )

            if squad_df is None:
                st.error(
                    "Optimization failed. Try increasing your budget or changing the formation. "
                    "The constraints may be too restrictive."
                )
                return

        st.success("Squad built!")

        starters = squad_df[squad_df["Is_Starter"]].copy()
        bench = squad_df[~squad_df["Is_Starter"]].copy()
        bench_gk = bench[bench["Position"] == "G"]
        bench_outfield = bench[bench["Position"] != "G"].sort_values("Player Score", ascending=False)
        bench_ordered = pd.concat([bench_gk, bench_outfield])
        bench_ordered["Bench_Order"] = range(1, len(bench_ordered) + 1)

        if "Is_Captain" in squad_df.columns and squad_df["Is_Captain"].any():
            cap_name = squad_df.loc[squad_df["Is_Captain"], "Player"].iloc[0]
        else:
            cap_name = starters.loc[starters["Player Score"].idxmax(), "Player"]

        total_cost = squad_df["Price"].sum()
        remaining = budget - total_cost

        st.markdown("### Squad Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(_score_card("Starting XI Score", f"{totals['starter_score']:.2f}"), unsafe_allow_html=True)
        with col2:
            st.markdown(_score_card("Captain", cap_name, accent="#f0c419"), unsafe_allow_html=True)
        with col3:
            st.markdown(_score_card("Total Cost", _format_money(total_cost), accent="#e0e0e0"), unsafe_allow_html=True)
        with col4:
            budget_color = "#00ff87" if remaining >= 0 else "#f87171"
            st.markdown(_score_card("Remaining Budget", _format_money(remaining), accent=budget_color), unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### Starting XI")
        formation_str = (
            f"{starters[starters['Position'] == 'D'].shape[0]}-"
            f"{starters[starters['Position'] == 'M'].shape[0]}-"
            f"{starters[starters['Position'] == 'F'].shape[0]}"
        )
        st.caption(f"Formation: {formation_str} | Captain: {cap_name}")

        render_styled_table(
            _display_rows(starters),
            col_formats=_SCORE_COL_FORMATS,
            positive_color_cols=["Score"],
        )

        st.markdown("---")
        st.markdown("### Bench")
        bench_display = _display_rows(bench_ordered)
        bench_display.insert(0, "Order", bench_ordered["Bench_Order"].values)
        render_styled_table(bench_display, col_formats=_SCORE_COL_FORMATS)

        st.markdown("---")
        st.markdown("### Opening Fixture Difficulty (Squad Teams)")
        try:
            _, fdr_diffs, _ = get_fixture_difficulty_grid(weeks=horizon)
            squad_teams = squad_df["Team"].unique().tolist()
            gw_cols = [f"GW{current_gw + i}" for i in range(horizon)]
            available_cols = [c for c in gw_cols if c in fdr_diffs.columns]
            if available_cols:
                fdr_filtered = fdr_diffs.loc[fdr_diffs.index.isin(squad_teams), available_cols].copy()
                avg_fdr = fdr_filtered.fillna(3).mean(axis=1)
                fdr_filtered = fdr_filtered.loc[avg_fdr.sort_values().index]
                palette = {1: "#00c853", 2: "#86efac", 3: "#ffc107", 4: "#ff9800", 5: "#dc3545"}

                def style_cell(val):
                    if pd.isna(val):
                        return "background-color: #e5e7eb; color: #111; text-align: center;"
                    fdr_int = max(1, min(5, int(round(val))))
                    color = palette[fdr_int]
                    txt = "#ffffff" if fdr_int >= 5 else "#111111"
                    return f"background-color: {color}; color: {txt}; text-align: center; font-weight: 600;"

                styled = fdr_filtered.style.applymap(style_cell)
                st.dataframe(styled, use_container_width=True)
            else:
                st.info("FDR data not available for the selected horizon.")
        except Exception:
            st.info("FDR data not available.")

        st.markdown("---")
        with st.expander("Team Breakdown"):
            team_counts = squad_df.groupby("Team").agg(
                {"Player": "count", "Price": "sum", "Player Score": "sum"}
            ).rename(columns={"Player": "Players", "Price": "Total Cost", "Player Score": "Total Score"})
            team_counts["Total Cost"] = team_counts["Total Cost"].apply(lambda x: f"£{x:.1f}m")
            team_counts["Total Score"] = team_counts["Total Score"].round(2)
            render_styled_table(team_counts.reset_index())

        with st.expander("Position Breakdown"):
            pos_breakdown = squad_df.groupby("Position").agg(
                {"Player": "count", "Price": "sum", "Player Score": "sum"}
            ).rename(columns={"Player": "Count", "Price": "Total Cost", "Player Score": "Total Score"})
            pos_breakdown["Total Cost"] = pos_breakdown["Total Cost"].apply(lambda x: f"£{x:.1f}m")
            pos_breakdown["Total Score"] = pos_breakdown["Total Score"].round(2)
            render_styled_table(pos_breakdown.reset_index())

        st.markdown("---")
        st.info(
            "**Tips:**\n"
            "- Season/Week1/Fixture scores are positional percentiles (0-1) — 0.85 means top 15% at that position\n"
            "- The captain is chosen by the optimizer itself, not picked afterward — a higher Captain Bonus Weight "
            "rewards owning one standout season-long performer over several merely-good players\n"
            "- Raise the Week 1 weight if you want the squad to lean harder into a strong opening fixture run\n"
            "- Transfers can reshape this squad as the season develops — treat this as a strong starting point, not a final answer"
        )
