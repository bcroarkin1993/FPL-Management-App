import streamlit as st
import pandas as pd
import config
from scripts.common.utils import get_rotowire_season_rankings
from scripts.common.styled_tables import render_styled_table

def _ensure_session():
    if "draft_taken_keys" not in st.session_state:
        st.session_state.draft_taken_keys = set()
    if "draft_mine_keys" not in st.session_state:
        st.session_state.draft_mine_keys = set()

def _player_key(row: pd.Series) -> str:
    """Unique key for a player row to persist selection reliably."""
    return f"{row.get('Player','')}|{row.get('Team','')}|{row.get('Position','')}"

def show_draft_helper_page():
    """
    Draft Helper — Season-long Top 400 board with in-place selection.

    - Loads rankings via get_rotowire_player_projections(config.ROTOWIRE_SEASON_RANKINGS_URL, limit=400)
    - Lets you mark players as Taken (any team) and Mine (your picks)
    - Persists selections with st.session_state
    - Filters to show only available (default) or all, with search and position filter
    """
    # Dark theme CSS for Draft Helper
    st.markdown("""
    <style>
    /* Draft Helper dark theme */
    .draft-title {
        background: linear-gradient(135deg, #37003c, #5a0060);
        padding: 16px 20px; border-radius: 10px; margin-bottom: 0.5rem;
    }
    .draft-title h2 { color: #00ff87; margin: 0; font-size: 1.5rem; }
    .draft-title p { color: rgba(255,255,255,0.8); margin: 4px 0 0 0; font-size: 0.9rem; }
    .draft-summary {
        background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
        padding: 12px 16px; color: #e0e0e0; margin-top: 0.5rem;
    }
    .draft-summary .num { color: #00ff87; font-weight: 700; }
    </style>
    <div class="draft-title">
        <h2>🧠 Draft Helper — Season Rankings</h2>
        <p>Mark draftees as <strong>Taken</strong> or <strong>Mine</strong> to keep your live board clean during the draft.</p>
    </div>
    """, unsafe_allow_html=True)

    # Guard: require configured URL
    if not getattr(config, "ROTOWIRE_SEASON_RANKINGS_URL", None):
        st.error("Missing `config.ROTOWIRE_SEASON_RANKINGS_URL`. Please set it to your Rotowire season rankings page.")
        return

    _ensure_session()

    # --- Load Rankings (Top 400) ---
    try:
        rankings = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL, limit=400).copy()
        # Format numeric columns for dataframe sorting
        numeric_cols = [
            "Overall Rank", "FW Rank", "MID Rank", "DEF Rank", "GK Rank",
            "Price", "TSB %", "Points", "PP/90", "Pos Rank", "Value"
        ]
        for c in numeric_cols:
            if c in rankings.columns:
                rankings[c] = pd.to_numeric(rankings[c], errors="coerce")
    except Exception as e:
        st.error(f"Failed to load season rankings: {e}")
        return

    # Normalize expected columns & types
    # Rename "Overall Rank" -> "Rank" if present
    if "Overall Rank" in rankings.columns:
        rankings = rankings.rename(columns={"Overall Rank": "Rank"})
    # Ensure columns exist
    for col in ["Rank", "Player", "Team", "Position", "Points", "PP/90", "Pos Rank"]:
        if col not in rankings.columns:
            rankings[col] = pd.NA

    # Coerce numerics
    rankings["Rank"] = pd.to_numeric(rankings["Rank"], errors="coerce")
    rankings["Points"] = pd.to_numeric(rankings["Points"], errors="coerce")
    rankings["PP/90"] = pd.to_numeric(rankings["PP/90"], errors="coerce")
    rankings["Pos Rank"] = pd.to_numeric(rankings["Pos Rank"], errors="coerce")

    # Drop exact duplicates (keep best-ranked)
    rankings = rankings.sort_values(["Rank", "Player"], na_position="last").drop_duplicates(
        subset=["Player", "Team", "Position"], keep="first"
    )

    # Session-state flags mapped to each row via a stable key
    rankings["key"] = rankings.apply(_player_key, axis=1)
    rankings["Taken"] = rankings["key"].isin(st.session_state.draft_taken_keys)
    rankings["Mine"] = rankings["key"].isin(st.session_state.draft_mine_keys)

    # --- Controls ---
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        search = st.text_input("Search player", "")
    with c2:
        pos_options = sorted([p for p in rankings["Position"].dropna().unique().tolist() if p != ""])
        pos_filter = st.multiselect("Filter positions", pos_options, default=[])
    with c3:
        show_only_available = st.checkbox("Show only available", value=True)
    with c4:
        if st.button("Reset Board", type="secondary"):
            st.session_state.draft_taken_keys = set()
            st.session_state.draft_mine_keys = set()
            st.rerun()

    # Filter
    df = rankings.copy()
    if search.strip():
        needle = search.lower()
        df = df[df["Player"].str.lower().str.contains(needle, na=False)]
    if pos_filter:
        df = df[df["Position"].isin(pos_filter)]
    if show_only_available:
        df = df[~df["Taken"]]

    # Display columns (keep it compact & useful)
    display_cols = ["Rank", "Player", "Team", "Position", "Points", "PP/90", "Pos Rank", "Taken", "Mine"]

    st.markdown(
        '<div style="background:linear-gradient(135deg,#37003c,#5a0060);padding:10px 16px;'
        'border-radius:8px;margin:0.5rem 0;color:#00ff87;font-weight:700;font-size:1.1rem;">'
        '📋 Rankings</div>',
        unsafe_allow_html=True,
    )
    st.caption("Tip: Uncheck **Show only available** if you need to mark players as taken.")

    edited = st.data_editor(
        df[display_cols],
        key="draft_editor",
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", help="Overall Rank", disabled=True),
            "Player": st.column_config.TextColumn("Player", disabled=True),
            "Team": st.column_config.TextColumn("Team", disabled=True),
            "Position": st.column_config.TextColumn("Position", disabled=True),
            "Points": st.column_config.NumberColumn("Points", disabled=True),
            "PP/90": st.column_config.NumberColumn("PP/90", disabled=True),
            "Pos Rank": st.column_config.NumberColumn("Pos Rank", disabled=True),
            "Taken": st.column_config.CheckboxColumn("Taken"),
            "Mine": st.column_config.CheckboxColumn("Mine"),
        },
    )

    # Save changes (only touches rows currently visible/edited)
    save_col, mine_col = st.columns([1, 2])
    with save_col:
        if st.button("💾 Save Changes", type="primary"):
            # Map edited rows back to session-state sets using their keys
            # Rebuild keys for the edited subset (since edited has no 'key' col)
            edited_keys = (
                edited[["Player", "Team", "Position"]]
                .assign(key=lambda x: x.apply(_player_key, axis=1))
                .merge(
                    edited[["Taken", "Mine"]],
                    left_index=True, right_index=True
                )
            )
            # Update session sets
            for _, row in edited_keys.iterrows():
                k = row["key"]
                if bool(row["Taken"]):
                    st.session_state.draft_taken_keys.add(k)
                else:
                    st.session_state.draft_taken_keys.discard(k)

                if bool(row["Mine"]):
                    st.session_state.draft_mine_keys.add(k)
                else:
                    st.session_state.draft_mine_keys.discard(k)

            st.success("Board updated.")
            st.rerun()

    with mine_col:
        with st.expander("📋 My Picks (selected as Mine)"):
            mine_df = rankings[rankings["key"].isin(st.session_state.draft_mine_keys)]
            mine_df = mine_df.sort_values("Rank", na_position="last")[["Rank", "Player", "Team", "Position"]]
            render_styled_table(mine_df)

    # Summary footer
    total_players = len(rankings)
    taken = len(st.session_state.draft_taken_keys)
    available = total_players - taken
    st.markdown(
        f'<div class="draft-summary">'
        f'<span class="num">{available}</span> Available &nbsp;&bull;&nbsp; '
        f'<span class="num">{taken}</span> Taken &nbsp;&bull;&nbsp; '
        f'<span class="num">{total_players}</span> Total'
        f'</div>',
        unsafe_allow_html=True,
    )
