# scripts/fpl/league_setup.py
#
# Streamlit "League Setup" admin page: set and lock the Draft/Classic
# league and team IDs used throughout the app. Values are validated against
# the live FPL APIs (resolving league/team names) before being saved to
# league_settings.json (gitignored, local-only — see scripts/common/league_config.py).

import os
import re

import streamlit as st

import config
from scripts.common.league_config import load_settings, save_settings, DEFAULT_SETTINGS
from scripts.common.fpl_draft_api import is_draft_league_reachable, get_league_entries
from scripts.common.fpl_classic_api import get_classic_or_h2h_league_standings, get_entry_details

_SEASON_RE = re.compile(r"^\d{4}/\d{2}$")


def show_league_setup_page():
    st.title("League Setup")
    st.caption(
        "Set your Draft and Classic league/team IDs here instead of editing `.env` by hand. "
        "Values are validated against the live FPL APIs, then locked to prevent accidental changes. "
        "Saved to `league_settings.json` on this machine (not committed to git)."
    )

    settings = load_settings()

    st.divider()
    _show_draft_section(settings)
    st.divider()
    _show_draft_history_section(settings)
    st.divider()
    _show_classic_section(settings)


# =================================================================
# Draft
# =================================================================

def _show_draft_section(settings: dict):
    st.header("Draft")
    draft = settings.get("draft", DEFAULT_SETTINGS["draft"])

    is_locked = draft.get("locked") and not st.session_state.get("draft_unlocked")
    if is_locked:
        _show_draft_locked_view(draft)
    else:
        _show_draft_edit_form(draft)


def _show_draft_locked_view(draft: dict):
    league_id = draft.get("league_id")
    team_id = draft.get("team_id")
    team_name = draft.get("team_name") or "Unknown"
    st.success(
        f"**League ID:** {league_id}  \n"
        f"**Your Team:** {team_name} (ID {team_id})"
    )

    if st.session_state.get("draft_unlock_pending"):
        st.warning(
            "Unlocking lets you change the Draft league/team ID. This affects live data "
            "across the app (Waiver Wire, Team Analysis, League Analysis, etc.). Are you sure?"
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Yes, unlock", key="draft_unlock_confirm", type="primary"):
                st.session_state["draft_unlock_pending"] = False
                st.session_state["draft_unlocked"] = True
                st.rerun()
        with col2:
            if st.button("Cancel", key="draft_unlock_cancel"):
                st.session_state["draft_unlock_pending"] = False
                st.rerun()
    else:
        if st.button("Unlock to Edit", key="draft_unlock_start"):
            st.session_state["draft_unlock_pending"] = True
            st.rerun()


def _show_draft_edit_form(draft: dict):
    st.caption(
        "Enter your Draft league ID, look it up to confirm it resolves, then pick your team "
        "from the list — no need to know your numeric entry ID."
    )

    league_id_input = st.text_input(
        "Draft League ID",
        value=str(draft.get("league_id") or ""),
        key="draft_league_id_input",
    )

    if st.button("Look Up League", key="draft_lookup_btn"):
        league_id_input = league_id_input.strip()
        if not league_id_input.isdigit():
            st.error("Please enter a numeric league ID.")
        else:
            league_id = int(league_id_input)
            with st.spinner("Looking up league..."):
                reachable = is_draft_league_reachable(league_id)
                entries = get_league_entries(league_id) if reachable else {}
            if not entries:
                st.error(
                    "Could not resolve that league ID. Draft leagues don't carry over "
                    "between seasons — make sure you've created this season's league at "
                    "[draft.premierleague.com](https://draft.premierleague.com) and are using its ID."
                )
                st.session_state.pop("draft_lookup_entries", None)
            else:
                st.session_state["draft_lookup_league_id"] = league_id
                st.session_state["draft_lookup_entries"] = entries
                st.success(f"Found {len(entries)} teams in this league.")

    entries = st.session_state.get("draft_lookup_entries")
    if entries:
        team_names = list(entries.values())
        default_idx = 0
        if draft.get("team_name") in team_names:
            default_idx = team_names.index(draft.get("team_name"))
        chosen_name = st.selectbox(
            "Which team is yours?",
            options=team_names,
            index=default_idx,
            key="draft_team_select",
        )
        chosen_id = next(eid for eid, name in entries.items() if name == chosen_name)
        st.caption(f"Selected team entry ID: {chosen_id}")

        if st.button("Save & Lock", key="draft_save_btn", type="primary"):
            full_settings = load_settings()
            full_settings["draft"] = {
                "league_id": st.session_state["draft_lookup_league_id"],
                "team_id": int(chosen_id),
                "team_name": chosen_name,
                "locked": True,
            }
            if save_settings(full_settings):
                config.refresh_league_settings()
                st.session_state.pop("draft_lookup_entries", None)
                st.session_state.pop("draft_lookup_league_id", None)
                st.session_state["draft_unlocked"] = False
                st.success("Draft league/team saved and locked!")
                st.rerun()
            else:
                st.error("Failed to save settings.")


# =================================================================
# Draft — cross-season history
# =================================================================

def _lookup_draft_team_name(league_id: int, preferred_name: str = None):
    """Best-effort team resolution for a historical league ID.

    Returns (team_id, team_name, entries_dict). entries_dict is {} if the
    league couldn't be reached live (old leagues can eventually stop
    resolving via the Draft API) — callers should degrade to manual entry.
    """
    try:
        if not is_draft_league_reachable(league_id):
            return None, None, {}
        entries = get_league_entries(league_id) or {}
    except Exception:
        return None, None, {}
    if preferred_name:
        for eid, name in entries.items():
            if name.strip().lower() == preferred_name.strip().lower():
                return int(eid), name, entries
    return None, None, entries


def _upsert_history_entry(season: str, league_id, team_id, team_name: str, manual_stats: dict = None):
    full_settings = load_settings()
    history = full_settings["draft"].setdefault("history", [])
    idx = next((i for i, h in enumerate(history) if h.get("season") == season), None)
    entry = {
        "season": season, "league_id": league_id, "team_id": team_id, "team_name": team_name,
        "manual_stats": manual_stats,
    }
    if idx is not None:
        history[idx] = entry
    else:
        history.append(entry)
    full_settings["draft"]["history"] = sorted(history, key=lambda h: h["season"])
    save_settings(full_settings)
    config.refresh_league_settings()


def _remove_history_entry(season: str):
    full_settings = load_settings()
    history = full_settings["draft"].setdefault("history", [])
    full_settings["draft"]["history"] = [h for h in history if h.get("season") != season]
    save_settings(full_settings)
    config.refresh_league_settings()


def _clear_history_lookup_state():
    for key in (
        "history_lookup_season", "history_lookup_league_id", "history_lookup_entries",
        "history_manual_mode", "history_new_season", "history_new_league_id",
        "history_overwrite_pending",
        "history_manual_stats_team", "history_manual_stats_rank", "history_manual_stats_points",
        "history_manual_stats_wins", "history_manual_stats_draws", "history_manual_stats_losses",
        "history_manual_overwrite_pending",
    ):
        st.session_state.pop(key, None)


def _show_draft_history_section(settings: dict):
    st.subheader("📜 Draft League History")
    st.caption(
        "Draft leagues are recreated every season, so old league IDs are easy to lose once "
        "you roll over to a new one. Save past seasons here so Season Wrapped's cross-season "
        "history (\"Looking Back\") and your career-arc chart keep working."
    )

    draft = settings.get("draft", DEFAULT_SETTINGS["draft"])
    json_history = list(draft.get("history", []))
    json_seasons = {h["season"] for h in json_history}

    env_history = config._parse_draft_league_history(os.getenv("FPL_DRAFT_LEAGUE_HISTORY", ""))
    env_only = [
        {"season": s, "league_id": lid, "team_id": None, "team_name": None, "_env_only": True}
        for s, lid in env_history if s not in json_seasons
    ]
    merged = sorted(json_history + env_only, key=lambda h: h["season"])

    if not merged:
        st.caption("No past seasons saved yet.")
    else:
        for entry in merged:
            col1, col2 = st.columns([4, 1])
            with col1:
                team_label = entry.get("team_name") or "Unresolved team"
                manual_stats = entry.get("manual_stats")
                if manual_stats:
                    line = (
                        f"**{entry['season']}** — {team_label} — "
                        f"Rank #{manual_stats.get('rank', '?')}, {manual_stats.get('total_points', 0)} pts "
                        f"({manual_stats.get('wins', 0)}-{manual_stats.get('draws', 0)}-{manual_stats.get('losses', 0)}) "
                        f"— *manually entered*"
                    )
                else:
                    line = f"**{entry['season']}** — {team_label} (League ID {entry['league_id']})"
                    if entry.get("_env_only"):
                        line += "  \n*From `.env` — not yet saved here*"
                st.markdown(line)
            with col2:
                if entry.get("_env_only"):
                    if st.button("Save here", key=f"history_save_{entry['season']}"):
                        with st.spinner("Resolving team..."):
                            team_id, team_name, _ = _lookup_draft_team_name(
                                entry["league_id"], draft.get("team_name")
                            )
                        _upsert_history_entry(entry["season"], entry["league_id"], team_id, team_name)
                        st.rerun()
                else:
                    if st.button("Remove", key=f"history_remove_{entry['season']}"):
                        _remove_history_entry(entry["season"])
                        st.rerun()

    st.markdown("**Add a past season**")
    entry_mode = st.radio(
        "How do you want to add this season?",
        ["Look up live league", "Enter final stats manually"],
        key="history_entry_mode", horizontal=True,
    )
    season_input = st.text_input("Season (e.g. 2024/25)", key="history_new_season")
    season_clean = season_input.strip()
    season_valid = bool(_SEASON_RE.match(season_clean)) if season_clean else False
    if season_clean and not season_valid:
        st.error("Season must look like `2024/25`.")

    if entry_mode == "Look up live league":
        league_id_input = st.text_input("League ID", key="history_new_league_id")

        if st.button("Look Up League", key="history_lookup_btn"):
            league_id_clean = league_id_input.strip()
            if not season_valid:
                st.error("Enter a valid season first.")
            elif not league_id_clean.isdigit():
                st.error("Please enter a numeric league ID.")
            else:
                league_id = int(league_id_clean)
                with st.spinner("Looking up league..."):
                    _, _, entries = _lookup_draft_team_name(league_id)
                if entries:
                    st.session_state["history_lookup_season"] = season_clean
                    st.session_state["history_lookup_league_id"] = league_id
                    st.session_state["history_lookup_entries"] = entries
                    st.success(f"Found {len(entries)} teams in this league.")
                else:
                    st.session_state.pop("history_lookup_season", None)
                    st.session_state.pop("history_lookup_league_id", None)
                    st.session_state.pop("history_lookup_entries", None)
                    st.warning(
                        "Could not resolve that league live — Draft league IDs get reissued to a "
                        "new league each season, so this is expected for anything but the current "
                        "or very recently ended season. Use \"Enter final stats manually\" above instead."
                    )

        pending_season = st.session_state.get("history_lookup_season")
        pending_league_id = st.session_state.get("history_lookup_league_id")
        entries = st.session_state.get("history_lookup_entries")

        if pending_season and pending_league_id and entries:
            is_duplicate = pending_season in json_seasons
            team_names = list(entries.values())
            chosen_name = st.selectbox("Which team is yours?", options=team_names, key="history_team_select")
            chosen_id = int(next(eid for eid, name in entries.items() if name == chosen_name))

            if is_duplicate and not st.session_state.get("history_overwrite_pending"):
                st.warning(f"`{pending_season}` is already saved — adding again will overwrite it.")
                if st.button("Overwrite existing entry", key="history_overwrite_btn"):
                    st.session_state["history_overwrite_pending"] = True
                    st.rerun()
            else:
                if st.button("Add to History", key="history_add_btn", type="primary"):
                    _upsert_history_entry(pending_season, pending_league_id, chosen_id, chosen_name)
                    _clear_history_lookup_state()
                    st.success(f"Saved {pending_season}.")
                    st.rerun()

    else:  # Enter final stats manually
        st.caption(
            "For a season whose league ID no longer resolves (Draft IDs get reissued to a new "
            "league each season) — enter the final standings directly, e.g. from a saved Season "
            "Wrapped PDF export."
        )
        manual_team_name = st.text_input("Team name", key="history_manual_stats_team")
        col_r, col_p = st.columns(2)
        with col_r:
            manual_rank = st.number_input("Final Rank", min_value=1, step=1, key="history_manual_stats_rank")
        with col_p:
            manual_points = st.number_input("Total Points", min_value=0, step=1, key="history_manual_stats_points")
        col_w, col_d, col_l = st.columns(3)
        with col_w:
            manual_wins = st.number_input("Wins", min_value=0, step=1, key="history_manual_stats_wins")
        with col_d:
            manual_draws = st.number_input("Draws", min_value=0, step=1, key="history_manual_stats_draws")
        with col_l:
            manual_losses = st.number_input("Losses", min_value=0, step=1, key="history_manual_stats_losses")

        is_duplicate = season_valid and season_clean in json_seasons
        add_disabled = not season_valid or not manual_team_name.strip()

        if is_duplicate and not st.session_state.get("history_manual_overwrite_pending"):
            st.warning(f"`{season_clean}` is already saved — adding again will overwrite it.")
            if st.button("Overwrite existing entry", key="history_manual_overwrite_btn", disabled=add_disabled):
                st.session_state["history_manual_overwrite_pending"] = True
                st.rerun()
        else:
            if st.button("Add to History", key="history_add_manual_btn", type="primary", disabled=add_disabled):
                _upsert_history_entry(
                    season_clean, None, None, manual_team_name.strip(),
                    manual_stats={
                        "rank": int(manual_rank), "total_points": int(manual_points),
                        "wins": int(manual_wins), "draws": int(manual_draws), "losses": int(manual_losses),
                    },
                )
                _clear_history_lookup_state()
                st.success(f"Saved {season_clean}.")
                st.rerun()


# =================================================================
# Classic
# =================================================================

def _show_classic_section(settings: dict):
    st.header("Classic")
    classic = settings.get("classic", DEFAULT_SETTINGS["classic"])

    is_locked = classic.get("locked") and not st.session_state.get("classic_unlocked")
    if is_locked:
        _show_classic_locked_view(classic)
    else:
        _show_classic_edit_form(classic)


def _show_classic_locked_view(classic: dict):
    leagues = classic.get("leagues", [])
    if leagues:
        leagues_str = "  \n".join(f"- {l.get('name') or 'Unnamed'} (ID {l.get('id')})" for l in leagues)
    else:
        leagues_str = "_No leagues set_"
    team_id = classic.get("team_id")
    team_name = classic.get("team_name") or "Unknown"
    st.success(
        f"**Leagues:**  \n{leagues_str}  \n\n"
        f"**Your Team:** {team_name} (ID {team_id})"
    )

    if st.session_state.get("classic_unlock_pending"):
        st.warning(
            "Unlocking lets you change your Classic league(s)/team ID. This affects live data "
            "across the app (Transfers, Team Analysis, League Analysis, etc.). Are you sure?"
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Yes, unlock", key="classic_unlock_confirm", type="primary"):
                st.session_state["classic_unlock_pending"] = False
                st.session_state["classic_unlocked"] = True
                st.rerun()
        with col2:
            if st.button("Cancel", key="classic_unlock_cancel"):
                st.session_state["classic_unlock_pending"] = False
                st.rerun()
    else:
        if st.button("Unlock to Edit", key="classic_unlock_start"):
            st.session_state["classic_unlock_pending"] = True
            st.rerun()


def _show_classic_edit_form(classic: dict):
    if "classic_working_leagues" not in st.session_state:
        st.session_state["classic_working_leagues"] = list(classic.get("leagues", []))

    st.subheader("Leagues")
    working_leagues = st.session_state["classic_working_leagues"]

    if working_leagues:
        for i, league in enumerate(working_leagues):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{league.get('name') or 'Unnamed'} (ID {league.get('id')})")
            with col2:
                if st.button("Remove", key=f"classic_remove_{i}"):
                    working_leagues.pop(i)
                    st.rerun()
    else:
        st.caption("No leagues added yet.")

    st.markdown("**Add a league**")
    new_league_id_input = st.text_input("Classic League ID", key="classic_new_league_id")
    if st.button("Look Up League", key="classic_lookup_league_btn"):
        new_league_id_input = new_league_id_input.strip()
        if not new_league_id_input.isdigit():
            st.error("Please enter a numeric league ID.")
        else:
            league_id = int(new_league_id_input)
            with st.spinner("Looking up league..."):
                result = get_classic_or_h2h_league_standings(league_id)
            if not result or not result.get("league"):
                st.error(
                    "Could not resolve that Classic league ID on either the Classic or H2H "
                    "endpoint. Double-check it, or note that league IDs aren't guaranteed to "
                    "carry over between seasons."
                )
            else:
                name = result["league"].get("name", "Unnamed")
                st.session_state["classic_lookup_result"] = {"id": league_id, "name": name}
                st.success(f"Found league: {name}")

    lookup_result = st.session_state.get("classic_lookup_result")
    if lookup_result:
        already_added = any(l["id"] == lookup_result["id"] for l in working_leagues)
        if already_added:
            st.caption("Already in your list.")
        elif st.button(f"Add \"{lookup_result['name']}\" to list", key="classic_add_league_btn"):
            working_leagues.append(lookup_result)
            st.session_state.pop("classic_lookup_result", None)
            st.rerun()

    st.subheader("Your Team")
    team_id_input = st.text_input(
        "Classic Team ID",
        value=str(classic.get("team_id") or ""),
        key="classic_team_id_input",
    )
    if st.button("Look Up Team", key="classic_lookup_team_btn"):
        team_id_input = team_id_input.strip()
        if not team_id_input.isdigit():
            st.error("Please enter a numeric team ID.")
        else:
            team_id = int(team_id_input)
            with st.spinner("Looking up team..."):
                entry = get_entry_details(team_id)
            if not entry:
                st.error("Could not resolve that team ID. Please double-check it.")
            else:
                manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
                team_name = entry.get("name", "Unnamed Team")
                st.session_state["classic_team_lookup"] = {
                    "id": team_id, "team_name": team_name, "manager_name": manager_name,
                }
                st.success(f"Found team: {team_name} (Manager: {manager_name})")

    team_lookup = st.session_state.get("classic_team_lookup")
    if team_lookup:
        st.caption(f"Selected team: {team_lookup['team_name']} (Manager: {team_lookup['manager_name']})")

    if st.button("Save & Lock", key="classic_save_btn", type="primary"):
        if not working_leagues:
            st.error("Add at least one Classic league before saving.")
        elif not team_lookup:
            st.error("Look up and confirm your Classic team ID before saving.")
        else:
            full_settings = load_settings()
            full_settings["classic"] = {
                "leagues": working_leagues,
                "team_id": team_lookup["id"],
                "team_name": team_lookup["team_name"],
                "locked": True,
            }
            if save_settings(full_settings):
                config.refresh_league_settings()
                st.session_state.pop("classic_working_leagues", None)
                st.session_state.pop("classic_lookup_result", None)
                st.session_state.pop("classic_team_lookup", None)
                st.session_state["classic_unlocked"] = False
                st.success("Classic leagues/team saved and locked!")
                st.rerun()
            else:
                st.error("Failed to save settings.")
