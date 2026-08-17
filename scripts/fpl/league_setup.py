# scripts/fpl/league_setup.py
#
# Streamlit "League Setup" admin page: set and lock the Draft/Classic
# league and team IDs used throughout the app. Values are validated against
# the live FPL APIs (resolving league/team names) before being saved to
# league_settings.json (gitignored, local-only — see scripts/common/league_config.py).

import os
import re

import streamlit as st
import streamlit.components.v1 as components

import config
from scripts.common.league_config import load_settings, save_settings, DEFAULT_SETTINGS
from scripts.common.fpl_draft_api import is_draft_league_reachable, get_league_entries
from scripts.common.fpl_classic_api import (
    get_classic_or_h2h_league_standings,
    get_entry_details,
    is_classic_league_reachable,
)

_SEASON_RE = re.compile(r"^\d{4}/\d{2}$")


# =================================================================
# Scroll-position helpers
# =================================================================
# A plain st.rerun() snaps the browser back to the top of the page, which is
# jarring when you're deep in a column (Classic history, several fields down)
# and just clicked something. _rerun_scrolled_to() remembers which column you
# were in and _apply_pending_scroll() scrolls back there after the rerun.

def _rerun_scrolled_to(anchor_id: str):
    st.session_state["_lsu_scroll_to"] = anchor_id
    st.rerun()


def _anchor(anchor_id: str):
    st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)


def _apply_pending_scroll():
    target = st.session_state.pop("_lsu_scroll_to", None)
    if not target:
        return
    # components.html renders in its own iframe, so the script must reach
    # into window.parent to find the anchor in the actual page. A short
    # retry loop covers the case where this component mounts slightly before
    # the rest of the rerun's DOM has painted.
    components.html(
        f"""
        <script>
        (function() {{
            const tryScroll = (attemptsLeft) => {{
                const doc = window.parent.document;
                const el = doc.getElementById("{target}");
                if (el) {{
                    el.scrollIntoView({{block: "start"}});
                }} else if (attemptsLeft > 0) {{
                    setTimeout(() => tryScroll(attemptsLeft - 1), 50);
                }}
            }};
            tryScroll(10);
        }})();
        </script>
        """,
        height=0,
    )


def show_league_setup_page():
    st.title("League Setup")
    st.caption(
        "Set your Draft and Classic league/team IDs here instead of editing `.env` by hand. "
        "Values are validated against the live FPL APIs, then locked to prevent accidental changes. "
        "Saved to `league_settings.json` on this machine (not committed to git)."
    )

    settings = load_settings()
    _show_season_nudge_banner(settings)

    st.divider()

    col_draft, col_classic = st.columns(2)
    with col_draft:
        _anchor("col-draft")
        _show_draft_section(settings)
        st.divider()
        _show_draft_history_section(settings)
    with col_classic:
        _anchor("col-classic")
        _show_classic_section(settings)
        st.divider()
        _show_classic_history_section(settings)

    _apply_pending_scroll()


def _show_season_nudge_banner(settings: dict):
    """If a locked section hasn't been re-confirmed since the PL season last
    rolled over, nudge the user to double-check it — a proactive warning that
    fires earlier than reachability alone (an old ID can keep resolving right
    up until a commissioner actually recreates the league)."""
    current_season = config.display_pl_season_label()
    stale_sections = []

    draft = settings.get("draft", {})
    if draft.get("locked") and draft.get("last_confirmed_season") not in (None, current_season):
        stale_sections.append("Draft")

    classic = settings.get("classic", {})
    if classic.get("locked") and classic.get("last_confirmed_season") not in (None, current_season):
        stale_sections.append("Classic")

    if stale_sections:
        st.info(
            f"📅 A new season may have started since you last confirmed your "
            f"{' and '.join(stale_sections)} league IDs (current season: {current_season}). "
            "Please double-check they're still correct below."
        )


# =================================================================
# Draft
# =================================================================

def _show_draft_section(settings: dict):
    st.header("📋 Draft")
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

    reachable = is_draft_league_reachable(league_id) if league_id else True
    if not reachable:
        season_label = draft.get("last_confirmed_season") or config.display_pl_season_label()
        already_archived = any(
            h.get("season") == season_label and h.get("league_id") == league_id
            for h in draft.get("history", [])
        )
        if not already_archived:
            _upsert_history_entry(season_label, league_id, team_id, team_name)
        st.warning(
            f"⚠️ **League ID {league_id} no longer resolves** — archived under `{season_label}` "
            "in Draft League History below. Draft leagues don't carry over between seasons; "
            "create this season's league at "
            "[draft.premierleague.com](https://draft.premierleague.com), then unlock to enter "
            "its ID."
        )
    else:
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
                _rerun_scrolled_to("col-draft")
        with col2:
            if st.button("Cancel", key="draft_unlock_cancel"):
                st.session_state["draft_unlock_pending"] = False
                _rerun_scrolled_to("col-draft")
    else:
        if st.button("Unlock to Edit", key="draft_unlock_start"):
            st.session_state["draft_unlock_pending"] = True
            _rerun_scrolled_to("col-draft")


def _archive_replaced_draft_league(old_draft: dict, new_league_id, season_label: str):
    """Snapshot whatever league ID was actually in effect before this save —
    whether it was previously locked in JSON, or (the common first-time case)
    only ever resolved via FPL_DRAFT_LEAGUE_ID in .env — if it's being
    replaced by a different one. Without this, the very first League Setup
    save silently drops the old ID with no record it ever existed."""
    previously_effective_id = config.FPL_DRAFT_LEAGUE_ID
    if not previously_effective_id or previously_effective_id == new_league_id:
        return
    history = old_draft.setdefault("history", [])
    already_archived = any(
        h.get("season") == season_label and h.get("league_id") == previously_effective_id
        for h in history
    )
    if not already_archived:
        history.append({
            "season": season_label, "league_id": previously_effective_id,
            "team_id": old_draft.get("team_id"), "team_name": old_draft.get("team_name"),
            "manual_stats": None,
        })


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
            old_draft = full_settings["draft"]
            old_season_label = old_draft.get("last_confirmed_season") or config.display_pl_season_label()
            new_league_id = st.session_state["draft_lookup_league_id"]
            _archive_replaced_draft_league(old_draft, new_league_id, old_season_label)
            # .update() rather than replace — preserves history/commish_seasons,
            # which a wholesale dict replacement would silently wipe out.
            full_settings["draft"].update({
                "league_id": new_league_id,
                "team_id": int(chosen_id),
                "team_name": chosen_name,
                "locked": True,
                "last_confirmed_season": config.display_pl_season_label(),
            })
            if save_settings(full_settings):
                config.refresh_league_settings()
                st.session_state.pop("draft_lookup_entries", None)
                st.session_state.pop("draft_lookup_league_id", None)
                st.session_state["draft_unlocked"] = False
                st.success("Draft league/team saved and locked!")
                _rerun_scrolled_to("col-draft")
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
        "history_new_season",
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
                        _rerun_scrolled_to("col-draft")
                else:
                    if st.button("Remove", key=f"history_remove_{entry['season']}"):
                        _remove_history_entry(entry["season"])
                        _rerun_scrolled_to("col-draft")

    with st.expander("➕ Add a past season", expanded=False):
        st.caption(
            "League IDs are already gone once a season ends, so live lookup can't recover them — "
            "enter the final standings directly, e.g. from a saved Season Wrapped PDF export. This "
            "is a backstop for seasons that predate the automatic archiving above; going forward, "
            "your league is snapshotted automatically once a season completes."
        )
        season_input = st.text_input("Season (e.g. 2024/25)", key="history_new_season")
        season_clean = season_input.strip()
        season_valid = bool(_SEASON_RE.match(season_clean)) if season_clean else False
        if season_clean and not season_valid:
            st.error("Season must look like `2024/25`.")

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
                _rerun_scrolled_to("col-draft")
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
                _rerun_scrolled_to("col-draft")


# =================================================================
# Classic
# =================================================================

def _upsert_classic_history_entry(
    season: str, league_id, league_name: str, manual_stats: dict = None, pct_finish: float = None,
):
    """Keyed on (season, league_id) rather than season alone — Classic
    supports multiple concurrent leagues, so more than one can belong to the
    same season.

    pct_finish is a season-level stat (not per-league — it's your percentile
    against the whole FPL player pool, e.g. "top 8%"), so it's stored
    separately in classic.season_notes rather than on this per-league entry.
    Optional here purely so the single "Add to History" button can persist
    both in one write; pass None to leave any existing note untouched.
    """
    full_settings = load_settings()
    history = full_settings["classic"].setdefault("league_history", [])
    idx = next(
        (i for i, h in enumerate(history) if h.get("season") == season and h.get("league_id") == league_id),
        None,
    )
    entry = {
        "season": season, "league_id": league_id, "league_name": league_name,
        "manual_stats": manual_stats,
    }
    if idx is not None:
        history[idx] = entry
    else:
        history.append(entry)
    full_settings["classic"]["league_history"] = sorted(
        history, key=lambda h: (h["season"], h.get("league_id") or 0)
    )
    if pct_finish is not None:
        full_settings["classic"].setdefault("season_notes", {})[season] = {"pct_finish": pct_finish}
    save_settings(full_settings)
    config.refresh_league_settings()


def _remove_classic_history_entry(season: str, league_id):
    full_settings = load_settings()
    history = full_settings["classic"].setdefault("league_history", [])
    full_settings["classic"]["league_history"] = [
        h for h in history if not (h.get("season") == season and h.get("league_id") == league_id)
    ]
    save_settings(full_settings)
    config.refresh_league_settings()


def _show_classic_section(settings: dict):
    st.header("🏆 Classic")
    classic = settings.get("classic", DEFAULT_SETTINGS["classic"])

    is_locked = classic.get("locked") and not st.session_state.get("classic_unlocked")
    if is_locked:
        _show_classic_locked_view(classic)
    else:
        _show_classic_edit_form(classic)


def _show_classic_locked_view(classic: dict):
    leagues = classic.get("leagues", [])
    team_id = classic.get("team_id")
    team_name = classic.get("team_name") or "Unknown"
    season_label = classic.get("last_confirmed_season") or config.display_pl_season_label()

    ok_leagues, stale_leagues = [], []
    for l in leagues:
        if is_classic_league_reachable(l.get("id")):
            ok_leagues.append(l)
        else:
            stale_leagues.append(l)

    if stale_leagues:
        history = classic.get("league_history", [])
        for l in stale_leagues:
            already_archived = any(
                h.get("season") == season_label and h.get("league_id") == l.get("id") for h in history
            )
            if not already_archived:
                _upsert_classic_history_entry(season_label, l.get("id"), l.get("name"))
        stale_str = ", ".join(f"{l.get('name') or 'Unnamed'} (ID {l.get('id')})" for l in stale_leagues)
        st.warning(
            f"⚠️ **No longer resolves:** {stale_str} — archived under `{season_label}` in "
            "Classic League History below. Private mini-leagues are sometimes recreated with a "
            "new ID each season; find the current one on "
            "[fantasy.premierleague.com](https://fantasy.premierleague.com), then unlock to "
            "update it."
        )

    if ok_leagues:
        # Match Draft's single-line "League ID: X" format when there's only one
        # league — the "Leagues:" bulleted list is only useful once there's
        # actually more than one to distinguish between.
        if len(ok_leagues) == 1:
            leagues_display = f"**League ID:** {ok_leagues[0].get('id')}"
        else:
            leagues_str = "  \n".join(f"- {l.get('name') or 'Unnamed'} (ID {l.get('id')})" for l in ok_leagues)
            leagues_display = f"**Leagues:**  \n{leagues_str}"
        st.success(
            f"{leagues_display}  \n"
            f"**Your Team:** {team_name} (ID {team_id})"
        )
    elif not stale_leagues:
        st.success(f"_No leagues set_  \n**Your Team:** {team_name} (ID {team_id})")

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
                _rerun_scrolled_to("col-classic")
        with col2:
            if st.button("Cancel", key="classic_unlock_cancel"):
                st.session_state["classic_unlock_pending"] = False
                _rerun_scrolled_to("col-classic")
    else:
        if st.button("Unlock to Edit", key="classic_unlock_start"):
            st.session_state["classic_unlock_pending"] = True
            _rerun_scrolled_to("col-classic")


def _archive_replaced_classic_leagues(old_classic: dict, new_league_ids: set, season_label: str):
    """Snapshot any league that was actually in effect before this save —
    whether it was previously locked in JSON, or (the common first-time case)
    only ever resolved via FPL_CLASSIC_LEAGUE_IDS in .env — if it's being
    dropped from the new list. Without this, the very first League Setup
    save silently drops the old league with no record it ever existed."""
    history = old_classic.setdefault("league_history", [])
    for league in (config.FPL_CLASSIC_LEAGUE_IDS or []):
        league_id = league.get("id") if isinstance(league, dict) else None
        if not league_id or league_id in new_league_ids:
            continue
        already_archived = any(
            h.get("season") == season_label and h.get("league_id") == league_id for h in history
        )
        if not already_archived:
            history.append({
                "season": season_label, "league_id": league_id,
                "league_name": league.get("name"), "manual_stats": None,
            })


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
                    _rerun_scrolled_to("col-classic")
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
                if any(l["id"] == league_id for l in working_leagues):
                    st.info(f'"{name}" is already in your list.')
                else:
                    working_leagues.append({"id": league_id, "name": name})
                    st.success(f'Added "{name}" to your list.')
                    _rerun_scrolled_to("col-classic")

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
            old_classic = full_settings["classic"]
            old_season_label = old_classic.get("last_confirmed_season") or config.display_pl_season_label()
            new_league_ids = {l["id"] for l in working_leagues}
            _archive_replaced_classic_leagues(old_classic, new_league_ids, old_season_label)
            # .update() rather than replace — preserves league_history, which a
            # wholesale dict replacement would silently wipe out.
            full_settings["classic"].update({
                "leagues": working_leagues,
                "team_id": team_lookup["id"],
                "team_name": team_lookup["team_name"],
                "locked": True,
                "last_confirmed_season": config.display_pl_season_label(),
            })
            if save_settings(full_settings):
                config.refresh_league_settings()
                st.session_state.pop("classic_working_leagues", None)
                st.session_state.pop("classic_team_lookup", None)
                st.session_state["classic_unlocked"] = False
                st.success("Classic leagues/team saved and locked!")
                _rerun_scrolled_to("col-classic")
            else:
                st.error("Failed to save settings.")


# =================================================================
# Classic — cross-season history
# =================================================================

def _clear_classic_history_lookup_state():
    for key in (
        "classic_history_new_season",
        "classic_history_manual_stats_league_name", "classic_history_manual_stats_league_id",
        "classic_history_manual_stats_rank", "classic_history_manual_stats_points",
        "classic_history_manual_overwrite_pending",
    ):
        st.session_state.pop(key, None)


def _show_classic_history_section(settings: dict):
    st.subheader("📜 Classic League History")
    st.caption(
        "Private Classic/H2H mini-leagues are sometimes recreated with a new ID each season — "
        "this is normally archived automatically once a season concludes, or flagged above if a "
        "league has already gone stale. Use this section to add older seasons by hand, or to fix "
        "up an auto-archived entry."
    )

    classic = settings.get("classic", DEFAULT_SETTINGS["classic"])
    json_history = list(classic.get("league_history", []))
    archived_ids = {h.get("league_id") for h in json_history}
    active_ids = {l.get("id") for l in classic.get("leagues", [])}

    # A league that only ever lived in FPL_CLASSIC_LEAGUE_IDS (never locked
    # via this page) has no season tag of its own — surface it here the same
    # way Draft's history section surfaces .env-only entries, rather than
    # silently losing it the moment a different league gets locked over it.
    env_leagues = config._parse_classic_leagues(os.getenv("FPL_CLASSIC_LEAGUE_IDS", ""))
    default_season = classic.get("last_confirmed_season") or config.display_pl_season_label()
    env_only = [
        {"season": default_season, "league_id": l["id"], "league_name": l.get("name"), "_env_only": True}
        for l in env_leagues if l["id"] not in archived_ids and l["id"] not in active_ids
    ]
    history = sorted(json_history + env_only, key=lambda h: (h["season"], h.get("league_id") or 0))

    if not history:
        st.caption("No past seasons saved yet.")
    else:
        for entry in history:
            col1, col2 = st.columns([4, 1])
            with col1:
                league_label = entry.get("league_name") or "Unnamed league"
                manual_stats = entry.get("manual_stats")
                if manual_stats:
                    line = (
                        f"**{entry['season']}** — {league_label} — "
                        f"Rank #{manual_stats.get('rank', '?')}, {manual_stats.get('total_points', 0)} pts "
                        f"— *manually entered*"
                    )
                else:
                    line = f"**{entry['season']}** — {league_label} (League ID {entry['league_id']})"
                    if entry.get("_env_only"):
                        line += "  \n*From `.env` — not yet saved here*"
                st.markdown(line)
            with col2:
                if entry.get("_env_only"):
                    if st.button("Save here", key=f"classic_history_save_env_{entry['league_id']}"):
                        _upsert_classic_history_entry(entry["season"], entry["league_id"], entry["league_name"])
                        _rerun_scrolled_to("col-classic")
                else:
                    if st.button(
                        "Remove", key=f"classic_history_remove_{entry['season']}_{entry.get('league_id')}"
                    ):
                        _remove_classic_history_entry(entry["season"], entry.get("league_id"))
                        _rerun_scrolled_to("col-classic")

    with st.expander("➕ Add a past season", expanded=False):
        st.caption(
            "League IDs are already gone once a season ends, so live lookup can't recover them — "
            "enter the league name and final standing directly. This is a backstop for seasons "
            "that predate the automatic archiving above; going forward, your leagues are "
            "snapshotted automatically once a season completes."
        )
        season_input = st.text_input("Season (e.g. 2024/25)", key="classic_history_new_season")
        season_clean = season_input.strip()
        season_valid = bool(_SEASON_RE.match(season_clean)) if season_clean else False
        if season_clean and not season_valid:
            st.error("Season must look like `2024/25`.")

        manual_league_name = st.text_input("League name", key="classic_history_manual_stats_league_name")
        manual_league_id_input = st.text_input(
            "League ID (if known, optional)", key="classic_history_manual_stats_league_id"
        )
        col_r, col_p = st.columns(2)
        with col_r:
            manual_rank = st.number_input(
                "Final Rank", min_value=1, step=1, key="classic_history_manual_stats_rank"
            )
        with col_p:
            manual_points = st.number_input(
                "Total Points", min_value=0, step=1, key="classic_history_manual_stats_points"
            )
        # No manual "% Finish" field here — Team Analysis's Season History table
        # already gets that live from FPL's entry-history endpoint (rank_percentage),
        # same as Points/Rank. Only League Placements has no live source.

        manual_league_id_clean = manual_league_id_input.strip()
        manual_league_id = int(manual_league_id_clean) if manual_league_id_clean.isdigit() else None
        is_duplicate = season_valid and any(
            h.get("season") == season_clean and h.get("league_id") == manual_league_id for h in history
        )
        add_disabled = not season_valid or not manual_league_name.strip()

        if is_duplicate and not st.session_state.get("classic_history_manual_overwrite_pending"):
            st.warning(f"`{season_clean}` is already saved for that league ID — adding again will overwrite it.")
            if st.button(
                "Overwrite existing entry", key="classic_history_manual_overwrite_btn", disabled=add_disabled
            ):
                st.session_state["classic_history_manual_overwrite_pending"] = True
                _rerun_scrolled_to("col-classic")
        else:
            if st.button(
                "Add to History", key="classic_history_add_manual_btn", type="primary", disabled=add_disabled
            ):
                _upsert_classic_history_entry(
                    season_clean, manual_league_id, manual_league_name.strip(),
                    manual_stats={"rank": int(manual_rank), "total_points": int(manual_points)},
                )
                _clear_classic_history_lookup_state()
                st.success(f"Saved {season_clean}.")
                _rerun_scrolled_to("col-classic")
