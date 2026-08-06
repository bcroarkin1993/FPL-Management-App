# scripts/fpl/league_setup.py
#
# Streamlit "League Setup" admin page: set and lock the Draft/Classic
# league and team IDs used throughout the app. Values are validated against
# the live FPL APIs (resolving league/team names) before being saved to
# league_settings.json (gitignored, local-only — see scripts/common/league_config.py).

import streamlit as st

import config
from scripts.common.league_config import load_settings, save_settings, DEFAULT_SETTINGS
from scripts.common.fpl_draft_api import is_draft_league_reachable, get_league_entries
from scripts.common.fpl_classic_api import get_classic_league_standings, get_entry_details


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
                result = get_classic_league_standings(league_id)
            if not result or not result.get("league"):
                st.error("Could not resolve that Classic league ID. Please double-check it.")
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
