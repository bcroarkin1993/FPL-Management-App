# scripts/draft/commish_mode.py
#
# "Commish Mode" — Draft league commissioner tools: track buy-in dues
# collection and compute prize payouts. Tied to a season label
# (league_settings.json's draft.commish_seasons) rather than the live
# league ID, so records survive Draft league rollover — Draft league IDs
# and their entries get reissued to an unrelated league every season (see
# Draft League History on the League Setup page for the same fix applied
# to cross-season standings).

from datetime import date

import streamlit as st

import config
from scripts.common.league_config import load_settings, save_settings, DEFAULT_SETTINGS
from scripts.common.fpl_draft_api import get_league_entries

_NEW_SEASON_OPTION = "+ Start a new season"
_PLACE_LABELS = {"1": "🥇 1st", "2": "🥈 2nd", "3": "🥉 3rd"}


def _money_md(amount) -> str:
    """Format a dollar amount for markdown-rendered text (st.caption/st.subheader/
    st.markdown). Streamlit's markdown renderer treats a pair of '$' as LaTeX math
    delimiters, so a string with multiple dollar amounts (e.g. a caption listing
    pool/1st/2nd/3rd) gets mangled unless each '$' is escaped. st.metric() doesn't
    markdown-parse its value, so plain f"${amount:,.0f}" is fine there instead."""
    return f"\\${amount:,.0f}"


def _current_pl_season_guess() -> str:
    """Best-effort current Premier League season as 'YYYY/YY' (Aug 1 rollover)."""
    today = date.today()
    start_year = today.year if today.month >= 8 else today.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def _is_paid(due: dict, buy_in: float) -> bool:
    """Whether a member's dues are marked paid. Reads the "paid" checkbox flag;
    falls back to the legacy amount_paid >= buy_in comparison for dues records
    saved before the switch to a binary paid/unpaid checkbox (nobody actually
    paid in partial installments, so this is a safe one-way migration)."""
    if "paid" in due:
        return bool(due["paid"])
    return due.get("amount_paid", 0) >= buy_in


def show_commish_mode_page():
    st.title("Commish Mode 💰")
    st.caption(
        "Track league dues and compute prize payouts for your Draft league. "
        "Saved to `league_settings.json` on this machine (not committed to git)."
    )

    if not config.FPL_DRAFT_LEAGUE_ID:
        st.warning("Set up your Draft league on the **🆔 League Setup** page first.")
        return

    settings = load_settings()
    draft = settings.get("draft", DEFAULT_SETTINGS["draft"])
    commish_seasons = draft.get("commish_seasons", {})
    existing_seasons = sorted(commish_seasons.keys())

    options = existing_seasons + [_NEW_SEASON_OPTION]
    # A just-saved season is requested via _commish_pending_season (set before
    # st.rerun()) rather than writing to the selectbox's own widget key directly —
    # Streamlit forbids mutating a widget's session_state after it's instantiated
    # in the same run, so the pending value is only consumed here, before creation.
    pending = st.session_state.pop("_commish_pending_season", None)
    if pending and pending in options:
        default_idx = options.index(pending)
    else:
        default_idx = len(existing_seasons) - 1 if existing_seasons else 0
    choice = st.selectbox("Season", options=options, index=default_idx, key="commish_season_select")

    st.divider()

    if choice == _NEW_SEASON_OPTION:
        _show_season_setup_form(season_label=None, existing=None)
    else:
        season_data = commish_seasons[choice]
        if st.session_state.get(f"commish_unlock_{choice}"):
            _show_season_setup_form(season_label=choice, existing=season_data)
        else:
            _show_season_dashboard(choice, season_data)


def _show_season_setup_form(season_label, existing):
    is_new = season_label is None
    st.subheader("Start a New Season" if is_new else f"Edit {season_label}")

    default_season = season_label or _current_pl_season_guess()
    season_input = st.text_input(
        "Season (e.g. 2026/27)", value=default_season,
        key="commish_setup_season", disabled=not is_new,
    )
    buy_in = st.number_input(
        "Buy-in per person ($)", min_value=0.0, step=5.0,
        value=float((existing or {}).get("buy_in", 75)), key="commish_setup_buyin",
    )

    st.markdown("**Payout split** (must total 100%)")
    default_pct = (existing or {}).get("payout_pct", {"1": 60, "2": 30, "3": 10})
    col1, col2, col3 = st.columns(3)
    with col1:
        pct1 = st.number_input("1st %", min_value=0, max_value=100, step=5,
                                value=int(default_pct.get("1", 60)), key="commish_pct_1")
    with col2:
        pct2 = st.number_input("2nd %", min_value=0, max_value=100, step=5,
                                value=int(default_pct.get("2", 30)), key="commish_pct_2")
    with col3:
        pct3 = st.number_input("3rd %", min_value=0, max_value=100, step=5,
                                value=int(default_pct.get("3", 10)), key="commish_pct_3")

    total_pct = pct1 + pct2 + pct3
    if total_pct != 100:
        st.error(f"Splits must total 100% (currently {total_pct}%).")

    try:
        entries_preview = get_league_entries(config.FPL_DRAFT_LEAGUE_ID)
    except Exception:
        entries_preview = None
    if entries_preview:
        n = len(entries_preview)
        pool = buy_in * n
        st.caption(
            f"Pool preview at {n} members: {_money_md(pool)} total — "
            f"1st {_money_md(pool * pct1 / 100)}, 2nd {_money_md(pool * pct2 / 100)}, "
            f"3rd {_money_md(pool * pct3 / 100)}"
        )

    season_clean = season_input.strip()
    disabled = total_pct != 100 or not season_clean
    button_label = "Save & Lock" if is_new else "Save Changes"
    if st.button(button_label, key="commish_save_setup_btn", type="primary", disabled=disabled):
        full_settings = load_settings()
        seasons = full_settings["draft"].setdefault("commish_seasons", {})
        dues = (existing or {}).get("dues")
        if dues is None:
            with st.spinner("Loading league members..."):
                try:
                    entries = get_league_entries(config.FPL_DRAFT_LEAGUE_ID) or {}
                except Exception:
                    entries = {}
            dues = {name: {"paid": False, "notes": ""} for name in entries.values()}
        seasons[season_clean] = {
            "buy_in": buy_in,
            "payout_pct": {"1": pct1, "2": pct2, "3": pct3},
            "locked": True,
            "dues": dues,
        }
        save_settings(full_settings)
        st.session_state.pop(f"commish_unlock_{season_label}", None)
        st.session_state["_commish_pending_season"] = season_clean
        st.success(f"Saved {season_clean}.")
        st.rerun()


def _show_season_dashboard(season_label: str, season_data: dict):
    buy_in = season_data.get("buy_in", 0)
    payout_pct = season_data.get("payout_pct", {})
    dues = season_data.get("dues", {})
    n_members = len(dues)
    pool = buy_in * n_members
    paid_count = sum(1 for d in dues.values() if _is_paid(d, buy_in))
    collected = buy_in * paid_count
    outstanding = [name for name, d in dues.items() if not _is_paid(d, buy_in)]

    st.subheader(f"{season_label} — {_money_md(buy_in)} buy-in, {n_members} members")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pool", f"${pool:,.0f}")
    col2.metric("Collected", f"${collected:,.0f} / ${pool:,.0f}")
    col3.metric("Outstanding", f"{len(outstanding)} of {n_members}")

    st.markdown("**Payout**")
    payout_cols = st.columns(3)
    for i, place in enumerate(["1", "2", "3"]):
        pct = payout_pct.get(place, 0)
        amount = pool * pct / 100
        payout_cols[i].metric(f"{_PLACE_LABELS[place]} ({pct}%)", f"${amount:,.0f}")

    if st.button("Unlock to Edit buy-in/split", key=f"commish_unlock_btn_{season_label}"):
        st.session_state[f"commish_unlock_{season_label}"] = True
        st.rerun()

    st.divider()
    st.markdown("**Dues Tracker**")
    if not dues:
        st.caption("No members on record for this season.")
        return
    if outstanding:
        st.caption(f"Outstanding: {', '.join(sorted(outstanding))}")
    else:
        st.caption("Everyone's paid in full! 🎉")

    with st.form(key=f"commish_dues_form_{season_label}"):
        updated = {}
        for name in sorted(dues.keys()):
            d = dues[name]
            c1, c2, c3 = st.columns([2, 1, 2])
            c1.write(name)
            paid = c2.checkbox(
                "Paid", value=_is_paid(d, buy_in),
                key=f"commish_due_paid_{season_label}_{name}", label_visibility="collapsed",
            )
            notes = c3.text_input(
                "Notes", value=d.get("notes", ""),
                key=f"commish_due_notes_{season_label}_{name}", label_visibility="collapsed",
                placeholder="Notes (optional)",
            )
            updated[name] = {"paid": paid, "notes": notes}

        if st.form_submit_button("Save Dues", type="primary"):
            full_settings = load_settings()
            full_settings["draft"]["commish_seasons"][season_label]["dues"] = updated
            save_settings(full_settings)
            config.refresh_league_settings()
            st.success("Dues saved.")
            st.rerun()
