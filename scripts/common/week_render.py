"""Rendering for a week checklist, shared by both formats.

Everything here reads only :class:`week_check.WeekItem`, so it is
format-agnostic by construction -- which is the point. The Draft strip and the
Classic strip are the same card, the same alert rows, the same navigation
buttons and the same per-row detail card; a second copy of them would drift
exactly as the two blends, the two lineup parsers and the two claim pipelines
in this codebase have.

What is *not* shared is what each format puts around them. Classic has a
confirm-XI button (it cannot see a pre-deadline lineup) and a connect-account
button; Draft has a check-for-claims button and neither of the others. Those
arrive as `leading` / `trailing` entries rather than being special-cased here,
so this module needs no knowledge of either format.

The view object is duck-typed deliberately: it needs `items` and `flagged`,
and nothing else. Both formats' view dataclasses satisfy that without
inheriting from anything.
"""

from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Sequence

import pandas as pd
import streamlit as st

from scripts.common import cards, week_check
from scripts.common.text_helpers import (compact_html, format_deadline_stamp,
                                         format_time_until)

#: Status -> (icon, colour). Unknown is grey rather than green: a question we
#: could not answer must not read as one answered "fine".
STATUS_STYLE = {
    week_check.STATUS_OK: ("✅", cards.ACCENT),
    week_check.STATUS_UNKNOWN: ("❔", cards.MUTED),
    week_check.STATUS_WARN: ("⚠️", cards.WARN),
    week_check.STATUS_ACT: ("❗", cards.BAD),
}

#: What each row is called in the strip. Deliberately shorter than the row's
#: own verdict: the bar is scanned, the rows are read. Covers both formats --
#: the keys do not collide, and a key with no entry falls back to its title.
PILL_TEXT = {
    # shared
    "deadline": "Deadline",
    "opponent": "This week",
    "squad": "Squad",
    "availability": "Fitness",
    "fixtures": "Fixtures",
    "lineup": "Your XI",
    # classic
    "transfers": "Transfers",
    "chips": "Chips",
    "prices": "Prices",
    # draft
    "waivers": "Waivers",
    "claim": "Waiver claim",
    "trades": "Trades",
}

#: Classic's squad row means something different from Draft's -- there it is a
#: staleness disclosure, here a fact -- so the Classic page overrides it.
PILL_TEXT_CLASSIC = dict(PILL_TEXT, squad="Squad not current")


class ActionButton(NamedTuple):
    """A button a format wants beside the navigation ones.

    `on_click` runs in the page's script run; return True to rerun. Kept as
    data rather than a callback into this module so that `week_render` needs
    no knowledge of confirmations, credentials or claim searches.
    """

    label: str
    key: str
    help: str = ""
    on_click: Optional[Callable[[], Any]] = None


def alert_row(item, pill_text=None) -> str:
    """One line of the alert list: status, what it is, and the number.

    This replaced a pill strip *and* a "Needs action" tab, which said the same
    thing twice. One list, each row carrying its own verdict, so nothing is
    repeated and nothing needs opening to be read.
    """
    icon, colour = STATUS_STYLE.get(item.status, ("•", "#9ca3af"))
    label = (pill_text or PILL_TEXT).get(item.key, item.title)
    return compact_html(f'''
        <div style="display:flex;align-items:baseline;gap:8px;padding:5px 0;
                    border-top:1px solid #2a2a3e;">
          <span style="font-size:0.9em;">{icon}</span>
          <span style="color:{colour};font-size:0.72em;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.5px;
                       min-width:78px;">{label}</span>
          <span style="color:#e0e0e0;font-size:0.9em;line-height:1.35;">
            {item.verdict}</span>
        </div>
    ''')


def deadline_stamp(items) -> str:
    """"GW11 · Fri Nov 21, 6:00 AM ET · in 3 days", off the deadline row.

    A bare timestamp does not say which deadline it is: two gameweeks a week
    apart render identically without the date, which is the failure
    `format_deadline_stamp` exists for.
    """
    row = next((i for i in items if i.key == "deadline"), None)
    when = (row.evidence or {}).get("deadline") if row else None
    if not when:
        return ""
    try:
        moment = pd.Timestamp(when).to_pydatetime()
    except (ValueError, TypeError):
        return ""
    return "GW%s &middot; %s &middot; %s" % (
        (row.evidence or {}).get("gameweek", ""),
        format_deadline_stamp(moment), format_time_until(moment))


def week_card_html(view, *, title: str = "Your week", caveat: str = "",
                   all_clear: str = "Nothing needs doing before the deadline.",
                   pill_text=None) -> str:
    """The strip as a card, for a Home page to drop into its top band.

    Returns markup rather than rendering, because the band is one CSS grid:
    grid items stretch to the tallest row member by default, which is what
    makes the cards level however their text wraps. Estimating pixel heights
    could not survive a verdict wrapping onto a second line, and this codebase
    tried that first.
    """
    rows = "".join(alert_row(i, pill_text) for i in view.flagged) or (
        '<div style="padding:8px 0 2px;border-top:1px solid %s;color:%s;'
        'font-size:0.92em;font-weight:600;">✅ %s</div>'
        % (cards.RULE, cards.ACCENT, all_clear))
    header = (
        '<div style="display:flex;justify-content:space-between;'
        'align-items:baseline;margin-bottom:4px;">'
        '<span style="color:%s;font-size:0.8em;font-weight:700;'
        'text-transform:uppercase;letter-spacing:1px;">%s</span>'
        '<span style="color:%s;font-size:0.78em;">%s</span></div>'
        % (cards.ACCENT, title, cards.MUTED, deadline_stamp(view.items)))
    return cards.card(header + rows + (cards.note(caveat) if caveat else ""))


def render_actions(view, *, key_prefix: str, leading: Sequence[ActionButton] = (),
                   trailing: Sequence[ActionButton] = ()) -> None:
    """One button per flagged row, plus whatever the format adds either side.

    Rendered under the band rather than inside the card, because a Streamlit
    widget cannot live inside one block of markup.

    `main.py` applies `_pending_nav_section` / `_pending_nav_page` before its
    sidebar radios are instantiated, so a button here lands on the page that
    owns the row -- Home's own "Go to League Setup" link is the precedent.
    """
    nav = [i for i in view.flagged if i.page]
    total = len(nav) + len(leading) + len(trailing)
    if not total:
        return

    cols = st.columns(total)
    slot = 0
    for button in leading:
        with cols[slot]:
            if st.button(button.label, key="%s_%s" % (key_prefix, button.key),
                         use_container_width=True, help=button.help or None):
                if button.on_click and button.on_click() is not False:
                    st.rerun()
        slot += 1
    for item in nav:
        with cols[slot]:
            if st.button(item.page.strip(),
                         key="%s_open_%s" % (key_prefix, item.key),
                         use_container_width=True,
                         help="Go to %s" % item.page.strip()):
                st.session_state["_pending_nav_section"] = item.section
                st.session_state["_pending_nav_page"] = item.page
                st.rerun()
        slot += 1
    for button in trailing:
        with cols[slot]:
            if st.button(button.label, key="%s_%s" % (key_prefix, button.key),
                         use_container_width=True, help=button.help or None):
                if button.on_click and button.on_click() is not False:
                    st.rerun()
        slot += 1


def item_card(item) -> str:
    """One checklist row, for the full-checklist tab.

    The accent is the status colour, which `cards.card` draws as a left edge.
    """
    icon, colour = STATUS_STYLE.get(item.status, ("•", cards.MUTED))
    body = (
        f'<div style="display:flex;align-items:baseline;gap:9px;">'
        f'<span>{icon}</span>'
        f'<span style="color:{cards.MUTED};font-size:0.74em;'
        f'text-transform:uppercase;letter-spacing:0.6px;min-width:104px;">'
        f'{item.title}</span>'
        f'<span style="font-size:0.95em;">{item.verdict}</span></div>')
    if item.detail:
        body += cards.note(item.detail, cards.MUTED)
    return cards.card(body, accent=colour, padding="10px 13px")


def render_checks(items: Iterable) -> None:
    """Every row, in order, as its own card."""
    for item in items or []:
        st.markdown(item_card(item), unsafe_allow_html=True)
