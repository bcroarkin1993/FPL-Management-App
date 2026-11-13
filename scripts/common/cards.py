"""The dark card, in one place.

Four different constructions of the same card had grown up across the app --
`home._league_card`, `home._record_card`, the My Week strip's own shell, and
`st.metric`, which the rest of the app had already stopped using. Nothing lined
up by construction, so making two cards level needed a tuned pixel constant per
page, and the next card added would have needed another.

So: one shell. A row of cards lines up because `card_row` gives them one height
and the shell makes them the same shape, not because someone measured.

**Every card goes through `compact_html`.** A blank line ends an HTML block in
Markdown and what follows is parsed as fresh Markdown -- indented four spaces,
that is a *code block* -- so an optional fragment that renders to "" on a line
of its own dumps the card's closing `</div>` on screen as literal text. That is
data-dependent, which is why it keeps shipping: the card is fine for a player
with a badge and breaks for one without.
"""

from typing import Iterable, List, Optional

from scripts.common.text_helpers import compact_html

#: The house palette, as used by every card already in the app.
ACCENT = "#00ff87"
ACCENT_ALT = "#4ecca3"
WARN = "#ffa502"
BAD = "#ff4757"
TEXT = "#e0e0e0"
MUTED = "#9ca3af"
DIM = "#6b7280"
BORDER = "#333"
RULE = "#2a2a3e"
GRADIENT = "linear-gradient(135deg,#1a1a2e 0%,#16213e 100%)"


def card(body: str, min_height: int = 0, accent: Optional[str] = None,
         padding: str = "14px 16px") -> str:
    """One card around ``body``.

    A **flex column**, so `push()` can hold a footer line to the bottom of a
    card that has been stretched to match its neighbours -- otherwise a taller
    card leaves a gap under its heading rather than filling.

    ``min_height`` rather than ``height``: a card may always grow past the row's
    measure, which fails safe as a taller row instead of clipped text.
    """
    edge = f"border-left:4px solid {accent};" if accent else ""
    return compact_html(
        f'<div style="border:1px solid {BORDER};{edge}border-radius:12px;'
        f'padding:{padding};box-sizing:border-box;'
        f'min-height:{max(0, int(min_height or 0))}px;'
        f'display:flex;flex-direction:column;background:{GRADIENT};'
        f'color:{TEXT};">{body}</div>')


def card_row(bodies: Iterable[str], min_height: int = 0,
             accent: Optional[str] = None) -> List[str]:
    """The same card, several times, at one height -- i.e. a row that lines up."""
    return [card(body, min_height=min_height, accent=accent)
            for body in bodies]


def eyebrow(text: str) -> str:
    """The small upper-case label at the top of a card."""
    return (f'<div style="color:{MUTED};font-size:0.74em;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.9px;'
            f'margin-bottom:6px;">{text}</div>')


def headline(text: str, size: str = "1.12em") -> str:
    """A card's primary line -- a league name, a team name."""
    return (f'<div style="font-size:{size};font-weight:700;'
            f'line-height:1.25;">{text}</div>')


def badge(text: str, colour: str = ACCENT) -> str:
    """A pill. Used for a league's format and a chip's name."""
    return (f'<span style="background:{colour}22;color:{colour};'
            f'border:1px solid {colour}66;padding:2px 9px;border-radius:10px;'
            f'font-size:0.72em;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.5px;white-space:nowrap;">{text}</span>')


def stat(value: str, label: str, colour: str = TEXT) -> str:
    """One number and what it is, stacked. The replacement for `st.metric`."""
    return (f'<div style="min-width:0;">'
            f'<div style="color:{colour};font-size:1.18em;font-weight:700;'
            f'line-height:1.2;">{value}</div>'
            f'<div style="color:{DIM};font-size:0.7em;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-top:2px;">{label}</div></div>')


def stats(items: Iterable[str], gap: str = "14px") -> str:
    """Several `stat` blocks across a card, wrapping rather than overflowing."""
    return (f'<div style="display:flex;gap:{gap};flex-wrap:wrap;'
            f'margin-top:9px;">{"".join(items)}</div>')


def note(text: str, colour: Optional[str] = None) -> str:
    """A quiet line -- a caveat, a provenance, a hint."""
    return (f'<div style="color:{colour or DIM};font-size:0.76em;'
            f'margin-top:7px;line-height:1.35;">{text}</div>')


def push(body: str = "") -> str:
    """Hold ``body`` to the bottom of a stretched card.

    `margin-top:auto` inside the shell's flex column. Without it, a card given a
    row's height renders its content at the top and the rest as dead space --
    which is the look this module exists to remove.
    """
    return f'<div style="margin-top:auto;padding-top:9px;">{body}</div>'
