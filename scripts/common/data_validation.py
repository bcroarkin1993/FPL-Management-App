# scripts/common/data_validation.py
#
# Plausibility checks for FPL data and the numbers derived from it.
#
# These exist because the app's worst bugs have not been logic errors — they were
# upstream data quietly changing shape while every unit test stayed green:
#
#   * Rotowire published a "gameweeks 1-5" article whose Points column was a
#     cumulative 5-week total. Nothing crashed; every projection was just 5x too
#     big (a goalkeeper showing 18.6 for one gameweek).
#   * The Draft API returned a full 38-gameweek score grid before the season
#     started, so the historical-score std was 0.0 and a 1.1-point projection
#     edge rendered as an 85% win probability.
#
# Neither is detectable by asserting on mocked inputs. What they have in common is
# that the *output* was wildly outside any plausible norm, which is cheap to assert.
# So: pure, network-free predicates over real numbers, called from the live
# plausibility tests (tests/live/) and usable as runtime tripwires.
#
# No Streamlit, no requests — importable from anywhere, including GitHub Actions.

import math
from typing import List, NamedTuple, Optional, Sequence

import pandas as pd

__all__ = [
    "Issue",
    "check_single_gw_projections",
    "check_score_std",
    "check_win_probability",
    "check_projected_team_total",
    "check_source_scale_agreement",
    "check_team_strength",
    "format_issues",
    "raise_on_error",
]


class Issue(NamedTuple):
    """One failed plausibility check.

    severity is "error" for a value that cannot be correct, "warning" for one
    that is merely suspicious and may be a legitimate edge case.
    """

    check: str
    severity: str
    message: str
    hint: str = ""

    def __str__(self) -> str:
        text = "[%s] %s: %s" % (self.severity.upper(), self.check, self.message)
        return "%s\n    -> %s" % (text, self.hint) if self.hint else text


# --------------------------------------------------------------------------
# Plausible ranges. Deliberately wide: these are "this cannot be right"
# boundaries, not "this looks unusual" ones. A check that cries wolf gets muted.
# --------------------------------------------------------------------------

# A single-gameweek FPL projection. The all-time record for one player in one
# gameweek is in the low 20s and requires a hat-trick plus bonus; a *projection*
# never approaches it. Rotowire's top weekly projection is typically 7-8.
MAX_PLAUSIBLE_PLAYER_GW_POINTS = 20.0
# Rotowire's weekly table lists projected starters, so the median sits near the
# points a nailed-on rotation-proof starter is worth: roughly 3-6.
MIN_PLAUSIBLE_MEDIAN_GW_POINTS = 1.0
MAX_PLAUSIBLE_MEDIAN_GW_POINTS = 10.0
# 20 teams x 11 projected starters = 220 is the usual shape; season-long tables
# run to 400+ and a truncated "top picks" list can be 100.
MIN_PLAUSIBLE_PROJECTION_ROWS = 80
MAX_PLAUSIBLE_PROJECTION_ROWS = 800

# Spread of single-gameweek team scores in a 10-team Draft league. Real leagues
# sit around 12-18; anything outside this band is a broken estimate, not a
# remarkable league.
MIN_PLAUSIBLE_SCORE_STD = 3.0
MAX_PLAUSIBLE_SCORE_STD = 40.0

# A projected XI total. The floor is permissive on purpose: a squad carrying
# several players who are not expected to start legitimately projects very low,
# because absence from Rotowire's starter list *is* the "not starting" signal.
MIN_PLAUSIBLE_XI_TOTAL = 8.0
MAX_PLAUSIBLE_XI_TOTAL = 120.0

# Win probability sanity. Two teams projected within a couple of points of each
# other is a coin flip under any sane sigma; conversely an extreme call has to be
# earned by an extreme scoreline gap.
NEAR_TIE_POINTS = 2.0
NEAR_TIE_MAX_DEVIATION = 0.12   # a near-tie must land inside 50% +/- 12
EXTREME_PROBABILITY = 0.80
MIN_GAP_FOR_EXTREME_PROBABILITY = 8.0


def check_single_gw_projections(df: Optional[pd.DataFrame],
                                source: str = "projections",
                                points_col: str = "Points") -> List[Issue]:
    """Assert a projection table really is single-gameweek player points.

    This is the check that would have caught the "gameweeks 1-5" article: its
    Points column had a median of 21.7, which is not a gameweek, it is a month.
    """
    issues = []

    if df is None or len(df) == 0:
        return [Issue(
            "single_gw_projections", "error",
            "%s returned no rows" % source,
            "The scrape or fetch failed, or the upstream table shape changed.",
        )]

    if points_col not in df.columns:
        return [Issue(
            "single_gw_projections", "error",
            "%s has no '%s' column (columns: %s)" % (source, points_col, list(df.columns)),
            "Upstream renamed a column; update the parser's header mapping.",
        )]

    n_rows = len(df)
    if not (MIN_PLAUSIBLE_PROJECTION_ROWS <= n_rows <= MAX_PLAUSIBLE_PROJECTION_ROWS):
        issues.append(Issue(
            "single_gw_projections", "error",
            "%s has %d rows, outside the plausible range %d-%d"
            % (source, n_rows, MIN_PLAUSIBLE_PROJECTION_ROWS, MAX_PLAUSIBLE_PROJECTION_ROWS),
            "Too few means a truncated or partly-parsed table; too many means a "
            "season-long table is being used where a weekly one is expected.",
        ))

    points = pd.to_numeric(df[points_col], errors="coerce").dropna()
    if points.empty:
        return issues + [Issue(
            "single_gw_projections", "error",
            "%s '%s' column parsed to no numeric values" % (source, points_col),
            "Check for a formatting change (currency symbols, thousands separators).",
        )]

    median = float(points.median())
    if median > MAX_PLAUSIBLE_MEDIAN_GW_POINTS:
        issues.append(Issue(
            "single_gw_projections", "error",
            "%s median %s is %.1f, above the single-gameweek maximum of %.0f"
            % (source, points_col, median, MAX_PLAUSIBLE_MEDIAN_GW_POINTS),
            "Almost certainly a multi-gameweek cumulative or season-long table. "
            "Check which article ROTOWIRE_URL resolved to.",
        ))
    elif median < MIN_PLAUSIBLE_MEDIAN_GW_POINTS:
        issues.append(Issue(
            "single_gw_projections", "error",
            "%s median %s is %.2f, below the plausible minimum of %.0f"
            % (source, points_col, median, MIN_PLAUSIBLE_MEDIAN_GW_POINTS),
            "Most rows are zero or near-zero -- the source has probably not "
            "published this gameweek yet, or name matching is failing wholesale.",
        ))

    max_points = float(points.max())
    if max_points > MAX_PLAUSIBLE_PLAYER_GW_POINTS:
        issues.append(Issue(
            "single_gw_projections", "error",
            "%s top projection is %.1f, above the single-gameweek maximum of %.0f"
            % (source, max_points, MAX_PLAUSIBLE_PLAYER_GW_POINTS),
            "No single player is projected for this many points in one gameweek.",
        ))

    n_negative = int((points < 0).sum())
    if n_negative:
        issues.append(Issue(
            "single_gw_projections", "warning",
            "%s has %d negative projection(s)" % (source, n_negative),
            "Possible column misalignment in the parsed table.",
        ))

    return issues


def check_score_std(sigma: Optional[float], n_samples: int = 0) -> List[Issue]:
    """Assert the historical-score spread used by the win-probability model is usable.

    A sigma of 0 is the specific failure this guards: the win probability is
    Phi(diff / sqrt(2*sigma^2)), so as sigma approaches 0 the model degenerates
    into a step function that calls a 1-point edge at ~100%.
    """
    if sigma is None:
        return [Issue(
            "score_std", "error", "sigma is None",
            "Fall back to the league prior rather than passing None downstream.",
        )]

    try:
        sigma = float(sigma)
    except (TypeError, ValueError):
        return [Issue(
            "score_std", "error", "sigma is not numeric: %r" % (sigma,), "",
        )]

    if not math.isfinite(sigma):
        return [Issue(
            "score_std", "error", "sigma is not finite (%r)" % sigma,
            "An empty or single-row history produces NaN; fall back to the prior.",
        )]

    if sigma <= 0:
        return [Issue(
            "score_std", "error",
            "sigma is %.4f -- a zero or negative spread degenerates the win "
            "probability model into a step function" % sigma,
            "Every historical score is identical (preseason, all zeros). Drop "
            "unplayed gameweeks and fall back to the league prior.",
        )]

    issues = []
    if not (MIN_PLAUSIBLE_SCORE_STD <= sigma <= MAX_PLAUSIBLE_SCORE_STD):
        issues.append(Issue(
            "score_std", "error",
            "sigma is %.2f, outside the plausible range %.0f-%.0f for weekly "
            "team scores" % (sigma, MIN_PLAUSIBLE_SCORE_STD, MAX_PLAUSIBLE_SCORE_STD),
            "Too low collapses every fixture to a near-certainty; too high "
            "flattens every fixture to 50/50. A std taken over cumulative season "
            "totals instead of per-gameweek scores lands high.",
        ))
    if 0 < n_samples < 2:
        issues.append(Issue(
            "score_std", "warning",
            "sigma estimated from only %d sample(s)" % n_samples, "",
        ))
    return issues


def check_win_probability(prob_a: Optional[float], score_a: float, score_b: float) -> List[Issue]:
    """Assert a win probability is consistent with the scoreline it came from.

    Catches the reported bug directly: 55.8 vs 54.7 shown as 85%/15%. The check
    is model-free -- it only asserts that near-ties are near coin flips and that
    extreme calls are backed by an extreme gap.
    """
    if prob_a is None or not isinstance(prob_a, (int, float)) or not math.isfinite(float(prob_a)):
        return [Issue(
            "win_probability", "error",
            "win probability is not a finite number (%r)" % (prob_a,), "",
        )]

    prob_a = float(prob_a)
    issues = []

    if not (0.0 <= prob_a <= 1.0):
        return [Issue(
            "win_probability", "error",
            "win probability %.3f is outside [0, 1]" % prob_a,
            "A probability was passed as a percentage, or the CDF is wrong.",
        )]

    gap = abs(float(score_a) - float(score_b))
    deviation = abs(prob_a - 0.5)

    if gap <= NEAR_TIE_POINTS and deviation > NEAR_TIE_MAX_DEVIATION:
        issues.append(Issue(
            "win_probability", "error",
            "%.1f vs %.1f is a %.1f-point gap but the win probability is %.0f%%"
            % (score_a, score_b, gap, prob_a * 100),
            "A near-tie must be near a coin flip. The score spread (sigma) used "
            "as the denominator is probably degenerate or missing.",
        ))

    if deviation >= (EXTREME_PROBABILITY - 0.5) and gap < MIN_GAP_FOR_EXTREME_PROBABILITY:
        issues.append(Issue(
            "win_probability", "error",
            "win probability %.0f%% is extreme for a gap of only %.1f points"
            % (prob_a * 100, gap),
            "An extreme call has to be earned by an extreme scoreline gap.",
        ))

    # Direction must follow the scoreline.
    if gap > NEAR_TIE_POINTS:
        favourite_is_a = score_a > score_b
        model_favours_a = prob_a > 0.5
        if favourite_is_a != model_favours_a:
            issues.append(Issue(
                "win_probability", "error",
                "%.1f vs %.1f but the win probability favours the lower-projected "
                "team (%.0f%%)" % (score_a, score_b, prob_a * 100),
                "The two teams' scores are swapped somewhere between projection "
                "and probability.",
            ))

    return issues


def check_projected_team_total(total: Optional[float], n_players: int,
                               label: str = "team") -> List[Issue]:
    """Assert a projected lineup total is in a range a real FPL XI can occupy.

    The floor is intentionally low. A squad with several players who are not
    expected to start legitimately projects in the high teens, because absence
    from the projected-starter list is itself the signal that they will not play.
    """
    if total is None or not math.isfinite(float(total)):
        return [Issue(
            "team_total", "error",
            "%s projected total is not a finite number (%r)" % (label, total), "",
        )]

    total = float(total)
    issues = []

    if total < 0:
        return [Issue(
            "team_total", "error",
            "%s projected total is negative (%.1f)" % (label, total), "",
        )]

    if n_players and n_players != 11:
        issues.append(Issue(
            "team_total", "error",
            "%s lineup has %d players, expected 11" % (label, n_players),
            "Formation constraints in find_optimal_lineup() are not being met.",
        ))

    if total > MAX_PLAUSIBLE_XI_TOTAL:
        issues.append(Issue(
            "team_total", "error",
            "%s projects %.1f points, above the plausible XI maximum of %.0f"
            % (label, total, MAX_PLAUSIBLE_XI_TOTAL),
            "Per-player projections are inflated -- check whether the projection "
            "source is single-gameweek.",
        ))
    elif total < MIN_PLAUSIBLE_XI_TOTAL:
        issues.append(Issue(
            "team_total", "warning",
            "%s projects only %.1f points" % (label, total),
            "Legitimate if most of the XI are not expected to start, but also "
            "what wholesale name-matching failure looks like.",
        ))

    return issues


def check_source_scale_agreement(values_a: Sequence[float], values_b: Sequence[float],
                                 label_a: str = "source A", label_b: str = "source B",
                                 max_ratio: float = 2.0) -> List[Issue]:
    """Assert two projection sources are denominated in the same units.

    Independent sources disagree about individual players all the time, but they
    should agree to within a factor of two on the *typical* player. A systematic
    multiple is the signature of a unit mismatch -- one source reporting a
    multi-gameweek total while the other reports a single gameweek.
    """
    a = pd.to_numeric(pd.Series(list(values_a)), errors="coerce").dropna()
    b = pd.to_numeric(pd.Series(list(values_b)), errors="coerce").dropna()
    a = a[a > 0]
    b = b[b > 0]

    if len(a) < 10 or len(b) < 10:
        return [Issue(
            "source_scale", "warning",
            "not enough overlapping non-zero values to compare %s (%d) and %s (%d)"
            % (label_a, len(a), label_b, len(b)),
            "One source has probably not published this gameweek yet.",
        )]

    median_a, median_b = float(a.median()), float(b.median())
    if median_b == 0:
        return []

    ratio = median_a / median_b
    if ratio > max_ratio or ratio < (1.0 / max_ratio):
        return [Issue(
            "source_scale", "error",
            "%s median is %.2f but %s median is %.2f (%.1fx apart)"
            % (label_a, median_a, label_b, median_b, max(ratio, 1 / ratio)),
            "The two sources are not in the same units. One is likely reporting a "
            "multi-gameweek or season total where a single gameweek is expected.",
        )]
    return []


DRAFT_SQUAD_SIZE = 15

#: A whole league scoring within this band of each other means the percentile join
#: silently failed and every player came back as the 0.5 default.
MIN_LEAGUE_SCORE_SPREAD = 0.5

#: Injuries cannot plausibly cost a team more than this much of its score.
MAX_PLAUSIBLE_INJURY_COST = 60.0


def check_team_strength(team_df: Optional[pd.DataFrame],
                        expected_teams: Optional[int] = None) -> List[Issue]:
    """Assert Draft power-ranking scores are in a range they could actually occupy.

    The failure this is really guarding against is silent: every function in
    analytics.py groups on position codes G/D/M/F, but the FPL bootstrap supplies
    GKP/DEF/MID/FWD. Feed it the wrong ones and positional_percentile() matches no
    group, returns its 0.5 default for every player, and the page renders a
    plausible-looking table in which every team scores exactly 50.0.
    """
    check = "team_strength"

    if team_df is None or not isinstance(team_df, pd.DataFrame) or team_df.empty:
        return [Issue(
            check, "error", "team strength table is empty",
            "Rosters or the bootstrap pool failed to load; the page should show an "
            "info message rather than an empty table.",
        )]

    issues = []
    score_cols = [c for c in ("Score", "Healthy_Score", "GK", "DEF", "MID", "FWD")
                  if c in team_df.columns]

    # --- Every score must sit on the 0-100 scale ---------------------------
    for col in score_cols:
        vals = pd.to_numeric(team_df[col], errors="coerce").dropna()
        if vals.empty:
            issues.append(Issue(
                check, "error", "column %s is entirely non-numeric or missing" % col,
                "A percentile computation returned NaN for every team.",
            ))
            continue
        if (vals < 0).any() or (vals > 100).any():
            issues.append(Issue(
                check, "error",
                "%s ranges %.1f to %.1f, outside the 0-100 scale"
                % (col, vals.min(), vals.max()),
                "Scores are positional percentiles x100 and cannot leave [0, 100]. "
                "Check that Raw_Strength is clipped before aggregation.",
            ))

    # --- The league must not be degenerate ---------------------------------
    if "Score" in team_df.columns and len(team_df) > 1:
        scores = pd.to_numeric(team_df["Score"], errors="coerce").dropna()
        if not scores.empty:
            spread = float(scores.max() - scores.min())
            if spread < MIN_LEAGUE_SCORE_SPREAD:
                issues.append(Issue(
                    check, "error",
                    "all %d teams score within %.2f of each other (around %.1f)"
                    % (len(scores), spread, scores.mean()),
                    "This is what a failed percentile join looks like -- every "
                    "player fell back to the 0.5 default. Check that Position holds "
                    "G/D/M/F, not GKP/DEF/MID/FWD, before scoring.",
                ))
            if float(scores.max()) == 0.0:
                issues.append(Issue(
                    check, "error", "every team scores zero",
                    "No scoring inputs resolved for any player.",
                ))

    # --- Squad shape --------------------------------------------------------
    if "Players" in team_df.columns:
        counts = pd.to_numeric(team_df["Players"], errors="coerce").fillna(0)
        wrong = team_df.loc[counts != DRAFT_SQUAD_SIZE]
        if not wrong.empty:
            issues.append(Issue(
                check, "error",
                "%d team(s) do not hold %d players (saw %s)"
                % (len(wrong), DRAFT_SQUAD_SIZE,
                   sorted(set(counts[counts != DRAFT_SQUAD_SIZE].astype(int)))),
                "Draft enforces 2 GK / 5 DEF / 5 MID / 3 FWD. A short squad means "
                "rostered element IDs were missing from the bootstrap pool.",
            ))

    if expected_teams is not None and len(team_df) != expected_teams:
        issues.append(Issue(
            check, "error",
            "table has %d teams, league has %d" % (len(team_df), expected_teams),
            "A team's roster failed to resolve and was dropped silently.",
        ))

    # --- Injury cost --------------------------------------------------------
    if "Injury_Cost" in team_df.columns:
        cost = pd.to_numeric(team_df["Injury_Cost"], errors="coerce").dropna()
        if not cost.empty:
            if (cost < -1e-6).any():
                issues.append(Issue(
                    check, "error",
                    "negative injury cost (%.1f)" % cost.min(),
                    "Injuries can only reduce a score; Player_Strength must never "
                    "exceed Raw_Strength.",
                ))
            if (cost > MAX_PLAUSIBLE_INJURY_COST).any():
                issues.append(Issue(
                    check, "error",
                    "injury cost of %.1f exceeds the plausible maximum of %.0f"
                    % (cost.max(), MAX_PLAUSIBLE_INJURY_COST),
                    "The injury multiplier is over-penalising -- check the floor in "
                    "injury_helpers.injury_multiplier().",
                ))

    return issues


def format_issues(issues: Sequence[Issue]) -> str:
    """Render issues as a readable multi-line block for a log or assertion message."""
    if not issues:
        return "no issues"
    return "\n".join(str(i) for i in issues)


def raise_on_error(issues: Sequence[Issue], context: str = "") -> None:
    """Raise AssertionError if any issue is an error. Warnings pass through.

    Intended for the live plausibility tests, where an implausible number should
    fail the build rather than print something nobody reads.
    """
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        header = "Implausible data%s:" % (" in %s" % context if context else "")
        raise AssertionError("%s\n%s" % (header, format_issues(errors)))
