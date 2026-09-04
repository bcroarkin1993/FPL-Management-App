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
    "check_merge_match_rate",
    "check_initial_squad",
    "check_element_states",
    "check_transfer_risk",
    "check_transfer_windows",
    "check_transfer_odds",
    "check_ffp_feed",
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


def check_merge_match_rate(matched: int, total: int, source_name: str,
                           min_rate: float = 0.90,
                           input_rows: Optional[int] = None) -> List[Issue]:
    """Assert a name-based merge actually matched most of its reference rows.

    This is the tripwire for the quietest failure mode in the app. Projection
    sources publish common names ("Bruno Fernandes"); the FPL bootstrap publishes
    full legal names ("Bruno Borges Fernandes"). When a merge misses, the caller
    sees NaN, substitutes a neutral default, and the page renders a table where
    the #2 asset in the game is scored as exactly average. Nothing raises,
    nothing looks wrong, and every mocked unit test stays green.

    A real miss rate is small and consists of genuinely unlisted players, so the
    floor is set well below any healthy run rather than at a target.

    ``input_rows`` is the size of the frame being merged *into*, and the check
    **abstains** when that frame is smaller than the reference table. The claim
    being tested is "the reference was mostly claimed", which only means anything
    when the caller could have claimed all of it. The Waiver Wire merges ~105
    available players against a 424-row reference, so it can never claim more
    than a quarter of them: it logged an ERROR at 24.8% on every page load while
    matching 100% of what it was given. Nor is the smaller frame a fair
    denominator -- a subset of the pool holds backups and unranked players that
    no projection source lists, and expecting them to match is the same false
    alarm wearing a different number. A check that cries wolf gets muted, which
    costs the tripwire its whole purpose.

    Full-pool callers still run it, and every page that merges a subset also
    enriches the full reference pool on the same load, so the tripwire is armed.
    """
    check = "merge_match_rate"
    issues: List[Issue] = []

    if not total:
        return [Issue(
            check, "warning",
            "%s: reference table was empty, nothing to match" % source_name,
            "Check the source URL and that the scrape returned rows.",
        )]

    if input_rows is not None and input_rows < total:
        return issues

    rate = matched / float(total)
    if rate < min_rate:
        severity = "error" if rate < min_rate * 0.8 else "warning"
        issues.append(Issue(
            check, severity,
            "%s: matched %d/%d reference rows (%.1f%%), below the %.0f%% floor"
            % (source_name, matched, total, rate * 100, min_rate * 100),
            "Unmatched players fall back to a neutral default, so elite assets "
            "silently score as average. Inspect the misses for a naming pattern "
            "and add a tier to player_matching.ReferenceMatcher.",
        ))
    return issues


# A starting XI's expected points per gameweek. Wide on purpose: a strong squad
# on a good week runs high, and an early-season squad with rotation risk runs
# low. Outside this band the objective is denominated in the wrong unit.
_MIN_XI_EXP_POINTS = 30.0
_MAX_XI_EXP_POINTS = 75.0
_MIN_BUDGET_SPEND = 0.95

_SQUAD_POSITION_QUOTA = {"G": 2, "D": 5, "M": 5, "F": 3}


def check_initial_squad(squad_df: Optional[pd.DataFrame], budget: float,
                        exp_points_col: str = "ExpPts") -> List[Issue]:
    """Assert an optimized 15-man Classic squad is legal and sensibly priced.

    Beyond the FPL rulebook, this catches a scale-free objective. When the ILP
    maximizes positional percentiles instead of expected points, percentile has
    no headroom above ~1.0, so a premium can never repay its price: the solver
    buys a flat mid-price squad, leaves money in the bank, and puts real money on
    the bench. Underspend is the visible symptom of that.
    """
    check = "initial_squad"
    issues: List[Issue] = []

    if squad_df is None or not isinstance(squad_df, pd.DataFrame) or squad_df.empty:
        return [Issue(
            check, "error", "squad is empty",
            "The ILP returned no solution. Check budget, eligibility filters, "
            "and that the score column has non-zero values.",
        )]

    if len(squad_df) != 15:
        issues.append(Issue(
            check, "error", "squad has %d players, expected 15" % len(squad_df),
            "solve_squad_ilp() constrains the squad to 15; a different count "
            "means the frame was filtered after solving.",
        ))

    if "Is_Starter" in squad_df.columns:
        n_start = int(squad_df["Is_Starter"].sum())
        if n_start != 11:
            issues.append(Issue(
                check, "error", "starting XI has %d players, expected 11" % n_start,
                "Check the formation constraints in solve_squad_ilp().",
            ))

    if "Position" in squad_df.columns:
        counts = squad_df["Position"].value_counts().to_dict()
        for pos, want in _SQUAD_POSITION_QUOTA.items():
            got = int(counts.get(pos, 0))
            if got != want:
                issues.append(Issue(
                    check, "error",
                    "squad has %d %s, expected %d" % (got, pos, want),
                    "Positions must be single-letter G/D/M/F. GKP/DEF/MID/FWD "
                    "codes silently match no quota -- convert with "
                    "_map_position_to_rw() first.",
                ))

    if "Team" in squad_df.columns:
        over = squad_df["Team"].value_counts()
        over = over[over > 3]
        if not over.empty:
            issues.append(Issue(
                check, "error",
                "more than 3 players from: %s" % ", ".join(
                    "%s (%d)" % (t, n) for t, n in over.items()),
                "FPL allows at most 3 per club. If the team column contains "
                "placeholders like '???', unresolved players collide into one bucket.",
            ))

    if "Price" in squad_df.columns and budget:
        cost = float(pd.to_numeric(squad_df["Price"], errors="coerce").sum())
        if cost > budget + 1e-6:
            issues.append(Issue(
                check, "error",
                "squad costs %.1f, over the %.1f budget" % (cost, budget),
                "Check the price column is in millions, not tenths.",
            ))
        elif cost < budget * _MIN_BUDGET_SPEND:
            issues.append(Issue(
                check, "warning",
                "squad spends only %.1f of %.1f (%.0f%%)"
                % (cost, budget, cost / budget * 100),
                "Leaving money unspent usually means the objective cannot trade "
                "points against price -- the classic symptom of optimizing on "
                "percentiles rather than expected points.",
            ))

    if exp_points_col in squad_df.columns and "Is_Starter" in squad_df.columns:
        xi = pd.to_numeric(
            squad_df.loc[squad_df["Is_Starter"], exp_points_col], errors="coerce")
        # Check for missing values before summing: pandas skips NaN, so a player
        # with no projection would silently contribute nothing to the total and
        # the squad would still look plausible.
        n_missing = int(xi.isna().sum())
        total = float(xi.sum())
        if n_missing:
            issues.append(Issue(
                check, "error",
                "%d of the starting XI have no %s value" % (n_missing, exp_points_col),
                "A missing projection contributes 0 to the objective, so those "
                "players were effectively picked at random. Check the merge that "
                "populates this column.",
            ))
        elif not math.isfinite(total):
            issues.append(Issue(
                check, "error",
                "starting XI expected points is not finite (%r)" % total,
                "An inf projection propagated into the objective.",
            ))
        elif total < _MIN_XI_EXP_POINTS or total > _MAX_XI_EXP_POINTS:
            issues.append(Issue(
                check, "error",
                "starting XI projects %.1f points/GW, outside the plausible "
                "%.0f-%.0f range" % (total, _MIN_XI_EXP_POINTS, _MAX_XI_EXP_POINTS),
                "Values near 10 suggest a percentile scale; values in the "
                "hundreds suggest season totals were never divided down to a "
                "per-gameweek rate.",
            ))

    return issues


#: Valid `status` codes on the Draft element-status endpoint.
VALID_ELEMENT_STATES = frozenset({"o", "a", "l"})

#: Locking is a transient state applied to a handful of recently-moved players.
#: A large fraction locked means the field is being misread, not that the league
#: dropped half the game.
MAX_PLAUSIBLE_LOCKED_FRACTION = 0.25


def check_element_states(states: Optional[dict],
                         expected_teams: Optional[int] = None) -> List[Issue]:
    """Assert Draft per-player transaction states could actually be what they say.

    The Waiver Wire decides what to suggest from these states. If the endpoint
    changes shape — a renamed status code, an owner field that stops populating —
    the failure is silent: every player reads as available and the page happily
    suggests someone who was dropped an hour ago and cannot be picked up.

    `expected_teams` lets the owned count be checked against the only number it can
    legally be: every team holds exactly DRAFT_SQUAD_SIZE players.
    """
    check = "element_states"

    if not states or not isinstance(states, dict):
        return [Issue(
            check, "error", "element state map is empty",
            "The element-status endpoint returned nothing. The Waiver Wire falls "
            "back to treating every player as available, so locked players will be "
            "suggested again.",
        )]

    issues = []

    codes = {s.get("status") for s in states.values()}
    unknown = codes - VALID_ELEMENT_STATES
    if unknown:
        issues.append(Issue(
            check, "error",
            "unrecognised status code(s) %s" % sorted(str(c) for c in unknown),
            "Draft publishes exactly 'o' (owned), 'a' (available) and 'l' (locked). "
            "A new code means ELEMENT_STATE_* in fpl_draft_api.py needs updating "
            "before the Waiver Wire can trust these states.",
        ))

    owned = [s for s in states.values() if s.get("status") == "o"]
    unowned = [s for s in states.values() if s.get("status") != "o"]
    locked = [s for s in states.values() if s.get("status") == "l"]

    # --- Owner field must agree with the status ----------------------------
    owned_without_owner = sum(1 for s in owned if s.get("owner") is None)
    if owned_without_owner:
        issues.append(Issue(
            check, "error",
            "%d owned player(s) have no owner" % owned_without_owner,
            "Status and owner disagree. Anything keying off owner alone (the "
            "ownership anti-join) will treat these players as free agents.",
        ))

    unowned_with_owner = sum(1 for s in unowned if s.get("owner") is not None)
    if unowned_with_owner:
        issues.append(Issue(
            check, "error",
            "%d unowned player(s) still carry an owner" % unowned_with_owner,
            "A locked or available player belongs to nobody by definition.",
        ))

    # --- Squad arithmetic ---------------------------------------------------
    if expected_teams is not None:
        expected_owned = int(expected_teams) * DRAFT_SQUAD_SIZE
        if len(owned) != expected_owned:
            issues.append(Issue(
                check, "error",
                "%d players are owned, but %d teams x %d players = %d"
                % (len(owned), expected_teams, DRAFT_SQUAD_SIZE, expected_owned),
                "Draft squads are fixed size, so this is arithmetic, not an "
                "estimate. A mismatch means the status codes are being misread.",
            ))

    # --- Locking is transient ----------------------------------------------
    locked_fraction = len(locked) / len(states)
    if locked_fraction > MAX_PLAUSIBLE_LOCKED_FRACTION:
        issues.append(Issue(
            check, "warning",
            "%.0f%% of players (%d of %d) are locked"
            % (locked_fraction * 100, len(locked), len(states)),
            "Locking normally applies to a handful of recently dropped or newly "
            "added players. This many suggests 'l' is being read as something else.",
        ))

    if not any(s.get("status") == "a" for s in states.values()):
        issues.append(Issue(
            check, "error", "no player is available",
            "Every player being owned or locked would leave the waiver wire empty; "
            "far more players exist than a league can roster.",
        ))

    return issues


#: A transfer window empties a handful of squads, not the league. Above this
#: fraction of the pool carrying real risk, the likelier explanation is that
#: keyword tiering or destination parsing broke — the Watkins fix turning into a
#: blanket discount that quietly flattens every ranking.
MAX_PLAUSIBLE_AT_RISK_FRACTION = 0.10

#: Below this many players the at-risk *fraction* carries no signal.
MIN_POOL_FOR_FRACTION_CHECK = 50

#: Risk this high off a single outlet means the corroboration gate is leaking.
#: One newspaper reporting a medical is a scoop or an error; four is a fact.
SINGLE_OUTLET_RISK_CEILING = 0.60


def check_transfer_risk(risk_df: Optional[pd.DataFrame],
                        today=None,
                        floor: float = 0.10) -> List[Issue]:
    """Assert transfer-risk discounts could plausibly be what they say.

    Every failure mode here is silent by construction. A broken feed returns an
    empty frame and renders a page identical to a working one, just with every
    player undiscounted — which is precisely the state that let Watkins be drafted
    in the first place. A broken *matcher* fails the other way, discounting the
    whole pool by a similar amount, which changes no rankings while looking busy.
    """
    check = "transfer_risk"

    if risk_df is None or not hasattr(risk_df, "columns") or risk_df.empty:
        return [Issue(
            check, "error", "transfer risk frame is empty",
            "The news feed returned nothing, so every player is undiscounted and "
            "the page looks exactly like a working one. Check "
            "transfer_feeds.fetch_transfer_news_batch and its network access.",
        )]

    issues = []
    required = {"Transfer_Risk", "Transfer_Mult"}
    missing = required - set(risk_df.columns)
    if missing:
        return [Issue(
            check, "error", "missing column(s) %s" % sorted(missing),
            "attach_transfer_risk() guarantees these columns. A caller has dropped "
            "them, so the multiplier silently defaults to 1.0 everywhere.",
        )]

    risk = pd.to_numeric(risk_df["Transfer_Risk"], errors="coerce")
    mult = pd.to_numeric(risk_df["Transfer_Mult"], errors="coerce")

    # Blending the market in must not push risk out of range, and a decayed
    # weight above 1.0 would let a stale quote outweigh a fresh one.
    if "Odds_Weight" in risk_df.columns:
        weight = pd.to_numeric(risk_df["Odds_Weight"], errors="coerce").dropna()
        bad_weight = weight[(weight < 0) | (weight > 1.0 + 1e-9)]
        if len(bad_weight):
            issues.append(Issue(
                check, "error",
                "%d odds weight(s) outside [0, 1] (e.g. %.3f)"
                % (len(bad_weight), bad_weight.iloc[0]),
                "transfer_odds.odds_age_weight is an exponential decay and cannot "
                "leave [0, 1]. A value above 1.0 means a stale quote is being "
                "amplified rather than discounted.",
            ))

    if "Odds_Risk" in risk_df.columns:
        odds_risk = pd.to_numeric(risk_df["Odds_Risk"], errors="coerce").dropna()
        bad_odds = odds_risk[(odds_risk < 0) | (odds_risk > 1.0 + 1e-9)]
        if len(bad_odds):
            issues.append(Issue(
                check, "error",
                "%d Odds_Risk value(s) outside [0, 1] (e.g. %.3f)"
                % (len(bad_odds), bad_odds.iloc[0]),
                "Odds_Risk is an implied probability. Out of range means "
                "transfer_odds.parse_fractional misread the price format.",
            ))

    bad_risk = risk.dropna()[(risk.dropna() < 0) | (risk.dropna() > 1)]
    if len(bad_risk):
        issues.append(Issue(
            check, "error",
            "%d player(s) with Transfer_Risk outside [0, 1] (e.g. %.3f)"
            % (len(bad_risk), bad_risk.iloc[0]),
            "Risk is a probability. score_headlines() clamps it, so an out-of-range "
            "value means something wrote the column directly.",
        ))

    bad_mult = mult.dropna()[(mult.dropna() < floor - 1e-9) | (mult.dropna() > 1.0 + 1e-9)]
    if len(bad_mult):
        issues.append(Issue(
            check, "error",
            "%d player(s) with Transfer_Mult outside [%.2f, 1.0] (e.g. %.3f)"
            % (len(bad_mult), floor, bad_mult.iloc[0]),
            "transfer_multiplier() floors at TRANSFER_FLOOR and never exceeds 1.0. "
            "A value above 1.0 would *inflate* a player's season projection.",
        ))

    # The fraction check is about *speculation* running away with itself, so it
    # counts only speculative rows. A completed departure is ground truth from the
    # bootstrap, scores 1.0 by construction, and says nothing about the matcher —
    # judged against the whole frame, a page that deliberately lists departed
    # players (the Availability tracker does) fails at 82% while working perfectly.
    status = (risk_df["Transfer_Status"] if "Transfer_Status" in risk_df.columns
              else pd.Series("", index=risk_df.index))
    resolved = status.astype(str).eq("Departed")
    if "Transfer_Note" in risk_df.columns:
        resolved = resolved | risk_df["Transfer_Note"].astype(str).str.startswith("Departed")

    speculative_risk = risk[~resolved]
    total = int(len(speculative_risk))
    at_risk = int((speculative_risk > 0.5).sum())
    # Only meaningful over a real pool. On a handful of rows a single genuinely
    # at-risk player is 25% of the frame, and a check that cries wolf gets muted.
    if total >= MIN_POOL_FOR_FRACTION_CHECK and at_risk / float(total) > MAX_PLAUSIBLE_AT_RISK_FRACTION:
        issues.append(Issue(
            check, "error",
            "%d of %d speculatively-scored players (%.0f%%) score above 0.5 "
            "transfer risk" % (at_risk, total, 100.0 * at_risk / total),
            "A window moves a handful of players, not a tenth of the league. Suspect "
            "keyword tiering matching too broadly, or headline_mentions_player() "
            "attaching one player's news to everybody.",
        ))

    if "Transfer_Outlets" in risk_df.columns:
        outlets = pd.to_numeric(risk_df["Transfer_Outlets"], errors="coerce").fillna(0)
        # Resolved departures legitimately carry zero outlets — the bootstrap said so.
        note = risk_df["Transfer_Note"] if "Transfer_Note" in risk_df.columns else pd.Series("", index=risk_df.index)
        speculative = ~note.astype(str).str.startswith("Departed")
        leaky = int(((risk > SINGLE_OUTLET_RISK_CEILING) & (outlets <= 1) & speculative).sum())
        if leaky:
            issues.append(Issue(
                check, "warning",
                "%d player(s) above %.2f risk on a single outlet"
                % (leaky, SINGLE_OUTLET_RISK_CEILING),
                "The corroboration gate should hold these near 0.5x. Check that "
                "Source is being populated from the feed.",
            ))

    return issues


def check_transfer_windows(windows: Optional[dict], today=None) -> List[Issue]:
    """Warn when the hardcoded transfer-window calendar has gone stale.

    The dates in TRANSFER_WINDOWS are season-specific and cannot be discovered.
    Once they are all in the past, exposure is permanently 0 and the entire
    feature silently becomes a no-op — indistinguishable from "nobody is at risk".
    """
    check = "transfer_windows"
    if not windows:
        return [Issue(check, "error", "transfer window calendar is empty",
                      "TRANSFER_WINDOWS drives exposure. With no windows every "
                      "multiplier is 1.0 and the feature is off.")]

    import datetime as _dt
    ref = today or _dt.date.today()
    closes = [close for spans in windows.values() for _open, close in spans]
    if not closes:
        return [Issue(check, "error", "no window close dates found", "")]

    latest = max(closes)
    if latest < ref:
        return [Issue(
            check, "warning",
            "latest transfer window closed %s, before today (%s)" % (latest, ref),
            "TRANSFER_WINDOWS in transfer_risk.py is season-specific and needs "
            "updating. Until then every transfer discount is silently 1.0.",
        )]
    return []


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


# --- Transfer odds ------------------------------------------------------------
#
# A next-club ladder fails in two directions and both look plausible on screen.
# Too low a total means a row failed to parse, and the missing row is usually the
# favourite. Too high means overlapping outcomes were counted separately -- "Any
# Saudi club" alongside "Al Ittihad" and "Al Hilal" -- which understates every
# destination by dividing one outcome three ways.

#: No real book prices a market at or below evens across all outcomes: that is an
#: arbitrage. Below 1.0 means a row is missing or misparsed.
MIN_LADDER_OVERROUND = 1.0

#: Next-club prices are offered as *independent* binary bets, not as one coupled
#: book, so their sum has no principled ceiling: the live Mateta ladder totals
#: 1.75 across six clubs with no overlap at all. This is therefore only an outer
#: "look at this" bound -- the precise overlap detector is the disjoint-total
#: comparison below, and a wide honest market must not be failed for being wide.
MAX_LADDER_OVERROUND = 3.0

#: Below this many priced rows a "ladder" is a single quote, not a market.
MIN_LADDER_ROWS = 2

#: A feed that has stopped updating without erroring. The live source was already
#: five months stale when it was re-evaluated, so this is a warning, not an error.
MAX_PLAUSIBLE_QUOTE_AGE_DAYS = 120


def check_transfer_odds(ladder_rows: Optional[Sequence[dict]],
                        normalised: Optional[Sequence[dict]] = None,
                        overround: Optional[float] = None,
                        age_days: Optional[float] = None) -> List[Issue]:
    """Assert a next-club odds ladder could plausibly be what it says.

    ``ladder_rows`` are raw rows carrying ``Implied`` (or ``Decimal``);
    ``normalised`` is ``transfer_odds.normalise_ladder`` output; ``overround`` is
    ``transfer_odds.ladder_overround``. Anything omitted is simply not checked.
    """
    check = "transfer_odds"
    issues: List[Issue] = []

    if not ladder_rows:
        return [Issue(check, "error", "odds ladder is empty",
                      "odds_feeds.fetch_player_odds_ladder returns an empty frame on "
                      "any failure, so a broken scrape renders a page identical to a "
                      "working one. Check the page shape at footballtransfers.co.uk.")]

    priced = 0
    for row in ladder_rows:
        implied = row.get("Implied") if isinstance(row, dict) else None
        if implied is None:
            continue
        try:
            value = float(implied)
        except (TypeError, ValueError):
            issues.append(Issue(check, "error",
                                "non-numeric implied probability %r" % (implied,),
                                "transfer_odds.parse_fractional returned something "
                                "unexpected for this row's price."))
            continue
        priced += 1
        if not (0.0 < value <= 1.0):
            issues.append(Issue(check, "error",
                                "implied probability %.3f outside (0, 1]" % value,
                                "Fractional odds parsed wrong. '8/11' is decimal "
                                "1.727 (57.9%), and a bare '2' means 2/1, not "
                                "decimal 2 -- see transfer_odds.parse_fractional."))

    if priced < MIN_LADDER_ROWS:
        issues.append(Issue(check, "warning",
                            "only %d priced row(s) -- not a market" % priced,
                            "A one-row ladder gives no destination distribution; "
                            "the page should say so rather than draw a 100% bar."))

    if overround is not None:
        try:
            total = float(overround)
        except (TypeError, ValueError):
            total = None

        # The overlap bug does not announce itself in the total. The live Salah
        # ladder sums to 1.58 uncollapsed and 1.09 collapsed -- both inside any
        # honest "cannot be right" band, so a threshold cannot separate them.
        # What does separate them is whether the caller collapsed at all, which
        # is exactly comparable against the reference implementation.
        if total is not None:
            try:
                from scripts.common.transfer_odds import ladder_overround
                expected = ladder_overround(ladder_rows)
            except Exception:
                expected = None
            if expected is not None and abs(total - expected) > 0.01:
                issues.append(Issue(check, "error",
                    "overround %.3f does not match the disjoint total %.3f"
                    % (total, expected),
                    "Overlapping outcomes were summed. 'Any Saudi club' already "
                    "contains 'Al Ittihad' and 'Al Hilal'; counting all three "
                    "divides one outcome three ways and understates every "
                    "destination. Sum transfer_odds.disjoint_ladder, not the raw "
                    "rows."))

        if total is not None:
            if total < MIN_LADDER_OVERROUND:
                issues.append(Issue(check, "error",
                    "disjoint ladder sums to %.3f, below 1.0" % total,
                    "A bookmaker does not offer an arbitrage, so a row is missing "
                    "or misparsed -- and the missing row is usually the favourite."))
            elif total > MAX_LADDER_OVERROUND:
                issues.append(Issue(check, "warning",
                    "disjoint ladder sums to %.3f, unusually high" % total,
                    "Overlapping outcomes were not collapsed. transfer_odds."
                    "disjoint_ladder must drop clubs already covered by a quoted "
                    "aggregate; an unresolved club region leaves them both in."))

    if normalised:
        total = 0.0
        for entry in normalised:
            try:
                total += float(entry.get("Probability") or 0.0)
            except (TypeError, ValueError):
                pass
        if abs(total - 1.0) > 0.01:
            issues.append(Issue(check, "error",
                "normalised destination shares sum to %.4f, not 1.0" % total,
                "transfer_odds.normalise_ladder divides by the disjoint total, so "
                "this can only drift if rows were added or dropped afterwards."))

    if age_days is not None:
        try:
            age = float(age_days)
        except (TypeError, ValueError):
            age = None
        if age is not None:
            if age < 0:
                issues.append(Issue(check, "error",
                    "quote is dated %.0f days in the future" % -age,
                    "A negative age means the timestamp was parsed in the wrong "
                    "timezone or the wrong field was read."))
            elif age > MAX_PLAUSIBLE_QUOTE_AGE_DAYS:
                issues.append(Issue(check, "warning",
                    "quote is %.0f days old" % age,
                    "The feed may have stopped updating. This is survivable -- "
                    "odds_age_weight decays it to near nothing -- but the page "
                    "must show the age, never present it as current."))

    return issues


# --- FFP feed ---------------------------------------------------------------

#: The site payload listed 368 players live. A sudden collapse means the parser
#: is picking up a partial table, not that FFP shed two thirds of the league.
MIN_PLAUSIBLE_FFP_ROWS = 150
#: FFP republishes every gameweek, so a stamp older than a fortnight is a feed
#: that has stopped moving.
MAX_PLAUSIBLE_FFP_AGE_DAYS = 14


def check_ffp_feed(df: Optional[pd.DataFrame],
                   gameweek: Optional[int] = None,
                   expected_gw: Optional[int] = None,
                   age_days: Optional[float] = None) -> List[Issue]:
    """Is this FFP table usable, and is it for the week we think it is?

    The failure this exists for is not a crash. FFP's published spreadsheet fell
    a gameweek behind their own site and kept serving 561 perfectly plausible
    rows; the app blended them at 40% against Rotowire's current-gameweek
    numbers and nothing anywhere said a word. Every value was individually
    reasonable. Only the *gameweek* was wrong.

    The other error this catches is the migration's own worst case. FFP's site
    names its two prediction columns the opposite way round from the sheet:
    ``predicted_points`` is conditional on starting while
    ``predicted_points_start`` already carries the start discount. Map them
    across by name instead of by basis and ``Predicted`` ends up larger than
    ``StartingPredicted`` -- which then charges the start probability twice, or
    not at all, depending on the consumer.
    """
    check = "ffp_feed"
    issues: List[Issue] = []

    if df is None or getattr(df, "empty", True):
        return [Issue(check, "error", "FFP feed is empty",
                      "A broken FFP feed renders a page identical to a working one -- "
                      "every score silently drops to Rotowire-only. Check "
                      "ffp_feed.fetch_points_predictor() and the spreadsheet fallback.")]

    if len(df) < MIN_PLAUSIBLE_FFP_ROWS:
        issues.append(Issue(check, "warning",
            "only %d FFP rows (expected a few hundred)" % len(df),
            "A short table usually means the payload parser captured a fragment "
            "rather than that FFP published less."))

    if expected_gw is not None and gameweek is not None and int(gameweek) != int(expected_gw):
        issues.append(Issue(check, "error",
            "FFP is published for GW%d but GW%d is being scored"
            % (int(gameweek), int(expected_gw)),
            "Blending two different gameweeks is undetectable by eye. Either gate "
            "FFP off for this gameweek or wait for FFP to publish."))
    elif gameweek is None:
        issues.append(Issue(check, "warning",
            "FFP feed does not state which gameweek it covers",
            "Without a gameweek there is nothing to gate on, and a stale table "
            "looks exactly like a current one."))

    if {"Predicted", "StartingPredicted"}.issubset(df.columns):
        pred = pd.to_numeric(df["Predicted"], errors="coerce")
        cond = pd.to_numeric(df["StartingPredicted"], errors="coerce")
        both = pred.notna() & cond.notna() & (cond > 0)
        if both.any():
            # Tolerance covers rounding only: the relation is exact by
            # construction (Predicted == StartingPredicted x Start%), and the
            # live sheet reproduces it to within 0.0003. A loose tolerance would
            # let an inversion pass unnoticed for every high-start player, who
            # are precisely the ones the blend leans on.
            inverted = (pred[both] > cond[both] + 0.05)
            if inverted.mean() > 0.10:
                issues.append(Issue(check, "error",
                    "Predicted exceeds StartingPredicted for %.0f%% of players"
                    % (100 * inverted.mean()),
                    "The two columns are on different bases: Predicted is "
                    "StartingPredicted x Start%%, so it can never be the larger of "
                    "the two. They have almost certainly been mapped across from "
                    "the site payload by name instead of by basis."))

    if "Start" in df.columns:
        start = pd.to_numeric(df["Start"], errors="coerce").dropna()
        if not start.empty and not start.between(0, 100).all():
            issues.append(Issue(check, "error",
                "Start%% outside 0-100 (min %.1f, max %.1f)" % (start.min(), start.max()),
                "The app divides this by 100 to scale projections, so an out-of-range "
                "value silently rescales every score that touches it."))

    if age_days is not None:
        try:
            age = float(age_days)
        except (TypeError, ValueError):
            age = None
        if age is not None:
            if age < 0:
                issues.append(Issue(check, "error",
                    "FFP stamp is %.0f days in the future" % -age,
                    "The publish time was parsed in the wrong timezone or the wrong "
                    "year was inferred."))
            elif age > MAX_PLAUSIBLE_FFP_AGE_DAYS:
                issues.append(Issue(check, "warning",
                    "FFP was last published %.0f days ago" % age,
                    "FFP republishes weekly. A stamp this old means the feed has "
                    "stopped moving -- show the age rather than presenting it as current."))

    return issues
