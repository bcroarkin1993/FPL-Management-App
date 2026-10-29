"""The 1GW vs rest-of-season balance, and the tier it applies to.

Reported from the app at GW9: a mid-price midfielder (Mathys Tel) led a rival
(Jack Rudoni) on *both* forward-looking signals -- 1GW percentile 0.937 against
0.730, and the 3-gameweek horizon 0.860 against 0.821 -- and still scored
lower, because the backward-looking `season_quality` term slightly outweighed
the forward `mgw` one and both of its inputs favoured the rival's longer track
record. At GW9 `season_quality` is still 63.5% *preseason* projection, so a
nine-week-old guess was outvoting what both live projections said.

These are policy numbers, not fitted ones -- there is no archived Transfer
Score to score against an outcome. The tests pin the *shape* of the policy.
"""

import config
from scripts.common.analytics import (
    _ros_weights, _DEFAULT_ROS_WEIGHTS, season_progress_weight,
    compute_dynamic_alpha,
)


class TestRosWeights:
    def test_they_sum_to_one_at_every_gameweek(self):
        """A set that does not sum to 1 silently rescales every ROS score
        rather than failing, so nothing downstream could reveal it."""
        for gw in range(1, 39):
            total = sum(_ros_weights(season_progress_weight(gw)).values())
            assert abs(total - 1.0) < 1e-9, (gw, total)

    def test_the_forward_term_leads_the_backward_one_early(self):
        """The reported bug: `sq` outweighed `mgw` at GW9, so a preseason
        projection outvoted two live ones."""
        w = _ros_weights(season_progress_weight(9))
        assert w["mgw"] > w["sq"], w

    def test_the_backward_term_takes_over_late(self):
        """Deliberate: by GW30 there are actuals worth trusting and barely any
        horizon left to forecast."""
        w = _ros_weights(season_progress_weight(30))
        assert w["sq"] > w["mgw"], w

    def test_the_fallback_tracks_config(self):
        """The app and its own config-unreadable fallback must not disagree
        about how it scores -- the rule `DEFAULT_START_FLOORS` carries too."""
        assert _DEFAULT_ROS_WEIGHTS == config.ROS_WEIGHTS

    def test_a_missing_config_still_yields_usable_weights(self, monkeypatch):
        monkeypatch.delattr(config, "ROS_WEIGHTS", raising=False)
        w = _ros_weights(0.5)
        assert abs(sum(w.values()) - 1.0) < 1e-9


class TestAlphaTiers:
    @staticmethod
    def _alpha(ros):
        return compute_dynamic_alpha("M", ros, "draft", None, 9)

    def test_an_elite_player_is_protected(self):
        """Hold a top asset through a bad week: weight ROS, not this gameweek."""
        assert self._alpha(0.90) < self._alpha(0.50)

    def test_a_mid_tier_player_tilts_towards_this_gameweek(self):
        """He is the one you are actively deciding whether to replace *now*, so
        the near term is what matters -- and protecting everyone above the 60th
        percentile treated him as irreplaceable."""
        baseline = config.ALPHA_BASELINE["draft"]
        assert self._alpha(0.67) > baseline

    def test_the_churn_zone_is_wide(self):
        """Everything at or below the top quartile is in it."""
        baseline = config.ALPHA_BASELINE["draft"]
        for ros in (0.30, 0.50, 0.67, 0.74):
            assert self._alpha(ros) > baseline, ros

    def test_protection_starts_above_the_top_quartile(self):
        baseline = config.ALPHA_BASELINE["draft"]
        assert self._alpha(0.80) < baseline
        assert self._alpha(0.90) < self._alpha(0.80)

    def test_classic_still_leans_further_on_this_gameweek_than_draft(self):
        """A Classic transfer is cheap and weekly; a dropped Draft player goes
        to the waiver wire."""
        assert (compute_dynamic_alpha("M", 0.67, "classic", None, 9)
                > compute_dynamic_alpha("M", 0.67, "draft", None, 9))

    def test_alpha_stays_clamped(self):
        for ros in (0.0, 0.5, 1.0):
            for fmt in ("draft", "classic"):
                a = compute_dynamic_alpha("G", ros, fmt, None, 38)
                assert 0.15 <= a <= 0.75, (ros, fmt, a)


class TestTheReportedShape:
    """A mid-tier player leading on both forward signals must come out ahead.

    Built from the real GW9 percentiles rather than invented ones.
    """

    @staticmethod
    def _score(one_gw, ros):
        alpha = compute_dynamic_alpha("M", ros, "draft", None, 9)
        return alpha * one_gw + (1 - alpha) * ros

    def test_the_forward_leader_now_wins(self):
        tel = self._score(0.937, 0.689)      # leads 1GW and the 3GW horizon
        rudoni = self._score(0.730, 0.783)   # leads only on season points
        assert tel > rudoni, (tel, rudoni)

    def test_the_old_policy_had_it_the_other_way(self, monkeypatch):
        """Pins what changed, so this cannot silently revert."""
        monkeypatch.setattr(config, "ALPHA_TIERS",
                            ((0.80, -0.10), (0.60, -0.05), (0.40, 0.0),
                             (0.00, +0.05)))
        tel = self._score(0.937, 0.666)
        rudoni = self._score(0.730, 0.770)
        assert tel < rudoni, "the old bands protected a merely above-average player"

    def test_an_elite_player_is_still_not_dropped_for_one_good_week(self):
        """The protection that survives: a 0.90-ROS asset outranks a one-week
        wonder with nothing behind him."""
        elite = self._score(0.60, 0.90)
        flash = self._score(0.95, 0.45)
        assert elite > flash, (elite, flash)
