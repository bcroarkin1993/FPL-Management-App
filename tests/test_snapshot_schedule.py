"""The snapshot collector must still run close to a deadline.

The accuracy harness is only as good as *when* its pre-deadline files were
frozen. GitHub does not honour cron reliably -- declared hourly, this workflow
was observed running every 2.5 to 6 hours -- so with an 8-hour capture window
typically one run landed inside it, the earliest, and the file froze there.
Measured: GW4 was captured 7.8 hours before its deadline and never refreshed,
so every source was scored on its pre-team-news state.

These tests parse the cron lines directly rather than importing PyYAML, which
is not a dependency of this project.
"""

import pathlib
import re

import pytest

WORKFLOW = (pathlib.Path(__file__).resolve().parents[1]
            / ".github" / "workflows" / "projection-snapshots.yml")

#: UTC hours in which this season's deadlines actually fall, read off the
#: bootstrap: {10, 11, 12, 13, 17, 18}. The dense schedule has to cover the
#: final hours before each of them.
DEADLINE_HOURS = (10, 11, 12, 13, 17, 18)


def _crons():
    text = WORKFLOW.read_text()
    return re.findall(r'-\s*cron:\s*["\']([^"\']+)["\']', text)


def _hours_covered(expr):
    """The set of UTC hours a cron expression fires in."""
    hour_field = expr.split()[1]
    if hour_field == "*":
        return set(range(24))
    hours = set()
    for part in hour_field.split(","):
        step = 1
        if "/" in part:
            part, step_s = part.split("/")
            step = int(step_s)
        if part == "*":
            lo, hi = 0, 23
        elif "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
        else:
            lo = hi = int(part)
        hours.update(range(lo, hi + 1, step))
    return hours


def _runs_per_hour(expr):
    minute_field = expr.split()[0]
    if minute_field == "*":
        return 60
    if minute_field.startswith("*/"):
        return 60 // int(minute_field[2:])
    return len(minute_field.split(","))


def test_the_workflow_declares_more_than_one_schedule():
    """One hourly cron is what froze GW4 eight hours early."""
    assert len(_crons()) >= 2, (
        "expected an hourly safety net plus a dense pre-deadline schedule, got %s"
        % _crons())


def test_an_hourly_safety_net_still_exists():
    """The wide net is what guarantees a gameweek is never missed outright, even
    if the dense schedule's hours stop matching the fixture calendar."""
    assert any(_runs_per_hour(c) >= 1 and _hours_covered(c) == set(range(24))
               for c in _crons()), _crons()


@pytest.mark.parametrize("hour", DEADLINE_HOURS)
def test_the_hour_before_each_deadline_is_densely_covered(hour):
    """A run must be able to land in the last hour before a deadline, which is
    where the team news that matters actually settles."""
    dense = [c for c in _crons() if _runs_per_hour(c) >= 4]
    assert dense, "no schedule runs more than hourly: %s" % _crons()
    covered = set()
    for c in dense:
        covered |= _hours_covered(c)
    assert (hour - 1) in covered or hour in covered, (
        "nothing dense covers %02d:00, an hour this season's deadlines fall in"
        % hour)


def test_the_dense_schedule_runs_every_day():
    """Seven of the 38 deadlines are midweek (Tue, Wed, Sun). Restricting the
    dense cover to Fri/Sat would lose exactly the awkward gameweeks."""
    for c in _crons():
        if _runs_per_hour(c) >= 4:
            assert c.split()[4] == "*", (
                "dense schedule is limited to weekdays %r, but deadlines fall on "
                "every day of the week" % c.split()[4])


def test_dependencies_are_cached():
    """The dense schedule multiplies runs that do nothing, and installing
    requirements is nearly all of their cost."""
    assert "cache: pip" in WORKFLOW.read_text()


def test_runs_are_still_serialised():
    """Two collectors writing the same file and both pushing is a guaranteed
    conflict -- and denser scheduling makes overlap more likely, not less."""
    text = WORKFLOW.read_text()
    assert "concurrency:" in text and "projection-snapshots" in text
