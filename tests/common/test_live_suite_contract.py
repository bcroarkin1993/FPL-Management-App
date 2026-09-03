"""The live suite's skip predicate must not file our bugs as outages.

`skip_if_unreachable` once caught bare `Exception`, so an `AttributeError` inside
`compute_player_scores` was reported as "Draft league strength unreachable" and
four Power Rankings checks stood down while the page was actually broken. A test
that skips itself when the code is wrong is worse than no test: it is a green
tick over a failure.

These run offline — the predicate is pure.
"""

import json
import xml.etree.ElementTree as ET

import pytest
import requests
from _pytest.outcomes import Skipped

from tests.live.conftest import _is_transport_failure, skip_if_unreachable


class TestTransportDetection:
    def test_request_failures_are_transport(self):
        assert _is_transport_failure(requests.ConnectionError("no route"))
        assert _is_transport_failure(requests.Timeout("too slow"))
        assert _is_transport_failure(requests.HTTPError("503"))

    def test_socket_and_decode_failures_are_transport(self):
        assert _is_transport_failure(ConnectionResetError("peer reset"))
        assert _is_transport_failure(json.JSONDecodeError("bad", "doc", 0))
        assert _is_transport_failure(ET.ParseError("truncated"))

    def test_our_own_bugs_are_not_transport(self):
        assert not _is_transport_failure(
            AttributeError("'numpy.float64' object has no attribute 'fillna'"))
        assert not _is_transport_failure(KeyError("Season_Points"))
        assert not _is_transport_failure(TypeError("unsupported operand"))

    def test_a_wrapped_outage_is_still_an_outage(self):
        """App code routinely catches a request error and raises its own."""
        try:
            try:
                raise requests.ConnectionError("no route")
            except requests.ConnectionError as exc:
                raise RuntimeError("could not load bootstrap") from exc
        except RuntimeError as wrapped:
            assert _is_transport_failure(wrapped)

    def test_an_unrelated_error_during_handling_is_not_laundered(self):
        """A bug raised while handling a bug stays a bug."""
        try:
            try:
                raise KeyError("Season_Points")
            except KeyError as exc:
                raise AttributeError("no attribute 'fillna'") from exc
        except AttributeError as wrapped:
            assert not _is_transport_failure(wrapped)


class TestSkipIfUnreachable:
    def test_transport_failure_skips(self):
        # Skipped derives from BaseException, so pytest.raises(Exception) would
        # let it through and skip *this* test instead of asserting on it.
        with pytest.raises(Skipped):
            skip_if_unreachable(
                lambda: (_ for _ in ()).throw(requests.Timeout("slow")), "Some feed")

    def test_code_bug_fails_loudly(self):
        with pytest.raises(AssertionError) as caught:
            skip_if_unreachable(
                lambda: (_ for _ in ()).throw(AttributeError("no 'fillna'")),
                "Draft league strength")
        message = str(caught.value)
        assert "bug in our code" in message
        assert "Draft league strength" in message

    def test_success_returns_the_value(self):
        assert skip_if_unreachable(lambda: 42, "Some feed") == 42

    def test_none_still_skips(self):
        """A source that returns nothing is unreachable for our purposes."""
        with pytest.raises(Skipped):
            skip_if_unreachable(lambda: None, "Some feed")
