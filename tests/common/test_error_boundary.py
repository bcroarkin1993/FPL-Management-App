"""Tests for page_error_boundary.

Two failure modes this session produced, both bad for different reasons:
  * Projections Hub raised, and the reader got Streamlit's raw traceback in
    place of the app.
  * My Leagues raised inside a bare `except`, and the reader got a blank space
    with no indication anything had gone wrong.

The boundary does neither: a legible message, the traceback behind a
disclosure, and an ERROR log with a stack trace for whoever is debugging.
"""

import logging

import pytest

from scripts.common.error_helpers import page_error_boundary


class TestPageErrorBoundary:
    def test_success_path_is_transparent(self, mock_streamlit):
        with page_error_boundary("Some Page"):
            result = 1 + 1
        assert result == 2
        mock_streamlit["error"].assert_not_called()

    def test_exception_is_swallowed_not_propagated(self, mock_streamlit):
        """The app must stay navigable -- the sidebar renders after this."""
        with page_error_boundary("Projections Hub"):
            raise ValueError("cannot convert float NaN to integer")

    def test_error_message_names_the_page_and_the_exception(self, mock_streamlit):
        with page_error_boundary("Projections Hub"):
            raise ValueError("cannot convert float NaN to integer")
        message = str(mock_streamlit["error"].call_args)
        assert "Projections Hub" in message
        assert "ValueError" in message
        assert "cannot convert float NaN to integer" in message

    def test_reader_is_told_the_rest_of_the_app_still_works(self, mock_streamlit):
        with page_error_boundary("Projections Hub"):
            raise RuntimeError("boom")
        assert "sidebar" in str(mock_streamlit["error"].call_args).lower()

    def test_traceback_is_logged_with_a_stack_trace(self, mock_streamlit, caplog):
        with caplog.at_level(logging.ERROR, logger="fpl_app"):
            with page_error_boundary("Projections Hub"):
                raise ValueError("boom")
        records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert records, "the failure must reach the log, not only the screen"
        assert records[0].exc_info is not None, "log the stack trace, not just the message"

    def test_keyboard_interrupt_is_not_swallowed(self, mock_streamlit):
        """Only real errors are caught; control-flow exceptions pass through."""
        with pytest.raises(KeyboardInterrupt):
            with page_error_boundary("Some Page"):
                raise KeyboardInterrupt
