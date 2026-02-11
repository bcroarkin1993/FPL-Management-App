"""Tests for scripts/common/error_helpers.py."""

import logging
from unittest.mock import patch, MagicMock

from scripts.common.error_helpers import get_logger, show_api_error


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_no_duplicate_handlers(self):
        """Calling get_logger multiple times shouldn't add duplicate handlers."""
        logger = get_logger("test_no_dup")
        handler_count = len(logger.handlers)
        get_logger("test_no_dup")
        assert len(logger.handlers) == handler_count

    def test_default_name(self):
        logger = get_logger()
        assert logger.name == "fpl_app"


class TestShowApiError:
    def test_calls_st_error(self):
        with patch("scripts.common.error_helpers.st") as mock_st:
            show_api_error("loading player data")
            mock_st.error.assert_called_once()
            call_arg = mock_st.error.call_args[0][0]
            assert "loading player data" in call_arg

    def test_logs_exception(self):
        with patch("scripts.common.error_helpers.st"):
            with patch("scripts.common.error_helpers.get_logger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                exc = ValueError("test error")
                show_api_error("fetching data", exception=exc)
                mock_logger.warning.assert_called_once()

    def test_stop_calls_st_stop(self):
        with patch("scripts.common.error_helpers.st") as mock_st:
            show_api_error("loading data", stop=True)
            mock_st.stop.assert_called_once()
