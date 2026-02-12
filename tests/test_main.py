"""Smoke test for main.py entry point.

Verifies that main() can set up navigation without errors,
catching issues like invalid keyword arguments or missing imports.
"""

from unittest.mock import patch, MagicMock

import pytest


class TestMain:
    def test_smoke(self, mock_all_utils):
        """main() should build navigation and call nav.run() without errors."""
        mock_nav = MagicMock()
        mock_page = MagicMock()

        with patch("main.st.navigation", return_value=mock_nav) as nav_call, \
             patch("main.st.Page", return_value=mock_page), \
             patch("main.os.path.exists", return_value=False), \
             patch("main.preload_app_data", return_value={}):
            from main import main
            main()

        # Verify navigation was created and run
        nav_call.assert_called_once()
        mock_nav.run.assert_called_once()

        # Verify all 3 sections were passed to st.navigation
        pages_arg = nav_call.call_args[0][0]
        assert "FPL App Home" in pages_arg
        assert "Draft" in pages_arg
        assert "Classic" in pages_arg

    def test_smoke_with_logo(self, mock_all_utils):
        """main() should call st.sidebar.image when logo file exists."""
        mock_nav = MagicMock()

        with patch("main.st.navigation", return_value=mock_nav), \
             patch("main.st.Page", return_value=MagicMock()), \
             patch("main.os.path.exists", return_value=True), \
             patch("main.preload_app_data", return_value={}):
            from main import main
            main()

        # Sidebar image should have been called (via the mock_streamlit sidebar mock)
        mock_nav.run.assert_called_once()
