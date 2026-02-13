"""Smoke test for main.py entry point.

Verifies that main() can set up sidebar navigation without errors,
catching issues like invalid keyword arguments or missing imports.
"""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


def _mock_sidebar_radio(label, options, **kwargs):
    """Return the first option, mimicking Streamlit's default radio behavior."""
    return options[0] if options else None


def _home_page_patches():
    """Context managers to mock functions used by the dashboard home page."""
    return [
        patch("main.requests.get", return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"team_h": 1, "team_a": 2, "started": False, "finished": False,
                 "team_h_score": None, "team_a_score": None},
            ]),
            raise_for_status=MagicMock(),
        )),
        patch("main.get_draft_league_details", return_value={
            "league": {"name": "Draft League"},
            "league_entries": [
                {"id": 1, "entry_id": 67890, "entry_name": "My Draft Team"},
                {"id": 2, "entry_id": 99, "entry_name": "Rival Team"},
            ],
            "standings": [
                {"league_entry": 2, "rank": 1, "matches_won": 7,
                 "matches_drawn": 1, "matches_lost": 0, "total": 80},
                {"league_entry": 1, "rank": 2, "matches_won": 5,
                 "matches_drawn": 2, "matches_lost": 1, "total": 60},
            ],
        }),
        patch("main.get_classic_league_standings", return_value={
            "league": {"name": "Test Classic League"},
            "standings": {
                "results": [
                    {"rank": 1, "entry": 22222, "entry_name": "Leader",
                     "total": 1500},
                    {"rank": 2, "entry": 11111, "entry_name": "My Team",
                     "total": 1480},
                ],
            },
        }),
    ]


class TestMain:
    def test_smoke(self, mock_all_utils):
        """main() should build sidebar navigation and route to a page without errors."""
        with patch("main.os.path.exists", return_value=False), \
             patch("main.preload_app_data", return_value={}), \
             patch("main.st.sidebar") as mock_sidebar:
            for p in _home_page_patches():
                p.start()
            mock_sidebar.radio = _mock_sidebar_radio
            from main import main
            try:
                main()  # Should complete without errors
            finally:
                patch.stopall()

    def test_smoke_with_logo(self, mock_all_utils):
        """main() should call st.sidebar.image when logo file exists."""
        with patch("main.os.path.exists", return_value=True), \
             patch("main.preload_app_data", return_value={}), \
             patch("main.st.sidebar") as mock_sidebar, \
             patch("main.render_app_home"):
            mock_sidebar.radio = _mock_sidebar_radio
            from main import main
            main()

        # Verify sidebar.image was called (logo exists)
        mock_sidebar.image.assert_called_once()

    def test_all_sections_defined(self):
        """All 3 sections should be present in SECTIONS."""
        from main import SECTIONS
        labels = list(SECTIONS.keys())
        assert len(labels) == 3
        assert any("FPL App Home" in l for l in labels)
        assert any("Draft" in l for l in labels)
        assert any("Classic" in l for l in labels)

    def test_all_pages_callable(self):
        """Every page in every section should map to a callable."""
        from main import SECTIONS
        for section_label, pages in SECTIONS.items():
            assert len(pages) > 0, f"Section {section_label} has no pages"
            for page_label, func in pages.items():
                assert callable(func), f"{section_label} > {page_label} is not callable"

    def test_render_app_home_smoke(self, mock_all_utils):
        """render_app_home() should render all dashboard sections without errors."""
        for p in _home_page_patches():
            p.start()
        try:
            from main import render_app_home
            render_app_home()
        finally:
            patch.stopall()

    def test_render_app_home_graceful_failures(self, mock_all_utils):
        """render_app_home() should handle API failures gracefully."""
        patches = [
            patch("main.requests.get", side_effect=Exception("Network error")),
            patch("main.get_draft_league_details", return_value=None),
            patch("main.get_classic_league_standings", return_value=None),
            patch("main.get_h2h_league_standings", return_value=None),
        ]
        for p in patches:
            p.start()
        try:
            from main import render_app_home
            render_app_home()  # Should not raise
        finally:
            patch.stopall()
