"""Smoke test for main.py entry point.

Verifies that main() can set up sidebar navigation without errors,
catching issues like invalid keyword arguments or missing imports.
"""

from unittest.mock import patch, MagicMock

import pytest


def _mock_sidebar_radio(label, options, **kwargs):
    """Return the first option, mimicking Streamlit's default radio behavior."""
    return options[0] if options else None


class TestMain:
    def test_smoke(self, mock_all_utils):
        """main() should build sidebar navigation and route to a page without errors."""
        with patch("main.os.path.exists", return_value=False), \
             patch("main.preload_app_data", return_value={}), \
             patch("main.st.sidebar") as mock_sidebar:
            mock_sidebar.radio = _mock_sidebar_radio
            from main import main
            main()  # Should complete without errors

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
