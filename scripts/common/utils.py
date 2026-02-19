"""
FPL Management App - Utility Functions (Re-export Shim)

This module re-exports all public names from the focused submodules so that
existing ``from scripts.common.utils import X`` statements continue to work
without modification.  New code should import from the specific modules instead.

Submodules:
  - text_helpers      : Constants, text normalization, position/team mapping
  - fpl_draft_api     : FPL Draft API functions
  - fpl_classic_api   : FPL Classic API functions
  - scraping          : Rotowire, FFP, and Odds API scraping
  - fixture_helpers   : Fixture difficulty grid, kickoff times, FDR styling
  - player_matching   : Player matching, fuzzy merge, normalization
  - analytics         : FDR/form enrichment, availability penalties
  - optimization      : Lineup validation and optimization
"""

# Re-export everything from submodules for backward compatibility.
from scripts.common.text_helpers import *          # noqa: F401,F403
from scripts.common.fpl_draft_api import *         # noqa: F401,F403
from scripts.common.fpl_classic_api import *       # noqa: F401,F403
from scripts.common.scraping import *              # noqa: F401,F403
from scripts.common.fixture_helpers import *       # noqa: F401,F403
from scripts.common.player_matching import *       # noqa: F401,F403
from scripts.common.analytics import *             # noqa: F401,F403
from scripts.common.optimization import *          # noqa: F401,F403
from scripts.common.styled_tables import *         # noqa: F401,F403

# Explicitly re-export underscore-prefixed helpers used by tests/consumers.
from scripts.common.text_helpers import (          # noqa: F401
    _clean_player_name,
    _map_position_to_rw,
    _norm_text,
    _strip_accents,
    _to_short_team_code,
)
from scripts.common.fixture_helpers import _bootstrap_teams_df  # noqa: F401
from scripts.common.player_matching import _backfill_player_ids  # noqa: F401
