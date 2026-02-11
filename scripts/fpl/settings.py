# scripts/fpl/settings.py
#
# Streamlit Settings page for configuring all alert types.

import streamlit as st
from scripts.common.alert_config import load_settings, save_settings, DEFAULT_SETTINGS


def show_settings_page():
    st.title("Alert Settings")
    st.caption(
        "Configure Discord notifications for deadline reminders and data source updates. "
        "Settings are saved to `alert_settings.json` and used by the GitHub Actions workflow."
    )

    settings = load_settings()

    # =================================================================
    # Discord Configuration
    # =================================================================
    st.header("Discord Configuration")
    st.info(
        "The Discord webhook URL is stored in your `.env` file (or GitHub Actions secrets) "
        "for security. Set `DISCORD_WEBHOOK_URL` there."
    )

    discord_cfg = settings.get("discord", {})
    col1, col2 = st.columns(2)
    with col1:
        mention_user = st.text_input(
            "Mention User ID",
            value=discord_cfg.get("mention_user_id", ""),
            help="Discord user ID to @mention in alerts (e.g., 123456789012345678)",
        )
    with col2:
        mention_role = st.text_input(
            "Mention Role ID",
            value=discord_cfg.get("mention_role_id", ""),
            help="Discord role ID to @mention in alerts (e.g., 987654321098765432)",
        )

    # =================================================================
    # Deadline Alerts
    # =================================================================
    st.header("Deadline Alerts")
    st.caption("Get reminded at ~24h, ~6h, and ~1h before each deadline.")

    dl = settings.get("deadline_alerts", {})
    draft_cfg = dl.get("draft", {})
    classic_cfg = dl.get("classic", {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Draft")
        draft_enabled = st.toggle(
            "Enable Draft deadline alerts",
            value=draft_cfg.get("enabled", False),
            key="draft_enabled",
        )
        draft_offset = st.slider(
            "Hours before kickoff",
            min_value=1.0,
            max_value=48.0,
            value=float(draft_cfg.get("offset_hours", 25.5)),
            step=0.5,
            key="draft_offset",
            help="Draft waiver/transaction deadline offset from the earliest GW kickoff",
        )

    with col2:
        st.subheader("Classic")
        classic_enabled = st.toggle(
            "Enable Classic deadline alerts",
            value=classic_cfg.get("enabled", False),
            key="classic_enabled",
        )
        classic_offset = st.slider(
            "Hours before kickoff",
            min_value=0.5,
            max_value=48.0,
            value=float(classic_cfg.get("offset_hours", 1.5)),
            step=0.5,
            key="classic_offset",
            help="Classic transfer deadline offset from the earliest GW kickoff",
        )

    # =================================================================
    # Data Source Alerts
    # =================================================================
    st.header("Data Source Alerts")
    st.caption(
        "Get notified on Discord when new gameweek data becomes available. "
        "Checked every 15 minutes via GitHub Actions."
    )

    ds = settings.get("data_source_alerts", {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rotowire")
        rotowire_enabled = st.toggle(
            "Enable Rotowire alerts",
            value=ds.get("rotowire", {}).get("enabled", False),
            key="rotowire_enabled",
        )
        st.caption("Alerts when the GW player rankings article is published.")

    with col2:
        st.subheader("Fantasy Football Pundit")
        ffp_enabled = st.toggle(
            "Enable FFP alerts",
            value=ds.get("ffp", {}).get("enabled", False),
            key="ffp_enabled",
        )
        st.caption("Alerts when FFP updates their projections spreadsheet for the current GW.")

    # =================================================================
    # Data Source Status
    # =================================================================
    st.header("Data Source Status")
    st.caption("Check whether data sources have published data for the current gameweek.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Check Rotowire", key="check_rotowire"):
            with st.spinner("Checking Rotowire..."):
                try:
                    from scripts.common.data_source_checks import is_rotowire_available_for_gw
                    from scripts.common.waiver_alerts import _get_current_gameweek
                    gw = _get_current_gameweek()
                    available = is_rotowire_available_for_gw(gw)
                    if available:
                        st.success(f"Rotowire GW {gw} rankings are available!")
                    else:
                        st.warning(f"Rotowire GW {gw} rankings not yet published.")
                except Exception as e:
                    st.error(f"Check failed: {e}")

    with col2:
        if st.button("Check FFP", key="check_ffp"):
            with st.spinner("Checking FFP..."):
                try:
                    from scripts.common.data_source_checks import is_ffp_available_for_gw
                    from scripts.common.waiver_alerts import _get_current_gameweek
                    gw = _get_current_gameweek()
                    available = is_ffp_available_for_gw(gw)
                    if available:
                        st.success(f"FFP GW {gw} projections are available!")
                    else:
                        st.warning(f"FFP GW {gw} projections not yet updated.")
                except Exception as e:
                    st.error(f"Check failed: {e}")

    # Show last alert state
    state = settings.get("alert_state", {})
    last_rw = state.get("last_rotowire_alert_gw", 0)
    last_ffp = state.get("last_ffp_alert_gw", 0)
    if last_rw or last_ffp:
        st.caption(
            f"Last alerts sent: Rotowire GW {last_rw}, FFP GW {last_ffp}"
        )

    # =================================================================
    # Save / Reset
    # =================================================================
    st.divider()
    col1, col2, _ = st.columns([1, 1, 4])

    with col1:
        if st.button("Save Settings", type="primary"):
            new_settings = {
                "version": 1,
                "discord": {
                    "mention_user_id": mention_user.strip(),
                    "mention_role_id": mention_role.strip(),
                },
                "deadline_alerts": {
                    "draft": {"enabled": draft_enabled, "offset_hours": draft_offset},
                    "classic": {"enabled": classic_enabled, "offset_hours": classic_offset},
                },
                "data_source_alerts": {
                    "rotowire": {"enabled": rotowire_enabled},
                    "ffp": {"enabled": ffp_enabled},
                },
                "alert_state": settings.get("alert_state", DEFAULT_SETTINGS["alert_state"]),
            }
            if save_settings(new_settings):
                st.success("Settings saved!")
            else:
                st.error("Failed to save settings.")

    with col2:
        if st.button("Reset to Defaults"):
            if save_settings(dict(DEFAULT_SETTINGS)):
                st.success("Settings reset to defaults! Refresh the page to see changes.")
            else:
                st.error("Failed to reset settings.")
