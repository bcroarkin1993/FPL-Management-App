# scripts/fpl/settings.py
#
# Streamlit Settings page for configuring all alert types.

import os
import requests
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
    st.caption(
        "Choose when to get reminded before each deadline. "
        "Draft waivers close 25.5h before the first kickoff; Classic transfers close 1.5h before."
    )

    all_window_options = [48, 24, 12, 6, 3, 1]

    dl = settings.get("deadline_alerts", {})
    draft_cfg = dl.get("draft", {})
    classic_cfg = dl.get("classic", {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Draft")
        st.caption("Deadline: 25.5h before kickoff")
        draft_enabled = st.toggle(
            "Enable Draft deadline alerts",
            value=draft_cfg.get("enabled", False),
            key="draft_enabled",
        )
        saved_draft_windows = draft_cfg.get("alert_windows", [24, 6, 1])
        draft_windows = st.multiselect(
            "Alert me before deadline",
            options=all_window_options,
            default=[w for w in saved_draft_windows if w in all_window_options],
            format_func=lambda h: f"{h}h before",
            key="draft_windows",
            help="Select which reminder intervals you want before the Draft deadline",
        )

    with col2:
        st.subheader("Classic")
        st.caption("Deadline: 1.5h before kickoff")
        classic_enabled = st.toggle(
            "Enable Classic deadline alerts",
            value=classic_cfg.get("enabled", False),
            key="classic_enabled",
        )
        saved_classic_windows = classic_cfg.get("alert_windows", [24, 6, 1])
        classic_windows = st.multiselect(
            "Alert me before deadline",
            options=all_window_options,
            default=[w for w in saved_classic_windows if w in all_window_options],
            format_func=lambda h: f"{h}h before",
            key="classic_windows",
            help="Select which reminder intervals you want before the Classic deadline",
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
                    from scripts.common.utils import get_current_gameweek
                    gw = get_current_gameweek()
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
                    from scripts.common.utils import get_current_gameweek
                    gw = get_current_gameweek()
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
    # Test Alerts
    # =================================================================
    st.header("Test Alerts")
    st.caption(
        "Send a test message to Discord for each alert type to verify your webhook and mention settings are working."
    )

    # Build mention string from current form values (unsaved is fine for testing)
    test_mention = ""
    if mention_user.strip():
        test_mention += f"<@{mention_user.strip()}> "
    if mention_role.strip():
        test_mention += f"<@&{mention_role.strip()}> "

    def _get_webhook() -> str:
        import config  # triggers dotenv load
        return os.getenv("DISCORD_WEBHOOK_URL", "")

    def _send_test(msg: str) -> bool:
        webhook = _get_webhook()
        if not webhook:
            st.error(
                "No `DISCORD_WEBHOOK_URL` found. "
                "Set it in your `.env` file or as a GitHub Actions secret."
            )
            return False
        try:
            resp = requests.post(webhook, json={"content": msg}, timeout=10)
            resp.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Failed to send: {e}")
            return False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Draft Deadline", key="test_draft"):
            windows_str = ", ".join(f"{w}h" for w in sorted(draft_windows, reverse=True)) if draft_windows else "none"
            msg = (
                f"{test_mention}\U0001f514 **[TEST]** FPL **Draft** deadline alert is working! "
                f"(reminders: {windows_str} before deadline)"
            )
            if _send_test(msg):
                st.success("Draft deadline test alert sent!")

        if st.button("Test Rotowire Alert", key="test_rotowire"):
            msg = f"{test_mention}\U0001f4ca **[TEST]** Rotowire data source alert is working!"
            if _send_test(msg):
                st.success("Rotowire test alert sent!")

    with col2:
        if st.button("Test Classic Deadline", key="test_classic"):
            windows_str = ", ".join(f"{w}h" for w in sorted(classic_windows, reverse=True)) if classic_windows else "none"
            msg = (
                f"{test_mention}\u23f0 **[TEST]** FPL **Classic** deadline alert is working! "
                f"(reminders: {windows_str} before deadline)"
            )
            if _send_test(msg):
                st.success("Classic deadline test alert sent!")

        if st.button("Test FFP Alert", key="test_ffp"):
            msg = f"{test_mention}\U0001f4ca **[TEST]** Fantasy Football Pundit data source alert is working!"
            if _send_test(msg):
                st.success("FFP test alert sent!")

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
                    "draft": {"enabled": draft_enabled, "alert_windows": sorted(draft_windows, reverse=True)},
                    "classic": {"enabled": classic_enabled, "alert_windows": sorted(classic_windows, reverse=True)},
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
