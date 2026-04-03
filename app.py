import os
import streamlit as st
from supabase import create_client, Client
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from calculations import compute_predictability

st.set_page_config(
    page_title="Sprint Predictability",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu          { visibility: hidden; }
footer             { visibility: hidden; }
header             { visibility: hidden; }

/* ── App background ── */
.stApp { background-color: #f8f9fb; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1e2a3a;
}
[data-testid="stSidebar"] * {
    color: #dce3ed !important;
}
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
    background-color: #2c3e55;
    color: #dce3ed !important;
    border: 1px solid #3d5166;
    border-radius: 6px;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover {
    background-color: #3d5166;
}

/* ── Selectbox dropdown options — white background needs dark text ── */
[data-baseweb="popover"] *,
[data-baseweb="menu"] *,
[data-baseweb="select"] ul li,
ul[role="listbox"] li {
    color: #1e2a3a !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover {
    background-color: #f0f4f8 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e6ea;
    border-radius: 8px;
    padding: 16px 20px;
}

/* ── Dividers ── */
hr { border-color: #e2e6ea; }

/* ── Subheaders ── */
h2 { color: #1e2a3a; }
h3 { color: #1e2a3a; }
</style>
""", unsafe_allow_html=True)

# ── Supabase client (one per browser session) ─────────────────────────────────
def get_supabase() -> Client:
    if "supabase_client" not in st.session_state:
        st.session_state["supabase_client"] = create_client(
            st.secrets["supabase_url"],
            st.secrets["supabase_anon_key"],
        )
    return st.session_state["supabase_client"]


# ── Session helpers ────────────────────────────────────────────────────────────
def restore_session() -> bool:
    if not st.session_state.get("access_token"):
        return False
    try:
        get_supabase().auth.set_session(
            st.session_state["access_token"],
            st.session_state["refresh_token"],
        )
        return True
    except Exception:
        clear_session()
        return False


def clear_session():
    for key in ["access_token", "refresh_token", "user_id", "user_email",
                "current_team_id", "current_team_name", "page", "supabase_client"]:
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token"))


def is_auth_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ["jwt expired", "invalid jwt", "token expired",
                                   "not authenticated", "session expired", "refresh token"])


# ── Auth ──────────────────────────────────────────────────────────────────────
def do_login(email: str, password: str):
    try:
        r = get_supabase().auth.sign_in_with_password({"email": email, "password": password})
        st.session_state["access_token"]  = r.session.access_token
        st.session_state["refresh_token"] = r.session.refresh_token
        st.session_state["user_id"]       = r.user.id
        st.session_state["user_email"]    = r.user.email
        st.session_state["page"]          = "teams"
        return None
    except Exception as e:
        return str(e)


def do_signup(email: str, password: str):
    try:
        get_supabase().auth.sign_up({"email": email, "password": password})
        return None, "Account created. Check your email to confirm before logging in."
    except Exception as e:
        return str(e), None


def do_logout():
    try:
        get_supabase().auth.sign_out()
    except Exception:
        pass
    clear_session()


def handle_password_recovery(token_hash: str = None, code: str = None,
                             access_token: str = None, refresh_token: str = None):
    st.title("Reset Your Password")

    # Establish the recovery session once and store it — avoids reusing one-time tokens on reruns.
    if "recovery_session" not in st.session_state:
        try:
            if token_hash:
                # Token-hash flow: verify the OTP token from the email template
                r = get_supabase().auth.verify_otp({"token_hash": token_hash, "type": "recovery"})
                st.session_state["recovery_session"] = {
                    "access_token":  r.session.access_token,
                    "refresh_token": r.session.refresh_token,
                }
            elif code:
                # PKCE flow: exchange the one-time code for a session
                r = get_supabase().auth.exchange_code_for_session({"auth_code": code})
                st.session_state["recovery_session"] = {
                    "access_token":  r.session.access_token,
                    "refresh_token": r.session.refresh_token,
                }
            elif access_token:
                # Implicit flow fallback
                get_supabase().auth.set_session(access_token, refresh_token or "")
                st.session_state["recovery_session"] = {
                    "access_token":  access_token,
                    "refresh_token": refresh_token or "",
                }
            else:
                st.error("Missing recovery credentials.")
                return
        except Exception:
            st.error("This reset link is invalid or has already been used. Please request a new one.")
            if st.button("Back to Login"):
                st.query_params.clear()
                st.rerun()
            return

    with st.form("reset_pw_form"):
        new_pw     = st.text_input("New Password",         type="password")
        confirm_pw = st.text_input("Confirm New Password", type="password")
        if st.form_submit_button("Set New Password", use_container_width=True):
            if not new_pw:
                st.warning("Please enter a password.")
            elif len(new_pw) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_pw != confirm_pw:
                st.error("Passwords do not match.")
            else:
                try:
                    sess = st.session_state["recovery_session"]
                    get_supabase().auth.set_session(sess["access_token"], sess["refresh_token"])
                    get_supabase().auth.update_user({"password": new_pw})
                    st.session_state.pop("recovery_session", None)
                    st.query_params.clear()
                    st.success("Password updated. You can now log in.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update password: {e}")


# ── Database ──────────────────────────────────────────────────────────────────
def db():
    return get_supabase()


def get_teams():
    return db().table("teams").select("*").eq("user_id", st.session_state["user_id"]).order("created_at").execute().data


def create_team(name: str):
    r = db().table("teams").insert({"user_id": st.session_state["user_id"], "name": name}).execute()
    team_id = r.data[0]["id"]
    db().table("team_config").insert({"team_id": team_id}).execute()
    return r.data[0]


def update_team(team_id: str, name: str):
    db().table("teams").update({"name": name}).eq("id", team_id).execute()


def delete_team(team_id: str):
    db().table("teams").delete().eq("id", team_id).execute()


DEFAULT_CONFIG = {
    "unit_of_work":              "Point",
    "analysis_mode":             "Rolling",
    "sprints_per_window":        5,
    "strong_threshold":          0.5,
    "moderate_threshold":        0.33,
    "needs_attention_threshold": 0.25,
    "conservative_percentile":   0.15,
    "trend_lookback":            5,
    "min_sprints_warning":       10,
}


def get_team_config(team_id: str) -> dict:
    r = db().table("team_config").select("*").eq("team_id", team_id).execute()
    if r.data:
        return {**DEFAULT_CONFIG, **r.data[0]}
    return {**DEFAULT_CONFIG, "team_id": team_id}


def save_team_config(team_id: str, data: dict):
    r = db().table("team_config").select("id").eq("team_id", team_id).execute()
    if r.data:
        db().table("team_config").update(data).eq("team_id", team_id).execute()
    else:
        db().table("team_config").insert({**data, "team_id": team_id}).execute()


def get_sprint_data(team_id: str):
    rows = db().table("sprint_data").select("*").eq("team_id", team_id).execute().data
    # sort_order (explicit user ordering) takes priority; fall back to date then created_at
    rows.sort(key=lambda s: (
        s.get("sort_order") is None,
        s.get("sort_order") or 0,
        s["sprint_date"] is None,
        s["sprint_date"] or "",
        s["created_at"],
    ))
    return rows


def replace_sprint_data(team_id: str, records: list):
    db().table("sprint_data").delete().eq("team_id", team_id).execute()
    if records:
        db().table("sprint_data").insert(records).execute()


def import_sprints(team_id: str, df: pd.DataFrame):
    records = []
    for _, row in df.iterrows():
        sprint_date = row.get("sprint_date")
        records.append({
            "team_id":          team_id,
            "sprint_name":      str(row["sprint_name"]),
            "sprint_date":      str(sprint_date) if pd.notna(sprint_date) else None,
            "completed_points": int(row["completed_points"]) if pd.notna(row.get("completed_points")) else 0,
            "completed_issues": int(row["completed_issues"]) if pd.notna(row.get("completed_issues")) else 0,
            "exclude":          str(row.get("exclude", "")).strip().lower() == "true",
        })
    db().table("sprint_data").insert(records).execute()


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_login():
    st.title("Sprint Predictability")
    st.write("Measure how consistently your team delivers sprint after sprint.")
    st.divider()

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email    = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log In", use_container_width=True):
                if email and password:
                    err = do_login(email, password)
                    if err:
                        st.error(f"Login failed: {err}")
                    else:
                        st.rerun()
                else:
                    st.warning("Please enter your email and password.")

        with st.expander("Forgot your password?"):
            with st.form("forgot_pw_form"):
                reset_email = st.text_input("Email address")
                if st.form_submit_button("Send Reset Email", use_container_width=True):
                    if reset_email.strip():
                        try:
                            get_supabase().auth.reset_password_for_email(
                                reset_email.strip(),
                                options={"redirect_to": st.secrets["app_url"]},
                            )
                            st.success("If an account exists with that email, a reset link has been sent.")
                        except Exception as e:
                            st.error(f"Failed to send reset email: {e}")
                    else:
                        st.warning("Please enter your email address.")

    with tab_signup:
        with st.form("signup_form"):
            email    = st.text_input("Email",            key="su_email")
            password = st.text_input("Password",         type="password", key="su_pw1")
            confirm  = st.text_input("Confirm Password", type="password", key="su_pw2")
            if st.form_submit_button("Create Account", use_container_width=True):
                if not email or not password:
                    st.warning("Please fill in all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    err, msg = do_signup(email, password)
                    if err:
                        st.error(f"Sign up failed: {err}")
                    else:
                        st.success(msg)


def get_team_summary(team_id: str) -> dict:
    """Return rating, avg ratio and trend for a team — used on the dashboard."""
    try:
        cfg     = get_team_config(team_id)
        sprints = get_sprint_data(team_id)
        unit    = cfg.get("unit_of_work", "Point")
        col_key = "completed_points" if unit == "Point" else "completed_issues"
        active  = [s for s in sprints if not s.get("exclude", False)]
        values  = [s.get(col_key) or 0 for s in active]

        if not values:
            return {"status": "no_data"}

        w_size = int(cfg.get("sprints_per_window", 5))
        if cfg.get("analysis_mode", "Rolling") == "Rolling" and len(values) < w_size:
            return {"status": "insufficient", "count": len(values), "needed": w_size}

        m = compute_predictability(values, cfg)
        return {
            "status":       "ok",
            "rating":       m.get("rating"),
            "avg_ratio":    m.get("avg_ratio"),
            "recent_trend": m.get("recent_trend"),
            "sprint_count": len(active),
        }
    except Exception:
        return {"status": "error"}


def page_teams():
    st.title("Your Teams")

    teams = get_teams()

    with st.expander("Add New Team", expanded=(len(teams) == 0)):
        with st.form("add_team"):
            name = st.text_input("Team Name")
            if st.form_submit_button("Add Team"):
                if name.strip():
                    create_team(name.strip())
                    st.success(f"Team '{name.strip()}' created.")
                    st.rerun()
                else:
                    st.warning("Please enter a team name.")

    if not teams:
        st.info("No teams yet. Add one above to get started.")
        return

    st.divider()

    for i, team in enumerate(teams):
        col_name, col_open, col_rename, col_delete = st.columns([5, 2, 2, 2])
        col_name.write(f"**{team['name']}**")

        summary = get_team_summary(team["id"])
        if summary["status"] == "ok":
            color      = RATING_COLORS.get(summary["rating"], "#999")
            trend_icon = TREND_ICONS.get(summary.get("recent_trend", ""), "—")
            trend_label = summary.get("recent_trend", "—")
            col_name.markdown(
                f'Rating: <span style="background:{color};color:white;padding:2px 7px;'
                f'border-radius:3px;font-size:0.75em;font-weight:bold">{summary["rating"]}</span>'
                f'&nbsp;&nbsp;Avg ratio: <b>{summary["avg_ratio"]:.0%}</b>'
                f'&nbsp;&nbsp;Trend: <b>{trend_icon} {trend_label}</b>'
                f'&nbsp;&nbsp;<span style="color:#888;font-size:0.85em">({summary["sprint_count"]} active sprints)</span>',
                unsafe_allow_html=True,
            )
        elif summary["status"] == "insufficient":
            col_name.caption(f'{summary["count"]} of {summary["needed"]} sprints needed for results')
        elif summary["status"] == "no_data":
            col_name.caption("No sprint data yet")

        if col_open.button("Open", key=f"open_{team['id']}"):
            st.session_state["current_team_id"]   = team["id"]
            st.session_state["current_team_name"] = team["name"]
            st.session_state["page"]              = "sprint_data"
            st.rerun()

        if col_rename.button("Rename", key=f"rename_{team['id']}"):
            st.session_state[f"renaming_{team['id']}"] = True

        if col_delete.button("Delete", key=f"delete_{team['id']}"):
            st.session_state[f"confirm_delete_{team['id']}"] = True

        if st.session_state.get(f"renaming_{team['id']}"):
            with st.form(f"rename_form_{team['id']}"):
                new_name  = st.text_input("New name", value=team["name"])
                c1, c2    = st.columns(2)
                saved     = c1.form_submit_button("Save")
                cancelled = c2.form_submit_button("Cancel")
            if saved:
                if new_name.strip():
                    update_team(team["id"], new_name.strip())
                    if st.session_state.get("current_team_id") == team["id"]:
                        st.session_state["current_team_name"] = new_name.strip()
                st.session_state.pop(f"renaming_{team['id']}", None)
                st.rerun()
            if cancelled:
                st.session_state.pop(f"renaming_{team['id']}", None)
                st.rerun()

        if st.session_state.get(f"confirm_delete_{team['id']}"):
            st.warning(f"Delete **{team['name']}**? This will also delete all sprint data for this team.")
            c1, c2 = st.columns(2)
            if c1.button("Yes, delete", key=f"yes_del_{team['id']}"):
                delete_team(team["id"])
                if st.session_state.get("current_team_id") == team["id"]:
                    st.session_state.pop("current_team_id",   None)
                    st.session_state.pop("current_team_name", None)
                st.session_state.pop(f"confirm_delete_{team['id']}", None)
                st.rerun()
            if c2.button("Cancel", key=f"no_del_{team['id']}"):
                st.session_state.pop(f"confirm_delete_{team['id']}", None)
                st.rerun()


def page_sprint_data():
    team_id   = st.session_state["current_team_id"]
    team_name = st.session_state.get("current_team_name", "Team")
    st.title(f"Sprint Data — {team_name}")

    sprints = get_sprint_data(team_id)

    if sprints:
        total    = len(sprints)
        excluded = sum(1 for s in sprints if s.get("exclude", False))
        active   = total - excluded
        st.caption(f"{total} sprints total — {active} active, {excluded} excluded")

    tab_data, tab_import = st.tabs(["Sprint Data", "Import CSV"])

    # ── Tab: Sprint Data ─────────────────────────────────────────────────────
    with tab_data:
        if sprints:
            df = pd.DataFrame(sprints)[
                ["sprint_name", "sprint_date", "completed_points", "completed_issues", "exclude"]
            ].copy()
            df["sprint_date"] = pd.to_datetime(df["sprint_date"], errors="coerce").dt.date
        else:
            df = pd.DataFrame(
                columns=["sprint_name", "sprint_date", "completed_points", "completed_issues", "exclude"]
            )

        df.columns = ["Sprint Name", "Sprint Date", "Completed Points", "Completed Issues", "Exclude"]

        # Prepend an Order column so users can control sort order explicitly.
        # When saved, rows are sorted by this column before inserting, so the
        # order is preserved on the next load (via created_at).
        df.insert(0, "Order", range(1, len(df) + 1))

        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Order":             st.column_config.NumberColumn(
                                         "Order", min_value=1, step=1,
                                         help="Change these numbers to reorder sprints. Lower numbers appear first."),
                "Sprint Name":       st.column_config.TextColumn("Sprint Name", required=True),
                "Sprint Date":       st.column_config.DateColumn("Sprint Date"),
                "Completed Points":  st.column_config.NumberColumn("Completed Points", min_value=0, step=1),
                "Completed Issues":  st.column_config.NumberColumn("Completed Issues", min_value=0, step=1),
                "Exclude":           st.column_config.CheckboxColumn("Exclude"),
            },
            key="sprint_editor",
        )

        if st.button("Save Changes", type="primary"):
            edited = edited.sort_values("Order", na_position="last").reset_index(drop=True)
            records = []
            for idx, (_, row) in enumerate(edited.iterrows()):
                name = row["Sprint Name"]
                if not name or (isinstance(name, float) and np.isnan(name)):
                    continue
                sprint_date = row["Sprint Date"]
                records.append({
                    "team_id":          team_id,
                    "sprint_name":      str(name),
                    "sprint_date":      str(sprint_date) if pd.notna(sprint_date) else None,
                    "completed_points": int(row["Completed Points"]) if pd.notna(row["Completed Points"]) else 0,
                    "completed_issues": int(row["Completed Issues"]) if pd.notna(row["Completed Issues"]) else 0,
                    "exclude":          bool(row["Exclude"]) if pd.notna(row["Exclude"]) else False,
                    "sort_order":       int(row["Order"]) if pd.notna(row["Order"]) else idx + 1,
                })
            replace_sprint_data(team_id, records)
            st.success("Sprint data saved.")
            st.rerun()

    # ── Tab: Import CSV ───────────────────────────────────────────────────────
    with tab_import:
        st.subheader("Import Sprints from CSV")
        st.markdown("""
**Expected columns** (column names must match exactly):

| sprint_name | sprint_date | completed_points | completed_issues | exclude |
|-------------|-------------|------------------|------------------|---------|
| Sprint 1    | 2024-01-15  | 42               | 8                | false   |
| Sprint 2    | 2024-01-29  | 38               | 7                | false   |

- `sprint_name` — required
- `sprint_date` — optional, format: YYYY-MM-DD
- `completed_points` — optional, whole number
- `completed_issues` — optional, whole number
- `exclude` — optional, true / false
        """)

        template = pd.DataFrame([
            {"sprint_name": "Sprint 1", "sprint_date": "2024-01-15",
             "completed_points": 42, "completed_issues": 8, "exclude": False},
            {"sprint_name": "Sprint 2", "sprint_date": "2024-01-29",
             "completed_points": 38, "completed_issues": 7, "exclude": False},
        ])
        st.download_button(
            "Download Template CSV",
            template.to_csv(index=False),
            "sprint_template.csv",
            "text/csv",
        )

        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            try:
                import_df = pd.read_csv(uploaded)
                import_df.columns = [c.strip().lower().replace(" ", "_") for c in import_df.columns]

                if "sprint_name" not in import_df.columns:
                    st.error("CSV must include a 'sprint_name' column.")
                else:
                    for col in ["sprint_date", "completed_points", "completed_issues"]:
                        if col not in import_df.columns:
                            import_df[col] = None
                    if "exclude" not in import_df.columns:
                        import_df["exclude"] = False

                    st.write(f"**Preview** ({len(import_df)} rows):")
                    st.dataframe(import_df.head(10), use_container_width=True)

                    replace_existing = st.checkbox(
                        "Replace all existing sprint data for this team", value=False
                    )

                    if st.button("Import", type="primary"):
                        if replace_existing:
                            db().table("sprint_data").delete().eq("team_id", team_id).execute()
                        import_sprints(team_id, import_df)
                        st.success(f"Imported {len(import_df)} sprint(s).")
                        st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")


def page_configuration():
    team_id   = st.session_state["current_team_id"]
    team_name = st.session_state.get("current_team_name", "Team")
    st.title(f"Configuration — {team_name}")

    # If reset was triggered on the previous run, apply default values to widget
    # states NOW — before any widgets are rendered — so they initialise correctly.
    if st.session_state.pop("cfg_reset_pending", False):
        st.session_state["cfg_unit"]         = "Point"
        st.session_state["cfg_mode"]         = "Rolling"
        st.session_state["cfg_window"]       = 5
        st.session_state["cfg_strong"]       = 0.5
        st.session_state["cfg_moderate"]     = 0.33
        st.session_state["cfg_needs"]        = 0.25
        st.session_state["cfg_conservative"] = 0.15
        st.session_state["cfg_trend"]        = 5
        st.session_state["cfg_min_warn"]     = 10

    cfg = get_team_config(team_id)

    # ── Analysis Settings ─────────────────────────────────────────────────────
    st.subheader("Analysis Settings")

    unit_of_work = st.selectbox(
        "Unit of Work",
        ["Point", "Issue"],
        index=0 if cfg.get("unit_of_work", "Point") == "Point" else 1,
        help="Which column from Sprint Data drives all calculations.",
        key="cfg_unit",
    )
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Rolling", "All"],
        index=0 if cfg.get("analysis_mode", "Rolling") == "Rolling" else 1,
        help="Rolling uses sliding windows. All treats every sprint as a single group.",
        key="cfg_mode",
    )

    # Sprints per Window is only relevant in Rolling mode
    if analysis_mode == "Rolling":
        sprints_per_window = st.number_input(
            "Sprints per Window",
            min_value=2, max_value=20,
            value=int(cfg.get("sprints_per_window", 5)),
            step=1,
            help="Larger = smoother but slower to react. Smaller = more responsive but more volatile.",
            key="cfg_window",
        )
    else:
        sprints_per_window = int(cfg.get("sprints_per_window", 5))
        st.caption("Sprints per Window is not used in All mode.")

    # ── Rating Thresholds ─────────────────────────────────────────────────────
    st.subheader("Rating Thresholds")
    st.caption("The avg predictability ratio is compared against these to assign a rating.")

    strong_threshold = st.number_input(
        "Strong Threshold",
        min_value=0.0, max_value=1.0,
        value=float(cfg.get("strong_threshold", 0.5)),
        step=0.01, format="%.2f",
        help="Ratio at or above this = Strong.",
        key="cfg_strong",
    )
    moderate_threshold = st.number_input(
        "Moderate Threshold",
        min_value=0.0, max_value=1.0,
        value=float(cfg.get("moderate_threshold", 0.33)),
        step=0.01, format="%.2f",
        help="Ratio at or above this (and below Strong) = Moderate.",
        key="cfg_moderate",
    )
    needs_attention_threshold = st.number_input(
        "Needs Attention Threshold",
        min_value=0.0, max_value=1.0,
        value=float(cfg.get("needs_attention_threshold", 0.25)),
        step=0.01, format="%.2f",
        help="Ratio at or above this (and below Moderate) = Needs Attention. Below this = Very Weak.",
        key="cfg_needs",
    )

    # ── Advanced Settings ─────────────────────────────────────────────────────
    st.subheader("Advanced Settings")

    conservative_percentile = st.number_input(
        "Conservative Percentile",
        min_value=0.05, max_value=0.5,
        value=float(cfg.get("conservative_percentile", 0.15)),
        step=0.01, format="%.2f",
        help="0.15 = 15th percentile. The team met or exceeded this level 85% of the time.",
        key="cfg_conservative",
    )

    # Trend Lookback is only meaningful in Rolling mode (All mode = 1 window, no trend)
    if analysis_mode == "Rolling":
        trend_lookback = st.number_input(
            "Trend Lookback (windows)",
            min_value=1, max_value=20,
            value=int(cfg.get("trend_lookback", 5)),
            step=1,
            help="How many windows back to compare when calculating the trend.",
            key="cfg_trend",
        )
    else:
        trend_lookback = int(cfg.get("trend_lookback", 5))
        st.caption("Trend Lookback is not used in All mode — trend analysis requires multiple windows.")

    min_sprints_warning = st.number_input(
        "Minimum Sprints Warning",
        min_value=1, max_value=50,
        value=int(cfg.get("min_sprints_warning", 10)),
        step=1,
        help="Show a warning if the sprint count falls below this number.",
        key="cfg_min_warn",
    )

    col_save, col_reset = st.columns([3, 1])

    if col_save.button("Save Configuration", type="primary", use_container_width=True):
        save_team_config(team_id, {
            "unit_of_work":              unit_of_work,
            "analysis_mode":             analysis_mode,
            "sprints_per_window":        sprints_per_window,
            "strong_threshold":          strong_threshold,
            "moderate_threshold":        moderate_threshold,
            "needs_attention_threshold": needs_attention_threshold,
            "conservative_percentile":   conservative_percentile,
            "trend_lookback":            trend_lookback,
            "min_sprints_warning":       min_sprints_warning,
        })
        st.success("Configuration saved.")

    if col_reset.button("Reset to Defaults", use_container_width=True):
        save_team_config(team_id, DEFAULT_CONFIG.copy())
        st.session_state["cfg_reset_pending"] = True
        st.rerun()


# ── Results helpers ────────────────────────────────────────────────────────────
RATING_COLORS = {
    "Strong":          "#2ecc71",
    "Moderate":        "#3498db",
    "Needs Attention": "#f39c12",
    "Very Weak":       "#e74c3c",
}

TREND_ICONS = {
    "Improving":          "↑",
    "Declining":          "↓",
    "Stable":             "→",
    "Not enough windows": "—",
}

RATING_EXPLANATIONS = {
    "Strong": (
        "This team delivers consistently. When they take on a workload, they reliably come close to "
        "completing it. Stakeholders can plan around this team's estimates with high confidence."
    ),
    "Moderate": (
        "This team delivers reasonably well but with some variation. Plans should include a small buffer. "
        "Confidence is reasonable but not high."
    ),
    "Needs Attention": (
        "This team's delivery is noticeably inconsistent. Estimates are frequently off, which makes "
        "planning difficult. It's worth investigating whether sprint sizing, scope changes, or other "
        "factors are driving the variance."
    ),
    "Very Weak": (
        "There is a significant gap between what this team takes on and what they complete. Plans should "
        "not rely on their estimates without a substantial buffer. The root cause should be investigated "
        "before making commitments based on this team's capacity."
    ),
}


def trend_text(recent: str, smoothed: str) -> str:
    parts = []
    messages = {
        "Improving": "The most recent window shows improvement compared to earlier — predictability is trending upward.",
        "Declining": "The most recent window shows a decline — predictability is trending downward. Monitor closely.",
        "Stable":    "Predictability has been stable — no significant change in recent windows.",
    }
    smoothed_messages = {
        "Improving": "The 3-window smoothed trend confirms improvement, suggesting a genuine sustained shift.",
        "Declining": "The 3-window smoothed trend also shows decline, suggesting this is not a one-off dip.",
        "Stable":    "The smoothed trend (3-window average) is also stable, reinforcing the overall picture.",
    }
    if recent in messages:
        parts.append(messages[recent])
    if smoothed in smoothed_messages:
        parts.append(smoothed_messages[smoothed])
    return " ".join(parts)


def page_results():
    team_id   = st.session_state["current_team_id"]
    team_name = st.session_state.get("current_team_name", "Team")
    st.title(f"Results — {team_name}")

    sprints = get_sprint_data(team_id)
    cfg     = get_team_config(team_id)

    if not sprints:
        st.info("No sprint data yet. Add sprints in the Sprint Data section.")
        return

    unit       = cfg.get("unit_of_work", "Point")
    col_key    = "completed_points" if unit == "Point" else "completed_issues"
    unit_label = "Points" if unit == "Point" else "Issues"

    active = [s for s in sprints if not s.get("exclude", False)]
    values = [s.get(col_key) or 0 for s in active]
    labels = [s.get("sprint_name", f"Sprint {i+1}") for i, s in enumerate(active)]

    if not values:
        st.warning("All sprints are excluded. Uncheck some in Sprint Data to see results.")
        return

    # Guard: not enough sprints for a full window — check before computing
    mode   = cfg.get("analysis_mode", "Rolling")
    w_size = int(cfg.get("sprints_per_window", 5))
    if mode == "Rolling" and len(values) < w_size:
        needed = w_size - len(values)
        st.info(
            f"Not enough data to calculate results yet. "
            f"You have {len(values)} active sprint(s). "
            f"Add {needed} more to reach the minimum of {w_size} required for one window."
        )
        return

    m = compute_predictability(values, cfg)

    rating       = m.get("rating") or "N/A"
    avg_ratio    = m.get("avg_ratio")
    most_recent  = m.get("most_recent_ratio")
    recent_trend = m.get("recent_trend") or "N/A"
    smooth_trend = m.get("smoothed_trend") or "N/A"
    warning      = m.get("data_volume_warning", "")

    # ── Summary cards ─────────────────────────────────────────────────────────
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)

    def summary_card(col, label, value, accent_color="#1e2a3a"):
        col.markdown(
            f'<div style="background:#ffffff;border:1px solid #e2e6ea;border-top:4px solid {accent_color};'
            f'border-radius:8px;padding:16px 20px;text-align:center;">'
            f'<div style="font-size:0.75em;color:#6c757d;margin-bottom:6px">{label}</div>'
            f'<div style="font-size:1.4em;font-weight:bold;color:#1e2a3a">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    color = RATING_COLORS.get(rating, "#999")
    summary_card(c1, "Predictability Rating", rating, accent_color=color)
    summary_card(c2, "Avg Ratio",         f"{avg_ratio:.2%}"   if avg_ratio   is not None else "—")
    summary_card(c3, "Most Recent Ratio", f"{most_recent:.2%}" if most_recent is not None else "—")
    trend_display = f"{TREND_ICONS.get(recent_trend, '—')} {recent_trend}"
    summary_card(c4, "Recent Trend", trend_display)

    if warning and "Warning" in warning:
        st.warning(warning)

    # ── What this means ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("What This Means")

    if rating in RATING_EXPLANATIONS:
        st.markdown(f"**Rating: {rating}**")
        st.write(RATING_EXPLANATIONS[rating])

    tt = trend_text(recent_trend, smooth_trend)
    if tt:
        st.markdown("**Trend**")
        st.write(tt)

    # ── How the ratio works ───────────────────────────────────────────────────
    with st.expander("How is the ratio calculated?"):
        pct        = int(cfg.get("conservative_percentile", 0.15) * 100)
        confidence = 100 - pct
        w_size     = cfg.get("sprints_per_window", 5)
        st.markdown(f"""
The predictability ratio answers: *"How much of what this team typically delivers can you count on?"*

For each rolling window of **{w_size} sprints**:
- **Typical delivery** = the median completed {unit_label.lower()} across the window
- **Conservative floor** = the {pct}th percentile — the team met or exceeded this level {confidence}% of the time
- **Ratio** = conservative floor ÷ typical delivery

A ratio of **0.80** means you can reliably count on 80% of what this team typically delivers.
A ratio of **0.30** means delivery is highly variable — you can only reliably count on 30%.

The overall rating is based on the **average ratio** across all windows:

| Rating | Threshold |
|---|---|
| Strong | ≥ {cfg.get("strong_threshold", 0.5):.0%} |
| Moderate | ≥ {cfg.get("moderate_threshold", 0.33):.0%} |
| Needs Attention | ≥ {cfg.get("needs_attention_threshold", 0.25):.0%} |
| Very Weak | below {cfg.get("needs_attention_threshold", 0.25):.0%} |
        """)

    # ── Chart ─────────────────────────────────────────────────────────────────
    windows = m.get("windows", [])
    if len(windows) > 1:
        st.divider()
        st.subheader("Predictability Ratio Over Time")

        w_size = int(cfg.get("sprints_per_window", 5))
        ratios = [w["ratio"] for w in windows]
        x      = [f"Window {w['window']}" for w in windows]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=ratios,
            mode="lines+markers",
            name="Predictability Ratio",
            line=dict(color="#3498db", width=2),
            marker=dict(size=6),
        ))

        for label, val, color in [
            ("Strong",          cfg.get("strong_threshold", 0.5),          "#2ecc71"),
            ("Moderate",        cfg.get("moderate_threshold", 0.33),        "#f39c12"),
            ("Needs Attention", cfg.get("needs_attention_threshold", 0.25), "#e74c3c"),
        ]:
            fig.add_hline(
                y=val, line_dash="dash", line_color=color,
                annotation_text=label, annotation_position="right",
            )

        fig.update_layout(
            yaxis=dict(
                tickformat=".0%",
                range=[0, max(1.05, max(ratios) + 0.05)],
                title="Ratio",
            ),
            xaxis=dict(title="Window"),
            height=400,
            margin=dict(r=130),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Statistical detail ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Statistical Detail")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Core Metrics**")
        rows_a = [
            ("Sprints in analysis",                       m["sprints_in_analysis"], "d"),
            ("Windows computed",                          len(windows),             "d"),
            (f"Avg typical {unit_label.lower()}/window",  m["avg_typical"],         ".1f"),
            ("Avg conservative floor",                    m["avg_conservative"],    ".1f"),
            ("Avg ratio",                                 m["avg_ratio"],           ".2%"),
            ("Min ratio",                                 m["min_ratio"],           ".2%"),
            ("Max ratio",                                 m["max_ratio"],           ".2%"),
            (f"Std dev of completed {unit_label.lower()}", m["std_dev"],            ".1f"),
        ]
        for label, val, fmt in rows_a:
            if val is not None:
                formatted = format(val, fmt) if fmt != "d" else str(val)
                st.write(f"{label}: **{formatted}**")

    with col_b:
        st.markdown("**Trend Detail**")
        lookback = cfg.get("trend_lookback", 5)
        rows_b = [
            ("Most recent window ratio",             m["most_recent_ratio"],   ".2%"),
            (f"Ratio {lookback} windows ago",        m["ratio_n_periods_ago"], ".2%"),
            ("Trend delta",                          m["trend_delta"],         "+.2%"),
            ("Recent trend",                         recent_trend,             None),
            ("Smoothed recent avg (last 3 windows)", m["recent_avg_ratio"],    ".2%"),
            ("Smoothed prior avg (previous 3)",      m["prior_avg_ratio"],     ".2%"),
            ("Smoothed trend",                       smooth_trend,             None),
        ]
        for label, val, fmt in rows_b:
            if val is not None:
                formatted = format(val, fmt) if fmt else str(val)
                st.write(f"{label}: **{formatted}**")

    # ── Window detail table ───────────────────────────────────────────────────
    if windows:
        st.divider()
        st.subheader("Window Detail")
        wdf = pd.DataFrame(windows)
        wdf.columns = ["Window", f"Typical {unit_label}", "Conservative Floor", "Ratio"]
        wdf["Ratio"]                 = wdf["Ratio"].map("{:.2%}".format)
        wdf[f"Typical {unit_label}"] = wdf[f"Typical {unit_label}"].map("{:.1f}".format)
        wdf["Conservative Floor"]   = wdf["Conservative Floor"].map("{:.1f}".format)
        st.dataframe(wdf, use_container_width=True, hide_index=True)


# ── Sidebar navigation ─────────────────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:
        st.write(f"Logged in as **{st.session_state.get('user_email', '')}**")
        st.divider()

        # ── Team switcher ──────────────────────────────────────────────────────
        teams = get_teams()
        if teams:
            team_names = [t["name"] for t in teams]
            team_ids   = [t["id"]   for t in teams]

            current_id  = st.session_state.get("current_team_id")
            current_idx = team_ids.index(current_id) if current_id in team_ids else 0

            selected_idx = st.selectbox(
                "Active Team",
                range(len(team_names)),
                format_func=lambda i: team_names[i],
                index=current_idx,
            )

            # Switch team if the ID or name changed (name can change after a rename)
            if (team_ids[selected_idx] != st.session_state.get("current_team_id") or
                    team_names[selected_idx] != st.session_state.get("current_team_name")):
                st.session_state["current_team_id"]   = team_ids[selected_idx]
                st.session_state["current_team_name"] = team_names[selected_idx]
                # Stay on the current page if it's a team page, otherwise go to Sprint Data
                if st.session_state.get("page") not in ("sprint_data", "configuration", "results", "teams"):
                    st.session_state["page"] = "sprint_data"
                st.rerun()

            # Auto-set team on first load if none selected
            if not st.session_state.get("current_team_id"):
                st.session_state["current_team_id"]   = team_ids[0]
                st.session_state["current_team_name"] = team_names[0]

        # ── Page navigation ────────────────────────────────────────────────────
        if st.session_state.get("current_team_id"):
            st.divider()
            for label, key in [
                ("Sprint Data",   "sprint_data"),
                ("Configuration", "configuration"),
                ("Results",       "results"),
            ]:
                if st.button(label, use_container_width=True):
                    st.session_state["page"] = key
                    st.rerun()

        # ── Manage Teams ───────────────────────────────────────────────────────
        st.divider()
        if st.button("Manage Teams", use_container_width=True):
            st.session_state["page"] = "teams"
            st.rerun()

        # ── PDF reference guide download ───────────────────────────────────────
        pdf_path = os.path.join(os.path.dirname(__file__), "Completion_Predictability_Guide.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download Reference Guide",
                    f.read(),
                    "Completion_Predictability_Guide.pdf",
                    "application/pdf",
                    use_container_width=True,
                )

        # ── Log out ────────────────────────────────────────────────────────────
        st.divider()
        if st.button("Log Out", use_container_width=True):
            do_logout()
            st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Handle password recovery — token_hash flow is primary (set via email template customisation)
    params = st.query_params
    if params.get("type") == "recovery":
        if "token_hash" in params:
            handle_password_recovery(token_hash=params["token_hash"])
            return
        if "code" in params:
            handle_password_recovery(code=params["code"])
            return
        if "access_token" in params:
            handle_password_recovery(
                access_token=params["access_token"],
                refresh_token=params.get("refresh_token", ""),
            )
            return

    if not is_authenticated():
        if not restore_session():
            page_login()
            return

    try:
        restore_session()
        show_sidebar()

        page    = st.session_state.get("page", "teams")
        team_id = st.session_state.get("current_team_id")

        if page == "teams":
            page_teams()
        elif page in ("sprint_data", "configuration", "results") and not team_id:
            st.warning("Please select a team first.")
            page_teams()
        elif page == "sprint_data":
            page_sprint_data()
        elif page == "configuration":
            page_configuration()
        elif page == "results":
            page_results()
        else:
            page_teams()

    except Exception as e:
        if is_auth_error(e):
            clear_session()
            st.error("Your session has expired. Please log in again.")
            page_login()
        else:
            raise


main()
