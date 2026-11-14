"""
MLPilot - Streamlit frontend (single file).

A read/operate dashboard for the deployed MLPilot FastAPI backend.
It contains NO ML logic: every number shown comes from a real API response.

Backend endpoints used (and nothing else):
    GET  /health                      public
    GET  /metrics                     public
    POST /predict                     public
    GET  /admin/remediation-history   admin  (X-Admin-Token)
    POST /admin/reload-model          admin  (X-Admin-Token)

Configuration (Streamlit secrets or environment variables):
    API_BASE_URL       optional, defaults to the deployed Render URL
    ADMIN_TOKEN        the backend's MLPILOT_ADMIN_TOKEN value (server-side only)
    ADMIN_PASSPHRASE   passphrase a person must enter to unlock admin features

Run:  streamlit run app.py
"""

from __future__ import annotations

import hmac
import html
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

APP_NAME = "MLPilot"
DEFAULT_API_URL = "https://mlpilot-api.onrender.com"

# Render free instances sleep when idle; the first request can take ~1 minute.
CONNECT_TIMEOUT = 10          # seconds
READ_TIMEOUT = 60             # seconds (cold start tolerant)
RELOAD_READ_TIMEOUT = 120
MONITORING_READ_TIMEOUT = 300     # model reload pulls artifacts from the registry
RETRY_DELAY = 2               # seconds between retries
CACHE_TTL_HEALTH = 20         # seconds
CACHE_TTL_ADMIN = 30          # seconds

MAX_UNLOCK_ATTEMPTS = 5
MAX_SAMPLES = 500

COLD_START_MESSAGE = (
    "Contacting the API... Render free instances can take up to a minute "
    "to wake from idle."
)

# Palette (charcoal + blue/cyan accents)
BG = "#0E1116"
PANEL = "#161B22"
BORDER = "#232B36"
TEXT = "#E6EDF3"
MUTED = "#8B98A9"
CYAN = "#22D3EE"
BLUE = "#3B82F6"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED = "#EF4444"
FONT = 'Inter, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

# The nine demand-forecasting features accepted by POST /predict.
# (Ranges follow src/pipeline/feature_engineering.py: pandas dayofweek 0-6,
# month 1-12, day 1-31, ISO week 1-53, non-negative demand values.)
FEATURE_SPECS: list[dict[str, Any]] = [
    {"name": "day_of_week", "label": "Day of week", "kind": "int", "min": 0, "max": 6,
     "default": 2, "help": "0 = Monday ... 6 = Sunday."},
    {"name": "month", "label": "Month", "kind": "int", "min": 1, "max": 12,
     "default": 6, "help": "Calendar month, 1-12."},
    {"name": "day_of_month", "label": "Day of month", "kind": "int", "min": 1, "max": 31,
     "default": 15, "help": "Calendar day, 1-31."},
    {"name": "week_of_year", "label": "Week of year", "kind": "int", "min": 1, "max": 53,
     "default": 24, "help": "ISO week number, 1-53."},
    {"name": "lag_1", "label": "Demand 1 day ago (lag_1)", "kind": "float", "min": 0.0,
     "default": 100.0, "help": "Daily demand one day before the forecast date."},
    {"name": "lag_7", "label": "Demand 7 days ago (lag_7)", "kind": "float", "min": 0.0,
     "default": 100.0, "help": "Daily demand seven days before the forecast date."},
    {"name": "lag_14", "label": "Demand 14 days ago (lag_14)", "kind": "float", "min": 0.0,
     "default": 100.0, "help": "Daily demand fourteen days before the forecast date."},
    {"name": "rolling_mean_7", "label": "7-day rolling mean", "kind": "float", "min": 0.0,
     "default": 100.0, "help": "Mean daily demand over the previous 7 days."},
    {"name": "rolling_mean_14", "label": "14-day rolling mean", "kind": "float", "min": 0.0,
     "default": 100.0, "help": "Mean daily demand over the previous 14 days."},
]

PAGES = [
    "Overview",
    "Predictions",
    "Monitoring",
    "AI Investigator",
    "Remediation History",
    "Model Management",
]



def _secret(name: str, default: Any = None) -> Any:
    """Read a value from Streamlit secrets, then the environment."""

    value = None

    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    if value in (None, ""):
        value = os.getenv(name)

    # Support the backend's existing admin-token environment variable.
    if value in (None, "") and name == "ADMIN_TOKEN":
        value = os.getenv("MLPILOT_ADMIN_TOKEN")

    return value if value not in (None, "") else default


def get_api_base_url() -> str:
    return str(_secret("API_BASE_URL", DEFAULT_API_URL)).strip().rstrip("/")


# =============================================================================
# 2. API CLIENT
# =============================================================================


@dataclass
class ApiResult:
    """Normalised outcome of one API call. Never contains credentials."""

    ok: bool
    status_code: int | None = None
    data: dict | None = None
    error: str | None = None
    # ok | timeout | connection | auth | client | server | unavailable | invalid
    kind: str = "ok"
    rtt_ms: float | None = None      # measured from this dashboard
    server_ms: float | None = None   # X-Process-Time-ms header from the API
    checked_at: datetime = field(default_factory=datetime.now)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_detail(body: Any, text: str) -> str:
    if isinstance(body, dict) and "detail" in body:
        detail = body["detail"]
        if isinstance(detail, str):
            return detail
        return json.dumps(detail, default=str)[:300]
    snippet = (text or "").strip()
    if snippet and not snippet.lstrip().startswith("<"):
        return snippet[:200]
    return "No error detail was returned by the API."


def _parse_response(
    response: requests.Response, rtt_ms: float, token: str | None
) -> ApiResult:
    server_ms = _to_float(response.headers.get("X-Process-Time-ms"))
    try:
        body = response.json()
    except ValueError:
        body = None

    code = response.status_code

    if response.ok:
        if isinstance(body, dict):
            return ApiResult(True, code, body, None, "ok", rtt_ms, server_ms)
        return ApiResult(
            False, code, None,
            "The API returned an unexpected (non-JSON-object) response.",
            "invalid", rtt_ms, server_ms,
        )

    detail = _extract_detail(body, response.text)

    if code == 401:
        kind, message = "auth", "The API rejected the admin token (HTTP 401)."
    elif code in (502, 503, 504) and body is None:
        kind = "unavailable"
        message = (
            f"The API is temporarily unavailable (HTTP {code}). "
            "The Render instance may still be starting up."
        )
    elif 400 <= code < 500:
        kind, message = "client", detail
    else:
        kind, message = "server", detail

    if token:
        message = message.replace(token, "[redacted]")
    return ApiResult(False, code, None, message, kind, rtt_ms, server_ms)


def _request(
    method: str,
    path: str,
    *,
    admin: bool = False,
    params: dict | None = None,
    json_body: dict | None = None,
    read_timeout: int = READ_TIMEOUT,
    retries: int = 1,
) -> ApiResult:
    """
    Perform one API call with timeouts and (optional) retries on timeouts or
    connection failures. The admin token is attached server-side only.
    """
    url = f"{get_api_base_url()}{path}"
    headers = {"Accept": "application/json"}
    token: str | None = None

    if admin:
        token = _secret("ADMIN_TOKEN")
        if not token:
            return ApiResult(
                ok=False, kind="auth",
                error="ADMIN_TOKEN is not configured in Streamlit secrets.",
            )
        token = str(token)
        headers["X-Admin-Token"] = token

    last: ApiResult | None = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            response = requests.request(
                method, url,
                headers=headers, params=params, json=json_body,
                timeout=(CONNECT_TIMEOUT, read_timeout),
            )
        except requests.exceptions.Timeout:
            last = ApiResult(
                ok=False, kind="timeout",
                error="The API did not respond before the timeout.",
            )
        except requests.exceptions.ConnectionError:
            last = ApiResult(
                ok=False, kind="connection",
                error="Could not connect to the API.",
            )
        except requests.exceptions.RequestException:
            return ApiResult(
                ok=False, kind="connection",
                error="An unexpected network error occurred.",
            )
        else:
            return _parse_response(
                response, (time.perf_counter() - started) * 1000, token
            )

        if attempt < retries:
            time.sleep(RETRY_DELAY)

    return last or ApiResult(ok=False, kind="connection", error="Request failed.")


# ---- session-scoped TTL cache (per browser session, never shared) -----------


def cached_call(key: str, ttl: int, fn: Callable[[], ApiResult]) -> ApiResult:
    store = st.session_state.setdefault("_api_cache", {})
    entry = store.get(key)
    now = time.time()
    if entry and now - entry[0] < ttl and entry[1].ok:
        return entry[1]
    result = fn()
    store[key] = (now, result)
    return result


def clear_api_cache() -> None:
    st.session_state["_api_cache"] = {}


# ---- endpoint wrappers ------------------------------------------------------


def get_health(force: bool = False) -> ApiResult:
    if force:
        st.session_state.get("_api_cache", {}).pop("health", None)
    return cached_call("health", CACHE_TTL_HEALTH, lambda: _request("GET", "/health"))


def get_metrics() -> ApiResult:
    """/metrics is not counted by the backend, so it is never cached."""
    result = _request("GET", "/metrics")
    if result.ok:
        record_metric_sample(result)
    return result


def api_predict(payload: dict[str, int | float]) -> ApiResult:
    return _request("POST", "/predict", json_body=payload, retries=1)


def get_remediation_history(limit: int) -> ApiResult:
    return cached_call(
        f"history:{limit}",
        CACHE_TTL_ADMIN,
        lambda: _request(
            "GET", "/admin/remediation-history",
            admin=True, params={"limit": limit},
        ),
    )


def api_reload_model() -> ApiResult:
    # No retry: a timed-out reload may already have happened server-side.
    return _request(
        "POST", "/admin/reload-model",
        admin=True, read_timeout=RELOAD_READ_TIMEOUT, retries=0,
    )

def api_run_monitoring() -> ApiResult:
    # No retry: monitoring downloads and processes the dataset.
    return _request(
        "POST",
        "/admin/monitoring/run",
        admin=True,
        read_timeout=MONITORING_READ_TIMEOUT,
        retries=0,
    )


def api_start_investigation() -> ApiResult:
    # Starting a cycle is intentionally user-triggered; never auto-run on page load.
    return _request(
        "POST", "/admin/investigator/start",
        admin=True, read_timeout=MONITORING_READ_TIMEOUT, retries=0,
    )


def api_get_investigation(thread_id: str) -> ApiResult:
    return _request(
        "GET", f"/admin/investigator/{thread_id}",
        admin=True, read_timeout=READ_TIMEOUT, retries=0,
    )


def api_approve_investigation(thread_id: str, approved: bool, feedback: str = "") -> ApiResult:
    # No retry: resuming a checkpoint is a state-changing operation.
    return _request(
        "POST", f"/admin/investigator/{thread_id}/approve",
        admin=True, json_body={"approved": approved, "feedback": feedback},
        read_timeout=MONITORING_READ_TIMEOUT, retries=0,
    )


# =============================================================================
# 3. ADMIN AUTHENTICATION (token stays server-side)
# =============================================================================


def admin_unlocked() -> bool:
    return st.session_state.get("admin_unlocked") is True


def render_admin_gate(feature: str) -> bool:
    """
    Returns True when admin features may be used. Fails closed: both
    ADMIN_TOKEN and ADMIN_PASSPHRASE must be configured. The token itself is
    never displayed or sent to the browser.
    """
    if not _secret("ADMIN_TOKEN"):
        st.warning(
            f"{feature} is disabled: `ADMIN_TOKEN` is not configured in "
            "Streamlit secrets."
        )
        return False

    passphrase = _secret("ADMIN_PASSPHRASE")
    if not passphrase:
        st.warning(
            f"{feature} is disabled: `ADMIN_PASSPHRASE` is not configured in "
            "Streamlit secrets. Admin access fails closed without it."
        )
        return False

    if admin_unlocked():
        return True

    attempts = st.session_state.get("unlock_attempts", 0)
    if attempts >= MAX_UNLOCK_ATTEMPTS:
        st.error("Too many failed attempts. Admin access is locked for this session.")
        return False

    st.info(f"{feature} requires administrator access.")
    with st.form("admin_unlock_form"):
        entered = st.text_input("Admin passphrase", type="password")
        submitted = st.form_submit_button("Unlock admin access")

    if submitted:
        if hmac.compare_digest(entered.encode("utf-8"), str(passphrase).encode("utf-8")):
            st.session_state["admin_unlocked"] = True
            st.session_state["unlock_attempts"] = 0
            st.rerun()
        else:
            st.session_state["unlock_attempts"] = attempts + 1
            remaining = MAX_UNLOCK_ATTEMPTS - st.session_state["unlock_attempts"]
            st.error(f"Incorrect passphrase. {remaining} attempt(s) remaining.")
    return False


# =============================================================================
# 4. THEME & UI COMPONENTS
# =============================================================================

CSS = """
<style>
html, body, .stApp, [class*="st-"] {
  font-family: Inter, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.stApp { background: #0E1116; color: #E6EDF3; }
[data-testid="stHeader"] { background: rgba(14, 17, 22, 0.85); }
[data-testid="stSidebar"] { background: #0B0E13; border-right: 1px solid #1F2733; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3, h4 { color: #E6EDF3; letter-spacing: -0.01em; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 0.4rem; }

.ml-brand { font-size: 22px; font-weight: 700; color: #E6EDF3; letter-spacing: -0.02em; }
.ml-brand span { color: #22D3EE; }
.ml-brand-sub { font-size: 11px; color: #8B98A9; text-transform: uppercase;
  letter-spacing: 0.08em; margin-bottom: 14px; }

.ml-page-title { font-size: 26px; font-weight: 600; margin: 0; color: #E6EDF3; }
.ml-page-sub { color: #8B98A9; font-size: 14px; margin: 2px 0 6px 0; }
.ml-source { font-size: 12px; color: #8B98A9; margin-bottom: 18px; }
.ml-source code { color: #22D3EE; background: #11161D; padding: 1px 5px; border-radius: 3px; }

.ml-card { background: #161B22; border: 1px solid #232B36; border-top: 3px solid #22D3EE;
  border-radius: 6px; padding: 14px 16px; min-height: 98px; margin-bottom: 12px; }
.ml-card.good { border-top-color: #22C55E; }
.ml-card.warn { border-top-color: #F59E0B; }
.ml-card.bad { border-top-color: #EF4444; }
.ml-card.neutral { border-top-color: #3B82F6; }
.ml-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
  color: #8B98A9; margin-bottom: 6px; }
.ml-value { font-size: 26px; font-weight: 600; color: #E6EDF3; line-height: 1.15;
  word-break: break-word; }
.ml-sub { font-size: 12px; color: #8B98A9; margin-top: 6px; }

.ml-pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px;
  border-radius: 999px; font-size: 12px; font-weight: 500; border: 1px solid; }
.ml-pill.good { color: #22C55E; border-color: #1F5F3A; background: rgba(34,197,94,0.10); }
.ml-pill.warn { color: #F59E0B; border-color: #6B4A12; background: rgba(245,158,11,0.10); }
.ml-pill.bad { color: #EF4444; border-color: #6B2222; background: rgba(239,68,68,0.10); }
.ml-pill.neutral { color: #3B82F6; border-color: #1E3A6B; background: rgba(59,130,246,0.10); }
.ml-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; display: inline-block; }

.ml-result { background: #161B22; border: 1px solid #1E3A4B; border-left: 4px solid #22D3EE;
  border-radius: 6px; padding: 18px 22px; margin: 8px 0 14px 0; }
.ml-result .ml-value { font-size: 40px; color: #22D3EE; }

[data-testid="stForm"] { background: #11161D; border: 1px solid #232B36; border-radius: 6px; }
div[data-testid="stExpander"] { border: 1px solid #232B36; border-radius: 6px; background: #11161D; }
hr { border-color: #232B36; }
</style>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def page_header(title: str, subtitle: str, source: str | None = None) -> None:
    st.markdown(
        f'<div class="ml-page-title">{_esc(title)}</div>'
        f'<div class="ml-page-sub">{_esc(subtitle)}</div>',
        unsafe_allow_html=True,
    )
    if source:
        st.markdown(
            f'<div class="ml-source">Data source: {source}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="ml-source"></div>', unsafe_allow_html=True)


def source_tags(*endpoints: str) -> str:
    return " ".join(f"<code>{_esc(e)}</code>" for e in endpoints)


def metric_card(label: str, value: str, sub: str = "", tone: str = "cyan") -> str:
    sub_html = f'<div class="ml-sub">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div class="ml-card {_esc(tone)}">'
        f'<div class="ml-label">{_esc(label)}</div>'
        f'<div class="ml-value">{_esc(value)}</div>'
        f"{sub_html}</div>"
    )


def cards_row(cards: list[tuple]) -> None:
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        column.markdown(metric_card(*card), unsafe_allow_html=True)


def pill(label: str, tone: str) -> str:
    return (
        f'<span class="ml-pill {_esc(tone)}"><span class="ml-dot"></span>'
        f"{_esc(label)}</span>"
    )


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# Newer Streamlit replaced use_container_width with width="stretch".
STRETCH: dict[str, Any] = (
    {"width": "stretch"}
    if _version_tuple(st.__version__) >= (1, 50)
    else {"use_container_width": True}
)


def show_table(df: pd.DataFrame) -> None:
    st.dataframe(df, hide_index=True, **STRETCH)


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, **STRETCH)


def style_fig(fig: go.Figure, title: str | None = None, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0, font=dict(size=14, color=TEXT)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#11161D",
        font=dict(color="#C9D1D9", family=FONT, size=12),
        margin=dict(l=48, r=24, t=56 if title else 24, b=44),
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="#1E2630", zeroline=False)
    fig.update_yaxes(gridcolor="#1E2630", zeroline=False)
    return fig


def show_error(result: ApiResult, context: str) -> None:
    """Translate an ApiResult failure into a clear, non-technical message."""
    detail = result.error or "Unknown error."
    kind = result.kind

    if kind == "timeout":
        st.error(
            f"{context}: the API did not respond in time. If the Render instance "
            "was idle it may still be waking up - wait a moment and try again."
        )
    elif kind == "connection":
        st.error(
            f"{context}: could not reach the API at `{get_api_base_url()}`. "
            "Check the URL (`API_BASE_URL`) and your network connection."
        )
    elif kind == "auth":
        st.error(
            f"{context}: admin authentication failed. {detail} "
            "Verify that `ADMIN_TOKEN` matches the backend's admin token."
        )
    elif kind == "unavailable":
        st.warning(f"{context}: {detail}")
    elif kind == "client":
        st.error(f"{context}: the request was rejected (HTTP {result.status_code}). {detail}")
    elif kind == "server":
        st.error(f"{context}: the API reported an error (HTTP {result.status_code}). {detail}")
    else:
        st.error(f"{context}: {detail}")


def fmt_number(value: Any, digits: int = 2, suffix: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{value:,.{digits}f}{suffix}"


def fmt_int(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{int(value):,}"


# =============================================================================
# 5. METRIC SAMPLING (client-side; the backend keeps no history)
# =============================================================================


def record_metric_sample(result: ApiResult) -> None:
    data = result.data or {}
    total = data.get("total_requests")
    avg = data.get("average_latency_ms")
    peak = data.get("max_latency_ms")
    values = (total, avg, peak)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        return

    samples: list[dict] = st.session_state.setdefault("metric_samples", [])
    now = datetime.now()
    if samples and (now - samples[-1]["ts"]).total_seconds() < 1.0:
        return

    samples.append(
        {
            "ts": now,
            "total_requests": float(total),
            "avg_ms": float(avg),
            "max_ms": float(peak),
            "rtt_ms": result.rtt_ms,
        }
    )
    del samples[:-MAX_SAMPLES]


def samples_frame() -> pd.DataFrame:
    samples = st.session_state.get("metric_samples", [])
    if not samples:
        return pd.DataFrame()
    df = pd.DataFrame(samples)
    df["delta"] = df["total_requests"].diff()
    df["reset"] = df["delta"] < 0
    # After a counter reset the new total is the count since the restart.
    df.loc[df["reset"], "delta"] = df.loc[df["reset"], "total_requests"]
    df["delta"] = df["delta"].fillna(0.0)
    return df


# =============================================================================
# 6. REMEDIATION RECORD HELPERS
# =============================================================================


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def approval_label(record: dict) -> str:
    approval = record.get("human_approval")
    if isinstance(approval, dict):
        if approval.get("approved") is True:
            return "Approved"
        if approval.get("approved") is False:
            return "Rejected"
    return "Unknown"


def flatten_record(record: dict) -> dict:
    evaluation = _dict(record.get("evaluation"))
    production = _dict(evaluation.get("production_metrics"))
    candidate = _dict(evaluation.get("candidate_metrics"))
    timestamp = pd.to_datetime(record.get("timestamp"), errors="coerce", utc=True)
    return {
        "timestamp": timestamp,
        "action": record.get("action_type") or "-",
        "decision": approval_label(record),
        "remediation_status": record.get("remediation_status") or "-",
        "evaluation_status": evaluation.get("status") or "-",
        "registered": evaluation.get("registered"),
        "production_mae": _num(production.get("MAE")),
        "candidate_mae": _num(candidate.get("MAE")),
        "production_rmse": _num(production.get("RMSE")),
        "candidate_rmse": _num(candidate.get("RMSE")),
        "root_cause": record.get("root_cause") or "",
    }


def fmt_timestamp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown time"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")


def history_records(result: ApiResult) -> list[dict]:
    raw = (result.data or {}).get("records", [])
    return [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []


def history_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([flatten_record(r) for r in records])


def truncate(text: str, length: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= length else text[: length - 1] + "..."


# =============================================================================
# 7. PAGES
# =============================================================================


def overall_status(health: ApiResult, metrics: ApiResult) -> tuple[str, str, str]:
    """Return (label, tone, detail) derived only from real API responses."""
    if not health.ok:
        return "Unreachable", "bad", health.error or "Health check failed."
    info = health.data or {}
    if info.get("status") != "healthy" or not info.get("model_loaded"):
        return "Outage", "bad", "API reports no loaded model."
    if not metrics.ok or not info.get("schema_loaded"):
        return "Degraded", "warn", "A non-critical check is failing."
    return "Operational", "good", "All checks passing."


# ---- Overview ---------------------------------------------------------------


def page_overview() -> None:
    page_header(
        "Overview",
        "Live API health, model status and request metrics",
        source_tags("GET /health", "GET /metrics"),
    )

    with st.spinner(COLD_START_MESSAGE):
        health = get_health()
        metrics = get_metrics()

    if not health.ok and not metrics.ok:
        show_error(health, "Health check")
        st.caption("Use **Refresh data** in the sidebar to retry.")
        return

    # ---- row 1: health / model / status
    if health.ok:
        info = health.data or {}
        healthy = info.get("status") == "healthy"
        health_card = (
            "API health",
            str(info.get("status", "unknown")).title(),
            f"HTTP {health.status_code}",
            "good" if healthy else "bad",
        )
        version = info.get("model_version")
        version_card = (
            "Production model version",
            f"v{version}" if version is not None else "-",
            "Reported by /health",
            "cyan" if version is not None else "warn",
        )
        loaded = info.get("model_loaded") is True
        model_card = (
            "Model status",
            "Loaded" if loaded else "Not loaded",
            "Feature schema loaded" if info.get("schema_loaded") else "Feature schema not loaded",
            "good" if loaded else "bad",
        )
    else:
        health_card = ("API health", "Unavailable", health.error or "", "bad")
        version_card = ("Production model version", "-", "Health check failed", "warn")
        model_card = ("Model status", "Unknown", "Health check failed", "warn")

    label, tone, detail = overall_status(health, metrics)
    status_card = ("Overall system status", label, detail, tone)
    cards_row([health_card, version_card, model_card, status_card])

    # ---- row 2: metrics
    if metrics.ok:
        data = metrics.data or {}
        cards_row(
            [
                ("Total API requests", fmt_int(data.get("total_requests")),
                 "Since last API restart", "cyan"),
                ("Average latency", fmt_number(data.get("average_latency_ms"), 2, " ms"),
                 "Server-side, all endpoints", "cyan"),
                ("Maximum latency", fmt_number(data.get("max_latency_ms"), 2, " ms"),
                 "Slowest request observed", "neutral"),
                ("API round-trip", fmt_number(metrics.rtt_ms, 0, " ms"),
                 "Measured from this dashboard", "neutral"),
            ]
        )
    else:
        show_error(metrics, "Metrics")

    st.markdown("### Service checks")
    left, right = st.columns([3, 2])

    with left:
        rows = []
        for endpoint, result in (("GET /health", health), ("GET /metrics", metrics)):
            rows.append(
                {
                    "Endpoint": endpoint,
                    "Status": "OK" if result.ok else "Error",
                    "HTTP": result.status_code if result.status_code else "-",
                    "Round-trip (ms)": round(result.rtt_ms, 1) if result.rtt_ms else None,
                    "Server time (ms)": result.server_ms,
                    "Checked": result.checked_at.strftime("%H:%M:%S"),
                    "Detail": "" if result.ok else (result.error or ""),
                }
            )
        show_table(pd.DataFrame(rows))
        st.caption(
            "Total requests counts all traffic except `/metrics`, including "
            "`/health` checks, and resets whenever the API restarts."
        )

    with right:
        if metrics.ok:
            data = metrics.data or {}
            avg, peak = _num(data.get("average_latency_ms")), _num(data.get("max_latency_ms"))
            if avg is not None and peak is not None:
                fig = go.Figure(
                    go.Bar(
                        x=["Average", "Maximum"],
                        y=[avg, peak],
                        marker_color=[CYAN, BLUE],
                        text=[f"{avg:,.1f} ms", f"{peak:,.1f} ms"],
                        textposition="outside",
                    )
                )
                style_fig(fig, "Server latency (ms)", 260)
                fig.update_layout(showlegend=False, hovermode="closest")
                show_chart(fig)


# ---- Predictions ------------------------------------------------------------


def _feature_input(spec: dict[str, Any]) -> int | float:
    key = f"feat_{spec['name']}"
    if spec["kind"] == "int":
        return st.number_input(
            spec["label"], min_value=spec["min"], max_value=spec["max"],
            value=spec["default"], step=1, help=spec["help"], key=key,
        )
    return st.number_input(
        spec["label"], min_value=spec["min"], value=spec["default"],
        step=1.0, format="%.2f", help=spec["help"], key=key,
    )


def page_predictions() -> None:
    page_header(
        "Predictions",
        "Forecast daily demand with the production model",
        source_tags("POST /predict"),
    )
    st.caption(
        "Pre-filled values are placeholders, not real data. Replace them with "
        "actual demand figures for the forecast date."
    )

    by_name = {spec["name"]: spec for spec in FEATURE_SPECS}
    values: dict[str, int | float] = {}

    with st.form("predict_form"):
        st.markdown("**Calendar features**")
        cols = st.columns(4)
        for col, name in zip(cols, ["day_of_week", "month", "day_of_month", "week_of_year"]):
            with col:
                values[name] = _feature_input(by_name[name])

        st.markdown("**Lag features**")
        cols = st.columns(3)
        for col, name in zip(cols, ["lag_1", "lag_7", "lag_14"]):
            with col:
                values[name] = _feature_input(by_name[name])

        st.markdown("**Rolling-mean features**")
        cols = st.columns(2)
        for col, name in zip(cols, ["rolling_mean_7", "rolling_mean_14"]):
            with col:
                values[name] = _feature_input(by_name[name])

        with st.expander("Advanced: choose which features to send"):
            st.caption(
                "The API validates the payload against the production model's "
                "stored feature schema. If the deployed model uses a subset of "
                "the nine features, deselect the unused ones."
            )
            selected = st.multiselect(
                "Features included in the request",
                options=[s["name"] for s in FEATURE_SPECS],
                default=[s["name"] for s in FEATURE_SPECS],
            )

        submitted = st.form_submit_button("Run forecast", type="primary")

    if submitted:
        if not selected:
            st.error("Select at least one feature to send.")
        else:
            payload = {
                name: (int(values[name]) if by_name[name]["kind"] == "int" else float(values[name]))
                for name in selected
            }
            with st.spinner(COLD_START_MESSAGE):
                result = api_predict(payload)

            if result.ok:
                data = result.data or {}
                prediction = _num(data.get("predicted_demand"))
                version = data.get("model_version")
                if prediction is None:
                    st.error("The API response did not contain a numeric `predicted_demand`.")
                else:
                    st.markdown(
                        '<div class="ml-result"><div class="ml-label">Predicted daily demand</div>'
                        f'<div class="ml-value">{prediction:,.2f}</div>'
                        f'<div class="ml-sub">Model version {_esc(version)}</div></div>',
                        unsafe_allow_html=True,
                    )
                    cards_row(
                        [
                            ("Model version", f"v{version}" if version is not None else "-",
                             "Served by the API", "cyan"),
                            ("Round-trip", fmt_number(result.rtt_ms, 0, " ms"),
                             "Measured from this dashboard", "neutral"),
                            ("Server processing", fmt_number(result.server_ms, 2, " ms"),
                             "X-Process-Time-ms header", "neutral"),
                        ]
                    )
                    log = st.session_state.setdefault("prediction_log", [])
                    log.append(
                        {
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Predicted demand": prediction,
                            "Model version": version,
                            **{k: v for k, v in payload.items()},
                        }
                    )
            else:
                show_error(result, "Prediction")
                if result.kind == "client" and result.status_code == 400:
                    st.caption(
                        "A 400 usually means the feature set differs from the production "
                        "model's schema. Adjust the selection under *Advanced*."
                    )

    log = st.session_state.get("prediction_log", [])
    if log:
        st.markdown("### Session prediction log")
        show_table(pd.DataFrame(log[::-1]))
        st.caption("Stored only in this browser session; not persisted anywhere.")
        if st.button("Clear log"):
            st.session_state["prediction_log"] = []
            st.rerun()


# ---- Monitoring -------------------------------------------------------------


def page_monitoring() -> int | None:
    page_header(
        "Monitoring",
        "API request metrics, data quality, drift and model performance",
        source_tags("GET /metrics", "POST /admin/monitoring/run"),
    )

    controls = st.columns([1, 1, 3])
    with controls[0]:
        auto = st.toggle("Auto-refresh", value=False)
    with controls[1]:
        interval = st.selectbox(
            "Interval",
            [5, 10, 30, 60],
            index=1,
            disabled=not auto,
            format_func=lambda s: f"{s}s",
        )

    with st.spinner(COLD_START_MESSAGE):
        metrics = get_metrics()

    if not metrics.ok:
        show_error(metrics, "Metrics")
        return interval if auto else None

    data = metrics.data or {}
    df = samples_frame()
    since_open = int(df["delta"].sum()) if not df.empty else 0

    cards_row(
        [
            ("Total API requests", fmt_int(data.get("total_requests")),
             "Since last API restart", "cyan"),
            ("Average latency", fmt_number(data.get("average_latency_ms"), 2, " ms"),
             "Server-side", "cyan"),
            ("Maximum latency", fmt_number(data.get("max_latency_ms"), 2, " ms"),
             "Server-side", "neutral"),
            ("Requests this session", fmt_int(since_open),
             f"{len(df)} sample(s) collected", "neutral"),
        ]
    )

    st.info(
        "The API exposes only live counters - it keeps no metric history. "
        "The charts below are built from samples this dashboard collected "
        "since this browser session opened."
    )

    if len(df) < 2:
        st.caption(
            "Collecting samples... press **Refresh data** or enable "
            "auto-refresh to build a trend."
        )
    else:
        left, right = st.columns(2)

        with left:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["ts"],
                    y=df["avg_ms"],
                    name="Average (server)",
                    mode="lines+markers",
                    line=dict(color=CYAN, width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["ts"],
                    y=df["max_ms"],
                    name="Maximum (server)",
                    mode="lines+markers",
                    line=dict(color=BLUE, width=2),
                )
            )
            if df["rtt_ms"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df["ts"],
                        y=df["rtt_ms"],
                        name="Round-trip (dashboard)",
                        mode="lines",
                        line=dict(color=MUTED, width=1.5, dash="dot"),
                    )
                )
            style_fig(fig, "Latency over this session (ms)")
            show_chart(fig)

        with right:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df["ts"],
                    y=df["delta"],
                    name="New requests per sample",
                    marker_color=BLUE,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["ts"],
                    y=df["total_requests"],
                    name="Cumulative total",
                    mode="lines",
                    yaxis="y2",
                    line=dict(color=CYAN, width=2),
                )
            )
            style_fig(fig, "Request volume over this session")
            fig.update_layout(
                yaxis=dict(title="Per sample"),
                yaxis2=dict(
                    title="Cumulative",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
            )
            show_chart(fig)

        resets = df[df["reset"]]
        if not resets.empty:
            when = ", ".join(resets["ts"].dt.strftime("%H:%M:%S"))
            st.warning(
                f"The API request counter reset at {when} - the backend "
                "was restarted or redeployed."
            )

    with st.expander("Collected samples"):
        if df.empty:
            st.caption("No samples yet.")
        else:
            table = df[
                ["ts", "total_requests", "avg_ms", "max_ms", "rtt_ms", "delta"]
            ].copy()
            table.columns = [
                "Time",
                "Total requests",
                "Avg latency (ms)",
                "Max latency (ms)",
                "Round-trip (ms)",
                "New requests",
            ]
            show_table(table.iloc[::-1])

            st.download_button(
                "Download samples (CSV)",
                table.to_csv(index=False).encode("utf-8"),
                file_name="mlpilot_metric_samples.csv",
                mime="text/csv",
            )

            if st.button("Clear samples"):
                st.session_state["metric_samples"] = []
                st.rerun()

    # ---------------------------------------------------------
    # Model and data monitoring
    # ---------------------------------------------------------

    st.markdown("### Data drift and model performance")

    st.caption(
        "Runs the monitoring pipeline against the historical dataset stored "
        "in Backblaze B2, using the currently loaded production model. "
        "This is a batch monitoring run, not live streaming monitoring."
    )

    admin_ok = render_admin_gate(
        "Data drift and model-performance monitoring"
    )

    if admin_ok:
        if st.button(
            "Run monitoring",
            type="primary",
            key="run_monitoring_button",
        ):
            with st.spinner(
                "Downloading monitoring dataset and running checks..."
            ):
                result = api_run_monitoring()

            if result.ok:
                st.session_state["latest_monitoring_report"] = result.data or {}
                st.success("Monitoring run completed successfully.")
            else:
                show_error(result, "Monitoring")

    report = st.session_state.get("latest_monitoring_report")

    if not report:
        st.info(
            "No monitoring report in this session yet. Unlock the admin "
            "controls and select **Run monitoring**."
        )
        return interval if auto else None

    st.caption("Latest monitoring report from this dashboard session.")

    status = report.get("monitoring_status", "UNKNOWN")
    quality = report.get("data_quality") or {}
    drift = report.get("data_drift") or {}
    performance = report.get("model_performance") or {}
    alerts = report.get("alerts") or {}

    status_color = (
        "success" if status == "HEALTHY"
        else "warning"
    )

    if status_color == "success":
        st.success(f"Monitoring status: {status}")
    else:
        st.warning(f"Monitoring status: {status}")

    cards_row(
        [
            (
                "Data quality",
                quality.get("quality_status", "Unknown"),
                f"{quality.get('total_records', 0)} records checked",
                "success" if quality.get("quality_status") == "HEALTHY" else "warning",
            ),
            (
                "Data drift",
                "Detected" if drift.get("drift_detected") else "Not detected",
                "Feature distribution comparison",
                "warning" if drift.get("drift_detected") else "success",
            ),
            (
                "Performance",
                "Degraded" if performance.get("performance_degraded") else "Not degraded",
                "Reference vs current MAE",
                "warning" if performance.get("performance_degraded") else "success",
            ),
            (
                "Model version",
                str(report.get("model_version", "Unknown")),
                report.get("model_name", "Unknown model"),
                "neutral",
            ),
        ]
    )

    # Data quality
    st.markdown("#### Data quality")

    issues = quality.get("issues") or []
    if issues:
        st.warning("Data quality issues were reported.")
        if isinstance(issues, list):
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.json(issues)
    else:
        st.success("No data quality issues reported.")

    # Drift results
    st.markdown("#### Feature drift results")

    feature_results = drift.get("feature_results") or {}

    if feature_results:
        drift_rows = []
        for feature, values in feature_results.items():
            p_value = values.get("p_value")
            drift_rows.append(
                {
                    "Feature": feature,
                    "KS statistic": values.get("ks_statistic"),
                    "P-value": (
                        "< 0.0001"
                        if p_value == 0
                        else p_value
                    ),
                    "Drift detected": values.get("drift_detected"),
                }
            )

        show_table(pd.DataFrame(drift_rows))
    else:
        st.caption("No feature-level drift results available.")

    # Model performance
    st.markdown("#### Model performance")

    perf_cols = st.columns(3)

    with perf_cols[0]:
        st.metric(
            "Reference MAE",
            fmt_number(performance.get("reference_mae"), 2),
        )

    with perf_cols[1]:
        st.metric(
            "Current MAE",
            fmt_number(performance.get("current_mae"), 2),
        )

    with perf_cols[2]:
        st.metric(
            "Error change",
            fmt_number(
                performance.get("error_increase_percentage"),
                2,
                "%",
            ),
            help="Negative means error decreased; positive means error increased.",
        )

    # Alerts
    st.markdown("#### Monitoring alerts")

    alert_labels = {
        "data_quality_issues": "Data quality issues",
        "data_drift_detected": "Data drift detected",
        "performance_degraded": "Model performance degraded",
    }

    for key, label in alert_labels.items():
        if alerts.get(key):
            st.warning(label)
        else:
            st.success(f"No alert: {label}")

    # Source information
    source = report.get("monitoring_source") or {}

    with st.expander("Monitoring run details"):
        st.write(f"**Model:** {report.get('model_name', 'Unknown')}")
        st.write(f"**Model version:** {report.get('model_version', 'Unknown')}")
        st.write(f"**Storage:** {source.get('storage', 'Unknown')}")
        st.write(f"**Dataset:** {source.get('object_key', 'Unknown')}")

    return interval if auto else None


# ---- AI Investigator --------------------------------------------------------


def _render_investigation(record: dict, expanded: bool) -> None:
    flat = flatten_record(record)
    title = (
        f"{fmt_timestamp(flat['timestamp'])}  |  {flat['action']}  |  {flat['decision']}"
    )
    with st.expander(title, expanded=expanded):
        st.markdown(
            pill(flat["decision"], {"Approved": "good", "Rejected": "bad"}.get(flat["decision"], "neutral"))
            + " &nbsp; "
            + pill(f"action: {flat['action']}", "neutral")
            + " &nbsp; "
            + pill(f"evaluation: {flat['evaluation_status']}", "neutral"),
            unsafe_allow_html=True,
        )

        st.markdown("**Root cause**")
        st.write(record.get("root_cause") or "Not recorded.")

        st.markdown("**Recommended action**")
        st.write(record.get("recommended_action") or "Not recorded.")

        parameters = record.get("parameters")
        if isinstance(parameters, dict) and parameters:
            st.markdown("**Action parameters**")
            st.json(parameters)

        st.markdown("**Outcome**")
        st.write(record.get("remediation_status") or "Not recorded.")

        evaluation = _dict(record.get("evaluation"))
        production = _dict(evaluation.get("production_metrics"))
        candidate = _dict(evaluation.get("candidate_metrics"))
        if production or candidate:
            st.markdown("**Candidate evaluation**")
            show_table(
                pd.DataFrame(
                    {
                        "Metric": ["MAE", "RMSE"],
                        "Production": [_num(production.get("MAE")), _num(production.get("RMSE"))],
                        "Candidate": [_num(candidate.get("MAE")), _num(candidate.get("RMSE"))],
                    }
                )
            )
        if evaluation.get("reason"):
            st.caption(f"Evaluation note: {evaluation['reason']}")
        if evaluation.get("registered") is not None:
            st.caption(f"Candidate registered: {'yes' if evaluation['registered'] else 'no'}")


def page_investigator() -> None:
    page_header(
        "AI Investigator",
        "Run investigations, inspect findings and control remediation through explicit approvals",
        source_tags(
            "POST /admin/investigator/start",
            "GET /admin/investigator/{thread_id}",
            "POST /admin/investigator/{thread_id}/approve",
        ),
    )

    if not render_admin_gate("AI Investigator controls"):
        return

    st.caption(
        "Starting an investigation runs monitoring against the current production model. "
        "Nothing is retrained or deployed unless you explicitly approve each separate stage."
    )

    saved_thread = st.session_state.get("investigator_thread_id", "")
    thread_input = st.text_input(
        "Investigation thread ID",
        value=saved_thread,
        key="investigator_thread_id_input",
        help="The ID is returned when an investigation starts. Paste an earlier ID here to resume viewing it.",
    )
    col_start, col_load = st.columns(2)
    with col_start:
        if st.button("Start / find active investigation", type="primary"):
            with st.spinner("Running monitoring and starting the Investigator..."):
                result = api_start_investigation()
            if result.ok:
                data = result.data or {}
                thread_id = data.get("thread_id") or st.session_state.get("investigator_thread_id")
                if thread_id:
                    st.session_state["investigator_thread_id"] = thread_id
                    st.session_state["investigator_thread_id_input"] = thread_id
                clear_api_cache()
                if data.get("status") == "already_pending":
                    st.info("An earlier investigation is already waiting for approval. Its thread has been loaded.")
                elif data.get("status") == "already_starting":
                    st.warning("An investigation is already starting. Refresh its status in a moment.")
                else:
                    st.success(f"Investigation started: {data.get('status', 'status unavailable')}")
                if data.get("root_cause"):
                    st.write("**Initial root cause:**", data.get("root_cause"))
                if data.get("recommended_action"):
                    st.write("**Recommended action:**", data.get("recommended_action"))
                st.rerun()
            else:
                show_error(result, "Start investigation")

    with col_load:
        if st.button("Load thread / refresh status"):
            if thread_input.strip():
                st.session_state["investigator_thread_id"] = thread_input.strip()
                st.rerun()
            else:
                st.warning("Enter an investigation thread ID first.")

    thread_id = st.session_state.get("investigator_thread_id") or thread_input.strip()
    if not thread_id:
        st.info("No investigation selected. Start a cycle or enter an existing thread ID.")
        return

    st.divider()
    st.markdown("### Investigation status")
    with st.spinner("Loading the latest workflow state..."):
        state_result = api_get_investigation(thread_id)

    if not state_result.ok:
        show_error(state_result, "Investigation status")
        st.caption("If this is an older thread, verify that its ID is correct.")
        return

    data = state_result.data or {}
    status = str(data.get("status") or "unknown")
    pending_nodes = data.get("pending_nodes") or []
    status_tone = (
        "warn" if status.startswith("awaiting_")
        else "good" if status in ("completed", "deployed")
        else "bad" if status in ("failed", "rolled_back")
        else "neutral"
    )
    st.markdown(pill(status.replace("_", " ").title(), status_tone), unsafe_allow_html=True)
    st.code(thread_id)

    report = data.get("monitoring_report") or {}
    if report:
        quality = report.get("data_quality") or {}
        drift = report.get("data_drift") or {}
        performance = report.get("model_performance") or {}
        cards_row([
            ("Monitoring", report.get("monitoring_status", "Unknown"), "Latest monitoring run", "good" if report.get("monitoring_status") == "HEALTHY" else "warn"),
            ("Data quality", quality.get("quality_status", "Unknown"), f"{quality.get('total_records', 0)} records", "good" if quality.get("quality_status") == "HEALTHY" else "warn"),
            ("Data drift", "Detected" if drift.get("drift_detected") else "Not detected", "Feature distribution", "warn" if drift.get("drift_detected") else "good"),
            ("Performance", "Degraded" if performance.get("performance_degraded") else "Not degraded", "Reference vs current MAE", "warn" if performance.get("performance_degraded") else "good"),
        ])
        with st.expander("Monitoring report details"):
            st.json(report)

    st.markdown("### Investigator findings")
    st.markdown("**Root cause**")
    st.write(data.get("root_cause") or "Not available yet.")
    st.markdown("**Recommended action**")
    st.write(data.get("recommended_action") or "No recommendation recorded.")
    action = data.get("remediation_action") or {}
    if action:
        st.markdown("**Proposed remediation parameters**")
        st.json(action)

    retraining = data.get("retraining_result") or {}
    if retraining:
        st.markdown("### Candidate evaluation")
        st.write("Candidate model version:", retraining.get("model_version", "Not registered"))
        production = retraining.get("production_metrics") or {}
        candidate = retraining.get("candidate_metrics") or {}
        if production or candidate:
            metrics = ["MAE", "RMSE"]
            show_table(pd.DataFrame({
                "Metric": metrics,
                "Production": [_num(production.get(k)) for k in metrics],
                "Candidate": [_num(candidate.get(k)) for k in metrics],
            }))
        if retraining.get("evaluation_reason"):
            st.caption(retraining.get("evaluation_reason"))

    deployment = data.get("deployment_result") or {}
    if deployment:
        st.markdown("### Deployment result")
        st.json(deployment)
    if data.get("deployment_status"):
        st.write("Deployment status:", data.get("deployment_status"))
    if data.get("remediation_status"):
        st.write("Remediation status:", data.get("remediation_status"))

    awaiting_remediation = status == "awaiting_remediation_approval" or "human_approval" in pending_nodes
    awaiting_deployment = status == "awaiting_deployment_approval" or "deployment_approval" in pending_nodes

    if awaiting_remediation or awaiting_deployment:
        stage = "remediation" if awaiting_remediation else "deployment"
        st.divider()
        st.markdown("### Human approval required")
        if awaiting_remediation:
            st.warning(
                "Stage 1 — Remediation approval. Approving allows the whitelisted remediation "
                "and candidate evaluation/registration. It does not deploy to production."
            )
        else:
            pending_approval = data.get("pending_approval") or {}
            st.warning(
                "Stage 2 — Deployment approval. Approving promotes the evaluated candidate "
                "to production. Rejecting leaves the current production model unchanged."
            )
            st.write("Candidate version:", pending_approval.get("candidate_version", "Unknown"))
            candidate_metrics = pending_approval.get("candidate_metrics") or {}
            production_metrics = pending_approval.get("production_metrics") or {}
            if candidate_metrics or production_metrics:
                show_table(pd.DataFrame({
                    "Metric": ["MAE", "RMSE"],
                    "Production": [_num(production_metrics.get("MAE")), _num(production_metrics.get("RMSE"))],
                    "Candidate": [_num(candidate_metrics.get("MAE")), _num(candidate_metrics.get("RMSE"))],
                }))

        feedback = st.text_area(
            "Approval / rejection feedback (optional)",
            key=f"investigator_feedback_{stage}",
        )
        confirm = st.checkbox(
            "I have reviewed the findings and understand the effect of this decision.",
            key=f"investigator_confirm_{thread_id}_{stage}",
        )
        approve_col, reject_col = st.columns(2)
        with approve_col:
            if st.button("Approve " + stage, type="primary", disabled=not confirm, key=f"approve_{thread_id}_{stage}"):
                with st.spinner("Resuming the approved workflow stage..."):
                    decision = api_approve_investigation(thread_id, True, feedback)
                if decision.ok:
                    st.success(f"Approval recorded. New status: {(decision.data or {}).get('status', 'updated')}")
                    st.rerun()
                else:
                    show_error(decision, "Approval")
        with reject_col:
            if st.button("Reject " + stage, disabled=not confirm, key=f"reject_{thread_id}_{stage}"):
                with st.spinner("Recording rejection..."):
                    decision = api_approve_investigation(thread_id, False, feedback)
                if decision.ok:
                    st.success(f"Rejection recorded. New status: {(decision.data or {}).get('status', 'updated')}")
                    st.rerun()
                else:
                    show_error(decision, "Rejection")
    else:
        st.info("No approval is currently pending for this investigation.")


# ---- Remediation History ----------------------------------------------------


def page_remediation_history() -> None:
    page_header(
        "Remediation History",
        "Audit trail of remediation actions, approvals and candidate evaluations",
        source_tags("GET /admin/remediation-history"),
    )

    if not render_admin_gate("Remediation history"):
        return

    limit = st.number_input("Records to load (max 200)", min_value=1, max_value=200, value=50, step=10)

    with st.spinner(COLD_START_MESSAGE):
        result = get_remediation_history(int(limit))

    if not result.ok:
        show_error(result, "Remediation history")
        return

    records = history_records(result)
    data = result.data or {}

    if not records:
        st.info("No remediation records have been stored yet.")
        return

    df = history_frame(records)
    cards_row(
        [
            ("Total records", fmt_int(data.get("total_records")), "In the database", "cyan"),
            ("Returned", fmt_int(data.get("returned_records")), "In this view", "neutral"),
            ("Approved", fmt_int((df["decision"] == "Approved").sum()), "Human decision", "good"),
            ("Rejected", fmt_int((df["decision"] == "Rejected").sum()), "Human decision", "warn"),
        ]
    )

    left, right = st.columns(2)
    with left:
        counts = df["action"].value_counts()
        fig = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), marker_color=CYAN))
        style_fig(fig, "Records by action type", 300)
        fig.update_layout(showlegend=False, hovermode="closest")
        show_chart(fig)

    with right:
        chart_df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        has_mae = chart_df[["production_mae", "candidate_mae"]].notna().any(axis=None)
        if has_mae:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["production_mae"],
                                     name="Production MAE", mode="lines+markers",
                                     line=dict(color=BLUE, width=2)))
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["candidate_mae"],
                                     name="Candidate MAE", mode="lines+markers",
                                     line=dict(color=CYAN, width=2)))
            style_fig(fig, "Evaluation MAE: production vs candidate", 300)
            show_chart(fig)
        else:
            st.info("No evaluation metrics are present in the loaded records.")

    st.markdown("### Records")
    table = pd.DataFrame(
        {
            "Timestamp": df["timestamp"].apply(fmt_timestamp),
            "Action": df["action"],
            "Decision": df["decision"],
            "Evaluation": df["evaluation_status"],
            "Registered": df["registered"].apply(
                lambda v: "-" if v is None or (isinstance(v, float) and pd.isna(v)) else ("Yes" if v else "No")
            ),
            "Prod MAE": df["production_mae"],
            "Cand MAE": df["candidate_mae"],
            "Status": df["remediation_status"].apply(lambda s: truncate(s, 70)),
            "Root cause": df["root_cause"].apply(truncate),
        }
    )
    show_table(table)
    st.download_button(
        "Download table (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="mlpilot_remediation_history.csv",
        mime="text/csv",
    )

    st.markdown("### Record detail")
    options = list(range(len(records)))
    chosen = st.selectbox(
        "Select a record",
        options,
        format_func=lambda i: f"{fmt_timestamp(df.loc[i, 'timestamp'])} - {df.loc[i, 'action']} - {df.loc[i, 'decision']}",
    )
    st.json(records[chosen])


# ---- Model Management -------------------------------------------------------


def _latest_evaluation(records: list[dict]) -> tuple[dict, dict, dict, dict] | None:
    """Newest record containing evaluation metrics (records are newest first)."""
    for record in records:
        evaluation = _dict(record.get("evaluation"))
        production = _dict(evaluation.get("production_metrics"))
        candidate = _dict(evaluation.get("candidate_metrics"))
        if production or candidate:
            return record, evaluation, production, candidate
    return None


CAPABILITIES = [
    ("Live model version and load status", "Implemented", "GET /health"),
    ("Request and latency counters", "Implemented", "GET /metrics"),
    ("Single-row demand prediction", "Implemented", "POST /predict"),
    ("Reload the production model", "Implemented (admin)", "POST /admin/reload-model"),
    ("Remediation and evaluation history", "Implemented (admin)", "GET /admin/remediation-history"),
    ("Model registry version list and per-version metrics", "Requires backend support",
     "No endpoint exists"),
    ("Promote a candidate / roll back production", "Requires backend support",
     "Pipeline modules exist in the repository but are not exposed by the API"),
    ("Inspect the active feature schema", "Requires backend support", "Not returned by any endpoint"),
    ("Drift and model-performance monitoring", "Requires backend support",
     "Monitoring pipeline runs offline; no endpoint"),
    ("Start or approve an AI investigation", "Requires backend support",
     "Agent runs in the backend codebase; no endpoint"),
]


def page_model_management() -> None:
    page_header(
        "Model Management",
        "Production model information and administrative controls",
        source_tags("GET /health", "POST /admin/reload-model", "GET /admin/remediation-history"),
    )

    # ---- live model info
    st.markdown("### Production model")
    with st.spinner(COLD_START_MESSAGE):
        health = get_health()

    if health.ok:
        info = health.data or {}
        version = info.get("model_version")
        loaded = info.get("model_loaded") is True
        cards_row(
            [
                ("Serving version", f"v{version}" if version is not None else "-",
                 "Reported by /health", "cyan"),
                ("Model status", "Loaded" if loaded else "Not loaded", "In API memory",
                 "good" if loaded else "bad"),
                ("Feature schema", "Loaded" if info.get("schema_loaded") else "Not loaded",
                 "Validates /predict inputs", "good" if info.get("schema_loaded") else "warn"),
                ("API status", str(info.get("status", "unknown")).title(), f"HTTP {health.status_code}",
                 "good" if info.get("status") == "healthy" else "bad"),
            ]
        )
    else:
        show_error(health, "Model information")

    # ---- reference evaluation metrics (admin)
    st.markdown("### Latest candidate evaluation")
    st.caption(
        "Reference metrics from the most recent remediation run that evaluated a candidate "
        "against production. These are offline evaluation results, not live monitoring."
    )

    admin_ok = render_admin_gate("Evaluation history and model reload")

    if admin_ok:
        history = get_remediation_history(50)
        if not history.ok:
            show_error(history, "Evaluation history")
        else:
            latest = _latest_evaluation(history_records(history))
            if latest is None:
                st.info("No stored remediation record contains evaluation metrics yet.")
            else:
                record, evaluation, production, candidate = latest
                st.caption(
                    f"From the record dated {fmt_timestamp(flatten_record(record)['timestamp'])} "
                    f"(action: {record.get('action_type') or '-'}, "
                    f"evaluation: {evaluation.get('status') or '-'})."
                )
                show_table(
                    pd.DataFrame(
                        {
                            "Metric": ["MAE", "RMSE"],
                            "Production": [_num(production.get("MAE")), _num(production.get("RMSE"))],
                            "Candidate": [_num(candidate.get("MAE")), _num(candidate.get("RMSE"))],
                        }
                    )
                )
                if evaluation.get("reason"):
                    st.caption(f"Evaluation note: {evaluation['reason']}")

        # ---- protected reload
        st.markdown("### Administrative actions")
        with st.container(border=True):
            st.markdown("**Reload production model**")
            st.write(
                "Asks the API to re-read the `production` alias from the model registry. The "
                "serving model is replaced only if the registered production version has "
                "changed; otherwise nothing happens."
            )
            confirmed = st.checkbox("I understand this reloads the model currently serving predictions.")
            if st.button("Reload model", type="primary", disabled=not confirmed):
                before = get_health(force=True)
                previous = (before.data or {}).get("model_version") if before.ok else None

                with st.spinner("Reloading the model - this can take a minute..."):
                    result = api_reload_model()

                log = st.session_state.setdefault("model_action_log", [])
                if result.ok:
                    data = result.data or {}
                    new_version = data.get("model_version")
                    if data.get("model_reloaded"):
                        st.success(
                            f"Model reloaded: v{previous if previous is not None else '?'} "
                            f"-> v{new_version}."
                        )
                        outcome = f"Reloaded to v{new_version}"
                    else:
                        st.info(f"No change - the API is already serving the production version (v{new_version}).")
                        outcome = f"No change (v{new_version})"
                    clear_api_cache()
                else:
                    show_error(result, "Model reload")
                    if result.kind == "timeout":
                        st.warning(
                            "The reload timed out; it may still have completed. Check the "
                            "serving version above after refreshing."
                        )
                    outcome = f"Failed: {result.kind}"
                log.append({"Time": datetime.now().strftime("%H:%M:%S"),
                            "Action": "Reload model", "Outcome": outcome})

        log = st.session_state.get("model_action_log", [])
        if log:
            st.caption("Actions performed in this session")
            show_table(pd.DataFrame(log[::-1]))
    else:
        st.markdown("### Administrative actions")
        st.caption("Model reload is protected and unavailable until admin access is unlocked.")

    # ---- feature matrix
    st.markdown("### Capability status")
    st.caption("What this dashboard can do today, and what needs additional backend endpoints.")
    show_table(pd.DataFrame(CAPABILITIES, columns=["Capability", "Status", "Backend source / gap"]))


# =============================================================================
# 8. APP SHELL
# =============================================================================

PAGE_FUNCTIONS: dict[str, Callable[[], int | None]] = {
    "Overview": page_overview,
    "Predictions": page_predictions,
    "Monitoring": page_monitoring,
    "AI Investigator": page_investigator,
    "Remediation History": page_remediation_history,
    "Model Management": page_model_management,
}


def render_sidebar_status(slot: Any) -> None:
    health = get_health()
    with slot.container():
        if health.ok:
            healthy = (health.data or {}).get("status") == "healthy"
            version = (health.data or {}).get("model_version")
            st.markdown(
                pill("API healthy" if healthy else "API unhealthy", "good" if healthy else "bad"),
                unsafe_allow_html=True,
            )
            if version is not None:
                st.caption(f"Production model v{version}")
        else:
            st.markdown(pill("API unreachable", "bad"), unsafe_allow_html=True)
        st.caption(f"Last check {health.checked_at.strftime('%H:%M:%S')}")


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} - MLOps Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    with st.sidebar:
        st.markdown(
            f'<div class="ml-brand">ML<span>Pilot</span></div>'
            '<div class="ml-brand-sub">MLOps &amp; Agentic AI Platform</div>',
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        status_slot = st.empty()

        if st.button("Refresh data"):
            clear_api_cache()

        if admin_unlocked():
            st.markdown(pill("Admin unlocked", "warn"), unsafe_allow_html=True)
            if st.button("Lock admin access"):
                st.session_state["admin_unlocked"] = False
                clear_api_cache()
                st.rerun()
        else:
            st.markdown(pill("Admin locked", "neutral"), unsafe_allow_html=True)

        st.caption(f"API: {get_api_base_url()}")

    refresh_seconds = PAGE_FUNCTIONS[page]()
    render_sidebar_status(status_slot)

    if refresh_seconds:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
