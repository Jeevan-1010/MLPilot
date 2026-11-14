"""
Regression tests for the Streamlit AI Investigator page.

Bug: ``StreamlitWidgetAlreadyInstantiatedError`` - ``st.session_state
['investigator_thread_id_input'] cannot be modified after the widget with that key
is instantiated``. The text_input widget was created first and the "Start" handler
then assigned to its key in the same script run.

Streamlit itself is not required to run these tests (CI does not install it).
``FakeStreamlit`` reproduces the one rule that matters here: writing a widget's
session-state key after that widget has been created in the current script run
raises, while writes made in an ``on_click`` callback (which runs before the next
script pass) or before the widget is created are allowed.

Set MLPILOT_APP_PATH to point the tests at another copy of app.py (used to confirm
they fail against the buggy version).
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

APP_PATH = Path(
    os.environ.get(
        "MLPILOT_APP_PATH",
        Path(__file__).resolve().parents[1] / "frontend" / "app.py",
    )
)
WIDGET_KEY = "investigator_thread_id_input"
ACTIVE_KEY = "investigator_thread_id"


# --------------------------------------------------------------------------- #
# Minimal Streamlit stand-in that enforces the widget-lifecycle rule
# --------------------------------------------------------------------------- #


class WidgetAlreadyInstantiated(Exception):
    pass


class _Rerun(Exception):
    pass


class GuardedSessionState(dict):
    def __init__(self, fake):
        super().__init__()
        self._fake = fake
        self.api_written: set[str] = set()

    def __setitem__(self, key, value):
        if key in self._fake.instantiated:
            raise WidgetAlreadyInstantiated(
                f"st.session_state[{key!r}] cannot be modified after the widget "
                "with that key is instantiated."
            )
        self.api_written.add(key)
        super().__setitem__(key, value)


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.__version__ = "1.40.0"  # read at import time by frontend/app.py
        self.instantiated: set[str] = set()
        self.session_state = GuardedSessionState(self)
        self.callbacks: dict = {}
        self.clicked: set[str] = set()
        self.messages: list[tuple[str, str]] = []
        self.default_value_conflicts: list[str] = []

    # widgets ---------------------------------------------------------------
    def _widget_value(self, key, default):
        if key is None:
            return default
        if key not in self.session_state:
            dict.__setitem__(self.session_state, key, default)
        self.instantiated.add(key)
        return self.session_state[key]

    def text_input(self, label, value="", key=None, **kwargs):
        if key is not None and value not in ("", None) and key in self.session_state.api_written:
            # Streamlit warns when a widget has a default value AND its state was set via the API.
            self.default_value_conflicts.append(key)
        return self._widget_value(key, value)

    def text_area(self, label, value="", key=None, **kwargs):
        return self._widget_value(key, value)

    def checkbox(self, label, value=False, key=None, **kwargs):
        return self._widget_value(key, value)

    def button(self, label, on_click=None, key=None, disabled=False, **kwargs):
        self.callbacks[label] = on_click
        return label in self.clicked and not disabled

    # layout / status -------------------------------------------------------
    def columns(self, spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [contextlib.nullcontext() for _ in range(count)]

    def spinner(self, *args, **kwargs):
        return contextlib.nullcontext()

    def expander(self, *args, **kwargs):
        return contextlib.nullcontext()

    def _message(self, level):
        def record(message, *args, **kwargs):
            self.messages.append((level, str(message)))

        return record

    def __getattr__(self, name):
        if name in ("info", "warning", "success", "error"):
            return self._message(name)
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None  # markdown, write, json, caption, ...

    def rerun(self):
        raise _Rerun()


class Harness:
    """Drives ``page_investigator`` the way Streamlit drives a script."""

    def __init__(self, app, fake):
        self.app, self.fake = app, fake

    def user_types(self, key, text):
        """A user edit: widget state set by the browser, not by app code."""
        dict.__setitem__(self.fake.session_state, key, text)

    def run(self, click=None):
        fake = self.fake
        fake.clicked = set()
        if click is not None:
            callback = fake.callbacks.get(click)
            fake.instantiated = set()  # callbacks run before the next script pass
            if callback:
                callback()
            fake.clicked = {click}
        for _ in range(5):
            fake.instantiated = set()
            fake.callbacks = {}
            fake.messages = []
            try:
                self.app.page_investigator()
                return
            except _Rerun:
                fake.clicked = set()
        raise AssertionError("page_investigator kept rerunning")


@pytest.fixture
def page(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    if importlib.util.find_spec("plotly") is None:
        monkeypatch.setitem(sys.modules, "plotly", MagicMock())
        monkeypatch.setitem(sys.modules, "plotly.graph_objects", MagicMock())

    module_name = "mlpilot_frontend_app_under_test"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    app = importlib.util.module_from_spec(spec)
    # @dataclass with `from __future__ import annotations` needs the module registered.
    monkeypatch.setitem(sys.modules, module_name, app)
    spec.loader.exec_module(app)

    calls = {"get": [], "approve": [], "start": 0}
    responses = {
        "start": app.ApiResult(ok=True, data={"thread_id": "thread-new", "status": "awaiting_remediation_approval"}),
        "state": app.ApiResult(ok=True, data={"status": "completed"}),
    }

    def fake_start():
        calls["start"] += 1
        return responses["start"]

    def fake_get(thread_id):
        calls["get"].append(thread_id)
        return responses["state"]

    def fake_approve(thread_id, approved, feedback=""):
        calls["approve"].append((thread_id, approved, feedback))
        return app.ApiResult(ok=True, data={"status": "updated"})

    monkeypatch.setattr(app, "render_admin_gate", lambda feature: True)
    monkeypatch.setattr(app, "api_start_investigation", fake_start)
    monkeypatch.setattr(app, "api_get_investigation", fake_get)
    monkeypatch.setattr(app, "api_approve_investigation", fake_approve)

    harness = Harness(app, fake)
    harness.calls, harness.responses, harness.app = calls, responses, app
    return harness


# --------------------------------------------------------------------------- #
# Behavioural tests
# --------------------------------------------------------------------------- #


def test_harness_enforces_the_streamlit_rule(page):
    """Guards against a vacuous test suite: the buggy pattern must raise here."""
    fake = page.fake
    page.run()  # creates the widget
    fake.instantiated = {WIDGET_KEY}
    with pytest.raises(WidgetAlreadyInstantiated):
        fake.session_state[WIDGET_KEY] = "x"


def test_initial_render_with_no_thread_selected(page):
    page.run()
    assert page.calls["get"] == []
    assert page.fake.session_state[WIDGET_KEY] == ""
    assert page.fake.default_value_conflicts == []


def test_start_new_investigation_from_blank_thread_id(page):
    page.run()
    page.run(click="Start / find active investigation")  # must not raise

    state = page.fake.session_state
    assert page.calls["start"] == 1
    assert state[ACTIVE_KEY] == "thread-new"
    assert state[WIDGET_KEY] == "thread-new"  # box reflects the started thread
    assert page.calls["get"][-1] == "thread-new"  # status is loaded for it
    assert ("success", "Investigation started: awaiting_remediation_approval") in page.fake.messages
    assert page.fake.default_value_conflicts == []


def test_start_reports_existing_pending_investigation(page):
    page.responses["start"] = page.app.ApiResult(
        ok=True, data={"thread_id": "thread-old", "status": "already_pending"}
    )
    page.run()
    page.run(click="Start / find active investigation")
    assert page.fake.session_state[WIDGET_KEY] == "thread-old"
    assert any(level == "info" for level, _ in page.fake.messages)


def test_start_failure_is_shown_and_does_not_change_selection(page):
    page.responses["start"] = page.app.ApiResult(ok=False, error="boom", kind="server")
    page.run()
    page.user_types(WIDGET_KEY, "typed-id")
    page.run(click="Start / find active investigation")
    assert page.fake.session_state[WIDGET_KEY] == "typed-id"
    assert ACTIVE_KEY not in page.fake.session_state
    assert any(level == "error" for level, _ in page.fake.messages)


def test_load_existing_thread_by_id(page):
    page.run()
    page.user_types(WIDGET_KEY, "  thread-existing  ")
    page.run(click="Load thread / refresh status")
    assert page.fake.session_state[ACTIVE_KEY] == "thread-existing"
    assert page.calls["get"][-1] == "thread-existing"


def test_load_with_blank_id_warns_and_does_not_call_api(page):
    page.run()
    page.run(click="Load thread / refresh status")
    assert ("warning", "Enter an investigation thread ID first.") in page.fake.messages
    assert page.calls["get"] == []


def test_refresh_after_start_reloads_status(page):
    page.run()
    page.run(click="Start / find active investigation")
    before = len(page.calls["get"])
    page.run(click="Load thread / refresh status")  # box already holds the started ID
    assert ("warning", "Enter an investigation thread ID first.") not in page.fake.messages
    assert len(page.calls["get"]) > before
    assert page.calls["get"][-1] == "thread-new"


def test_box_is_reseeded_when_returning_to_page(page):
    page.run()
    page.run(click="Start / find active investigation")
    # Streamlit drops widget state for widgets not rendered on a run (user switched pages).
    dict.pop(page.fake.session_state, WIDGET_KEY)
    page.run()
    assert page.fake.session_state[WIDGET_KEY] == "thread-new"
    assert page.fake.default_value_conflicts == []


@pytest.mark.parametrize(
    "status, stage",
    [
        ("awaiting_remediation_approval", "remediation"),
        ("awaiting_deployment_approval", "deployment"),
    ],
)
@pytest.mark.parametrize("approve", [True, False])
def test_approval_and_rejection_for_both_stages(page, status, stage, approve):
    page.responses["state"] = page.app.ApiResult(ok=True, data={"status": status})
    page.run()
    page.user_types(WIDGET_KEY, "thread-x")
    page.run(click="Load thread / refresh status")

    button = ("Approve " if approve else "Reject ") + stage
    page.run(click=button)  # disabled until confirmed
    assert page.calls["approve"] == []

    page.user_types(f"investigator_confirm_thread-x_{stage}", True)
    page.user_types(f"investigator_feedback_{stage}", "looks fine")
    page.run(click=button)
    assert page.calls["approve"] == [("thread-x", approve, "looks fine")]


def test_findings_are_displayed(page):
    page.responses["state"] = page.app.ApiResult(
        ok=True,
        data={"status": "completed", "root_cause": "Window too short", "recommended_action": "Widen window"},
    )
    page.run()
    page.user_types(WIDGET_KEY, "thread-x")
    page.run(click="Load thread / refresh status")
    assert page.calls["get"][-1] == "thread-x"
    assert ("info", "No approval is currently pending for this investigation.") in page.fake.messages


# --------------------------------------------------------------------------- #
# Static guard: nobody may write the widget key after the widget is created
# --------------------------------------------------------------------------- #


def _module_string_constants(tree):
    constants = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            constants[node.targets[0].id] = node.value.value
    return constants


def _resolve(node, constants):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _is_session_state(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "session_state"
        and isinstance(node.value, ast.Name)
        and node.value.id == "st"
    )


def _widget_key_writes(func, constants):
    lines = []
    for node in ast.walk(func):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and _is_session_state(target.value)
                and _resolve(target.slice, constants) == WIDGET_KEY
            ):
                lines.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_session_state(node.func.value)
            and node.func.attr in {"setdefault", "update"}
            and node.args
            and _resolve(node.args[0], constants) == WIDGET_KEY
        ):
            lines.append(node.lineno)
    return lines


def _is_st_call(node, name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
    )


def test_widget_key_is_only_written_in_callback_or_before_widget_creation():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    constants = _module_string_constants(tree)
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    for func in functions:
        writes = _widget_key_writes(func, constants)
        if not writes:
            continue
        if func.name == "_start_investigation_callback":
            continue  # on_click callback: runs before the widget exists in the next run
        assert func.name == "page_investigator", (
            f"{func.name}() writes st.session_state[{WIDGET_KEY!r}] outside a callback "
            "(only legal in an on_click callback or before the widget is created)."
        )
        creations = [
            n.lineno
            for n in ast.walk(func)
            if _is_st_call(n, "text_input")
            and any(kw.arg == "key" and _resolve(kw.value, constants) == WIDGET_KEY for kw in n.keywords)
        ]
        assert creations, "page_investigator writes the widget key but never creates the widget"
        assert all(line < min(creations) for line in writes), (
            f"st.session_state[{WIDGET_KEY!r}] is written after the text_input is created "
            f"(write lines {writes}, widget created at line {min(creations)})."
        )


def test_start_button_uses_callback_not_inline_handler():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    page_func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "page_investigator")
    start_buttons = [
        n
        for n in ast.walk(page_func)
        if _is_st_call(n, "button")
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "Start / find active investigation"
    ]
    assert len(start_buttons) == 1
    assert any(
        kw.arg == "on_click" and isinstance(kw.value, ast.Name) and kw.value.id == "_start_investigation_callback"
        for kw in start_buttons[0].keywords
    )