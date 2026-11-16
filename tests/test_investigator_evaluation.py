"""Offline Investigator evaluation.

Uses deterministic LLM/tool doubles. Scores workflow contracts and orchestration,
not the accuracy of a live Groq model.
"""
import json
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from src.agents import investigator as agent


TOOLS = {
    "check_data_quality",
    "detect_data_drift",
    "analyze_prediction_errors",
    "analyze_feature_drift",
}


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, prompt):
        response = self.responses[self.calls]
        self.calls += 1
        return FakeResponse(response)


def test_tool_selection_contract_metrics(monkeypatch):
    cases = [
        ("quality", ["check_data_quality"], ["check_data_quality"]),
        ("drift", ["detect_data_drift"], ["detect_data_drift"]),
        ("errors", ["analyze_prediction_errors"], ["analyze_prediction_errors"]),
        ("features", ["analyze_feature_drift"], ["analyze_feature_drift"]),
        ("combined", sorted(TOOLS), sorted(TOOLS)),
    ]
    exact = 0
    for label, response, expected in cases:
        monkeypatch.setattr(
            agent, "get_llm",
            lambda response=response: FakeLLM([",".join(response)]),
        )
        result = agent.select_diagnostic_tools({"monitoring_report": {"scenario": label}})
        selected = sorted(result["selected_tools"])
        exact += selected == sorted(expected)
    score = exact / len(cases)
    print(f"INVESTIGATOR_TOOL_SELECTION_CONTRACT cases={len(cases)} exact_match={score:.3f}")
    assert score == 1.0


def test_root_cause_guardrail_and_response_contract(monkeypatch):
    cases = []
    drift_only = {
        "monitoring_report": {
            "alerts": {"data_drift_detected": True, "performance_degraded": False},
            "data_quality": {"quality_status": "HEALTHY"},
            "model_performance": {"performance_degraded": False},
        },
        "diagnostic_results": {"feature_drift": {"drift_detected": True}},
    }
    cases.append((drift_only, "none"))

    no_issue = {
        "monitoring_report": {},
        "diagnostic_results": {},
    }
    cases.append((no_issue, "none"))

    degraded = {
        "monitoring_report": {
            "alerts": {"data_drift_detected": True, "performance_degraded": True},
            "data_quality": {"quality_status": "HEALTHY"},
            "model_performance": {"performance_degraded": True},
        },
        "diagnostic_results": {"prediction_errors": {"mae": 9000}},
    }
    response = json.dumps({
        "root_cause": "Synthetic performance degradation in the test fixture.",
        "recommended_action": "Evaluate a controlled hyperparameter change.",
        "action_type": "update_hyperparameters",
        "parameters": {"n_estimators": 50},
    })
    monkeypatch.setattr(agent, "get_llm", lambda: FakeLLM([response]))

    correct = 0
    for index, (state, expected_action) in enumerate(cases):
        result = agent.root_cause_analysis_node(state)
        correct += result["remediation_action"]["action_type"] == expected_action
    result = agent.root_cause_analysis_node(degraded)
    correct += result["remediation_action"]["action_type"] == "update_hyperparameters"

    score = correct / 3
    print(f"INVESTIGATOR_ROOT_CAUSE_CONTRACT cases=3 correct_decisions={correct} exact_match={score:.3f}")
    assert score == 1.0


def test_full_investigator_graph_completion_time(monkeypatch):
    llm = FakeLLM([
        ",".join(sorted(TOOLS)),
        json.dumps({
            "root_cause": "Synthetic performance degradation identified.",
            "recommended_action": "Evaluate a controlled hyperparameter update.",
            "action_type": "update_hyperparameters",
            "parameters": {"n_estimators": 50},
        }),
    ])
    monkeypatch.setattr(agent, "get_llm", lambda: llm)
    monkeypatch.setattr(agent, "save_remediation_record", lambda state: None)
    monkeypatch.setattr(
        agent, "execute_remediation_action",
        lambda action, params: {"status": "validated", "candidate_params": params},
    )
    monkeypatch.setattr(
        agent, "run_pipeline",
        lambda **kwargs: {
            "status": "passed", "registered": True, "model_version": "test-2",
            "production_model_version": "test-1",
            "production_metrics": {"MAE": 10.0}, "candidate_metrics": {"MAE": 8.0},
        },
    )

    class FakeModel:
        def predict(self, frame):
            return np.zeros(len(frame))

    features = [
        "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14",
    ]
    rng = np.random.default_rng(20261115)

    def rows(offset):
        frame = pd.DataFrame({
            feature: rng.normal(offset, 1, 30) for feature in features
        })
        frame["daily_demand"] = rng.uniform(10, 100, 30)
        return frame.to_dict(orient="records")

    state = {
        "monitoring_report": {
            "alerts": {"data_drift_detected": True, "performance_degraded": True},
            "data_quality": {"quality_status": "HEALTHY"},
            "model_performance": {"performance_degraded": True},
        },
        "reference_data": rows(0),
        "current_data": rows(1.5),
        "features": features,
        "selected_tools": [],
        "diagnostic_results": {},
        "test_feature_data": None,
    }
    deployment_calls = []
    def fake_deploy(version, **kwargs):
        deployment_calls.append(version)
        return {"status": "success", "previous_version": "test-1", "deployed_version": version}

    graph = agent.build_investigator_graph(
        model_provider=lambda: FakeModel(),
        data_context_provider=lambda: nullcontext("offline-fixture.csv"),
        deployment_provider=fake_deploy,
    )
    config = {"configurable": {"thread_id": "offline-eval-investigator"}}

    started = time.perf_counter()
    graph.invoke(state, config=config)
    snapshot = graph.get_state(config)
    assert "human_approval" in snapshot.next

    graph.invoke(Command(resume={"approved": True, "feedback": "offline test approval"}), config=config)
    snapshot = graph.get_state(config)
    assert "deployment_approval" in snapshot.next

    graph.invoke(Command(resume={"approved": True, "feedback": "offline test deployment approval"}), config=config)
    elapsed_ms = (time.perf_counter() - started) * 1000
    final = graph.get_state(config)
    assert not final.next
    assert final.values["deployment_status"] == "deployed"
    assert deployment_calls == ["test-2"]
    print(f"INVESTIGATOR_OFFLINE_GRAPH elapsed_ms={elapsed_ms:.2f} approval_stages=2 completed=1")
