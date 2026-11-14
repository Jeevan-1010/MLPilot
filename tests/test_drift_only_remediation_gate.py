from src.agents.investigator import root_cause_analysis_node


def test_drift_without_performance_degradation_does_not_recommend_remediation():
    state = {
        "monitoring_report": {
            "alerts": {
                "data_drift_detected": True,
                "performance_degraded": False,
            },
            "data_quality": {"quality_status": "HEALTHY"},
            "model_performance": {"performance_degraded": False},
        },
        "diagnostic_results": {
            "feature_drift": {
                "strongest_drift_feature": "rolling_mean_14",
            }
        },
    }

    result = root_cause_analysis_node(state)

    assert result["remediation_action"] == {
        "action_type": "none",
        "parameters": {},
    }
    assert "Drift alone" in result["root_cause"]
    assert "No remediation" in result["recommended_action"]


def test_performance_degradation_does_not_get_blocked_by_drift_only_gate(monkeypatch):
    class FakeResponse:
        content = (
            '{"root_cause":"Performance degraded.",'
            '"recommended_action":"Evaluate a corrective change.",'
            '"action_type":"change_rolling_window",'
            '"parameters":{"feature":"rolling_mean_14","window":7}}'
        )

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse()

    monkeypatch.setattr("src.agents.investigator.get_llm", lambda: FakeLLM())

    state = {
        "monitoring_report": {
            "alerts": {
                "data_drift_detected": True,
                "performance_degraded": True,
            },
            "data_quality": {"quality_status": "HEALTHY"},
            "model_performance": {"performance_degraded": True},
        },
        "diagnostic_results": {"prediction_errors": {"mae": 9000}},
    }

    result = root_cause_analysis_node(state)

    assert result["remediation_action"]["action_type"] == "change_rolling_window"
