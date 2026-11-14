
import os
import json
from typing import TypedDict, Any, Annotated
from operator import or_

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langgraph.graph import StateGraph, START, END

from src.pipeline.train_pipeline import run_pipeline
from src.pipeline.feature_engineering import DEFAULT_ROLLING_WINDOWS
from src.agents.remediation_history import save_remediation_record

from src.agents.diagnostic_tools import (
    run_data_quality_check,
    run_data_drift_check,
    run_feature_drift_analysis,
    run_prediction_error_analysis,
)

from src.agents.remediation_tools import (
    verify_and_correct_rolling_features,
    execute_remediation_action,
)


class InvestigatorState(TypedDict, total=False):
    monitoring_report: dict
    reference_data: list[dict]
    current_data: list[dict]
    features: list[str]
    selected_tools: list[str]
    diagnostic_results: Annotated[dict[str, Any], or_]
    root_cause: str
    recommended_action: str
    remediation_action: dict
    human_approval: Any
    remediation_status: str
    retraining_result: dict
    test_feature_data: Any
    deployment_approval: Any
    deployment_status: str
    deployment_result: dict


load_dotenv()

def get_llm():
    """Create the Groq client only when an investigation actually needs it."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to run the AI Investigator.")
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
    )


AVAILABLE_DIAGNOSTIC_TOOLS = {
    "check_data_quality",
    "detect_data_drift",
    "analyze_prediction_errors",
    "analyze_feature_drift",
}

DATA_PATH = "data/prepwise_cleaned_dataset (3).csv"


def select_diagnostic_tools(state: InvestigatorState):
    prompt = f"""
You are the AI Investigator Agent of MLPilot.

Monitoring report:
{state["monitoring_report"]}

Available tools:
- check_data_quality
- detect_data_drift
- analyze_prediction_errors
- analyze_feature_drift

Select relevant tools.
Return ONLY comma-separated tool names.
"""

    response = get_llm().invoke(prompt)

    selected_tools = [
        tool.strip()
        for tool in str(response.content).split(",")
        if tool.strip() in AVAILABLE_DIAGNOSTIC_TOOLS
    ]

    return {"selected_tools": selected_tools}


def data_quality_node(state):
    if "check_data_quality" not in state["selected_tools"]:
        return {"diagnostic_results": {}}

    return {
        "diagnostic_results": {
            "data_quality": run_data_quality_check(state)
        }
    }


def data_drift_node(state):
    if "detect_data_drift" not in state["selected_tools"]:
        return {"diagnostic_results": {}}

    return {
        "diagnostic_results": {
            "data_drift": run_data_drift_check(state)
        }
    }


def feature_drift_node(state):
    if "analyze_feature_drift" not in state["selected_tools"]:
        return {"diagnostic_results": {}}

    return {
        "diagnostic_results": {
            "feature_drift": run_feature_drift_analysis(state)
        }
    }


def prediction_errors_node(state):
    if "analyze_prediction_errors" not in state["selected_tools"]:
        return {"diagnostic_results": {}}

    return {
        "diagnostic_results": {
            "prediction_errors": run_prediction_error_analysis(state)
        }
    }


def root_cause_analysis_node(state):
    diagnostic_results = state.get("diagnostic_results", {})

    if not diagnostic_results:
        return {
            "root_cause": "No issues requiring investigation were detected.",
            "recommended_action": "No action required.",
            "remediation_action": {
                "action_type": "none",
                "parameters": {},
            },
        }

    prompt = f"""
You are MLPilot's AI Investigator.

Analyze these diagnostic results:
{diagnostic_results}

Identify the most likely root cause using only the available evidence.
Do not claim correlation proves causation.
Do not automatically recommend retraining.

You may recommend ONE of these executable remediation actions:

1. update_hyperparameters
Parameters example:
{{
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2
}}

2. remove_feature
Parameters example:
{{
    "feature": "rolling_mean_14"
}}

3. change_rolling_window
Parameters example:
{{
    "feature": "rolling_mean_14",
    "window": 7
}}

If no specific corrective action is supported by the evidence, use:
{{
    "action_type": "none",
    "parameters": {{}}
}}

Return ONLY valid JSON in this exact structure:
{{
    "root_cause": "Evidence-based explanation",
    "recommended_action": "Human-readable recommendation",
    "action_type": "update_hyperparameters | remove_feature | change_rolling_window | none",
    "parameters": {{}}
}}
"""

    response = llm.invoke(prompt)
    content = str(response.content).strip()

    try:
        # Handle responses wrapped in Markdown code fences.
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)

        action_type = result.get("action_type", "none")
        parameters = result.get("parameters", {})

        allowed_actions = {
            "update_hyperparameters",
            "remove_feature",
            "change_rolling_window",
            "none",
        }

        if action_type not in allowed_actions or not isinstance(parameters, dict):
            raise ValueError("Invalid remediation action returned by the LLM.")

        return {
            "root_cause": str(result.get("root_cause", "Not established.")),
            "recommended_action": str(
                result.get("recommended_action", "No action recommended.")
            ),
            "remediation_action": {
                "action_type": action_type,
                "parameters": parameters,
            },
        }

    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "root_cause": "The investigator could not produce a valid structured analysis.",
            "recommended_action": "Manual investigation required.",
            "remediation_action": {
                "action_type": "none",
                "parameters": {},
            },
        }


def human_approval_node(state):
    approval = interrupt({
        "message": "Investigation completed. Human approval required.",
        "root_cause": state["root_cause"],
        "recommended_action": state["recommended_action"],
        "remediation_action": state.get("remediation_action", {}),
    })

    return {"human_approval": approval}


def route_after_approval(state):
    approval = state.get("human_approval") or {}

    if isinstance(approval, dict) and approval.get("approved") is True:
        return "remediation"

    return "end"


def remediation_node(state):
    print("\n========== REMEDIATION ==========")

    def finish(updates):
        record_state = {**state, **updates}

        try:
            save_remediation_record(record_state)
            print("Remediation history saved.")
        except Exception as error:
            print(f"Warning: Could not save remediation history: {error}")

        return updates

    approval = state.get("human_approval") or {}

    if not isinstance(approval, dict) or approval.get("approved") is not True:
        return finish({
            "remediation_status": "Rejected by human.",
            "retraining_result": {"status": "rejected"},
        })

    action = state.get("remediation_action", {})
    action_type = action.get("action_type", "none")
    parameters = action.get("parameters", {})

    if action_type == "none":
        return finish({
            "remediation_status": (
                "No executable corrective action was recommended."
            ),
            "retraining_result": {"status": "skipped"},
        })

    try:
        validated_action = execute_remediation_action(
            action_type,
            parameters,
        )

        if validated_action.get("status") == "failed":
            return finish({
                "remediation_status": "Remediation action validation failed.",
                "retraining_result": validated_action,
            })

        pipeline_kwargs = {
            "data_path": DATA_PATH,
            "register_candidate": False,
        }

        if action_type == "update_hyperparameters":
            pipeline_kwargs["candidate_params"] = (
                validated_action["candidate_params"]
            )

        elif action_type == "remove_feature":
            pipeline_kwargs["feature_columns"] = (
                validated_action["feature_columns"]
            )

        elif action_type == "change_rolling_window":
            rolling_windows = validated_action["rolling_windows"]

            verification = verify_and_correct_rolling_features(
                DATA_PATH,
                state.get("test_feature_data"),
                rolling_windows=rolling_windows,
            )

            if verification.get("status") == "failed":
                return finish({
                    "remediation_status": (
                        "Rolling-window verification failed. Retraining skipped."
                    ),
                    "retraining_result": verification,
                })

            pipeline_kwargs["rolling_windows"] = rolling_windows

            if verification.get("corrected_data") is not None:
                pipeline_kwargs["feature_data"] = (
                    verification["corrected_data"]
                )

        print(f"Approved action: {action_type}")
        print("Starting candidate evaluation...")

        result = run_pipeline(**pipeline_kwargs)

        if result.get("status") != "evaluation_passed":
            return finish({
                "remediation_status": (
                    "Corrective action executed, but candidate evaluation "
                    "did not pass. No candidate was registered."
                ),
                "retraining_result": result,
            })

        return finish({
            "remediation_status": (
                "Corrective action executed and candidate evaluation passed. "
                "Candidate was not registered."
            ),
            "retraining_result": result,
        })

    except Exception as error:
        return finish({
            "remediation_status": f"Remediation failed: {error}",
            "retraining_result": {
                "status": "failed",
                "error": str(error),
            },
        })

def build_investigator_graph():
    graph = StateGraph(InvestigatorState)

    graph.add_node("select_tools", select_diagnostic_tools)
    graph.add_node("data_quality", data_quality_node)
    graph.add_node("data_drift", data_drift_node)
    graph.add_node("feature_drift", feature_drift_node)
    graph.add_node("prediction_errors", prediction_errors_node)

    graph.add_node("root_cause_analysis", root_cause_analysis_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("remediation", remediation_node)

    graph.add_edge(START, "select_tools")

    graph.add_edge("select_tools", "data_quality")
    graph.add_edge("select_tools", "data_drift")
    graph.add_edge("select_tools", "feature_drift")
    graph.add_edge("select_tools", "prediction_errors")

    graph.add_edge(
        [
            "data_quality",
            "data_drift",
            "feature_drift",
            "prediction_errors",
        ],
        "root_cause_analysis",
    )

    graph.add_edge("root_cause_analysis", "human_approval")

    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "remediation": "remediation",
            "end": END,
        },
    )

    graph.add_edge("remediation", END)

    memory = MemorySaver()

    return graph.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# MLPilot approval workflow (API/checkpointer-aware implementation)
# ---------------------------------------------------------------------------
from contextlib import nullcontext
from src.pipeline.deploy_model import deploy_model as _deploy_model


def _workflow_save_history(state):
    try:
        save_remediation_record(state)
    except Exception as error:
        print(f"Warning: could not save remediation history: {error}")


def _workflow_no_action_node(state):
    updates = {
        "remediation_status": "No actionable issue detected; no remediation required.",
        "retraining_result": {"status": "skipped"},
        "deployment_status": "not_required",
    }
    _workflow_save_history({**state, **updates})
    return updates


def _workflow_remediation_rejected_node(state):
    updates = {
        "remediation_status": "Remediation rejected by human; no candidate was trained.",
        "retraining_result": {"status": "rejected"},
        "deployment_status": "not_started",
    }
    _workflow_save_history({**state, **updates})
    return updates


def _workflow_route_after_root_cause(state):
    action = state.get("remediation_action") or {}
    return "no_action" if action.get("action_type", "none") == "none" else "approval"


def _workflow_route_after_approval(state):
    approval = state.get("human_approval") or {}
    return (
        "remediation"
        if isinstance(approval, dict) and approval.get("approved") is True
        else "rejected"
    )


def _workflow_remediation_node(state, data_context_provider=None):
    approval = state.get("human_approval") or {}
    if not isinstance(approval, dict) or approval.get("approved") is not True:
        return _workflow_remediation_rejected_node(state)

    action = state.get("remediation_action") or {}
    action_type = action.get("action_type", "none")
    parameters = action.get("parameters", {})
    if action_type == "none":
        return _workflow_no_action_node(state)

    try:
        validated = execute_remediation_action(action_type, parameters)
        if validated.get("status") != "validated":
            updates = {
                "remediation_status": "Remediation validation failed.",
                "retraining_result": validated,
                "deployment_status": "not_started",
            }
            _workflow_save_history({**state, **updates})
            return updates

        provider = data_context_provider or (lambda: nullcontext(DATA_PATH))
        with provider() as data_path:
            kwargs = {"data_path": data_path, "register_candidate": True}
            if action_type == "update_hyperparameters":
                kwargs["candidate_params"] = validated["candidate_params"]
            elif action_type == "remove_feature":
                kwargs["feature_columns"] = validated["feature_columns"]
            elif action_type == "change_rolling_window":
                windows = validated["rolling_windows"]
                verification = verify_and_correct_rolling_features(
                    data_path, state.get("test_feature_data"), rolling_windows=windows
                )
                if verification.get("status") == "failed":
                    updates = {
                        "remediation_status": "Rolling-window verification failed; training skipped.",
                        "retraining_result": verification,
                        "deployment_status": "not_started",
                    }
                    _workflow_save_history({**state, **updates})
                    return updates
                kwargs["rolling_windows"] = windows
                if verification.get("corrected_data") is not None:
                    kwargs["feature_data"] = verification["corrected_data"]
            result = run_pipeline(**kwargs)

        ready = (
            result.get("status") == "passed"
            and result.get("registered") is True
            and bool(result.get("model_version"))
        )
        if not ready:
            updates = {
                "remediation_status": "Candidate did not pass validation or registration; production remains unchanged.",
                "retraining_result": result,
                "deployment_status": "not_started",
            }
            _workflow_save_history({**state, **updates})
            return updates

        return {
            "remediation_status": "Candidate passed evaluation and was registered; awaiting separate deployment approval.",
            "retraining_result": result,
            "deployment_status": "awaiting_human_approval",
        }
    except Exception as error:
        updates = {
            "remediation_status": f"Remediation failed: {error}",
            "retraining_result": {"status": "failed", "error": str(error)},
            "deployment_status": "not_started",
        }
        _workflow_save_history({**state, **updates})
        return updates


def _workflow_route_after_remediation(state):
    result = state.get("retraining_result") or {}
    if (
        state.get("deployment_status") == "awaiting_human_approval"
        and result.get("registered") is True
        and result.get("model_version")
    ):
        return "deployment_approval"
    return "end"


def _workflow_deployment_approval_node(state):
    result = state.get("retraining_result") or {}
    approval = interrupt({
        "stage": "deployment",
        "message": "Candidate passed evaluation. Explicit approval is required before production changes.",
        "candidate_version": str(result.get("model_version")),
        "production_metrics": result.get("production_metrics"),
        "candidate_metrics": result.get("candidate_metrics"),
        "candidate_features": result.get("features"),
    })
    return {"deployment_approval": approval}


def _workflow_route_after_deployment_approval(state):
    approval = state.get("deployment_approval") or {}
    return (
        "deploy"
        if isinstance(approval, dict) and approval.get("approved") is True
        else "rejected"
    )


def _workflow_deployment_rejected_node(state):
    updates = {
        "deployment_status": "rejected; candidate was not promoted and production remains unchanged.",
        "deployment_result": {"status": "rejected"},
    }
    _workflow_save_history({**state, **updates})
    return updates


def _workflow_deployment_node(state, deployment_provider=None):
    result = state.get("retraining_result") or {}
    candidate_version = result.get("model_version")
    if not candidate_version:
        return {
            "deployment_status": "failed",
            "deployment_result": {"status": "failed", "error": "Candidate version missing."},
        }
    provider = deployment_provider or _deploy_model
    try:
        deployed = provider(
            str(candidate_version),
            approved=True,
            expected_previous_version=result.get("production_model_version"),
        )
        status = deployed.get("status", "unknown")
        updates = {
            "deployment_status": "deployed" if status == "success" else (
                "rolled_back" if status == "rolled_back" else status
            ),
            "deployment_result": deployed,
        }
        _workflow_save_history({**state, **updates})
        return updates
    except Exception as error:
        updates = {
            "deployment_status": "failed",
            "deployment_result": {"status": "failed", "error": str(error)},
        }
        _workflow_save_history({**state, **updates})
        return updates


def build_investigator_graph(
    checkpointer=None,
    model_provider=None,
    data_context_provider=None,
    deployment_provider=None,
):
    graph = StateGraph(InvestigatorState)
    graph.add_node("select_tools", select_diagnostic_tools)
    graph.add_node("data_quality", data_quality_node)
    graph.add_node("data_drift", data_drift_node)
    graph.add_node("feature_drift", feature_drift_node)

    if model_provider is None:
        graph.add_node("prediction_errors", prediction_errors_node)
    else:
        def prediction_errors_with_active_model(state):
            if "analyze_prediction_errors" not in state.get("selected_tools", []):
                return {"diagnostic_results": {}}
            return {
                "diagnostic_results": {
                    "prediction_errors": run_prediction_error_analysis(
                        state, model=model_provider()
                    )
                }
            }
        graph.add_node("prediction_errors", prediction_errors_with_active_model)

    def remediation_with_data(state):
        return _workflow_remediation_node(state, data_context_provider)

    def deployment_with_provider(state):
        return _workflow_deployment_node(state, deployment_provider)

    graph.add_node("root_cause_analysis", root_cause_analysis_node)
    graph.add_node("no_action", _workflow_no_action_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("remediation", remediation_with_data)
    graph.add_node("remediation_rejected", _workflow_remediation_rejected_node)
    graph.add_node("deployment_approval", _workflow_deployment_approval_node)
    graph.add_node("deployment_rejected", _workflow_deployment_rejected_node)
    graph.add_node("deployment", deployment_with_provider)

    graph.add_edge(START, "select_tools")
    graph.add_edge("select_tools", "data_quality")
    graph.add_edge("select_tools", "data_drift")
    graph.add_edge("select_tools", "feature_drift")
    graph.add_edge("select_tools", "prediction_errors")
    graph.add_edge(
        ["data_quality", "data_drift", "feature_drift", "prediction_errors"],
        "root_cause_analysis",
    )
    graph.add_conditional_edges(
        "root_cause_analysis",
        _workflow_route_after_root_cause,
        {"no_action": "no_action", "approval": "human_approval"},
    )
    graph.add_edge("no_action", END)
    graph.add_conditional_edges(
        "human_approval",
        _workflow_route_after_approval,
        {"remediation": "remediation", "rejected": "remediation_rejected"},
    )
    graph.add_edge("remediation_rejected", END)
    graph.add_conditional_edges(
        "remediation",
        _workflow_route_after_remediation,
        {"deployment_approval": "deployment_approval", "end": END},
    )
    graph.add_conditional_edges(
        "deployment_approval",
        _workflow_route_after_deployment_approval,
        {"deploy": "deployment", "rejected": "deployment_rejected"},
    )
    graph.add_edge("deployment", END)
    graph.add_edge("deployment_rejected", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
