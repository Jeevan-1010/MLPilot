from pathlib import Path
import joblib
import json

from langgraph.types import Command

from src.data_preparation import prepare_data
from src.pipeline.feature_engineering import create_features
from src.monitoring.monitoring_pipeline import run_monitoring
from src.agents.investigator import build_investigator_graph


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "prepwise_cleaned_dataset (3).csv"
FEATURES_PATH = BASE_DIR / "artifacts" / "demand_forecasting_features.joblib"


REMEDIATION_SCENARIOS = [
    {
        "action_type": "update_hyperparameters",
        "parameters": {
            "n_estimators": 400,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
    },
    {
        "action_type": "remove_feature",
        "parameters": {
            "feature": "rolling_mean_14",
        },
    },
    {
        "action_type": "change_rolling_window",
        "parameters": {
            "feature": "rolling_mean_14",
            "window": 7,
        },
    },
]


def test_investigator():
    print("\nPreparing investigator input...")

    _, daily_demand = prepare_data(DATA_PATH)
    feature_df = create_features(daily_demand)
    features = joblib.load(FEATURES_PATH)

    reference_data = feature_df.iloc[-60:-30]
    current_data = feature_df.tail(30)

    monitoring_report = run_monitoring()
    graph = build_investigator_graph()

    base_state = {
        "monitoring_report": monitoring_report,
        "reference_data": json.loads(
            reference_data.to_json(orient="records", date_format="iso")
        ),
        "current_data": json.loads(
            current_data.to_json(orient="records", date_format="iso")
        ),
        "features": features,
        "selected_tools": [],
        "diagnostic_results": {},
        "root_cause": "",
        "recommended_action": "",
        "human_approval": None,
        "remediation_status": "Not started",
    }

    for index, scenario in enumerate(REMEDIATION_SCENARIOS, start=1):
        action_type = scenario["action_type"]

        print("\n" + "=" * 55)
        print(f"TEST {index}: {action_type}")
        print("=" * 55)

        config = {
            "configurable": {
                "thread_id": f"mlpilot-remediation-test-{index}"
            }
        }

        # Run investigation until the human approval interrupt.
        result = graph.invoke(base_state.copy(), config=config)

        print("\nAI Investigation:")
        print("Root cause:", result.get("root_cause"))
        print("Recommendation:", result.get("recommended_action"))

        # Override the AI-selected action to test this specific action.
        graph.update_state(
            config,
            {
                "remediation_action": scenario,
                "recommended_action": (
                    f"Test execution of {action_type}"
                ),
            },
        )

        # Resume after human approval.
        result = graph.invoke(
            Command(
                resume={
                    "approved": True,
                    "feedback": "Approved for remediation testing",
                }
            ),
            config=config,
        )

        print("\nRemediation Result:")
        print("Action:", action_type)
        print("Status:", result.get("remediation_status"))

        print("\nCandidate Evaluation:")
        print(json.dumps(result.get("retraining_result", {}), indent=4, default=str))
        
        print("\nHuman Approval:")
        print(result.get("human_approval"))

    print("\nAll three remediation scenarios completed.")


if __name__ == "__main__":
    test_investigator()