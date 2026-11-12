
from src.agents.remediation_tools import execute_remediation_action

tests = [
    (
        "update_hyperparameters",
        {
            "n_estimators": 400,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
    ),
    (
        "remove_feature",
        {"feature": "rolling_mean_14"},
    ),
    (
        "change_rolling_window",
        {"feature": "rolling_mean_14", "window": 7},
    ),
]

for action_type, parameters in tests:
    result = execute_remediation_action(action_type, parameters)
    print(f"\n{action_type}:")
    print(result)
    assert result.get("status") != "failed", result

print("\nAll remediation action tests passed.")
