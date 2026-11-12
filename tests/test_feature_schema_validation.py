
import json
import pytest
from types import SimpleNamespace

from src.api.main import load_model_schema


class FakeClient:
    def __init__(self, schema):
        self.schema = schema

    def list_artifacts(self, run_id):
        return [SimpleNamespace(path="feature_schema.json")]

    def download_artifacts(self, run_id, artifact_path):
        path = "test_feature_schema.json"

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.schema, file)

        return path


def valid_schema():
    return {
        "schema_version": 1,
        "feature_names": ["lag_1", "rolling_mean_7"],
        "target": "daily_demand",
        "rolling_windows": {
            "rolling_mean_7": 7
        },
    }


@pytest.mark.parametrize(
    "modify",
    [
        lambda s: s.update(schema_version=2),
        lambda s: s.update(feature_names=[]),
        lambda s: s.update(feature_names=["lag_1", "lag_1"]),
        lambda s: s.update(feature_names=["unknown_feature"]),
        lambda s: s.update(target="wrong_target"),
        lambda s: s.update(rolling_windows=[]),
        lambda s: s.update(
            rolling_windows={"rolling_mean_7": 1}
        ),
        lambda s: s.update(
            rolling_windows={"rolling_mean_14": 7}
        ),
        lambda s: s.update(
            rolling_windows={}
        ),
    ],
)
def test_invalid_schema_is_rejected(modify):
    schema = valid_schema()
    modify(schema)

    client = FakeClient(schema)
    production = SimpleNamespace(run_id="test", version="test")

    with pytest.raises(ValueError):
        load_model_schema(client, production)


def test_valid_schema_is_accepted():
    schema = valid_schema()

    client = FakeClient(schema)
    production = SimpleNamespace(run_id="test", version="test")

    result = load_model_schema(client, production)

    assert result == schema
    print("Valid schema accepted!")


def test_legacy_schema_fallback():
    class LegacyClient:
        def list_artifacts(self, run_id):
            return []

    production = SimpleNamespace(run_id="test", version="1")

    result = load_model_schema(LegacyClient(), production)

    assert result["schema_version"] == 0
    assert result["feature_names"] == [
        "day_of_week",
        "month",
        "day_of_month",
        "week_of_year",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
    ]

    print("Legacy schema fallback passed!")
