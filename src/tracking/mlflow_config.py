
import os

import mlflow
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def configure_mlflow():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI,
    )

    mlflow.set_tracking_uri(tracking_uri)

    return tracking_uri
