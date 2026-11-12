
import pandas as pd


DEFAULT_ROLLING_WINDOWS = {
    "rolling_mean_7": 7,
    "rolling_mean_14": 14,
}


def create_features(
    daily_demand: pd.DataFrame,
    rolling_windows: dict = None,
) -> pd.DataFrame:

    df = daily_demand.copy()

    windows = (
        rolling_windows.copy()
        if rolling_windows is not None
        else DEFAULT_ROLLING_WINDOWS.copy()
    )

    if not windows:
        raise ValueError("At least one rolling feature is required.")

    for feature, window in windows.items():
        if not isinstance(feature, str) or not feature.startswith("rolling_mean_"):
            raise ValueError(f"Unsupported rolling feature: {feature}")

        if (
            isinstance(window, bool)
            or not isinstance(window, int)
            or not 2 <= window <= 30
        ):
            raise ValueError(
                f"Invalid rolling window for {feature}: {window}"
            )

    if "daily_demand" not in df.columns or "date" not in df.columns:
        raise ValueError("Input must contain date and daily_demand columns.")

    df["date"] = pd.to_datetime(df["date"])

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df["lag_1"] = df["daily_demand"].shift(1)
    df["lag_7"] = df["daily_demand"].shift(7)
    df["lag_14"] = df["daily_demand"].shift(14)

    for feature, window in windows.items():
        df[feature] = (
            df["daily_demand"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    df = df.dropna().reset_index(drop=True)

    return df
