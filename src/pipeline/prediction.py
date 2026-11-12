
import pandas as pd


def predict_demand(
    model,
    daily_demand: pd.DataFrame,
    features: list
) -> tuple[pd.Timestamp, float]:

    df = daily_demand.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    demand = df["daily_demand"]

    next_date = df["date"].iloc[-1] + pd.Timedelta(days=1)

    feature_row = pd.DataFrame([{
        "day_of_week": next_date.dayofweek,
        "month": next_date.month,
        "day_of_month": next_date.day,
        "week_of_year": int(next_date.isocalendar().week),
        "lag_1": demand.iloc[-1],
        "lag_7": demand.iloc[-7],
        "lag_14": demand.iloc[-14],
        "rolling_mean_7": demand.tail(7).mean(),
        "rolling_mean_14": demand.tail(14).mean(),
    }])

    feature_row = feature_row[features]

    prediction = float(model.predict(feature_row)[0])

    return next_date, prediction
