import pandas as pd
import numpy as np
from scipy import stats


def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    method: str = "both",
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    result = df.copy()
    series = pd.to_numeric(result[column], errors="coerce")

    zscore_mask = pd.Series([False] * len(series), index=series.index)
    iqr_mask = pd.Series([False] * len(series), index=series.index)

    if method in ("zscore", "both"):
        clean = series.dropna()
        if len(clean) > 1:
            z = np.abs(stats.zscore(clean))
            z_full = pd.Series(np.nan, index=series.index)
            z_full[clean.index] = z
            zscore_mask = z_full > zscore_threshold

    if method in ("iqr", "both"):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            iqr_mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

    is_anomaly = (zscore_mask | iqr_mask).fillna(False)
    result["is_anomaly"] = is_anomaly

    mean, std = series.mean(), series.std()
    if std and std > 0:
        result["severity"] = ((series - mean).abs() / std).round(2)
    else:
        result["severity"] = 0.0

    return result


def detect_trends(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    window: int = 7,
) -> pd.DataFrame:
    result = df.copy().sort_values(date_col).reset_index(drop=True)
    result["rolling_avg"] = (
        pd.to_numeric(result[value_col], errors="coerce")
        .rolling(window=window, min_periods=1)
        .mean()
        .round(2)
    )
    result["growth_rate"] = (
        pd.to_numeric(result[value_col], errors="coerce")
        .pct_change()
        .mul(100)
        .round(2)
    )
    return result
