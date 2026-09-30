import re
from pathlib import Path
from typing import Any
import pandas as pd
from config import clean_dir, staging_dir, raw_dir

# Define the input and output file locations used by the cleaning pipeline.
input_path = raw_dir / "merged_market.csv"
output_path = staging_dir / "data_020118_300522.csv"

# Keep the study period explicit instead of relying on a hard-coded row limit.
DEFAULT_ROW_LIMIT = 1097

# COVID-related series are treated as zero when the original value is missing,
# since missing entries usually indicate "no new cases/deaths recorded".
zero_impute_cols = (
    "new",
    "cumulative",
    "phase_1",
    "phase_2",
    "phase_3",
    "phase_4",
    "deaths",
)


def _moving_average_impute(series: pd.Series, window: int) -> pd.Series:
    """Fill missing values with local moving averages while keeping edge cases sensible.

    This is used for market indicators like RSI, PLI, and return series where
    gaps are common and should be estimated from nearby observations rather than
    dropped outright.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")

    # Estimate the missing value from the surrounding observations on both sides.
    trailing_average = numeric_series.rolling(window=window, min_periods=1).mean()
    leading_average = (
        numeric_series.iloc[::-1]
        .rolling(window=window, min_periods=1)
        .mean()
        .iloc[::-1]
    )
    imputed = numeric_series.fillna(trailing_average).fillna(leading_average)
    observed = numeric_series.dropna()

    if not observed.empty:
        first_valid = numeric_series.first_valid_index()
        last_valid = numeric_series.last_valid_index()

        # Fill any leading gaps with the early observed average to avoid bias at the beginning of the time series.
        if first_valid is not None and first_valid > 0:
            leading_mean = observed.iloc[:window].mean()
            imputed.iloc[:first_valid] = imputed.iloc[:first_valid].fillna(leading_mean)

        # Fill any trailing gaps with the recent observed average to avoid bias at the end of the series.
        if last_valid is not None and last_valid < len(numeric_series) - 1:
            trailing_mean = observed.iloc[-window:].mean()
            imputed.iloc[last_valid + 1 :] = imputed.iloc[last_valid + 1 :].fillna(trailing_mean)

        # Interpolate the remaining interior missing points between observed data.
        moving_estimates = trailing_average.fillna(leading_average).interpolate(
            limit_area="inside",
            limit_direction="both",
        )
        imputed = imputed.fillna(moving_estimates)

    return imputed


def assess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, normalize, and impute a market/COVID dataset.

    The function converts dates, catches duplicate and invalid rows, repairs known
    spelling issues in column names, fills missing indicator data, and returns a
    summary explaining what was corrected.
    """
    if "date" not in data.columns:
        raise ValueError("Input data does not contain a 'date' column.")

    assessed = data.copy()

    # Convert the date column to a proper datetime dtype and record basic quality issues.
    assessed["date"] = pd.to_datetime(assessed["date"], errors="coerce")
    invalid_date_rows = int(assessed["date"].isna().sum())
    duplicate_date_rows = int(assessed["date"].duplicated().sum())
    duplicate_date_values = int(
        assessed.loc[assessed["date"].duplicated(keep=False), "date"].nunique()
    )
    assessed = assessed.sort_values("date", kind="stable", na_position="last").reset_index(drop=True)

    # Fix recurring spelling mistakes in column names that are used later in analysis.
    rename_map = {
        column: re.sub(
            r"entertaiment|groce",
            lambda match: {
                "entertaiment": "entertainment",
                "groce": "grocery",
            }[match.group(0).lower()],
            column,
            flags=re.IGNORECASE,
        )
        for column in assessed.columns
        if re.search(r"entertaiment|groce", column, flags=re.IGNORECASE)
    }
    assessed = assessed.rename(columns=rename_map)

    # Impute missing values for market indicator fields using short rolling windows.
    imputed_counts: dict[str, int] = {}
    moving_average_columns = [
        column
        for column in assessed.columns
        if column in {"rsi", "pli"} or column.startswith("return_")
    ]
    for column in moving_average_columns:
        window = 14 if column in {"rsi", "pli"} else 10
        missing_before = int(assessed[column].isna().sum())
        assessed[column] = _moving_average_impute(assessed[column], window)
        imputed_counts[column] = missing_before - int(assessed[column].isna().sum())

    # For pandemic variables, a missing value is treated as zero rather than leaving a gap.
    for column in zero_impute_cols:
        if column in assessed.columns:
            missing_before = int(assessed[column].isna().sum())
            assessed[column] = assessed[column].fillna(0)
            imputed_counts[column] = missing_before

    summary = {
        "rows": len(assessed),
        "invalid_date_rows": invalid_date_rows,
        "duplicate_date_values": duplicate_date_values,
        "duplicate_date_rows": duplicate_date_rows,
        "renamed_columns": rename_map,
        "imputed_values": imputed_counts,
        "remaining_missing_values": assessed.isna().sum().loc[lambda counts: counts > 0].to_dict(),
    }
    return assessed, summary


def run_assessment(
    input_path: Path = input_path,
    output_path: Path = output_path,
    row_limit: int | None = DEFAULT_ROW_LIMIT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, assess, and export a cleaned market dataset.

    If a row limit is supplied, the script keeps only the first N rows to match the
    study window. This makes the historical cutoff explicit and easier to change.
    """
    data = pd.read_csv(input_path)
    if row_limit is not None:
        data = data.iloc[:row_limit]

    assessed, summary = assess_data(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assessed.to_csv(output_path, index=False)
    return assessed, summary


if __name__ == "__main__":
    _, assessment_summary = run_assessment()
    print("Data assessment completed.")
    print(f"Rows: {assessment_summary['rows']:,}")
    print(f"Invalid date rows: {assessment_summary['invalid_date_rows']:,}")
    print(f"Duplicate date values: {assessment_summary['duplicate_date_values']:,}")
    print(f"Duplicate date rows: {assessment_summary['duplicate_date_rows']:,}")
    print(f"Renamed columns: {assessment_summary['renamed_columns']}")
    print(f"Imputed values: {assessment_summary['imputed_values']}")
    print(f"Remaining missing values: {assessment_summary['remaining_missing_values']}")
    print(f"Output file: {output_path}")

