import pandas as pd
from pathlib import Path
from config import raw_dir

# File paths
market_data = raw_dir/"market_data.csv"
covid_cases = raw_dir/"covid_cases.csv"
death_cases = raw_dir/"death_cases.csv"
merged_cases_deaths = raw_dir/"cases_deaths.csv"
merged_market = raw_dir/"merged_market.csv"

# Load CSV files
df_covid_cases = pd.read_csv(covid_cases)
df_death_cases = pd.read_csv(death_cases)
df_market_data = pd.read_csv(market_data)

print(f"Covid cases loaded: {len(df_covid_cases):,} rows")
print(f"Death cases loaded: {len(df_death_cases):,} rows")
print(f"Market data loaded: {len(df_market_data):,} rows")

# Validate date columns
if "date" not in df_covid_cases.columns:
    raise ValueError("Covid cases file does not contain a 'date' column.")

if "date" not in df_death_cases.columns:
    raise ValueError("Death cases file does not contain a 'date' column.")
    
if "date" not in df_market_data.columns:
    raise ValueError("Market data file does not contain a 'date' column.")

# Check missing values in all columns
missing_covid_cases = df_covid_cases.isna().sum()
missing_death_cases = df_death_cases.isna().sum()

print("Missing values in each dataset:")
print(f"Covid cases: {missing_covid_cases.to_dict()}")
print(f"Death cases: {missing_death_cases.to_dict()}")

# Convert date columns
df_covid_cases["date"] = pd.to_datetime(
    df_covid_cases["date"],
    errors="coerce"
).dt.date

df_death_cases["date"] = pd.to_datetime(
    df_death_cases["date"],
    errors="coerce"
).dt.date

df_market_data["date"] = pd.to_datetime(
    df_market_data["date"],
    errors="coerce"
).dt.date

# LEFT JOIN keeps every date from the COVID cases file.
print("\nMerging Covid cases and death cases...")

first_merged = pd.merge(
    df_covid_cases,
    df_death_cases,
    on="date",
    how="left"
)

first_merged.to_csv(merged_cases_deaths, index=False)

# Summary
print("\nFirst merge completed successfully.")
print(f"Covid cases rows:      {len(df_covid_cases):,}")
print(f"Death cases rows:      {len(df_death_cases):,}")
print(f"Matched rows:     {len(first_merged):,}")
print(f"Output columns:   {len(first_merged.columns):,}")
print(f"Output file:      {merged_cases_deaths}")

# Load CSV files
df_cases_deaths = pd.read_csv(merged_cases_deaths)

print(f"Covid cases deaths loaded: {len(df_cases_deaths):,} rows")
print(f"Market data loaded: {len(df_market_data):,} rows")

# Convert date columns
df_cases_deaths["date"] = pd.to_datetime(
    df_cases_deaths["date"],
    errors="coerce"
).dt.date

# LEFT JOIN keeps every date from the market data file.
print("\nMerging market data with covid cases and death cases...")

second_merged = pd.merge(
    df_market_data,
    df_cases_deaths,
    on="date",
    how="left"
)

second_merged.to_csv(merged_market, index=False)

# Summary
print("\nSecond merge completed successfully.")
print(f"Covid cases and deaths rows:      {len(df_cases_deaths):,}")
print(f"Market data rows:      {len(df_market_data):,}")
print(f"Matched rows:     {len(second_merged):,}")
print(f"Output columns:   {len(second_merged.columns):,}")
print(f"Output file:      {merged_market}")