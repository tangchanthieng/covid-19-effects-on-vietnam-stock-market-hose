import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import clean_dir, assets_dir, pca_dir


input_file = clean_dir / "data_020118_300522_subset.csv"
output_file = pca_dir / "investor_sentiment_index.csv"
plot_file = assets_dir / "isi_correlation_map.png"
comparison_plot_file = assets_dir / "isi_isis_comparison.png"

PCA_VARIABLES = [
    "market_return",
    "market_volume",
    "market_liquidity",
    "rsi",
    "pli",
    "oil",
    "gold",
    "exchange",
]
ISIS_VARIABLES = [
    "market_return_lag1",
    "market_volume",
    "market_liquidity",
    "rsi",
    "pli",
    "oil",
    "gold",
    "exchange",
]


def calculate_first_component(
    data: pd.DataFrame,
    variables: list[str],
    index_name: str,
) -> tuple[pd.Series, PCA, int]:
    numeric_data = data[variables].apply(pd.to_numeric, errors="coerce")
    complete_rows = numeric_data.notna().all(axis=1)
    if complete_rows.sum() < 2:
        raise ValueError(f"At least two complete observations are required to calculate {index_name}.")

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numeric_data.loc[complete_rows])
    pca = PCA(n_components=1)
    component = pca.fit_transform(standardized_data).ravel()

    # PCA signs are arbitrary; orient both indices toward the RSI and PLI indicators.
    sentiment_anchor = sum(
        pca.components_[0, variables.index(variable)] for variable in ("rsi", "pli")
    )
    if sentiment_anchor < 0:
        component *= -1

    index = pd.Series(float("nan"), index=data.index, name=index_name)
    index.loc[complete_rows] = component
    return index, pca, int(complete_rows.sum())


def main() -> None:
    data = pd.read_csv(input_file, parse_dates=["date"])
    required_columns = {"date", *PCA_VARIABLES}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {sorted(missing_columns)}")

    data = data.sort_values("date", kind="stable").reset_index(drop=True)
    lagged_variables = [f"{variable}_lag1" for variable in PCA_VARIABLES]
    for variable, lagged_variable in zip(PCA_VARIABLES, lagged_variables):
        data[lagged_variable] = pd.to_numeric(data[variable], errors="coerce").shift(1)

    data["ISI"], isi_pca, isi_observations = calculate_first_component(
        data, PCA_VARIABLES, "ISI"
    )
    data["ISIS"], isis_pca, isis_observations = calculate_first_component(
        data, ISIS_VARIABLES, "ISIS"
    )

    correlation_variables = PCA_VARIABLES + lagged_variables
    correlations = data[["ISI", *correlation_variables]].corr().loc[["ISI"], correlation_variables]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False)

    plt.figure(figsize=(16, 2.8))
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("ISI Correlation with Market Indicators and Their One-Day Lags")
    plt.xlabel("Variable (lag1 = previous trading-day observation)")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()

    paired_indices = data[["ISI", "ISIS"]].dropna()
    index_correlation = paired_indices["ISI"].corr(paired_indices["ISIS"])
    comparison = pd.DataFrame(
        {
            "observations": [isi_observations, isis_observations],
            "variance_explained": [
                isi_pca.explained_variance_ratio_[0],
                isis_pca.explained_variance_ratio_[0],
            ],
            "score_mean": [paired_indices["ISI"].mean(), paired_indices["ISIS"].mean()],
            "score_std": [paired_indices["ISI"].std(), paired_indices["ISIS"].std()],
        },
        index=["ISI", "ISIS"],
    )
    loading_variables = list(dict.fromkeys(PCA_VARIABLES + ISIS_VARIABLES))
    loadings = pd.DataFrame(
        {
            "ISI_loading": pd.Series(isi_pca.components_[0], index=PCA_VARIABLES),
            "ISIS_loading": pd.Series(isis_pca.components_[0], index=ISIS_VARIABLES),
        }
    ).reindex(loading_variables)

    standardized_indices = paired_indices.apply(lambda column: (column - column.mean()) / column.std())
    assets_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(data.loc[paired_indices.index, "date"], standardized_indices["ISI"], label="ISI")
    plt.plot(data.loc[paired_indices.index, "date"], standardized_indices["ISIS"], label="ISIS", alpha=0.8)
    plt.title("ISI and ISIS Comparison (Standardized Scores)")
    plt.xlabel("Date")
    plt.ylabel("Standardized index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_plot_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"ISI observations: {isi_observations:,}; variance explained: {isi_pca.explained_variance_ratio_[0]:.2%}")
    print(f"ISIS observations: {isis_observations:,}; variance explained: {isis_pca.explained_variance_ratio_[0]:.2%}")
    print(f"Correlation between ISI and ISIS: {index_correlation:.4f}")
    print("Index summary:")
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    print("PCA loadings:")
    print(loadings.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"ISI data saved to: {output_file}")
    print(f"Correlation map saved to: {plot_file}")
    print(f"Index comparison plot saved to: {comparison_plot_file}")

if __name__ == "__main__":
    main()