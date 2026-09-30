from pathlib import Path
import pandas as pd
from config import clean_dir, staging_dir

data_file = staging_dir / "data_020118_300522.csv"
data_original_file = clean_dir / "data_020118_300522.csv"
data_subset_file = clean_dir / "data_020118_300522_subset.csv"


def main():
	clean_dir.mkdir(parents=True, exist_ok=True)

	data = pd.read_csv(data_file)
	data.to_csv(data_original_file, index=False)

	prefixes = ("return_", "mtb_", "me_")
	print(f"Removing columns with prefixes: {prefixes}")
	subset = data.loc[:, [not name.startswith(prefixes) for name in data.columns]]
	subset.to_csv(data_subset_file, index=False)
	print(f"Subset of data complteted. Output file: {data_subset_file}")


if __name__ == "__main__":
	main()
