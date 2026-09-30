import os
from pathlib import Path

# File System Paths
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir/"data"
raw_dir = data_dir/"raw"
staging_dir = data_dir/"staging"
clean_dir = data_dir/"clean"