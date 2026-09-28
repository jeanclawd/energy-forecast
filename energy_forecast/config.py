import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ENERGY_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("ENERGY_MODEL_DIR", ROOT / "models"))
DB_PATH = Path(os.environ.get("ENERGY_DB", DATA_DIR / "energy.db"))
MARKET = os.environ.get("ENERGY_MARKET", "france")

TARGETS = ("load", "pv")
# Forecast horizon: from the run time to the end of the next local day.
HISTORY_DAYS = int(os.environ.get("ENERGY_HISTORY_DAYS", 730))
