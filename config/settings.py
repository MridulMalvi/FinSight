import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:root@localhost:3306/finsight_ai"
)
_raw_model_path = os.getenv("MODEL_PATH", "data/models/stock_model.joblib")
if Path(_raw_model_path).is_absolute():
    MODEL_PATH = _raw_model_path
else:
    MODEL_PATH = str(BASE_DIR / _raw_model_path)

# ─── Marketaux Financial News API ────────────────────────────────────────────
MARKETAUX_API_KEY: str = os.getenv("MARKETAUX_API_KEY", "")
MARKETAUX_BASE_URL: str = "https://api.marketaux.com/v1"

