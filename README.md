# FinSight AI — Multi-Horizon Stock Prediction & News Intelligence Engine

**FinSight AI** is an advanced quantitative machine learning and natural language processing (NLP) platform designed for probabilistic stock-direction forecasting (`UP`, `NEUTRAL`, `DOWN`), real-time financial news intelligence, company-aware semantic impact analysis, and multi-horizon target price forecasting.

The platform combines a **FastAPI** backend, an ensemble of **XGBoost + LightGBM** classifiers, Indian market & global data providers (**Yahoo Finance** & historical CSV ingestion), domain-specific **Indian financial NLP**, **role-based access control (RBAC)**, **per-user analysis quotas**, and a **fintech glassmorphism web dashboard**.

---

## 🌟 Key Features

### 1. Multi-Horizon ML Forecasting
Dedicated retrained ensemble models provide calibrated directional probabilities and projected price targets for four distinct forecast windows:
- **Next Day** (1 trading day, `stock_model_1d.joblib`)
- **1 Week** (5 trading days, `stock_model_5d.joblib`)
- **15 Days** (10 trading days, `stock_model_10d.joblib`)
- **1 Month** (21 trading days, `stock_model_21d.joblib`)
- **News-Enhanced Next Day** (1 trading day with live sentiment, `stock_model_1d_news.joblib`)

### 2. Trained on 10 Benchmark Company Datasets
Trained on **44,658 combined trading sessions** across 10 diverse Indian equities and index ETFs:
1. **ASHOKLEY** — Ashok Leyland Ltd (Automotive / Commercial Vehicles & EVs)
2. **BEL** — Bharat Electronics Ltd (Aerospace & Defence Electronics PSU)
3. **GOLDBEES** — Nippon India ETF Gold BeES (Precious Metals / Gold Commodity)
4. **HUDCO** — Housing and Urban Development Corp Ltd (Housing & Urban Infra Lending PSU)
5. **IDBI** — IDBI Bank Ltd (Banking & Strategic Disinvestment candidate)
6. **IOC** — Indian Oil Corporation Ltd (Oil Refining & Retail Fuel Marketing PSU)
7. **IRFC** — Indian Railway Finance Corporation Ltd (Railway Capex Financing PSU)
8. **MID150BEES** — Nippon India ETF Nifty Midcap 150 (Broader Market Midcap Index)
9. **POWERGRID** — Power Grid Corporation of India Ltd (Power Transmission Utility PSU)
10. **SUZLON** — Suzlon Energy Ltd (Renewable Wind Turbine Manufacturing & EPC)

### 3. Dual-Engine Intelligence (Technical + NLP News)
- **43-Feature Quantitative Pipeline**:
  - **Returns**: 1d, 3d, 5d, 10d, 20d, 30d, 60d
  - **Momentum & Oscillators**: RSI, MACD, MACD Signal, MACD Hist, Stochastic Oscillator, Commodity Channel Index (CCI), Williams %R
  - **Volatility & Trend**: Bollinger Bands (Width & %B), Average True Range (ATR %), 20-Day Realized Volatility, Daily Range, ADX, Moving Average Deviations (SMA-20, SMA-50, SMA-200), Price Acceleration
  - **Structural & Volume Flow**: On-Balance Volume (OBV), Volume Ratio, Ichimoku Cloud Direction, Price vs Cloud, 52-Week Range Position, % from 52-Week High, 10-day Momentum
  - **News & Sentiment Indicators**: 12 aggregated features (sentiment score, impact, importance, volume, positive/negative ratios, event counters)
- **Company-Aware Semantic Engine**:
  - Maps news to company-specific themes (e.g. defense orders for BEL, CV sales for ASHOKLEY, GRMs for IOC, WTG orders for SUZLON, PMAY outlays for HUDCO, disinvestment for IDBI, railway capex for IRFC).
  - Directional sensitivity weights (`[-1.0, +1.0]`) scale probability distributions and price targets.


### 4. Balanced, Anti-Bias Machine Learning
- **SMOTE & Resampling**: Eliminates majority-class (`NEUTRAL`) bias by balancing training samples across all classes.
- **Soft-Voting Ensemble**: 55% XGBoost + 45% LightGBM booster ensemble.
- **Per-Class Threshold Calibration**: Calibrated decision boundaries ensure balanced recall (>30%) for `DOWN`, `NEUTRAL`, and `UP`.

### 5. Security, Governance & Quota Management
- **Strict Access Control**: Public registration is disabled. Accounts are provisioned exclusively by administrators.
- **Per-User Quota Enforcement**: Regular users receive a quota of **7 stock analyses**, with visual progress tracking and modal alerts.
- **Audit Logging & Activity Tracking**: Every forecast is recorded in the audit database with user identity, ticker symbol, probabilities, and price targets.

---


## 🏗️ Project Structure

```text
stock_ai_news_engine/
├── README.md                       # Comprehensive system documentation
└── stock_ai/                       # Application root
    ├── api/                        # REST API Layer (FastAPI)
    │   ├── admin.py                # Admin dashboard, audit logs, and user management
    │   ├── auth.py                 # Bearer token authentication & JWT sessions
    │   └── main.py                 # Core prediction routes and frontend static mount
    ├── config/                     # Configuration & environment loaders
    │   └── settings.py
    ├── data/
    │   ├── models/                 # Trained Ensemble joblib bundles (.joblib)
    │   │   ├── stock_model.joblib          # Default 1d alias
    │   │   ├── stock_model_1d.joblib       # Dedicated next-day model
    │   │   ├── stock_model_5d.joblib       # Dedicated 1-week model
    │   │   ├── stock_model_10d.joblib      # Dedicated 15-day model
    │   │   ├── stock_model_21d.joblib      # Dedicated 1-month model
    │   │   └── stock_model_1d_news.joblib  # News-enhanced 1-day model
    │   ├── raw/                    # 10 lifetime benchmark CSV datasets
    │   │   ├── ASHOKLEY_lifetime_historical.csv
    │   │   ├── BEL_lifetime_historical.csv
    │   │   ├── GOLDBEES_lifetime_historical.csv
    │   │   ├── HUDCO_lifetime_historical.csv
    │   │   ├── IDBI_lifetime_historical.csv
    │   │   ├── IOC_lifetime_historical.csv
    │   │   ├── IRFC_lifetime_historical.csv
    │   │   ├── MID150BEES_lifetime_historical.csv
    │   │   ├── POWERGRID_lifetime_historical.csv
    │   │   └── SUZLON_lifetime_historical.csv
    │   └── finsight_ai.db          # Local SQLite database (auto-initialized)
    ├── database/                   # Database abstraction layer
    │   ├── connection.py           # MySQL engine setup with zero-config SQLite fallback
    │   ├── models.py               # SQLAlchemy models (User, UserActivity, AllowedSymbol)
    │   └── schema.sql              # Database table schemas
    ├── features/                   # Technical indicator computations
    │   └── technical.py            # 43 technical & statistical feature extractors
    ├── frontend/                   # Web application
    │   └── index.html              # Vanilla JS/CSS glassmorphic dashboard
    ├── ingestion/                  # Market data ingestion
    │   ├── csv_provider.py         # Multi-format historical CSV parser (Handles formats A, B, C)
    │   └── yfinance_provider.py    # Real-time Yahoo Finance market data provider
    ├── ml/                         # Machine learning engine
    │   ├── dataset.py              # Balanced dataset preparation & SMOTE resampling
    │   ├── model.py                # EnsembleModel (XGBoost 55% + LightGBM 45%)
    │   ├── predict.py              # Probabilistic inference & multi-horizon projections
    │   └── train.py                # Walk-forward CV & threshold calibration
    ├── news/                       # NLP & News Intelligence Engine
    │   ├── company_context.py      # 10 company profiles, aliases, and sensitivity weights
    │   ├── features.py             # Entity matching & time-decayed feature aggregator
    │   ├── models.py               # NewsArticle dataclass schemas
    │   ├── nlp.py                  # Lexicon analyzer with Indian financial overrides
    │   ├── provider.py             # Unified news fetcher
    │   ├── semantic.py             # Sector theme detection & company impact engine
    │   └── yahoo.py                # Yahoo Finance news provider & precision query builder
    ├── scripts/                    # Automation and management scripts
    │   └── retrain_all.py          # Complete multi-horizon retraining pipeline
    └── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10, 3.11, or 3.12**
- Git
- Optional: MySQL 8.0+ (Automatic zero-setup fallback to SQLite at `stock_ai/data/finsight_ai.db`)

### 2. Clone the Repository
```bash
git clone https://github.com/GravitInfo/FinSight_AI.git
cd FinSight_AI
```

### 3. Virtual Environment Setup
- **Windows (PowerShell):**
  ```powershell
  python -m venv stock_ai\.venv
  .\stock_ai\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv stock_ai/.venv
  source stock_ai/.venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r stock_ai/requirements.txt
```

### 5. Run the Application
From the `stock_ai/` directory:
```bash
cd stock_ai
uvicorn api.main:app --reload --port 8000
```

Access the application:
- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Redoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔐 Default Access & User Governance

The system starts with an initialized administrator account:

| Username | Default Password | Role | Analysis Quota | Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| `admin` | `admin123` | **Admin** | **Unlimited** | Full prediction access, Admin Panel, User Provisioning, System Audit Logs |

### User Management Workflow:
1. Sign in with the `admin` account.
2. Open the **Admin Panel** via the shield icon in the navigation bar.
3. In the **Users** tab, click **+ Add User** to provision a new user.
4. Newly provisioned users receive a lifetime quota of **7 stock analyses** with isolated tracking.

---

## 🧠 Model Training & Retraining

You can retrain all multi-horizon models using `scripts/retrain_all.py`:

```bash
cd stock_ai

# Full retrain with cross-validation (all 10 companies):
python scripts/retrain_all.py

# Fast retrain (skipping cross-validation folds):
python scripts/retrain_all.py --no-cv
```

To train an individual horizon manually:
```bash
python -m ml.train \
  --data-dir data/raw \
  --horizon 1 \
  --threshold 0.012 \
  --output data/models/stock_model_1d.joblib
```

---

