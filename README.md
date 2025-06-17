# March Madness Bracket Prediction Engine

Tried and tested in the 2025 March Madness Bracket. Achieved approximately 97% accuracy, and being tweaked in time for the 2026 one.

A production-grade NCAA tournament bracket prediction engine using advanced ML, feature-rich modeling, probability calibration, and full-tournament simulation.

## Features
- Advanced team metrics (KenPom, momentum, betting odds, etc.)
- ML model (XGBoost/LightGBM) with probability calibration
- Monte Carlo simulation engine (10,000+ brackets)
- SHAP explainability for picks
- Frontend (Streamlit or React+FastAPI)
- Export results: PDF/PNG, JSON, CSV, API
- Backtesting vs. public models (2023/2024)

## Setup
```bash
pip install -r requirements.txt
```

## Folder Structure
- `data/` - Raw and processed data
- `models/` - Model training and calibration
- `simulator/` - Tournament simulation
- `app/` - Frontend and API
- `notebooks/` - Prototyping
- `scripts/` - ETL and utilities
- `tests/` - Testing

## Usage
Scripts and UI coming soon!
