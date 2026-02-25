# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview
Quantra is a Python-based quantitative/algorithmic trading system for backtesting and live-trading VWAP/Bollinger Band strategies on US ETFs (QQQ, SPY, TQQQ) and a pairs trading strategy on Hang Seng Tech stocks.

### Key Services

| Service | Required? | Notes |
|---------|-----------|-------|
| Python 3.10+ with pip dependencies | **Yes** | `pip install -r requirements.txt` |
| Longport API credentials | Only for data fetching / live trading | Set via `.env` (copy from `.env.template`) |
| Interactive Brokers Gateway | Optional | Only for `ib_data_fetch.py` / `ib_test.py` |

### Running the application

- **Backtesting (core functionality):** `python3 backtest.py` — requires a CSV data file (default: `qqq_longport.csv`). The data path is configured inside `backtest.py`'s `config` dict.
- **FTMO analysis:** `python3 ftmo_analysis.py` — runs Monte Carlo simulations of prop firm challenge pass rates.
- **Prop firm test:** `python3 prop_firm_test.py` — simulates FTMO exam scenarios.
- **Pairs trading (HStech):** `cd pairs_trading_hstech && python3 daily_reversal_hedge_strategy.py` — requires daily CSV files in `pairs_trading_hstech/daily_data/`.
- **Data processing:** `python3 process_data.py <input.csv>` — filters market hours and adds MACD indicators.
- **Live trading:** `python3 simulate.py` — connects to Longport API for real-time trading (requires valid API credentials in `.env`).

### Important caveats

1. **No CSV data files are committed to the repo.** Backtesting scripts will fail without data. You must either:
   - Fetch data using `data_fetch_from_longport*.py` scripts (requires Longport API credentials), or
   - Generate synthetic data for testing purposes.
2. **Data file paths are hardcoded** in each script's `config` dict (e.g., `'data_path': 'qqq_longport.csv'`). Adjust as needed.
3. **matplotlib plots** use the Agg backend by default in headless environments. If you need to display plots, set `MPLBACKEND=Agg` and save to files.
4. **No automated test suite exists.** Validate changes by running the backtest scripts and checking output.
5. **Linting:** Use `python3 -m pyflakes <file>` or `python3 -m pycodestyle --max-line-length=200 <file>`. Pre-existing warnings exist (unused imports, f-string placeholders).
