#!/usr/bin/env python3
"""
12 个月滚动收益统计：仅跑一次 2020 年起的回测，取每月收益，再在结果上做滚动 12 个月叠加，
统计所有可能的 12 个月区间的收益（无需多次跑回测）。需在 futu 环境下运行。
"""
import sys
import os
import io
import numpy as np
import pandas as pd
from datetime import date

# 与 backtest.py __main__ 中 config 完全一致，仅 data_path/start_date/end_date 覆盖
BASE_CONFIG = {
    'data_path': None,
    'ticker': 'QQQ',
    'initial_capital': 100000,
    'lookback_days': 1,
    'start_date': None,
    'end_date': None,
    'check_interval_minutes': 15,
    'enable_transaction_fees': True,
    'transaction_fee_per_share': 0.008166,
    'slippage_per_share': 0.01,
    'trading_start_time': (9, 40),
    'trading_end_time': (15, 40),
    'max_positions_per_day': 10,
    'print_daily_trades': False,
    'print_trade_details': False,
    'K1': 1,
    'K2': 1,
    'leverage': 3,
    'use_vwap': False,
    'enable_intraday_stop_loss': False,
    'intraday_stop_loss_pct': 0.045,
    'enable_trailing_take_profit': True,
    'trailing_tp_activation_pct': 0.01,
    'trailing_tp_callback_pct': 0.7,
}

# 2020 年起，每个数据文件的回测区间（end 取数据可用末尾）
DATA_RANGES = {
    'qqq_longport.csv': (date(2024, 2, 1), date(2026, 2, 17)),   # 仅有 2024 年后
    'qqq_market_hours_with_indicators.csv': (date(2020, 1, 1), date(2025, 4, 4)),
}


def run_once_and_rolling_12m(data_path, start_date, end_date):
    """
    跑一次回测（2020 年后），取月度收益，再计算滚动 12 个月叠加收益。
    返回: (data_path, monthly_returns_series, rolling_12m_series) 或 (data_path, None, None) 若报错。
    """
    config = {**BASE_CONFIG, 'data_path': data_path, 'start_date': start_date, 'end_date': end_date}
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        import backtest
        daily_df, monthly, trades_df, metrics = backtest.run_backtest(config)
        sys.stdout, sys.stderr = old_stdout, old_stderr
        if monthly is None or 'monthly_return' not in monthly.columns:
            return (data_path, None, None)
        mr = monthly['monthly_return'].dropna()
        if len(mr) < 12:
            return (data_path, mr, None)
        # 滚动 12 个月：连续 12 个月收益叠加 (1+r1)*(1+r2)*...*(1+r12) - 1
        roll = (1 + mr).rolling(12).apply(lambda x: x.prod() - 1.0, raw=True)
        roll = roll.dropna()
        return (data_path, mr, roll)
    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        print(f"[{data_path}] 回测失败: {e}", file=sys.stderr)
        return (data_path, None, None)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("仅跑一次 2020 年后的回测，用每月收益做滚动 12 个月叠加统计。")
    print()

    for data_path, (start_date, end_date) in DATA_RANGES.items():
        print(f"回测: {data_path}  ({start_date} ~ {end_date})")
        _, monthly_ret, rolling_12 = run_once_and_rolling_12m(data_path, start_date, end_date)
        if monthly_ret is None:
            print(f"  跳过（无月度数据或失败）\n")
            continue
        if rolling_12 is None or len(rolling_12) == 0:
            print(f"  月度数 {len(monthly_ret)} < 12，无法计算滚动 12 个月。\n")
            continue

        pos = (rolling_12 > 0).sum()
        neg = (rolling_12 <= 0).sum()
        total = len(rolling_12)
        print(f"  滚动 12 个月窗口数: {total}  正收益: {pos}  非正收益: {neg}")
        print(f"  12 个月回报范围: {rolling_12.min()*100:.1f}% ~ {rolling_12.max()*100:.1f}%")

        if neg > 0:
            neg_periods = rolling_12[rolling_12 <= 0].sort_values()
            print(f"  非正收益的 12 个月区间（结束月）:")
            for end_ts, r in neg_periods.items():
                try:
                    start_ts = end_ts - pd.DateOffset(months=11)
                    print(f"    {start_ts.strftime('%Y-%m')} ~ {end_ts.strftime('%Y-%m')}  回报: {r*100:.2f}%")
                except Exception:
                    print(f"    {end_ts}  回报: {r*100:.2f}%")
        print()

    print("=" * 60)
    print("说明: 每个 12 个月区间为「结束月」对应的过去 12 个月收益叠加，无需多次跑回测。")


if __name__ == "__main__":
    main()
