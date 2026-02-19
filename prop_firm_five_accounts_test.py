"""
五账户持续在考模拟：从指定起始日到结束日，始终保持 5 个账户在考试。
- 爆仓则新买考试（$500），下一交易日开新账户补位；
- 通过（两轮都过）则累计通过时资金，再新买考试补位。
统计：通过考试的资金合计、考试费合计、通过/爆仓次数。
"""

import pandas as pd
from datetime import date
from math import floor

from backtest import simulate_day
from prop_firm_test import prepare_backtest_data

EXAM_FEE = 500  # 单次考试费用（美元）


def run_one_exam_day(price_df, allowed_times, filtered_dates, config,
                     phase, capital, day_idx, initial_capital,
                     max_dd_pct_so_far, max_daily_dd_pct_so_far,
                     profit_target_pct, max_total_drawdown_pct, max_daily_drawdown_pct,
                     leverage):
    """
    模拟单个交易日的考试步进。phase=1 目标+10%, phase=2 目标+5%。
    返回: dict with status 'continue'|'passed'|'failed'|'insufficient_data',
         new_capital, new_max_dd_pct, new_max_daily_dd_pct, fail_reason (可选)
    """
    T = len(filtered_dates)
    if day_idx >= T:
        return {'status': 'insufficient_data', 'new_capital': capital}
    trade_date = filtered_dates[day_idx]
    equity_floor = initial_capital * (1 - max_total_drawdown_pct)
    target_capital = initial_capital * (1 + profit_target_pct)

    day_data = price_df[price_df['Date'] == trade_date].copy()
    day_data = day_data.sort_values('DateTime').reset_index(drop=True)
    if len(day_data) < 10:
        return {'status': 'continue', 'new_capital': capital,
                'new_max_dd_pct': max_dd_pct_so_far, 'new_max_daily_dd_pct': max_daily_dd_pct_so_far}
    prev_close_val = day_data['prev_close'].iloc[0]
    if pd.isna(prev_close_val):
        return {'status': 'continue', 'new_capital': capital,
                'new_max_dd_pct': max_dd_pct_so_far, 'new_max_daily_dd_pct': max_daily_dd_pct_so_far}

    day_open_price = day_data['day_open'].iloc[0]
    position_size = floor(capital * leverage / day_open_price)
    if position_size <= 0:
        return {'status': 'continue', 'new_capital': capital,
                'new_max_dd_pct': max_dd_pct_so_far, 'new_max_daily_dd_pct': max_daily_dd_pct_so_far}

    day_start_equity = capital
    exam_config = config.copy()
    exam_config['initial_capital'] = initial_capital
    exam_config['print_trade_details'] = False
    exam_config['print_daily_trades'] = False

    trades, _, intraday_low, intraday_high = simulate_day(
        day_data, prev_close_val, allowed_times, position_size, exam_config, capital
    )
    day_pnl = sum(t['pnl'] for t in trades)
    capital += day_pnl

    daily_dd_pct = max(0, (day_start_equity - intraday_low) / day_start_equity)
    max_daily_dd_pct_new = max(max_daily_dd_pct_so_far, daily_dd_pct)
    lowest_equity = min(capital, intraday_low)
    total_dd_pct = max(0, (initial_capital - lowest_equity) / initial_capital)
    max_dd_pct_new = max(max_dd_pct_so_far, total_dd_pct)

    if daily_dd_pct >= max_daily_drawdown_pct:
        return {'status': 'failed', 'new_capital': capital,
                'new_max_dd_pct': max_dd_pct_new, 'new_max_daily_dd_pct': max_daily_dd_pct_new,
                'fail_reason': f'日内回撤超限 {daily_dd_pct*100:.2f}% (日期:{trade_date})'}
    if lowest_equity <= equity_floor:
        return {'status': 'failed', 'new_capital': capital,
                'new_max_dd_pct': max_dd_pct_new, 'new_max_daily_dd_pct': max_daily_dd_pct_new,
                'fail_reason': f'总回撤超限 最低${lowest_equity:,.0f} (日期:{trade_date})'}
    if capital >= target_capital:
        return {'status': 'passed', 'new_capital': capital,
                'new_max_dd_pct': max_dd_pct_new, 'new_max_daily_dd_pct': max_daily_dd_pct_new}
    return {'status': 'continue', 'new_capital': capital,
            'new_max_dd_pct': max_dd_pct_new, 'new_max_daily_dd_pct': max_daily_dd_pct_new}


def run_five_accounts_full_period(price_df, allowed_times, filtered_dates, config,
                                  initial_capital=100000, leverage=3, verbose=True):
    """
    从第一个交易日到最后一个交易日，始终保持 5 个账户在考试。
    返回: total_passed_capital, total_exam_fees, num_passed, num_failed, total_exams_started
    """
    T = len(filtered_dates)
    if T < 5:
        return 0.0, 5 * EXAM_FEE, 0, 0, 5

    slots = []
    for i in range(5):
        slots.append({
            'start_idx': i,
            'phase': 1,
            'capital': initial_capital,
            'day_offset': 0,
            'max_dd_pct': 0.0,
            'max_daily_dd_pct': 0.0,
        })
    total_exam_fees = 5 * EXAM_FEE
    total_passed_capital = 0.0
    num_passed = 0
    num_failed = 0
    total_exams_started = 5

    p1_target, p1_max_dd, p1_daily_dd = 0.10, 0.10, 0.05
    p2_target, p2_max_dd, p2_daily_dd = 0.05, 0.10, 0.05
    progress_interval = max(1, T // 20)

    for t in range(T):
        trade_date = filtered_dates[t]
        if verbose and t > 0 and t % progress_interval == 0:
            print(f"  [进度] 已模拟到 {trade_date} (第 {t+1}/{T} 个交易日) | 通过 {num_passed} 个, 爆仓 {num_failed} 个")

        for i, slot in enumerate(slots):
            local_day_idx = slot['start_idx'] + slot['day_offset']
            if local_day_idx != t:
                continue
            if slot['phase'] == 1:
                target_pct, max_total_dd, max_daily_dd = p1_target, p1_max_dd, p1_daily_dd
            else:
                target_pct, max_total_dd, max_daily_dd = p2_target, p2_max_dd, p2_daily_dd

            res = run_one_exam_day(
                price_df, allowed_times, filtered_dates, config,
                phase=slot['phase'],
                capital=slot['capital'],
                day_idx=local_day_idx,
                initial_capital=initial_capital,
                max_dd_pct_so_far=slot['max_dd_pct'],
                max_daily_dd_pct_so_far=slot['max_daily_dd_pct'],
                profit_target_pct=target_pct,
                max_total_drawdown_pct=max_total_dd,
                max_daily_drawdown_pct=max_daily_dd,
                leverage=leverage,
            )
            status = res['status']

            if status == 'continue':
                slot['capital'] = res['new_capital']
                slot['max_dd_pct'] = res['new_max_dd_pct']
                slot['max_daily_dd_pct'] = res['new_max_daily_dd_pct']
                slot['day_offset'] += 1
                continue
            if status == 'insufficient_data':
                continue

            if status == 'failed':
                num_failed += 1
                total_exam_fees += EXAM_FEE
                total_exams_started += 1
                if verbose:
                    phase_str = "第一轮" if slot['phase'] == 1 else "第二轮"
                    reason = res.get('fail_reason', '未知')
                    print(f"  [爆仓] {trade_date} 槽位{i+1} ({phase_str}): {reason}")
                next_start = t + 1
                slots[i] = {
                    'start_idx': next_start if next_start < T else T,
                    'phase': 1,
                    'capital': initial_capital,
                    'day_offset': 0,
                    'max_dd_pct': 0.0,
                    'max_daily_dd_pct': 0.0,
                }
                continue

            if status == 'passed':
                if slot['phase'] == 1:
                    if verbose:
                        print(f"  [通过P1] {trade_date} 槽位{i+1} 第一轮通过，进入第二轮")
                    next_start = t + 1
                    slots[i] = {
                        'start_idx': next_start if next_start < T else T,
                        'phase': 2,
                        'capital': initial_capital,
                        'day_offset': 0,
                        'max_dd_pct': 0.0,
                        'max_daily_dd_pct': 0.0,
                    }
                else:
                    cap = res['new_capital']
                    total_passed_capital += cap
                    num_passed += 1
                    total_exam_fees += EXAM_FEE
                    total_exams_started += 1
                    if verbose:
                        print(f"  [通过考试] {trade_date} 槽位{i+1} 两轮通过，通过时资金 ${cap:,.0f} (累计通过 {num_passed} 个)")
                    next_start = t + 1
                    slots[i] = {
                        'start_idx': next_start if next_start < T else T,
                        'phase': 1,
                        'capital': initial_capital,
                        'day_offset': 0,
                        'max_dd_pct': 0.0,
                        'max_daily_dd_pct': 0.0,
                    }
                continue

    return total_passed_capital, total_exam_fees, num_passed, num_failed, total_exams_started


if __name__ == "__main__":
    INITIAL_CAPITAL = 100000
    config = {
        'data_path': 'qqq_longport.csv',
        'ticker': 'QQQ',
        'initial_capital': INITIAL_CAPITAL,
        'lookback_days': 1,
        'start_date': date(2024, 2, 1),
        'end_date': date(2026, 2, 20),
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
        'enable_intraday_stop_loss': True,
        'intraday_stop_loss_pct': 0.045,
        'enable_trailing_take_profit': True,
        'trailing_tp_activation_pct': 0.01,
        'trailing_tp_callback_pct': 0.7,
    }
    print("五账户持续在考模拟 (2024-02-01 ~ 2026-02-20, 3x 杠杆)")
    print("预处理回测数据...")
    price_df, allowed_times, filtered_dates = prepare_backtest_data(config)
    print(f"可用交易日: {len(filtered_dates)} ({filtered_dates[0]} ~ {filtered_dates[-1]})")
    print()
    total_passed_capital, total_exam_fees, num_passed, num_failed, total_exams = run_five_accounts_full_period(
        price_df, allowed_times, filtered_dates, config,
        initial_capital=INITIAL_CAPITAL,
        leverage=3,
        verbose=True,
    )
    print()
    print("=" * 70)
    print("五账户持续在考模拟结果 (2024-02-01 ~ 2026-02-20, 3x 杠杆)")
    print("=" * 70)
    print(f"  通过考试的资金合计: ${total_passed_capital:,.2f}")
    print(f"  考试费合计: ${total_exam_fees:,.2f} (单次 ${EXAM_FEE}, 共 {total_exams} 次考试)")
    print(f"  通过账户数: {num_passed}  |  爆仓/失败数: {num_failed}")
    print("=" * 70)
