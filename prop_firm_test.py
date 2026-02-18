"""
Prop Firm (FTMO) 考试通过率模拟测试

复用 backtest.py 的 simulate_day 逻辑，均匀分布不同的考试开始日期，
模拟在不同时间段参加 FTMO 考试的通过率。

FTMO 规则:
- 第一轮: 初始 $100,000, 目标盈利 10%, 最大总回撤 ≤ 10%, 日内回撤 ≤ 5%
- 第二轮: 初始 $100,000, 目标盈利 5%, 其他限制相同
- 两轮都通过才算考试通过

关键定义:
- 日内回撤 (Daily Loss Limit): 当天任意时刻浮动权益不能比当天开始余额低 5%
  即: (day_start_equity - lowest_equity_of_day) / day_start_equity <= 5%
- 最大总回撤 (Max Loss): 账户权益任意时刻不能低于初始余额的 90%
  即: equity 不能跌破 $90,000 (硬底线)
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from math import floor
from backtest import simulate_day


def prepare_backtest_data(config):
    """预处理回测数据，返回处理好的 price_df 和 allowed_times。"""
    data_path = config['data_path']
    lookback_days = config.get('lookback_days', 90)
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    trading_start_time = config.get('trading_start_time', (10, 0))
    trading_end_time = config.get('trading_end_time', (15, 40))
    check_interval_minutes = config.get('check_interval_minutes', 30)
    K1 = config.get('K1', 1)
    K2 = config.get('K2', 1)

    price_df = pd.read_csv(data_path, parse_dates=['DateTime'])
    price_df.sort_values('DateTime', inplace=True)
    price_df['Date'] = price_df['DateTime'].dt.date
    price_df['Time'] = price_df['DateTime'].dt.strftime('%H:%M')

    if start_date is not None:
        price_df = price_df[price_df['Date'] >= start_date]
    if end_date is not None:
        price_df = price_df[price_df['Date'] <= end_date]

    if 'DayOpen' not in price_df.columns or 'DayClose' not in price_df.columns:
        opening_prices = price_df.groupby('Date').first().reset_index()[['Date', 'Open']].rename(columns={'Open': 'DayOpen'})
        closing_prices = price_df.groupby('Date').last().reset_index()[['Date', 'Close']].rename(columns={'Close': 'DayClose'})
        price_df = pd.merge(price_df, opening_prices, on='Date', how='left')
        price_df = pd.merge(price_df, closing_prices, on='Date', how='left')

    price_df['prev_close'] = price_df.groupby('Date')['DayClose'].transform('first').shift(1)
    price_df['day_open'] = price_df.groupby('Date')['DayOpen'].transform('first')

    unique_dates = price_df['Date'].unique()
    date_refs = []
    for d in unique_dates:
        day_data = price_df[price_df['Date'] == d].iloc[0]
        day_open = day_data['day_open']
        prev_close = day_data['prev_close']
        if not pd.isna(prev_close):
            upper_ref = max(day_open, prev_close)
            lower_ref = min(day_open, prev_close)
        else:
            upper_ref = day_open
            lower_ref = day_open
        date_refs.append({'Date': d, 'upper_ref': upper_ref, 'lower_ref': lower_ref})

    date_refs_df = pd.DataFrame(date_refs)
    if date_refs_df.empty:
        raise ValueError("没有有效的交易数据")

    price_df = price_df.drop(columns=['upper_ref', 'lower_ref'], errors='ignore')
    price_df = pd.merge(price_df, date_refs_df, on='Date', how='left')
    price_df['ret'] = price_df['Close'] / price_df['day_open'] - 1

    pivot = price_df.pivot(index='Date', columns='Time', values='ret').abs()
    sigma = pivot.rolling(window=lookback_days, min_periods=lookback_days).mean().shift(1)
    sigma = sigma.stack().reset_index(name='sigma')

    price_df = pd.merge(price_df, sigma, on=['Date', 'Time'], how='left')

    incomplete_sigma_dates = set()
    for d in price_df['Date'].unique():
        dd = price_df[price_df['Date'] == d]
        na_count = dd['sigma'].isna().sum()
        total_count = len(dd)
        if total_count > 0 and na_count / total_count > 0.1:
            incomplete_sigma_dates.add(d)

    price_df = price_df[~price_df['Date'].isin(incomplete_sigma_dates)]
    price_df['sigma'] = price_df.groupby('Date')['sigma'].ffill()
    price_df['sigma'] = price_df.groupby('Date')['sigma'].bfill()
    price_df['sigma'] = price_df['sigma'].fillna(0)

    price_df['UpperBound'] = price_df['upper_ref'] * (1 + K1 * price_df['sigma'])
    price_df['LowerBound'] = price_df['lower_ref'] * (1 - K2 * price_df['sigma'])

    allowed_times = []
    start_hour, start_minute = trading_start_time
    end_hour, end_minute = trading_end_time
    current_hour, current_minute = start_hour, start_minute
    while current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute):
        allowed_times.append(f"{current_hour:02d}:{current_minute:02d}")
        current_minute += check_interval_minutes
        if current_minute >= 60:
            current_hour += current_minute // 60
            current_minute = current_minute % 60

    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    if end_time_str not in allowed_times:
        allowed_times.append(end_time_str)
        allowed_times.sort()

    filtered_dates = sorted(price_df['Date'].unique())
    return price_df, allowed_times, filtered_dates


def simulate_exam_phase(price_df, allowed_times, filtered_dates, config,
                        exam_start_idx, initial_capital, profit_target_pct,
                        max_total_drawdown_pct, max_daily_drawdown_pct,
                        leverage):
    """
    模拟一轮 FTMO 考试，无时间限制。

    返回 dict:
        status: 'passed' | 'failed' | 'insufficient_data'
        fail_reason, days_used, final_capital, max_dd_pct, max_daily_dd_pct,
        start_date, end_date, end_idx
    """
    capital = initial_capital
    equity_floor = initial_capital * (1 - max_total_drawdown_pct)
    target_capital = initial_capital * (1 + profit_target_pct)

    max_dd_pct_actual = 0
    max_daily_dd_pct_actual = 0

    remaining_days = len(filtered_dates) - exam_start_idx
    if remaining_days <= 0:
        return {'status': 'insufficient_data'}

    start_date_val = filtered_dates[exam_start_idx]
    days_used = 0

    for day_offset in range(remaining_days):
        day_idx = exam_start_idx + day_offset
        trade_date = filtered_dates[day_idx]
        days_used += 1

        day_data = price_df[price_df['Date'] == trade_date].copy()
        day_data = day_data.sort_values('DateTime').reset_index(drop=True)

        if len(day_data) < 10:
            continue
        prev_close_val = day_data['prev_close'].iloc[0]
        if pd.isna(prev_close_val):
            continue

        day_open_price = day_data['day_open'].iloc[0]
        position_size = floor(capital * leverage / day_open_price)
        if position_size <= 0:
            continue

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

        # 日内回撤: (当天开始余额 - 当天最低权益) / 当天开始余额
        daily_dd_pct = max(0, (day_start_equity - intraday_low) / day_start_equity)
        max_daily_dd_pct_actual = max(max_daily_dd_pct_actual, daily_dd_pct)

        # 总回撤: 任意时刻权益不能低于硬底线
        lowest_equity = min(capital, intraday_low)
        total_dd_pct = max(0, (initial_capital - lowest_equity) / initial_capital)
        max_dd_pct_actual = max(max_dd_pct_actual, total_dd_pct)

        base = {
            'days_used': days_used,
            'final_capital': capital,
            'max_dd_pct': max_dd_pct_actual,
            'max_daily_dd_pct': max_daily_dd_pct_actual,
            'start_date': start_date_val,
            'end_date': trade_date,
            'end_idx': day_idx + 1,
        }

        if daily_dd_pct >= max_daily_drawdown_pct:
            return {**base, 'status': 'failed',
                    'fail_reason': f'日内回撤超限 {daily_dd_pct*100:.2f}% (日期:{trade_date})'}

        if lowest_equity <= equity_floor:
            return {**base, 'status': 'failed',
                    'fail_reason': f'总回撤超限 最低${lowest_equity:,.0f} (日期:{trade_date})'}

        if capital >= target_capital:
            return {**base, 'status': 'passed', 'fail_reason': None}

    # 数据用完未达标也未爆仓 → 数据不足，丢弃
    return {'status': 'insufficient_data'}


def run_prop_firm_simulation(leverage_list, num_samples=100):
    """运行 Prop Firm 考试模拟，均匀分布开始日期。"""
    INITIAL_CAPITAL = 100000

    base_config = {
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

    print("=" * 70)
    print("FTMO Prop Firm 考试通过率模拟测试")
    print("=" * 70)
    print(f"标的: QQQ | 数据范围: {base_config['start_date']} ~ {base_config['end_date']}")
    print(f"每个杠杆倍数均匀采样: {num_samples} 次")
    print(f"初始资金: ${INITIAL_CAPITAL:,}")
    print(f"FTMO规则:")
    print(f"  第一轮: 盈利目标 10%, 最大回撤 ≤ 10% (底线 ${INITIAL_CAPITAL*0.9:,.0f}), 日内回撤 ≤ 5%")
    print(f"  第二轮: 盈利目标 5%,  最大回撤 ≤ 10% (底线 ${INITIAL_CAPITAL*0.9:,.0f}), 日内回撤 ≤ 5%")
    print(f"  日内止损: {base_config['intraday_stop_loss_pct']*100:.1f}% (留 0.5% 余量)")
    print(f"  无时间限制 | 数据不足的样本不计入统计")
    print()

    print("预处理回测数据...")
    price_df, allowed_times, filtered_dates = prepare_backtest_data(base_config)
    total_days = len(filtered_dates)
    print(f"可用交易日: {total_days} ({filtered_dates[0]} ~ {filtered_dates[-1]})")
    print()

    # 均匀分布采样起始索引
    step = max(1, total_days // num_samples)
    sample_indices = list(range(0, total_days, step))[:num_samples]

    all_leverage_results = {}

    for leverage in leverage_list:
        print(f"\n{'='*70}")
        print(f"杠杆倍数: {leverage}x")
        print(f"{'='*70}")

        config = base_config.copy()
        config['leverage'] = leverage

        records = []  # 有效记录
        skipped = 0

        for i, start_idx in enumerate(sample_indices):
            # ===== 第一轮 =====
            r1 = simulate_exam_phase(
                price_df, allowed_times, filtered_dates, config,
                exam_start_idx=start_idx,
                initial_capital=INITIAL_CAPITAL,
                profit_target_pct=0.10,
                max_total_drawdown_pct=0.10,
                max_daily_drawdown_pct=0.05,
                leverage=leverage
            )

            if r1['status'] == 'insufficient_data':
                skipped += 1
                continue

            # ===== 第二轮 (仅第一轮通过时) =====
            r2 = None
            if r1['status'] == 'passed':
                r2 = simulate_exam_phase(
                    price_df, allowed_times, filtered_dates, config,
                    exam_start_idx=r1['end_idx'],
                    initial_capital=INITIAL_CAPITAL,
                    profit_target_pct=0.05,
                    max_total_drawdown_pct=0.10,
                    max_daily_drawdown_pct=0.05,
                    leverage=leverage
                )
                if r2['status'] == 'insufficient_data':
                    skipped += 1
                    continue

            p1_passed = r1['status'] == 'passed'
            p2_passed = r2 is not None and r2['status'] == 'passed'
            both_passed = p1_passed and p2_passed

            # 计算自然日总用时
            exam_start_date = filtered_dates[start_idx]
            if both_passed:
                exam_end_date = r2['end_date']
            elif p1_passed:
                exam_end_date = r2['end_date'] if r2 else r1['end_date']
            else:
                exam_end_date = r1['end_date']

            calendar_days = (exam_end_date - exam_start_date).days

            records.append({
                'start_date': exam_start_date,
                'p1_passed': p1_passed,
                'p1_days': r1['days_used'],
                'p1_end_date': r1['end_date'],
                'p1_profit': (r1['final_capital'] / INITIAL_CAPITAL - 1) * 100,
                'p1_max_dd': r1['max_dd_pct'] * 100,
                'p1_max_daily_dd': r1['max_daily_dd_pct'] * 100,
                'p1_fail_reason': r1.get('fail_reason'),
                'p2_passed': p2_passed,
                'p2_days': r2['days_used'] if r2 else 0,
                'p2_end_date': r2['end_date'] if r2 else None,
                'p2_profit': (r2['final_capital'] / INITIAL_CAPITAL - 1) * 100 if r2 else 0,
                'p2_max_dd': r2['max_dd_pct'] * 100 if r2 else 0,
                'p2_max_daily_dd': r2['max_daily_dd_pct'] * 100 if r2 else 0,
                'p2_fail_reason': r2.get('fail_reason') if r2 else None,
                'both_passed': both_passed,
                'calendar_days': calendar_days,
            })

            if (i + 1) % 20 == 0 or i == len(sample_indices) - 1:
                n = len(records)
                if n > 0:
                    p1_ok = sum(1 for r in records if r['p1_passed'])
                    both_ok = sum(1 for r in records if r['both_passed'])
                    print(f"  进度 {i+1}/{len(sample_indices)} | 有效:{n} 丢弃:{skipped} | "
                          f"P1通过:{p1_ok}/{n}({p1_ok/n*100:.0f}%) | "
                          f"总通过:{both_ok}/{n}({both_ok/n*100:.0f}%)")

        # ===== 统计 =====
        n = len(records)
        if n == 0:
            print("  无有效样本")
            continue

        df = pd.DataFrame(records)
        p1_pass = df['p1_passed'].sum()
        p2_attempted = df[df['p1_passed']].shape[0]
        p2_pass = df['p2_passed'].sum()
        both_pass = df['both_passed'].sum()

        print(f"\n{'─'*60}")
        print(f"杠杆 {leverage}x 结果 (有效样本: {n}, 丢弃: {skipped})")
        print(f"{'─'*60}")

        # 第一轮
        print(f"\n第一轮 (目标: +10%)")
        print(f"  通过: {p1_pass}/{n} = {p1_pass/n*100:.1f}%")
        passed1 = df[df['p1_passed']]
        failed1 = df[~df['p1_passed']]
        if len(passed1) > 0:
            print(f"  通过者: 平均{passed1['p1_days'].mean():.0f}交易日, "
                  f"平均盈利{passed1['p1_profit'].mean():.1f}%, "
                  f"平均最大回撤{passed1['p1_max_dd'].mean():.1f}%, "
                  f"平均日内最大回撤{passed1['p1_max_daily_dd'].mean():.1f}%")
        if len(failed1) > 0:
            print(f"  失败者: 平均盈利{failed1['p1_profit'].mean():.1f}%, "
                  f"平均最大回撤{failed1['p1_max_dd'].mean():.1f}%")
            reasons = failed1['p1_fail_reason'].apply(lambda x: x.split('(')[0].strip() if x else '未知')
            for reason, cnt in reasons.value_counts().items():
                print(f"    {reason}: {cnt}次")

        # 第二轮
        if p2_attempted > 0:
            print(f"\n第二轮 (目标: +5%, 仅第一轮通过者)")
            print(f"  通过: {p2_pass}/{p2_attempted} = {p2_pass/p2_attempted*100:.1f}%")
            passed2 = df[df['p2_passed']]
            failed2 = df[df['p1_passed'] & ~df['p2_passed']]
            if len(passed2) > 0:
                print(f"  通过者: 平均{passed2['p2_days'].mean():.0f}交易日, "
                      f"平均盈利{passed2['p2_profit'].mean():.1f}%")
            if len(failed2) > 0:
                print(f"  失败者: 平均盈利{failed2['p2_profit'].mean():.1f}%")
                reasons2 = failed2['p2_fail_reason'].apply(lambda x: x.split('(')[0].strip() if x else '未知')
                for reason, cnt in reasons2.value_counts().items():
                    print(f"    {reason}: {cnt}次")

        # 总通过
        print(f"\n总通过率 (两轮都通过): {both_pass}/{n} = {both_pass/n*100:.1f}%")

        # 通过者用时
        both_df = df[df['both_passed']]
        if len(both_df) > 0:
            print(f"  通过者总用时(自然日): 平均{both_df['calendar_days'].mean():.0f}天, "
                  f"最短{both_df['calendar_days'].min()}天, "
                  f"最长{both_df['calendar_days'].max()}天")

        # 逐条打印通过的案例
        if len(both_df) > 0:
            print(f"\n通过的案例:")
            print(f"  {'开始日期':<12} {'P1结束':<12} {'P2结束':<12} {'总天数':>6} "
                  f"{'P1盈利':>7} {'P2盈利':>7} {'P1回撤':>7} {'P2回撤':>7}")
            print(f"  {'─'*78}")
            for _, r in both_df.iterrows():
                print(f"  {r['start_date']}  {r['p1_end_date']}  {r['p2_end_date']}  "
                      f"{r['calendar_days']:>5}天 "
                      f"{r['p1_profit']:>+6.1f}% {r['p2_profit']:>+6.1f}% "
                      f"{r['p1_max_dd']:>6.1f}% {r['p2_max_dd']:>6.1f}%")

        # 失败案例
        failed_all = df[~df['both_passed']]
        if len(failed_all) > 0:
            print(f"\n失败的案例:")
            print(f"  {'开始日期':<12} {'阶段':>4} {'结束日期':<12} {'盈利':>7} {'回撤':>7} {'日内DD':>7} {'原因'}")
            print(f"  {'─'*85}")
            for _, r in failed_all.iterrows():
                if not r['p1_passed']:
                    print(f"  {r['start_date']}   P1  {r['p1_end_date']}  "
                          f"{r['p1_profit']:>+6.1f}% {r['p1_max_dd']:>6.1f}% "
                          f"{r['p1_max_daily_dd']:>6.1f}% {r['p1_fail_reason']}")
                else:
                    print(f"  {r['start_date']}   P2  {r['p2_end_date']}  "
                          f"{r['p2_profit']:>+6.1f}% {r['p2_max_dd']:>6.1f}% "
                          f"{r['p2_max_daily_dd']:>6.1f}% {r['p2_fail_reason']}")

        # 按月统计
        df['start_month'] = pd.to_datetime(df['start_date']).dt.to_period('M')
        monthly = df.groupby('start_month').agg(
            total=('both_passed', 'count'),
            p1_ok=('p1_passed', 'sum'),
            both_ok=('both_passed', 'sum')
        )
        monthly['p1_rate'] = (monthly['p1_ok'] / monthly['total'] * 100).round(0)
        monthly['both_rate'] = (monthly['both_ok'] / monthly['total'] * 100).round(0)

        print(f"\n按开始月份:")
        print(f"  {'月份':<10} {'样本':>4} {'P1通过':>8} {'总通过':>8}")
        print(f"  {'─'*34}")
        for month, row in monthly.iterrows():
            print(f"  {str(month):<10} {int(row['total']):>4} "
                  f"{row['p1_rate']:>7.0f}% {row['both_rate']:>7.0f}%")

        all_leverage_results[leverage] = {
            'valid_samples': n,
            'skipped': skipped,
            'p1_pass_rate': p1_pass / n * 100,
            'p2_pass_rate': p2_pass / max(p2_attempted, 1) * 100,
            'overall_pass_rate': both_pass / n * 100,
            'avg_calendar_days': both_df['calendar_days'].mean() if len(both_df) > 0 else None,
            'records': records,
        }

    # ===== 汇总报告 =====
    print(f"\n\n{'='*70}")
    print(f"汇 总 报 告")
    print(f"{'='*70}")
    print(f"标的: QQQ | 数据: {base_config['start_date']} ~ {base_config['end_date']}")
    print(f"初始资金: ${INITIAL_CAPITAL:,} | 日内止损: {base_config['intraday_stop_loss_pct']*100:.1f}%")
    print(f"FTMO: P1目标+10%, P2目标+5%, 总回撤≤10%, 日内回撤≤5%")
    print()
    print(f"{'杠杆':>6} {'有效样本':>8} {'P1通过率':>10} {'P2通过率':>10} "
          f"{'总通过率':>10} {'通过者平均用时':>14}")
    print(f"{'─'*62}")
    for lev in leverage_list:
        if lev not in all_leverage_results:
            continue
        r = all_leverage_results[lev]
        days_str = f"{r['avg_calendar_days']:.0f}天" if r['avg_calendar_days'] is not None else "N/A"
        print(f"  {lev}x   {r['valid_samples']:>8} "
              f"{r['p1_pass_rate']:>9.1f}% "
              f"{r['p2_pass_rate']:>9.1f}% "
              f"{r['overall_pass_rate']:>9.1f}% "
              f"{days_str:>14}")

    print(f"\n说明:")
    print(f"  - P1通过率 = 第一轮考试通过的比例")
    print(f"  - P2通过率 = 在第一轮通过的前提下，第二轮通过的比例")
    print(f"  - 总通过率 = 两轮都通过的比例 (= P1通过率 × P2通过率)")
    print(f"  - 通过者平均用时 = 从第一轮开始到第二轮通过的自然日天数")
    print(f"  - 数据不足的样本已丢弃，不计入成功也不计入失败")
    print(f"{'='*70}")

    return all_leverage_results


if __name__ == "__main__":
    results = run_prop_firm_simulation(
        leverage_list=[2, 3, 4, 5],
        num_samples=100,
    )
