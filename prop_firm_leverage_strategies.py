"""
比较两阶段不同杠杆组合的 Prop Firm 通过率和用时。

规则与 prop_firm_test 一致：P1 盈利 10%、P2 盈利 5%；总回撤 ≤10%；日内回撤 ≤5%。
配置（数据路径、日期、手续费、日内止损、趋势门控等）与 backtest.py 主入口对齐。

用法:
  conda activate futu
  python prop_firm_leverage_strategies.py
"""

import argparse
import time
from datetime import date

import pandas as pd

from prop_firm_test import prepare_backtest_data, simulate_exam_phase


# 与 backtest 主配置对齐；初始资金与 prop firm 账户统一为 100000
BACKTEST_ALIGNED_CONFIG = {
    'data_path': 'qqq_longport.csv',
    'ticker': 'QQQ',
    'initial_capital': 100000,
    'lookback_days': 1,
    'start_date': date(2024, 4, 1),
    'end_date': date(2026, 3, 31),
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
    'leverage': 2,
    'use_vwap': False,
    'enable_intraday_stop_loss': True,
    'intraday_stop_loss_pct': 0.045,
    'enable_trailing_take_profit': True,
    'trailing_tp_activation_pct': 0.01,
    'trailing_tp_callback_pct': 0.7,
    'entry_trend_filter': {'metric': 'er5', 'min': 0.1},
}


def run_dual_leverage_simulation(
    strategy_name: str,
    leverage_phase1: float,
    leverage_phase2: float,
    base_config: dict,
    num_samples: int = 100,
    prepared=None,
):
    """
    P1 / P2 使用不同杠杆；其余参数来自 base_config。
    prepared: 可选 (price_df, allowed_times, filtered_dates)，由 prepare_backtest_data 生成；
              传入则跳过重复预处理（多策略对比时只算一次）。
    返回统计 dict。
    """
    cfg = base_config.copy()
    initial_capital = cfg['initial_capital']

    if prepared is not None:
        price_df, allowed_times, filtered_dates = prepared
    else:
        price_df, allowed_times, filtered_dates = prepare_backtest_data(cfg)
    total_days = len(filtered_dates)
    step = max(1, total_days // num_samples)
    sample_indices = list(range(0, total_days, step))[:num_samples]

    records = []
    skipped = 0

    n_idx = len(sample_indices)
    for i, start_idx in enumerate(sample_indices):
        if n_idx >= 20 and (i + 1) % max(1, n_idx // 4) == 0:
            print(f"    [{strategy_name}] 进度 {i + 1}/{n_idx}", flush=True)

        cfg_run = cfg.copy()
        cfg_run['leverage'] = leverage_phase1

        r1 = simulate_exam_phase(
            price_df, allowed_times, filtered_dates, cfg_run,
            exam_start_idx=start_idx,
            initial_capital=initial_capital,
            profit_target_pct=0.10,
            max_total_drawdown_pct=0.10,
            max_daily_drawdown_pct=0.05,
            leverage=leverage_phase1,
        )

        if r1['status'] == 'insufficient_data':
            skipped += 1
            continue

        r2 = None
        if r1['status'] == 'passed':
            cfg_p2 = cfg.copy()
            cfg_p2['leverage'] = leverage_phase2
            r2 = simulate_exam_phase(
                price_df, allowed_times, filtered_dates, cfg_p2,
                exam_start_idx=r1['end_idx'],
                initial_capital=initial_capital,
                profit_target_pct=0.05,
                max_total_drawdown_pct=0.10,
                max_daily_drawdown_pct=0.05,
                leverage=leverage_phase2,
            )
            if r2['status'] == 'insufficient_data':
                skipped += 1
                continue

        p1_ok = r1['status'] == 'passed'
        p2_ok = r2 is not None and r2['status'] == 'passed'
        both_ok = p1_ok and p2_ok

        exam_start_date = filtered_dates[start_idx]
        if both_ok:
            exam_end_date = r2['end_date']
        elif p1_ok:
            exam_end_date = r2['end_date'] if r2 else r1['end_date']
        else:
            exam_end_date = r1['end_date']

        calendar_days = (exam_end_date - exam_start_date).days
        p1_days = r1['days_used']
        p2_days = r2['days_used'] if r2 else 0
        trading_days_total = p1_days + p2_days if both_ok else None

        records.append({
            'start_date': exam_start_date,
            'p1_passed': p1_ok,
            'p2_passed': p2_ok,
            'both_passed': both_ok,
            'p1_days': p1_days,
            'p2_days': p2_days,
            'trading_days_total': trading_days_total,
            'calendar_days': calendar_days,
        })

    df = pd.DataFrame(records)
    n = len(df)
    if n == 0:
        return {
            'strategy_name': strategy_name,
            'leverage_p1': leverage_phase1,
            'leverage_p2': leverage_phase2,
            'valid_samples': 0,
            'skipped': skipped,
        }

    both_df = df[df['both_passed']]
    p1_pass = df['p1_passed'].sum()
    p2_attempt = df['p1_passed'].sum()
    p2_pass = df['p2_passed'].sum()
    both_pass = df['both_passed'].sum()

    return {
        'strategy_name': strategy_name,
        'leverage_p1': leverage_phase1,
        'leverage_p2': leverage_phase2,
        'valid_samples': n,
        'skipped': skipped,
        'p1_pass_rate': p1_pass / n * 100,
        'p2_conditional_pass_rate': p2_pass / p2_attempt * 100 if p2_attempt else 0.0,
        'overall_pass_rate': both_pass / n * 100,
        'avg_calendar_days_both': both_df['calendar_days'].mean() if len(both_df) else None,
        'median_calendar_days_both': both_df['calendar_days'].median() if len(both_df) else None,
        'avg_trading_days_both': both_df['trading_days_total'].mean() if len(both_df) else None,
        'median_trading_days_both': both_df['trading_days_total'].median() if len(both_df) else None,
        'min_calendar_days_both': both_df['calendar_days'].min() if len(both_df) else None,
        'max_calendar_days_both': both_df['calendar_days'].max() if len(both_df) else None,
    }


def main():
    ap = argparse.ArgumentParser(description='两阶段杠杆 Prop Firm 模拟对比')
    ap.add_argument(
        '--samples', type=int, default=100,
        help='均匀采样的考试起点数量（越大越慢，统计越稳）',
    )
    args = ap.parse_args()
    num_samples = max(1, args.samples)

    t0 = time.perf_counter()
    print('=' * 72, flush=True)
    print('Prop Firm 两阶段杠杆策略对比（与 backtest.py 配置对齐）', flush=True)
    print('=' * 72, flush=True)
    c = BACKTEST_ALIGNED_CONFIG
    print(f"数据: {c['data_path']} | {c['start_date']} ~ {c['end_date']}", flush=True)
    print(f"初始资金: ${c['initial_capital']:,} | 日内止损: {c['intraday_stop_loss_pct']*100:.2f}%", flush=True)
    print(f"趋势门控: {c.get('entry_trend_filter')}", flush=True)
    print(f"规则: P1 +10% / P2 +5%；总回撤≤10%；日内回撤≤5%", flush=True)
    print(f"采样起点数: {num_samples}（命令行: python prop_firm_leverage_strategies.py --samples N）", flush=True)
    print()
    print('预处理行情与趋势特征（仅一次）...', flush=True)
    prepared = prepare_backtest_data(c)
    n_days = len(prepared[2])
    step = max(1, n_days // num_samples)
    print(f'可用交易日: {n_days} | 起点步长≈每 {step} 个交易日取 1 个起点', flush=True)
    print()

    strategies = [
        ('策略A P3/P2', 3, 2),
        ('策略B P2/P1', 2, 1),
        ('策略C P2/P1.5', 2, 1.5),
    ]

    rows = []
    for name, l1, l2 in strategies:
        r = run_dual_leverage_simulation(
            name, l1, l2, BACKTEST_ALIGNED_CONFIG,
            num_samples=num_samples, prepared=prepared,
        )
        rows.append(r)
        print(f"--- {name} (P1={l1:g}x, P2={l2:g}x) ---")
        print(f"  有效样本: {r['valid_samples']} (丢弃数据不足: {r['skipped']})")
        if r['valid_samples'] == 0:
            continue
        print(f"  P1 通过率: {r['p1_pass_rate']:.1f}%")
        print(f"  P2 通过率 (在 P1 通过前提下): {r['p2_conditional_pass_rate']:.1f}%")
        print(f"  两轮都通过: {r['overall_pass_rate']:.1f}%")
        if r['avg_calendar_days_both'] is not None:
            print(f"  通过者 — 自然日: 平均 {r['avg_calendar_days_both']:.0f} 天, "
                  f"中位 {r['median_calendar_days_both']:.0f} 天, "
                  f"范围 [{r['min_calendar_days_both']:.0f}, {r['max_calendar_days_both']:.0f}]")
            print(f"  通过者 — 交易日合计: 平均 {r['avg_trading_days_both']:.1f} 天, "
                  f"中位 {r['median_trading_days_both']:.0f} 天")
        else:
            print('  无「两轮都通过」样本')
        print()

    print('=' * 72)
    print('汇总表')
    print('=' * 72)
    hdr = (
        f"{'策略':<16} {'P1/P2杠杆':>12} {'总通过率':>10} "
        f"{'通过均自然日':>14} {'通过均交易日':>14}"
    )
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        lev = f"{r['leverage_p1']:g}x/{r['leverage_p2']:g}x"
        if r['valid_samples'] == 0:
            print(f"{r['strategy_name']:<16} {lev:>12}  (无数据)")
            continue
        cal = f"{r['avg_calendar_days_both']:.0f}" if r['avg_calendar_days_both'] is not None else 'N/A'
        trd = f"{r['avg_trading_days_both']:.1f}" if r['avg_trading_days_both'] is not None else 'N/A'
        print(
            f"{r['strategy_name']:<16} "
            f"{lev:>12} "
            f"{r['overall_pass_rate']:>9.1f}% "
            f"{cal:>14} "
            f"{trd:>14}"
        )
    print()
    print('说明: 起始日在整个回测窗口内均匀采样，与 prop_firm_test 相同；'
          '「通过时间」仅统计两轮均通过的样本。')

    elapsed = time.perf_counter() - t0
    print()
    print('=' * 100)
    print('整体汇总（全策略对照）')
    print('=' * 100)
    hdr = (
        f"{'策略':<14} {'P1/P2':>10} {'有效':>5} {'丢弃':>5} "
        f"{'P1%':>7} {'P2|P1%':>8} {'总%':>7} "
        f"{'自然日均':>8} {'自然日中位':>10} {'交易日均':>8} {'交易日中位':>10}"
    )
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        lev = f"{r['leverage_p1']:g}/{r['leverage_p2']:g}"
        if r['valid_samples'] == 0:
            print(f"{r['strategy_name']:<14} {lev:>10} {'0':>5} {'-':>5} {'-':>7} {'-':>8} {'-':>7} {'-':>8} {'-':>10} {'-':>8} {'-':>10}")
            continue
        p1 = f"{r['p1_pass_rate']:.1f}"
        p2c = f"{r['p2_conditional_pass_rate']:.1f}"
        tot = f"{r['overall_pass_rate']:.1f}"
        ac = f"{r['avg_calendar_days_both']:.0f}" if r['avg_calendar_days_both'] is not None else '-'
        mc = f"{r['median_calendar_days_both']:.0f}" if r['median_calendar_days_both'] is not None else '-'
        at = f"{r['avg_trading_days_both']:.1f}" if r['avg_trading_days_both'] is not None else '-'
        mt = f"{r['median_trading_days_both']:.0f}" if r['median_trading_days_both'] is not None else '-'
        print(
            f"{r['strategy_name']:<14} {lev:>10} "
            f"{r['valid_samples']:>5} {r['skipped']:>5} "
            f"{p1:>7} {p2c:>8} {tot:>7} "
            f"{ac:>8} {mc:>10} {at:>8} {mt:>10}"
        )
    print('-' * len(hdr))
    print(
        f"采样配置: {num_samples} 个起点 | 总耗时: {elapsed / 60:.1f} 分钟\n"
        f"丢弃: 该起点下考试在数据结束前仍未结束（未达标也未淘汰），不计入有效样本。"
    )


if __name__ == '__main__':
    main()
