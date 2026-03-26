#!/usr/bin/env python3
"""
以 er5 为主、与其它日频趋势因子做 AND 组合扫描，比较夏普与总回报。
用法: python3 er5_combo_scan.py
依赖 backtest.run_backtest，不写 CSV。日期与 trend_factor_scan.base_config 一致。
"""
import io
import contextlib
import itertools
from datetime import date

from backtest import run_backtest


def silent_run(cfg):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, _, _, m = run_backtest(cfg)
    return m


def base():
    return {
        'data_path': 'qqq_longport.csv',
        'ticker': 'QQQ',
        'initial_capital': 25000,
        'lookback_days': 1,
        'start_date': date(2024, 4, 1),
        'end_date': date(2026, 3, 25),
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
        'leverage': 2.8,
        'use_vwap': False,
        'enable_intraday_stop_loss': False,
        'enable_trailing_take_profit': True,
        'trailing_tp_activation_pct': 0.01,
        'trailing_tp_callback_pct': 0.7,
    }


def build_candidates():
    c = []
    c.append(('baseline', None))

    # 单因子 er5
    for t in [0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16, 0.18]:
        c.append((f'er5_min_{t}', [{'metric': 'er5', 'min': t}]))

    # er5 + linreg5_r2（控制规模）
    for er, lr in itertools.product(
        [0.10, 0.11, 0.12, 0.14],
        [0.28, 0.30, 0.32, 0.35],
    ):
        c.append((f'er{er}_AND_linreg{lr}', [
            {'metric': 'er5', 'min': er},
            {'metric': 'linreg5_r2', 'min': lr},
        ]))

    # er5 + weekly_sn 下限
    for er, sn in itertools.product(
        [0.09, 0.10, 0.11, 0.12],
        [0.50, 0.55, 0.60, 0.65],
    ):
        c.append((f'er{er}_AND_snmin{sn}', [
            {'metric': 'er5', 'min': er},
            {'metric': 'weekly_sn', 'min': sn},
        ]))

    # er5 + weekly_sn 上限
    for er, sx in itertools.product(
        [0.09, 0.10, 0.12],
        [1.2, 1.35, 1.5, 1.65],
    ):
        c.append((f'er{er}_AND_snmax{sx}', [
            {'metric': 'er5', 'min': er},
            {'metric': 'weekly_sn', 'max': sx},
        ]))

    # er5 + 偏离 MA20
    for er, d in itertools.product(
        [0.09, 0.10, 0.12],
        [0.006, 0.01, 0.014, 0.018],
    ):
        c.append((f'er{er}_AND_dist{d}', [
            {'metric': 'er5', 'min': er},
            {'metric': 'dist_ma20_abs', 'min_abs': d},
        ]))

    # 三因子
    for er, lr, sn in itertools.product(
        [0.10, 0.11],
        [0.30, 0.32],
        [0.55, 0.60, 0.65],
    ):
        c.append((f'er{er}_lr{lr}_snmin{sn}', [
            {'metric': 'er5', 'min': er},
            {'metric': 'linreg5_r2', 'min': lr},
            {'metric': 'weekly_sn', 'min': sn},
        ]))

    return c


def main():
    b = base()
    cand = build_candidates()
    n = len(cand)
    print(f'候选数: {n}（含基准）')
    rows = []
    for i, (name, filt) in enumerate(cand):
        print(f'  [{i + 1}/{n}] {name}', flush=True)
        cfg = {**b}
        if filt is not None:
            cfg['entry_trend_filter'] = filt
        m = silent_run(cfg)
        rows.append({
            'name': name,
            'sharpe': m['sharpe_ratio'],
            'ret': m['total_return'],
            'irr': m['irr'],
            'mdd': m['mdd'],
            'trades': m['total_trades'],
            'hit': m['hit_ratio'],
        })

    bs = next(r for r in rows if r['name'] == 'baseline')

    by_sh = sorted(rows, key=lambda r: r['sharpe'], reverse=True)
    by_ret = sorted(rows, key=lambda r: r['ret'], reverse=True)

    def line(r):
        return (
            f"{r['name']:<40} {r['sharpe']:>8.4f} {r['ret']*100:>9.2f}% "
            f"{r['mdd']*100:>8.2f}% {r['trades']:>6} {r['hit']*100:>6.1f}%"
        )

    print()
    print('基准:', line(bs))
    print()
    print('=== 按夏普 Top 25 ===')
    print(f"{'name':<40} {'sharpe':>8} {'ret':>10} {'MDD%':>8} {'trades':>6} {'win%':>7}")
    print('-' * 92)
    for r in by_sh[:25]:
        print(line(r))

    print()
    print('=== 按总回报 Top 15 ===')
    for r in by_ret[:15]:
        print(line(r))

    best_sh = by_sh[0]
    best_ret = by_ret[0]
    print()
    print('小结:')
    print(f"  夏普最高: {best_sh['name']}  Sharpe={best_sh['sharpe']:.4f}  ret={best_sh['ret']*100:.2f}%  (vs baseline Δ夏普 {best_sh['sharpe']-bs['sharpe']:+.4f})")
    print(f"  回报最高: {best_ret['name']}  ret={best_ret['ret']*100:.2f}%  Sharpe={best_ret['sharpe']:.4f}")

    better = [r for r in rows if r['sharpe'] > bs['sharpe'] and r['ret'] > bs['ret'] and r['name'] != 'baseline']
    better.sort(key=lambda r: (r['sharpe'], r['ret']), reverse=True)
    print()
    if better:
        print('同时优于基准「夏普」且「总回报」的候选:')
        for r in better[:12]:
            print(' ', line(r))
    else:
        print('未发现同时优于基准夏普与总回报的组合；常见权衡是夏普↑回报略↓或反之。')


if __name__ == '__main__':
    main()
