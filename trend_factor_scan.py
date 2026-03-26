#!/usr/bin/env python3
"""
批量尝试 entry_trend_filter 单因子与 AND 组合，比较夏普 / Calmar 等。
用法:
  python3 trend_factor_scan.py           # 完整网格（较慢）
  python3 trend_factor_scan.py --quick     # 约 1/3 规模
不改变 CSV；依赖 backtest.run_backtest。
"""
import argparse
import io
import contextlib
import itertools
from datetime import date

from backtest import run_backtest


def silent_run(cfg):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        daily_df, _, _, metrics = run_backtest(cfg)
    return daily_df, metrics


def base_config():
    # 与 data_fetch / 当前 qqq_longport 覆盖区间对齐时可改此处
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


def build_candidates(quick: bool):
    candidates = []
    candidates.append(('baseline', None))

    if quick:
        linreg_grid = [round(x, 3) for x in [0.44, 0.48, 0.52, 0.56]]
        er_grid = [round(x, 3) for x in [0.2, 0.24, 0.28, 0.32]]
        sn_max_grid = [1.1, 1.3]
        dist_grid = [0.01, 0.014]
    else:
        # 全量网格：略收窄单因子数量，把算力留给双因子组合
        linreg_grid = [round(0.38 + i * 0.02, 3) for i in range(12)]  # 0.38..0.60
        er_grid = [round(0.14 + i * 0.02, 3) for i in range(12)]  # 0.14..0.36
        sn_max_grid = [0.95, 1.05, 1.15, 1.25, 1.4, 1.55]
        sn_min_grid = [0.7, 0.85, 1.0, 1.15]
        dist_grid = [round(0.005 + i * 0.0025, 4) for i in range(7)]  # 0.005..0.02

    for thr in linreg_grid:
        candidates.append((f'linreg5_r2_min_{thr}', {'metric': 'linreg5_r2', 'min': thr}))

    for thr in er_grid:
        candidates.append((f'er5_min_{thr}', {'metric': 'er5', 'min': thr}))

    if quick:
        for thr in sn_max_grid:
            candidates.append((f'weekly_sn_max_{thr}', {'metric': 'weekly_sn', 'max': thr}))
        for thr in dist_grid:
            candidates.append((f'dist_ma20_abs_min_{thr}', {'metric': 'dist_ma20_abs', 'min_abs': thr}))
    else:
        for thr in sn_max_grid:
            candidates.append((f'weekly_sn_max_{thr}', {'metric': 'weekly_sn', 'max': thr}))
        for thr in sn_min_grid:
            candidates.append((f'weekly_sn_min_{thr}', {'metric': 'weekly_sn', 'min': thr}))
        for thr in dist_grid:
            candidates.append((f'dist_ma20_abs_min_{thr}', {'metric': 'dist_ma20_abs', 'min_abs': thr}))

    # 双因子 AND：在关键区间做笛卡尔积（控制数量）
    if quick:
        lr = [0.48, 0.52]
        er = [0.24, 0.28]
    else:
        lr = [0.44, 0.48, 0.52, 0.56]
        er = [0.2, 0.24, 0.28, 0.32]

    for a, b in itertools.product(lr, er):
        name = f'AND_linreg{a}_er{b}'
        filt = [
            {'metric': 'linreg5_r2', 'min': a},
            {'metric': 'er5', 'min': b},
        ]
        candidates.append((name, filt))

    # linreg + weekly_sn 上限（避免极端噪声）
    if quick:
        pairs = [(0.5, 1.2), (0.52, 1.3)]
    else:
        pairs = list(itertools.product([0.46, 0.5, 0.54], [1.05, 1.2, 1.35, 1.5]))
    for lr_v, sn_v in pairs:
        name = f'AND_linreg{lr_v}_snmax{sn_v}'
        candidates.append((name, [
            {'metric': 'linreg5_r2', 'min': lr_v},
            {'metric': 'weekly_sn', 'max': sn_v},
        ]))

    # er5 + dist_ma20
    if quick:
        pairs2 = [(0.28, 0.012)]
    else:
        pairs2 = list(itertools.product([0.22, 0.26, 0.3], [0.008, 0.012, 0.016]))
    for er_v, d_v in pairs2:
        name = f'AND_er{er_v}_dist{d_v}'
        candidates.append((name, [
            {'metric': 'er5', 'min': er_v},
            {'metric': 'dist_ma20_abs', 'min_abs': d_v},
        ]))

    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='缩小网格以快速试跑')
    args = ap.parse_args()

    base = base_config()
    candidates = build_candidates(quick=args.quick)
    print(f'候选数: {len(candidates)} (quick={args.quick})')
    print()

    rows = []
    ntot = len(candidates)
    for i, (name, filt) in enumerate(candidates):
        print(f'  [{i + 1}/{ntot}] {name} ...', flush=True)
        cfg = {**base}
        if filt is not None:
            cfg['entry_trend_filter'] = filt
        _, m = silent_run(cfg)
        calmar = m.get('calmar_ratio')
        if calmar is None or (isinstance(calmar, float) and calmar == float('inf')):
            calmar = float('nan')
        rows.append({
            'name': name,
            'sharpe': m['sharpe_ratio'],
            'calmar': calmar,
            'irr': m['irr'],
            'mdd': m['mdd'],
            'total_return': m['total_return'],
            'trades': m['total_trades'],
            'hit': m['hit_ratio'],
            'vol': m['volatility'],
        })

    baseline = next(x for x in rows if x['name'] == 'baseline')
    b_sh = baseline['sharpe']

    rows_sh = sorted(rows, key=lambda r: r['sharpe'], reverse=True)
    rows_cal = sorted(rows, key=lambda r: (float('nan') if r['calmar'] != r['calmar'] else -r['calmar']))

    def fmt_row(r):
        cm = r['calmar']
        cms = f"{cm:.3f}" if cm == cm else '  nan'
        return (
            f"{r['name']:<34} {r['sharpe']:>8.4f} {cms:>8} {r['irr']*100:>8.2f} {r['mdd']*100:>8.2f} "
            f"{r['total_return']*100:>8.2f} {r['vol']*100:>7.2f} {r['trades']:>7} {r['hit']*100:>7.1f}"
        )

    print('=== 按夏普排序（前 35 名）===')
    print(f"{'name':<34} {'sharpe':>8} {'calmar':>8} {'IRR%':>8} {'MDD%':>8} {'ret%':>8} {'vol%':>7} {'trades':>7} {'win%':>7}")
    print('-' * 118)
    for r in rows_sh[:35]:
        print(fmt_row(r))

    print()
    print('=== 按 Calmar 排序（前 15 名，排除无效）===')
    valid_cal = [r for r in rows_cal if r['calmar'] == r['calmar'] and r['calmar'] < 1e6]
    for r in valid_cal[:15]:
        print(fmt_row(r))

    best = rows_sh[0]
    print()
    print(f"基准夏普: {b_sh:.4f} | 基准 Calmar: {baseline['calmar'] if baseline['calmar'] == baseline['calmar'] else float('nan'):.4f}")
    print(f"夏普最优: {best['name']} -> Sharpe={best['sharpe']:.4f} (Δ {best['sharpe'] - b_sh:+.4f})")
    if best['name'] != 'baseline':
        print(f"          总回报 {best['total_return']*100:.2f}% | 回撤 {best['mdd']*100:.2f}% | 交易 {best['trades']}")

    min_tr = 120
    robust = [r for r in rows_sh if r['trades'] >= min_tr and r['name'] != 'baseline']
    print()
    print(f'=== 稳健榜（交易数 >= {min_tr}，降低偶然性）按夏普 ===')
    if not robust:
        print('  (无满足条件的候选)')
    else:
        rb = sorted(robust, key=lambda r: r['sharpe'], reverse=True)[:15]
        for r in rb:
            print(fmt_row(r))
        br = rb[0]
        print(f'稳健榜最优: {br["name"]} Sharpe={br["sharpe"]:.4f} (Δ {br["sharpe"] - b_sh:+.4f})  trades={br["trades"]}')

    # 是否「显著」优于基准：样本内启发式阈值（非统计检验）
    delta = best['sharpe'] - b_sh
    if best['trades'] < 80:
        print()
        print('提示: 夏普榜首若交易笔数很少，指标方差大，需结合「稳健榜」或拉长样本验证。')
    if delta >= 0.05:
        print(f"说明: 样本内夏普提升 >= 0.05，可视为较强改进（仍可能过拟合，需样本外验证）。")
    elif delta >= 0.02:
        print(f"说明: 样本内夏普提升约 {delta:.4f}，温和改进。")
    else:
        print(f"说明: 样本内最优相对基准提升 {delta:.4f}，未见大幅度优化。")


if __name__ == '__main__':
    main()
