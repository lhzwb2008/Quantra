#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
理想汽车(LiAuto)港美股套利策略回测

策略逻辑：
1. 港股T日收盘时，记录港股收盘价作为基准价格
2. 美股T日开盘后，监控美股价格相对港股收盘价的偏离
3. 当偏离超过阈值时，在美股开仓
4. 港股T+1开盘时，通过港股对冲锁定利润

注意：
- 理想汽车港股代码: 2015.HK
- 理想汽车美股代码: LI.US  
- 美股1 ADR = 2 普通股
- 汇率假设: 7.8
"""

from datetime import date
from longport.openapi import QuoteContext, Config, Period, AdjustType
import pandas as pd
import numpy as np

# ———— 配置 & 初始化 ————
config = Config.from_env()
ctx = QuoteContext(config)

# ———— 常量 ————
USD_HKD_RATE = 7.8
ADR_RATIO = 2


def fetch_daily_data(symbol, start_date, end_date):
    """获取日K线数据"""
    resp = ctx.history_candlesticks_by_date(
        symbol, Period.Day, AdjustType.NoAdjust, start_date, end_date
    )
    
    rows = []
    for c in resp:
        rows.append({
            'Date': c.timestamp.strftime('%Y-%m-%d'),
            'Open': float(c.open),
            'High': float(c.high),
            'Low': float(c.low),
            'Close': float(c.close),
        })
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['Date'], keep='last')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def convert_us_to_hkd(us_price):
    """美股价格转港币等效价格"""
    return us_price / ADR_RATIO * USD_HKD_RATE


def run_arbitrage_backtest(cfg):
    """运行套利策略回测"""
    start_date = cfg.get('start_date', date(2024, 1, 1))
    end_date = cfg.get('end_date', date(2026, 2, 3))
    initial_capital = cfg.get('initial_capital', 100000)
    premium_threshold = cfg.get('premium_threshold', 0.02)
    position_pct = cfg.get('position_pct', 1.0)
    transaction_cost_pct = cfg.get('transaction_cost_pct', 0.001)
    
    # 获取数据
    hk_df = fetch_daily_data("2015.HK", start_date, end_date)
    us_df = fetch_daily_data("LI.US", start_date, end_date)
    
    # 转换美股价格
    us_df['High_HKD'] = us_df['High'].apply(convert_us_to_hkd)
    us_df['Low_HKD'] = us_df['Low'].apply(convert_us_to_hkd)
    
    hk_dates = list(hk_df['Date'].dt.date)
    us_dates = list(us_df['Date'].dt.date)
    
    hk_df.set_index('Date', inplace=True)
    us_df.set_index('Date', inplace=True)
    
    # 回测
    capital = initial_capital
    trades = []
    
    for i, hk_date in enumerate(hk_dates[:-1]):
        hk_date_ts = pd.Timestamp(hk_date)
        if hk_date_ts not in hk_df.index:
            continue
        
        hk_close = hk_df.loc[hk_date_ts]['Close']
        
        # 找对应美股交易日
        us_trade_date = None
        for us_date in us_dates:
            if us_date >= hk_date:
                us_trade_date = us_date
                break
        
        if us_trade_date is None:
            continue
        
        us_date_ts = pd.Timestamp(us_trade_date)
        if us_date_ts not in us_df.index:
            continue
        
        us_data = us_df.loc[us_date_ts]
        us_high_hkd = us_data['High_HKD']
        us_low_hkd = us_data['Low_HKD']
        
        max_premium = (us_high_hkd - hk_close) / hk_close
        min_premium = (us_low_hkd - hk_close) / hk_close
        
        # 找下一个港股交易日
        next_hk_date = None
        for future_hk_date in hk_dates:
            if future_hk_date > hk_date:
                next_hk_date = future_hk_date
                break
        
        if next_hk_date is None:
            continue
        
        next_hk_date_ts = pd.Timestamp(next_hk_date)
        if next_hk_date_ts not in hk_df.index:
            continue
        
        next_hk_open = hk_df.loc[next_hk_date_ts]['Open']
        
        trade_executed = False
        
        # 美股溢价 -> 做空美股 + 做多港股
        if max_premium > premium_threshold and not trade_executed:
            entry_price_hkd = hk_close * (1 + premium_threshold)
            entry_price_us = entry_price_hkd / USD_HKD_RATE * ADR_RATIO
            
            position_value = capital * position_pct
            position_size_us = int(position_value / entry_price_us)
            position_size_hk = position_size_us * ADR_RATIO
            
            if position_size_us > 0:
                hk_hedge_price = next_hk_open
                locked_spread_hkd = entry_price_hkd - hk_hedge_price
                locked_pnl_hkd = locked_spread_hkd * position_size_hk
                locked_pnl_usd = locked_pnl_hkd / USD_HKD_RATE
                
                us_cost = entry_price_us * position_size_us * transaction_cost_pct
                hk_cost = hk_hedge_price * position_size_hk / USD_HKD_RATE * transaction_cost_pct
                total_cost = us_cost + hk_cost
                
                total_pnl = locked_pnl_usd - total_cost
                capital += total_pnl
                trades.append({'pnl': total_pnl, 'capital': capital})
                trade_executed = True
        
        # 美股折价 -> 做多美股 + 做空港股
        if min_premium < -premium_threshold and not trade_executed:
            entry_price_hkd = hk_close * (1 - premium_threshold)
            entry_price_us = entry_price_hkd / USD_HKD_RATE * ADR_RATIO
            
            position_value = capital * position_pct
            position_size_us = int(position_value / entry_price_us)
            position_size_hk = position_size_us * ADR_RATIO
            
            if position_size_us > 0:
                hk_hedge_price = next_hk_open
                locked_spread_hkd = hk_hedge_price - entry_price_hkd
                locked_pnl_hkd = locked_spread_hkd * position_size_hk
                locked_pnl_usd = locked_pnl_hkd / USD_HKD_RATE
                
                us_cost = entry_price_us * position_size_us * transaction_cost_pct
                hk_cost = hk_hedge_price * position_size_hk / USD_HKD_RATE * transaction_cost_pct
                total_cost = us_cost + hk_cost
                
                total_pnl = locked_pnl_usd - total_cost
                capital += total_pnl
                trades.append({'pnl': total_pnl, 'capital': capital})
                trade_executed = True
    
    # 计算指标
    trades_df = pd.DataFrame(trades)
    total_return = (capital - initial_capital) / initial_capital
    
    # 计算夏普比率
    sharpe_ratio = 0
    max_drawdown = 0
    if len(trades_df) > 1:
        trades_df['return'] = trades_df['pnl'] / (initial_capital * position_pct)
        trading_days = len(hk_dates)
        years = trading_days / 252
        annualized_return = (capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        trades_per_year = len(trades_df) / years if years > 0 else 0
        annualized_vol = trades_df['return'].std() * np.sqrt(trades_per_year) if trades_per_year > 0 else 0
        
        if annualized_vol > 0:
            sharpe_ratio = (annualized_return - 0.02) / annualized_vol
        
        # 最大回撤
        trades_df['peak'] = trades_df['capital'].cummax()
        trades_df['drawdown'] = (trades_df['capital'] - trades_df['peak']) / trades_df['peak']
        max_drawdown = trades_df['drawdown'].min()
    
    # 胜率
    win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    
    # 输出结果
    print(f"\n{'='*50}")
    print(f"理想汽车港美股套利策略回测结果")
    print(f"{'='*50}")
    print(f"数据范围: {start_date} 到 {end_date}")
    print(f"溢价阈值: {premium_threshold*100:.1f}%")
    print(f"{'='*50}")
    print(f"初始资金: ${initial_capital:,.0f}")
    print(f"最终资金: ${capital:,.2f}")
    print(f"总收益率: {total_return*100:.2f}%")
    print(f"交易次数: {len(trades_df)}")
    print(f"胜率: {win_rate:.1f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"{'='*50}")
    
    return capital, total_return, sharpe_ratio, max_drawdown


# ———— 主程序 ————
if __name__ == "__main__":
    config = {
        'start_date': date(2024, 1, 1),
        'end_date': date(2026, 2, 3),
        'initial_capital': 100000,
        'premium_threshold': 0.03,
        'position_pct': 1.0,
        'transaction_cost_pct': 0.001,
    }
    
    run_arbitrage_backtest(config)
