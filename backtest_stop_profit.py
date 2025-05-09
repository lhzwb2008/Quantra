import pandas as pd
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta, date
import random
import os
from plot_trading_day import plot_trading_day

def calculate_vwap(prices, volumes):
    """
    计算VWAP (成交量加权平均价格)
    """
    return sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if sum(volumes) > 0 else prices[-1]

def simulate_day(day_df, prev_close, allowed_times, position_size, transaction_fee_per_share=0.01, trading_end_time=(15, 50), max_positions_per_day=float('inf'), take_profit_pct=None, print_details=False):
    """
    模拟单日交易，使用噪声空间策略 + VWAP
    
    参数:
        day_df: 包含日内数据的DataFrame
        prev_close: 前一日收盘价
        allowed_times: 允许交易的时间列表
        position_size: 仓位大小
        transaction_fee_per_share: 每股交易费用
        trading_end_time: 交易结束时间 (小时, 分钟)
        max_positions_per_day: 每日最大开仓次数
        take_profit_pct: 止盈百分比（例如：0.02表示2%）
        print_details: 是否打印交易详情
    """
    position = 0  # 0: 无仓位, 1: 多头, -1: 空头
    entry_price = np.nan
    trailing_stop = np.nan
    trade_entry_time = None
    trades = []
    positions_opened_today = 0  # 今日开仓计数器
    
    # 存储用于计算VWAP的数据
    prices = []
    volumes = []
    
    for idx, row in day_df.iterrows():
        current_time = row['Time']
        price = row['Close']
        volume = row['Volume']
        upper = row['UpperBound']
        lower = row['LowerBound']
        
        # 更新VWAP计算数据
        prices.append(price)
        volumes.append(volume)
        
        # 计算当前VWAP
        vwap = calculate_vwap(prices, volumes)
        
        # 在允许时间内的入场信号
        if position == 0 and current_time in allowed_times and positions_opened_today < max_positions_per_day:
            # 检查潜在多头入场 - 加入VWAP条件
            if price > upper and price > vwap:
                # 打印边界计算详情（如果需要）
                if print_details:
                    date_str = row['DateTime'].strftime('%Y-%m-%d')
                    sigma = row.get('sigma', 0)
                    upper_ref = row.get('upper_ref', 0)
                    lower_ref = row.get('lower_ref', 0)
                    day_open = row.get('day_open', 0)
                    
                    print(f"\n交易点位详情 [{date_str} {current_time}] - 多头入场:")
                    print(f"  价格: {price:.2f} > 上边界: {upper:.2f} 且 > VWAP: {vwap:.2f}")
                    print(f"  边界计算详情:")
                    print(f"    - 日开盘价: {day_open:.2f}, 前日收盘价: {prev_close:.2f}")
                    print(f"    - 上边界参考价: max({day_open:.2f}, {prev_close:.2f}) = {upper_ref:.2f}")
                    print(f"    - 下边界参考价: min({day_open:.2f}, {prev_close:.2f}) = {lower_ref:.2f}")
                    print(f"    - Sigma值: {sigma:.6f}")
                    print(f"    - 上边界计算: {upper_ref:.2f} * (1 + {sigma:.6f}) = {upper:.2f}")
                    print(f"    - 下边界计算: {lower_ref:.2f} * (1 - {sigma:.6f}) = {lower:.2f}")
                
                # 允许多头入场
                position = 1
                entry_price = price
                trade_entry_time = row['DateTime']
                positions_opened_today += 1  # 增加开仓计数器
                # 追踪止损设为上边界和VWAP的最大值
                trailing_stop = max(upper, vwap)
                    
            # 检查潜在空头入场 - 加入VWAP条件
            if price < lower and price < vwap:
                # 打印边界计算详情（如果需要）
                if print_details:
                    date_str = row['DateTime'].strftime('%Y-%m-%d')
                    sigma = row.get('sigma', 0)
                    upper_ref = row.get('upper_ref', 0)
                    lower_ref = row.get('lower_ref', 0)
                    day_open = row.get('day_open', 0)
                    
                    print(f"\n交易点位详情 [{date_str} {current_time}] - 空头入场:")
                    print(f"  价格: {price:.2f} < 下边界: {lower:.2f} 且 < VWAP: {vwap:.2f}")
                    print(f"  边界计算详情:")
                    print(f"    - 日开盘价: {day_open:.2f}, 前日收盘价: {prev_close:.2f}")
                    print(f"    - 上边界参考价: max({day_open:.2f}, {prev_close:.2f}) = {upper_ref:.2f}")
                    print(f"    - 下边界参考价: min({day_open:.2f}, {prev_close:.2f}) = {lower_ref:.2f}")
                    print(f"    - Sigma值: {sigma:.6f}")
                    print(f"    - 上边界计算: {upper_ref:.2f} * (1 + {sigma:.6f}) = {upper:.2f}")
                    print(f"    - 下边界计算: {lower_ref:.2f} * (1 - {sigma:.6f}) = {lower:.2f}")
                
                # 允许空头入场
                position = -1
                entry_price = price
                trade_entry_time = row['DateTime']
                positions_opened_today += 1  # 增加开仓计数器
                # 追踪止损设为下边界和VWAP的最小值
                trailing_stop = min(lower, vwap)
        
        # 更新追踪止损并检查出场信号
        if position != 0:
            if position == 1:  # 多头仓位
                # 计算止损水平（使用上边界和VWAP的最大值）
                new_stop = max(upper, vwap)
                # 只在有利方向更新（提高止损）
                trailing_stop = max(trailing_stop, new_stop)
                
                # 如果价格跌破追踪止损，则平仓
                exit_condition = price < trailing_stop
                
                # 主动止盈条件: 如果设置了止盈比例，并且当前盈利达到了止盈水平，则触发主动止盈
                take_profit_condition = False
                if take_profit_pct is not None:
                    profit_pct = (price / entry_price) - 1
                    take_profit_condition = profit_pct >= take_profit_pct
                
                # 检查是否出场（止损或者止盈）
                if (exit_condition or take_profit_condition) and current_time in allowed_times:
                    # 打印出场详情（如果需要）
                    if print_details:
                        date_str = row['DateTime'].strftime('%Y-%m-%d')
                        if take_profit_condition:
                            profit_pct = (price / entry_price) - 1
                            print(f"\n交易点位详情 [{date_str} {current_time}] - 多头止盈出场:")
                            print(f"  价格: {price:.2f}, 入场价: {entry_price:.2f}, 盈利: {profit_pct*100:.2f}% >= 止盈目标: {take_profit_pct*100:.2f}%")
                        else:
                            print(f"\n交易点位详情 [{date_str} {current_time}] - 多头止损出场:")
                            print(f"  价格: {price:.2f} < 追踪止损: {trailing_stop:.2f}")
                            print(f"  止损计算: max(上边界={upper:.2f}, VWAP={vwap:.2f}) = {new_stop:.2f}")
                    
                    # 平仓多头
                    exit_time = row['DateTime']
                    # 计算交易费用（开仓和平仓）
                    transaction_fees = position_size * transaction_fee_per_share * 2  # 买入和卖出费用
                    pnl = position_size * (price - entry_price) - transaction_fees
                    
                    exit_reason = 'Take Profit' if take_profit_condition else 'Stop Loss'
                    trades.append({
                        'entry_time': trade_entry_time,
                        'exit_time': exit_time,
                        'side': 'Long',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    trailing_stop = np.nan
                    
            elif position == -1:  # 空头仓位
                # 计算止损水平（使用下边界和VWAP的最小值）
                new_stop = min(lower, vwap)
                # 只在有利方向更新（降低止损）
                trailing_stop = min(trailing_stop, new_stop)
                
                # 如果价格涨破追踪止损，则平仓
                exit_condition = price > trailing_stop
                
                # 主动止盈条件: 如果设置了止盈比例，并且当前盈利达到了止盈水平，则触发主动止盈
                take_profit_condition = False
                if take_profit_pct is not None:
                    profit_pct = (entry_price / price) - 1  # 空头盈利计算
                    take_profit_condition = profit_pct >= take_profit_pct
                
                # 检查是否出场（止损或者止盈）
                if (exit_condition or take_profit_condition) and current_time in allowed_times:
                    # 打印出场详情（如果需要）
                    if print_details:
                        date_str = row['DateTime'].strftime('%Y-%m-%d')
                        if take_profit_condition:
                            profit_pct = (entry_price / price) - 1
                            print(f"\n交易点位详情 [{date_str} {current_time}] - 空头止盈出场:")
                            print(f"  价格: {price:.2f}, 入场价: {entry_price:.2f}, 盈利: {profit_pct*100:.2f}% >= 止盈目标: {take_profit_pct*100:.2f}%")
                        else:
                            print(f"\n交易点位详情 [{date_str} {current_time}] - 空头止损出场:")
                            print(f"  价格: {price:.2f} > 追踪止损: {trailing_stop:.2f}")
                            print(f"  止损计算: min(下边界={lower:.2f}, VWAP={vwap:.2f}) = {new_stop:.2f}")
                    
                    # 平仓空头
                    exit_time = row['DateTime']
                    # 计算交易费用（开仓和平仓）
                    transaction_fees = position_size * transaction_fee_per_share * 2  # 买入和卖出费用
                    pnl = position_size * (entry_price - price) - transaction_fees
                    
                    exit_reason = 'Take Profit' if take_profit_condition else 'Stop Loss'
                    trades.append({
                        'entry_time': trade_entry_time,
                        'exit_time': exit_time,
                        'side': 'Short',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    trailing_stop = np.nan
    
    # 获取交易结束时间字符串，格式为HH:MM
    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    
    # 寻找结束时间的数据点（如果存在）
    close_time_rows = day_df[day_df['Time'] == end_time_str]
    
    # 如果有结束时间的数据点且仍有未平仓位，则平仓
    if not close_time_rows.empty and position != 0:
        close_row = close_time_rows.iloc[0]
        exit_time = close_row['DateTime']
        close_price = close_row['Close']
        
        if position == 1:  # 多头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {end_time_str}] - 多头收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {close_price:.2f}")
            
            # 计算交易费用（开仓和平仓）
            transaction_fees = position_size * transaction_fee_per_share * 2  # 买入和卖出费用
            pnl = position_size * (close_price - entry_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Long',
                'entry_price': entry_price,
                'exit_price': close_price,
                'pnl': pnl,
                'exit_reason': 'Intraday Close'
            })
            
            position = 0
            trailing_stop = np.nan
                
        else:  # 空头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {end_time_str}] - 空头收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {close_price:.2f}")
            
            # 计算交易费用（开仓和平仓）
            transaction_fees = position_size * transaction_fee_per_share * 2  # 买入和卖出费用
            pnl = position_size * (entry_price - close_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Short',
                'entry_price': entry_price,
                'exit_price': close_price,
                'pnl': pnl,
                'exit_reason': 'Intraday Close'
            })
            
            position = 0
            trailing_stop = np.nan
    
    # 如果仍有未平仓位且没有结束时间数据点，则在一天结束时平仓
    elif position != 0:
        exit_time = day_df.iloc[-1]['DateTime']
        last_price = day_df.iloc[-1]['Close']
        last_time = day_df.iloc[-1]['Time']
        
        if position == 1:  # 多头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {last_time}] - 多头市场收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {last_price:.2f}")
            
            # 计算交易费用（开仓和平仓）
            transaction_fees = position_size * transaction_fee_per_share * 2  # 买入和卖出费用
            pnl = position_size * (last_price - entry_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Long',
                'entry_price': entry_price,
                'exit_price': last_price,
                'pnl': pnl,
                'exit_reason': 'Market Close'
            })
                
        else:  # 空头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {last_time}] - 空头市场收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {last_price:.2f}")
            
            # 计算交易费用（开仓和平仓）
            transaction_fees = position_size * transaction_fee_per_share * 2  # 买入和卖出费用
            pnl = position_size * (entry_price - last_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Short',
                'entry_price': entry_price,
                'exit_price': last_price,
                'pnl': pnl,
                'exit_reason': 'Market Close'
            })
    
    return trades 

def run_backtest(data_path, ticker=None, initial_capital=100000, lookback_days=90, start_date=None, end_date=None, 
                plot_days=None, random_plots=0, plots_dir='trading_plots',
                check_interval_minutes=30, transaction_fee_per_share=0.01,
                trading_start_time=(10, 00), trading_end_time=(15, 40), max_positions_per_day=float('inf'),
                take_profit_pct=None, print_daily_trades=True, print_trade_details=False):
    """
    运行回测 - 噪声空间策略 + VWAP
    
    参数:
        data_path: 分钟数据CSV文件的路径
        ticker: 标的代码（如果为None，则从文件名中提取）
        initial_capital: 初始资金
        lookback_days: 用于计算噪声区域的天数
        start_date: 回测开始日期
        end_date: 回测结束日期
        plot_days: 要绘制的特定日期列表
        random_plots: 要随机绘制的交易日数量
        plots_dir: 保存绘图的目录
        check_interval_minutes: 交易检查间隔（分钟）
        transaction_fee_per_share: 每股交易费用
        trading_start_time: 交易开始时间
        trading_end_time: 交易结束时间
        max_positions_per_day: 每日最大开仓次数
        take_profit_pct: 止盈百分比（例如：0.02表示2%）
        print_daily_trades: 是否打印每日交易详情
        print_trade_details: 是否打印交易详细信息
        
    返回:
        日度结果DataFrame
        月度结果DataFrame
        交易记录DataFrame
        性能指标字典
    """
    # 如果未提供ticker，从文件名中提取
    if ticker is None:
        # 从文件名中提取ticker
        file_name = os.path.basename(data_path)
        # 移除_market_hours.csv（如果存在）
        ticker = file_name.replace('_market_hours.csv', '')
    
    # 加载和处理数据
    print(f"加载{ticker}数据，从{data_path}...")
    price_df = pd.read_csv(data_path, parse_dates=['DateTime'])
    price_df.sort_values('DateTime', inplace=True)
    
    # 提取日期和时间组件
    price_df['Date'] = price_df['DateTime'].dt.date
    price_df['Time'] = price_df['DateTime'].dt.strftime('%H:%M')
    
    # 按日期范围过滤数据（如果指定）
    if start_date is not None:
        price_df = price_df[price_df['Date'] >= start_date]
        print(f"筛选数据，开始日期为{start_date}")
    
    if end_date is not None:
        price_df = price_df[price_df['Date'] <= end_date]
        print(f"筛选数据，结束日期为{end_date}")
    
    # 检查DayOpen和DayClose列是否存在，如果不存在则创建
    if 'DayOpen' not in price_df.columns or 'DayClose' not in price_df.columns:
        # 对于每一天，获取第一行（9:30 AM开盘价）
        opening_prices = price_df.groupby('Date').first().reset_index()
        opening_prices = opening_prices[['Date', 'Open']].rename(columns={'Open': 'DayOpen'})

        # 对于每一天，获取最后一行（4:00 PM收盘价）
        closing_prices = price_df.groupby('Date').last().reset_index()
        closing_prices = closing_prices[['Date', 'Close']].rename(columns={'Close': 'DayClose'})

        # 将开盘价和收盘价合并回主DataFrame
        price_df = pd.merge(price_df, opening_prices, on='Date', how='left')
        price_df = pd.merge(price_df, closing_prices, on='Date', how='left')
    
    # 使用筛选后数据的DayOpen和DayClose
    # 这些代表9:30 AM开盘价和4:00 PM收盘价
    price_df['prev_close'] = price_df.groupby('Date')['DayClose'].transform('first').shift(1)
    
    # 使用9:30 AM价格作为当天的开盘价
    price_df['day_open'] = price_df.groupby('Date')['DayOpen'].transform('first')
    
    # 为每个交易日计算一次参考价格，并将其应用于该日的所有时间点
    # 这确保了整个交易日使用相同的参考价格
    unique_dates = price_df['Date'].unique()
    
    # 创建临时DataFrame来存储每个日期的参考价格
    date_refs = []
    for d in unique_dates:
        day_data = price_df[price_df['Date'] == d].iloc[0]  # 获取该日第一行数据
        day_open = day_data['day_open']
        prev_close = day_data['prev_close']
        
        # 计算该日的参考价格
        if not pd.isna(prev_close):
            upper_ref = max(day_open, prev_close)
            lower_ref = min(day_open, prev_close)
        else:
            upper_ref = day_open
            lower_ref = day_open
            
        date_refs.append({
            'Date': d,
            'upper_ref': upper_ref,
            'lower_ref': lower_ref
        })
    
    # 创建日期参考价格DataFrame
    date_refs_df = pd.DataFrame(date_refs)
    
    # 将参考价格合并回主DataFrame
    price_df = price_df.drop(columns=['upper_ref', 'lower_ref'], errors='ignore')
    price_df = pd.merge(price_df, date_refs_df, on='Date', how='left')
    
    # 计算每分钟相对开盘的回报（使用day_open保持一致性）
    price_df['ret'] = price_df['Close'] / price_df['day_open'] - 1 

    # 计算噪声区域边界
    print(f"计算噪声区域边界...")
    # 将时间点转为列
    pivot = price_df.pivot(index='Date', columns='Time', values='ret').abs()
    # 计算每个时间点的绝对回报的滚动平均值
    # 这确保我们对每个时间点使用前lookback_days天的数据
    sigma = pivot.rolling(window=lookback_days, min_periods=1).mean().shift(1)
    # 转回长格式
    sigma = sigma.stack().reset_index(name='sigma')
    
    # 将sigma合并回主DataFrame
    price_df = pd.merge(price_df, sigma, on=['Date', 'Time'], how='left')
    
    # 检查每个交易日是否有足够的sigma数据
    # 创建一个标记，记录哪些日期的sigma数据不完整
    incomplete_sigma_dates = set()
    for date in price_df['Date'].unique():
        day_data = price_df[price_df['Date'] == date]
        if day_data['sigma'].isna().any():
            incomplete_sigma_dates.add(date)
    
    # 移除sigma数据不完整的日期
    price_df = price_df[~price_df['Date'].isin(incomplete_sigma_dates)]
    
    # 确保所有剩余的sigma值都有有效数据
    if price_df['sigma'].isna().any():
        print(f"警告: 仍有{price_df['sigma'].isna().sum()}个缺失的sigma值")
    
    # 使用正确的参考价格计算噪声区域的上下边界
    price_df['UpperBound'] = price_df['upper_ref'] * (1 + price_df['sigma'])
    price_df['LowerBound'] = price_df['lower_ref'] * (1 - price_df['sigma'])
    
    # 根据检查间隔生成允许的交易时间
    allowed_times = []
    start_hour, start_minute = trading_start_time  # 使用可配置的开始时间
    end_hour, end_minute = trading_end_time        # 使用可配置的结束时间
    
    current_hour, current_minute = start_hour, start_minute
    while current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute):
        # 将当前时间添加到allowed_times
        allowed_times.append(f"{current_hour:02d}:{current_minute:02d}")
        
        # 增加check_interval_minutes
        current_minute += check_interval_minutes
        if current_minute >= 60:
            current_hour += current_minute // 60
            current_minute = current_minute % 60
    
    # 始终确保trading_end_time包含在内，用于平仓
    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    if end_time_str not in allowed_times:
        allowed_times.append(end_time_str)
        allowed_times.sort()
    
    print(f"使用{check_interval_minutes}分钟的检查间隔")
    
    # 初始化回测变量
    capital = initial_capital
    daily_results = []
    all_trades = []
    total_transaction_fees = 0  # 跟踪总交易费用
    
    # 如果指定了随机生成图表的数量，随机选择交易日
    days_with_trades = []
    if random_plots > 0:
        # 先运行回测，记录有交易的日期
        for trade_date in unique_dates:
            day_data = price_df[price_df['Date'] == trade_date].copy()
            if len(day_data) < 10:  # 跳过数据不足的日期
                continue
                
            prev_close = day_data['prev_close'].iloc[0] if not pd.isna(day_data['prev_close'].iloc[0]) else None
            if prev_close is None:
                continue
                
            # 模拟当天交易
            simulation_result = simulate_day(day_data, prev_close, allowed_times, 100, 
                                           transaction_fee_per_share=transaction_fee_per_share,
                                           take_profit_pct=take_profit_pct,
                                           print_details=print_trade_details)
            
            # 从结果中提取交易
            trades = simulation_result
                
            if trades:  # 如果有交易
                days_with_trades.append(trade_date)
        
        # 如果有交易的日期少于请求的随机图表数量，调整随机图表数量
        random_plots = min(random_plots, len(days_with_trades))
        # 随机选择日期
        if random_plots > 0:
            random_plot_days = random.sample(days_with_trades, random_plots)
        else:
            random_plot_days = []
    else:
        random_plot_days = []
    
    # 合并指定的绘图日期和随机选择的日期
    if plot_days is None:
        plot_days = []
    all_plot_days = list(set(plot_days + random_plot_days))
    
    # 确保绘图目录存在
    if all_plot_days and plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    
    # 创建买入持有回测数据
    buy_hold_data = []
    
    for i, trade_date in enumerate(unique_dates):
        # 获取当天的数据
        day_data = price_df[price_df['Date'] == trade_date].copy()
        day_data = day_data.sort_values('DateTime').reset_index(drop=True)
        
        # 跳过数据不足的日期
        if len(day_data) < 10:  # 任意阈值
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
        
        # 获取前一天的收盘价
        prev_close = day_data['prev_close'].iloc[0] if not pd.isna(day_data['prev_close'].iloc[0]) else None
        
        # 获取当天的开盘价和收盘价（用于计算买入持有）
        open_price = day_data['day_open'].iloc[0]
        close_price = day_data['DayClose'].iloc[0]
        
        # 存储买入持有数据
        buy_hold_data.append({
            'Date': trade_date,
            'Open': open_price,
            'Close': close_price
        })
        
        # 将trade_date转换为字符串格式以便统一显示
        date_str = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        
        # 计算仓位大小
        position_size = floor(capital / open_price)
        
        # 如果资金不足，跳过当天
        if position_size <= 0:
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
                
        # 模拟当天的交易
        simulation_result = simulate_day(day_data, prev_close, allowed_times, position_size,
                           transaction_fee_per_share=transaction_fee_per_share,
                           trading_end_time=trading_end_time, 
                           max_positions_per_day=max_positions_per_day,
                           take_profit_pct=take_profit_pct,
                           print_details=print_trade_details)
        
        # 从结果中提取交易
        trades = simulation_result
        
        # 打印每天的交易信息
        if trades and print_daily_trades:
            # 计算当天总盈亏
            day_total_pnl = sum(trade['pnl'] for trade in trades)
            
            # 创建交易方向与时间的简要信息
            trade_summary = []
            for trade in trades:
                direction = "多" if trade['side'] == 'Long' else "空"
                entry_time = trade['entry_time'].strftime('%H:%M')
                exit_time = trade['exit_time'].strftime('%H:%M')
                pnl = trade['pnl']
                # 添加简要信息: 方向(入场时间->出场时间) 盈亏
                trade_summary.append(f"{direction}({entry_time}->{exit_time}) ${pnl:.2f}")
            
            # 打印单行交易日志
            trade_info = ", ".join(trade_summary)
            print(f"{date_str} | 交易数: {len(trades)} | 总盈亏: ${day_total_pnl:.2f} | {trade_info}")
        
        # 检查是否需要为这一天生成图表
        if trade_date in all_plot_days:
            # 为当天的交易生成图表
            plot_path = os.path.join(plots_dir, f"{ticker}_trade_visualization_{trade_date}")
            
            # 添加交易类型到文件名
            sides = [trade['side'] for trade in trades]
            if 'Long' in sides and 'Short' not in sides:
                plot_path += "_Long.png"
            elif 'Short' in sides and 'Long' not in sides:
                plot_path += "_Short.png"
            elif 'Long' in sides and 'Short' in sides:
                plot_path += "_Mixed.png"
            else:
                plot_path += ".png"  # 没有交易
                
            # 生成并保存图表
            plot_trading_day(day_data, trades, save_path=plot_path)
        
        # 计算每日盈亏和交易费用
        day_pnl = 0
        day_transaction_fees = 0
        for trade in trades:
            day_pnl += trade['pnl']
            # 从每笔交易中提取交易费用
            if 'transaction_fees' not in trade:
                # 如果交易数据中没有交易费用，则计算
                trade['transaction_fees'] = position_size * transaction_fee_per_share * 2  # 买入和卖出费用
            day_transaction_fees += trade['transaction_fees']
        
        # 添加到总交易费用
        total_transaction_fees += day_transaction_fees
        
        # 更新资金并计算每日回报
        capital_start = capital
        capital += day_pnl
        daily_return = day_pnl / capital_start
        
        # 存储每日结果
        daily_results.append({
            'Date': trade_date,
            'capital': capital,
            'daily_return': daily_return
        })
        
        # 存储交易
        for trade in trades:
            trade['Date'] = trade_date
            all_trades.append(trade)
    
    # 创建每日结果DataFrame
    daily_df = pd.DataFrame(daily_results)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df.set_index('Date', inplace=True)
    
    # 创建买入持有DataFrame
    buy_hold_df = pd.DataFrame(buy_hold_data)
    buy_hold_df['Date'] = pd.to_datetime(buy_hold_df['Date'])
    buy_hold_df.set_index('Date', inplace=True)
    
    # 计算买入持有策略的表现
    if not buy_hold_df.empty:
        # 计算每日收益率
        buy_hold_df['daily_return'] = buy_hold_df['Close'] / buy_hold_df['Close'].shift(1) - 1
        
        # 计算累积资本
        buy_hold_df['capital'] = initial_capital * (1 + buy_hold_df['daily_return']).cumprod().fillna(1)
    
    # 计算月度回报
    monthly = daily_df.resample('ME').first()[['capital']].rename(columns={'capital': 'month_start'})
    monthly['month_end'] = daily_df.resample('ME').last()['capital']
    monthly['monthly_return'] = monthly['month_end'] / monthly['month_start'] - 1
    
    # 打印月度回报
    print("\n月度回报:")
    print(monthly[['month_start', 'month_end', 'monthly_return']])
    
    # 计算总体表现
    total_return = capital / initial_capital - 1
    print(f"\n总回报: {total_return*100:.2f}%")
    
    # 创建交易DataFrame
    trades_df = pd.DataFrame(all_trades)
    
    # 计算策略性能指标
    print(f"\n计算策略性能指标...")
    metrics = calculate_performance_metrics(daily_df, trades_df, initial_capital, buy_hold_df=buy_hold_df)
    
    # 打印交易费用统计
    print(f"\n交易费用统计:")
    print(f"总交易费用: ${total_transaction_fees:.2f}")
    if len(trades_df) > 0:
        print(f"平均每笔交易费用: ${total_transaction_fees / len(trades_df):.2f}")
    if len(daily_df) > 0:
        print(f"平均每日交易费用: ${total_transaction_fees / len(daily_df):.2f}")
    print(f"交易费用占初始资金比例: {total_transaction_fees / initial_capital * 100:.2f}%")
    print(f"交易费用占总收益比例: {total_transaction_fees / (capital - initial_capital) * 100:.2f}%" if capital > initial_capital else "交易费用占总收益比例: N/A (无盈利)")
    
    # 打印简化的性能指标
    print(f"\n策略性能指标:")
    strategy_name = f"{ticker} Curr.Band + VWAP"
    print(f"策略: {strategy_name}")
    
    # 创建表格格式对比策略与买入持有的指标
    print("\n性能指标对比:")
    print(f"{'指标':<20} | {'策略':<15} | {f'{ticker} Buy & Hold':<15}")
    print("-" * 55)
    
    # 总回报率
    print(f"{'总回报率':<20} | {metrics['total_return']*100:>14.1f}% | {metrics['buy_hold_return']*100:>14.1f}%")
    
    # 年化收益率
    print(f"{'年化收益率':<20} | {metrics['irr']*100:>14.1f}% | {metrics['buy_hold_irr']*100:>14.1f}%")
    
    # 波动率
    print(f"{'波动率':<20} | {metrics['volatility']*100:>14.1f}% | {metrics['buy_hold_volatility']*100:>14.1f}%")
    
    # 夏普比率
    print(f"{'夏普比率':<20} | {metrics['sharpe_ratio']:>14.2f} | {metrics['buy_hold_sharpe']:>14.2f}")
    
    # 最大回撤
    print(f"{'最大回撤':<20} | {metrics['mdd']*100:>14.1f}% | {metrics['buy_hold_mdd']*100:>14.1f}%")
    
    # 策略特有指标
    print(f"\n策略特有指标:")
    print(f"胜率: {metrics['hit_ratio']*100:.1f}%")
    print(f"总交易次数: {metrics['total_trades']}")
    print(f"平均每日交易次数: {metrics['avg_daily_trades']:.2f}")
    
    # 打印超额收益
    print(f"\n策略超额收益: {(metrics['total_return'] - metrics['buy_hold_return'])*100:.1f}%")
    
    return daily_df, monthly, trades_df, metrics 

def calculate_performance_metrics(daily_df, trades_df, initial_capital, risk_free_rate=0.02, trading_days_per_year=252, buy_hold_df=None):
    """
    计算策略的性能指标
    
    参数:
        daily_df: 包含每日回测结果的DataFrame
        trades_df: 包含所有交易的DataFrame
        initial_capital: 初始资金
        risk_free_rate: 无风险利率，默认为2%
        trading_days_per_year: 一年的交易日数量，默认为252
        buy_hold_df: 买入持有策略的DataFrame
        
    返回:
        包含各种性能指标的字典
    """
    metrics = {}
    
    # 确保daily_df有数据
    if len(daily_df) == 0:
        print("警告: 没有足够的数据来计算性能指标")
        # 返回默认值
        return {
            'total_return': 0, 'irr': 0, 'volatility': 0, 'sharpe_ratio': 0,
            'hit_ratio': 0, 'mdd': 0, 'buy_hold_return': 0, 'buy_hold_irr': 0,
            'buy_hold_volatility': 0, 'buy_hold_sharpe': 0, 'buy_hold_mdd': 0
        }
    
    # 1. 总回报率 (Total Return)
    final_capital = daily_df['capital'].iloc[-1]
    metrics['total_return'] = final_capital / initial_capital - 1
    
    # 2. 年化收益率 (IRR - Internal Rate of Return)
    # 获取回测的开始和结束日期
    start_date = daily_df.index[0]
    end_date = daily_df.index[-1]
    # 计算实际年数（考虑实际日历日而不仅仅是交易日）
    years = (end_date - start_date).days / 365.25
    # 如果时间跨度太短，使用交易日计算
    if years < 0.1:  # 少于约36天
        trading_days = len(daily_df)
        years = trading_days / trading_days_per_year
    
    # 计算年化收益率 (CAGR - Compound Annual Growth Rate)
    if years > 0:
        metrics['irr'] = (1 + metrics['total_return']) ** (1 / years) - 1
    else:
        metrics['irr'] = 0
    
    # 3. 波动率 (Vol - Volatility)
    # 计算日收益率的标准差，然后年化
    daily_returns = daily_df['daily_return']
    # 移除异常值（如果有）
    daily_returns = daily_returns[daily_returns.between(daily_returns.quantile(0.001), 
                                                      daily_returns.quantile(0.999))]
    metrics['volatility'] = daily_returns.std() * np.sqrt(trading_days_per_year)
    
    # 4. 夏普比率 (Sharpe Ratio)
    if metrics['volatility'] > 0:
        metrics['sharpe_ratio'] = (metrics['irr'] - risk_free_rate) / metrics['volatility']
    else:
        metrics['sharpe_ratio'] = 0
    
    # 5. 胜率 (Hit Ratio)和交易统计
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        metrics['hit_ratio'] = len(winning_trades) / len(trades_df)
        
        # 计算平均盈利和平均亏损
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # 计算盈亏比
        metrics['profit_loss_ratio'] = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 总交易次数
        metrics['total_trades'] = len(trades_df)
        
        # 计算每日交易次数
        daily_trade_counts = trades_df.groupby('Date').size()
        metrics['avg_daily_trades'] = daily_trade_counts.mean() if len(daily_trade_counts) > 0 else 0
        metrics['max_daily_trades'] = daily_trade_counts.max() if len(daily_trade_counts) > 0 else 0
        
        # 计算每日盈亏
        daily_pnl = trades_df.groupby('Date')['pnl'].sum()
        metrics['max_daily_loss'] = daily_pnl.min() if len(daily_pnl) > 0 and daily_pnl.min() < 0 else 0
        metrics['max_daily_gain'] = daily_pnl.max() if len(daily_pnl) > 0 else 0
    else:
        metrics['hit_ratio'] = 0
        metrics['profit_loss_ratio'] = 0
        metrics['total_trades'] = 0
        metrics['avg_daily_trades'] = 0
        metrics['max_daily_trades'] = 0
        metrics['max_daily_loss'] = 0
        metrics['max_daily_gain'] = 0
    
    # 6. 最大回撤 (MDD - Maximum Drawdown)
    # 计算每日资金的累计最大值
    daily_df['peak'] = daily_df['capital'].cummax()
    # 计算每日回撤
    daily_df['drawdown'] = (daily_df['capital'] - daily_df['peak']) / daily_df['peak']
    # 最大回撤
    metrics['mdd'] = daily_df['drawdown'].min() * -1
    
    # 计算回撤持续时间
    # 找到每个回撤开始的点
    drawdown_begins = (daily_df['peak'] != daily_df['peak'].shift(1)) & (daily_df['peak'] != daily_df['capital'])
    # 找到每个回撤结束的点（资金达到新高）
    drawdown_ends = daily_df['capital'] == daily_df['peak']
    
    # 计算最长回撤持续时间（交易日）
    if drawdown_begins.any() and drawdown_ends.any():
        begin_dates = daily_df.index[drawdown_begins]
        end_dates = daily_df.index[drawdown_ends]
        
        max_duration = 0
        for begin_date in begin_dates:
            # 找到这个回撤之后的第一个结束点
            end_date = end_dates[end_dates > begin_date]
            if len(end_date) > 0:
                duration = (end_date.min() - begin_date).days
                max_duration = max(max_duration, duration)
        
        metrics['max_drawdown_duration'] = max_duration
    else:
        metrics['max_drawdown_duration'] = 0
    
    # 计算Calmar比率 (年化收益率/最大回撤)
    if metrics['mdd'] > 0:
        metrics['calmar_ratio'] = metrics['irr'] / metrics['mdd']
    else:
        metrics['calmar_ratio'] = float('inf')  # 如果没有回撤，设为无穷大
        
    # 计算曝光时间 (Exposure Time)
    if len(trades_df) > 0:
        # 计算每笔交易的持仓时间（以分钟为单位）
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
        
        # 计算总交易时间（分钟）
        total_trade_minutes = trades_df['duration'].sum()
        
        # 计算回测总时间（分钟）
        # 假设每个交易日有6.5小时（390分钟）
        trading_minutes_per_day = 390
        total_backtest_minutes = len(daily_df) * trading_minutes_per_day
        
        # 计算曝光时间百分比
        metrics['exposure_time'] = total_trade_minutes / total_backtest_minutes
    else:
        metrics['exposure_time'] = 0
    
    # 计算买入持有策略的表现
    if buy_hold_df is not None and not buy_hold_df.empty:
        # 计算买入持有策略的总回报率
        if 'capital' in buy_hold_df.columns:
            final_buy_hold_capital = buy_hold_df['capital'].iloc[-1]
            metrics['buy_hold_return'] = final_buy_hold_capital / initial_capital - 1
            
            # 计算买入持有策略的年化收益率
            if years > 0:
                metrics['buy_hold_irr'] = (1 + metrics['buy_hold_return']) ** (1 / years) - 1
            else:
                metrics['buy_hold_irr'] = 0
            
            # 计算买入持有策略的波动率
            if 'daily_return' in buy_hold_df.columns:
                buy_hold_returns = buy_hold_df['daily_return'].dropna()
                # 移除异常值
                buy_hold_returns = buy_hold_returns[buy_hold_returns.between(
                    buy_hold_returns.quantile(0.001), buy_hold_returns.quantile(0.999))]
                metrics['buy_hold_volatility'] = buy_hold_returns.std() * np.sqrt(trading_days_per_year)
                
                # 计算买入持有策略的夏普比率
                if metrics['buy_hold_volatility'] > 0:
                    metrics['buy_hold_sharpe'] = (metrics['buy_hold_irr'] - risk_free_rate) / metrics['buy_hold_volatility']
                else:
                    metrics['buy_hold_sharpe'] = 0
            else:
                metrics['buy_hold_volatility'] = 0
                metrics['buy_hold_sharpe'] = 0
            
            # 计算买入持有策略的最大回撤
            if 'capital' in buy_hold_df.columns:
                buy_hold_df['peak'] = buy_hold_df['capital'].cummax()
                buy_hold_df['drawdown'] = (buy_hold_df['capital'] - buy_hold_df['peak']) / buy_hold_df['peak']
                metrics['buy_hold_mdd'] = buy_hold_df['drawdown'].min() * -1
            else:
                metrics['buy_hold_mdd'] = 0
        else:
            # 如果buy_hold_df中没有capital列，则计算起始日期和结束日期的价格变化
            if 'Close' in buy_hold_df.columns:
                start_price = buy_hold_df['Close'].iloc[0]
                end_price = buy_hold_df['Close'].iloc[-1]
                metrics['buy_hold_return'] = end_price / start_price - 1
                
                # 计算买入持有策略的年化收益率
                if years > 0:
                    metrics['buy_hold_irr'] = (1 + metrics['buy_hold_return']) ** (1 / years) - 1
                else:
                    metrics['buy_hold_irr'] = 0
                
                # 其他指标设为0
                metrics['buy_hold_volatility'] = 0
                metrics['buy_hold_sharpe'] = 0
                metrics['buy_hold_mdd'] = 0
            else:
                # 没有足够的数据计算买入持有策略的表现
                metrics['buy_hold_return'] = 0
                metrics['buy_hold_irr'] = 0
                metrics['buy_hold_volatility'] = 0
                metrics['buy_hold_sharpe'] = 0
                metrics['buy_hold_mdd'] = 0
    else:
        # 没有买入持有的数据，设置默认值
        metrics['buy_hold_return'] = 0
        metrics['buy_hold_irr'] = 0
        metrics['buy_hold_volatility'] = 0
        metrics['buy_hold_sharpe'] = 0
        metrics['buy_hold_mdd'] = 0
    
    return metrics 

def plot_specific_days(data_path, dates_to_plot, lookback_days=90, plots_dir='trading_plots', 
                      check_interval_minutes=30, transaction_fee_per_share=0.01,
                      trading_start_time=(9, 40), trading_end_time=(15, 50), 
                      max_positions_per_day=float('inf'), take_profit_pct=None):
    """
    为指定的日期生成交易图表
    
    参数:
        data_path: 分钟数据CSV文件的路径
        dates_to_plot: 要绘制的日期列表 (datetime.date 对象列表)
        lookback_days: 用于计算Noise Area的天数
        plots_dir: 保存图表的目录
        check_interval_minutes: 交易检查间隔（分钟）
        take_profit_pct: 止盈百分比
    """
    # 运行回测，指定要绘制的日期
    _, _, _, _ = run_backtest(
        data_path=data_path,
        lookback_days=lookback_days,
        plot_days=dates_to_plot,
        plots_dir=plots_dir,
        check_interval_minutes=check_interval_minutes,
        transaction_fee_per_share=transaction_fee_per_share,
        trading_start_time=trading_start_time,
        trading_end_time=trading_end_time,
        max_positions_per_day=max_positions_per_day,
        take_profit_pct=take_profit_pct
    )
    
    print(f"\n已为以下日期生成图表:")
    for d in dates_to_plot:
        print(f"- {d}")
    print(f"图表保存在 '{plots_dir}' 目录中")

# 示例用法
if __name__ == "__main__":  
    # 运行回测
    daily_results, monthly_results, trades, metrics = run_backtest(
        'tqqq_market_hours_with_indicators.csv',  # 数据文件
        ticker='TQQQ',                     # 指定ticker
        initial_capital=10000, 
        lookback_days=10,
        start_date=date(2024, 1, 20), 
        end_date=date(2025, 1, 20),
        check_interval_minutes=10,
        transaction_fee_per_share=0.005,  # 每股交易费用
        # 交易时间配置
        trading_start_time=(9, 40),  # 交易开始时间
        trading_end_time=(15, 40),      # 交易结束时间
        max_positions_per_day=3,  # 每天最多开仓3次
        take_profit_pct=0.03,     # 2%止盈
        # random_plots=3,  # 随机选择3天生成图表
        # plots_dir='trading_plots',  # 图表保存目录
        print_daily_trades=False,  # 是否打印每日交易详情
        print_trade_details=False  # 是否打印交易细节
    ) 