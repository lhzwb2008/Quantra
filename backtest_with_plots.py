import pandas as pd
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta, date
import random
import os
from plot_trading_day import plot_trading_day

def calculate_vwap_incrementally(prices, volumes):
    """
    Calculate VWAP incrementally for a series of prices and volumes
    """
    cum_vol = 0
    cum_pv = 0
    vwaps = []
    
    for price, volume in zip(prices, volumes):
        cum_vol += volume
        cum_pv += price * volume
        
        if cum_vol > 0:
            vwap = cum_pv / cum_vol
        else:
            vwap = price  # If no volume yet, use current price
            
        vwaps.append(vwap)
        
    return vwaps

def simulate_day(day_df, prev_close, allowed_times, position_size, debug=False):
    """
    Simulate trading for a single day using curr.band + VWAP strategy
    """
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = np.nan
    trailing_stop = np.nan
    trade_entry_time = None
    trades = []
    trade_completed = False  # Flag to track if a trade has been completed for the day
    
    # Calculate VWAP incrementally during simulation
    prices = []
    volumes = []
    vwaps = []
    
    if debug:
        print(f"\nSimulating day: {day_df['Date'].iloc[0]}")
        print(f"Open: {day_df['Open'].iloc[0]}, Prev Close: {prev_close}")
        print(f"Upper Ref: {day_df['upper_ref'].iloc[0]}, Lower Ref: {day_df['lower_ref'].iloc[0]}")
    
    for idx, row in day_df.iterrows():
        current_time = row['Time']
        price = row['Close']
        volume = row['Volume']
        upper = row['UpperBound']
        lower = row['LowerBound']
        
        # Update prices and volumes for VWAP calculation
        prices.append(price)
        volumes.append(volume)
        
        # Calculate VWAP up to this point
        vwap = calculate_vwap_incrementally(prices, volumes)[-1]
        vwaps.append(vwap)
        
        # Entry signals at allowed times only if no trade has been completed for the day
        if position == 0 and current_time in allowed_times and not trade_completed:
            if price > upper:
                # Long entry
                position = 1
                entry_price = price
                trade_entry_time = row['DateTime']
                # Trailing stop: max(UpperBound, VWAP)
                trailing_stop = max(upper, vwap)
                
                if debug:
                    print(f"LONG ENTRY at {current_time}: Price={price:.2f}, Upper={upper:.2f}, VWAP={vwap:.2f}, Stop={trailing_stop:.2f}")
                    
            elif price < lower:
                # Short entry
                position = -1
                entry_price = price
                trade_entry_time = row['DateTime']
                # Trailing stop: min(LowerBound, VWAP)
                trailing_stop = min(lower, vwap)
                
                if debug:
                    print(f"SHORT ENTRY at {current_time}: Price={price:.2f}, Lower={lower:.2f}, VWAP={vwap:.2f}, Stop={trailing_stop:.2f}")
        
        # Update trailing stop and check for exit signals
        if position != 0:
            if position == 1:  # Long position
                new_stop = max(upper, vwap)
                # Only update in favorable direction (raise the stop)
                trailing_stop = max(trailing_stop, new_stop)
                
                if debug and current_time in allowed_times:
                    print(f"LONG UPDATE at {current_time}: Price={price:.2f}, Upper={upper:.2f}, VWAP={vwap:.2f}, Stop={trailing_stop:.2f}")
                    
                if price < trailing_stop and current_time in allowed_times:
                    # Exit long position
                    exit_time = row['DateTime']
                    pnl = position_size * (price - entry_price)
                    trades.append({
                        'entry_time': trade_entry_time,
                        'exit_time': exit_time,
                        'side': 'Long',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    })
                    
                    if debug:
                        print(f"LONG EXIT at {current_time}: Price={price:.2f}, Stop={trailing_stop:.2f}, P&L=${pnl:.2f}")
                        
                    position = 0
                    trailing_stop = np.nan
                    trade_completed = True  # Mark that a trade has been completed for the day
                    
            elif position == -1:  # Short position
                new_stop = min(lower, vwap)
                # Only update in favorable direction (lower the stop)
                trailing_stop = min(trailing_stop, new_stop)
                
                if debug and current_time in allowed_times:
                    print(f"SHORT UPDATE at {current_time}: Price={price:.2f}, Lower={lower:.2f}, VWAP={vwap:.2f}, Stop={trailing_stop:.2f}")
                    
                if price > trailing_stop and current_time in allowed_times:
                    # Exit short position
                    exit_time = row['DateTime']
                    pnl = position_size * (entry_price - price)
                    trades.append({
                        'entry_time': trade_entry_time,
                        'exit_time': exit_time,
                        'side': 'Short',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    })
                    
                    if debug:
                        print(f"SHORT EXIT at {current_time}: Price={price:.2f}, Stop={trailing_stop:.2f}, P&L=${pnl:.2f}")
                        
                    position = 0
                    trailing_stop = np.nan
                    trade_completed = True  # Mark that a trade has been completed for the day
    
    # Close any open position at the end of the day
    if position != 0:
        exit_time = day_df.iloc[-1]['DateTime']
        last_price = day_df.iloc[-1]['Close']
        last_time = day_df.iloc[-1]['Time']
        
        if position == 1:  # Long position
            pnl = position_size * (last_price - entry_price)
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Long',
                'entry_price': entry_price,
                'exit_price': last_price,
                'pnl': pnl,
                'exit_reason': 'Market Close'
            })
            
            if debug:
                print(f"LONG EXIT at {last_time} (CLOSE): Price={last_price:.2f}, P&L=${pnl:.2f}")
                
        else:  # Short position
            pnl = position_size * (entry_price - last_price)
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Short',
                'entry_price': entry_price,
                'exit_price': last_price,
                'pnl': pnl,
                'exit_reason': 'Market Close'
            })
            
            if debug:
                print(f"SHORT EXIT at {last_time} (CLOSE): Price={last_price:.2f}, P&L=${pnl:.2f}")
    
    return trades

def run_backtest(data_path, initial_capital=100000, lookback_days=14, start_date=None, end_date=None, 
                debug_days=None, plot_days=None, random_plots=0, plots_dir='trading_plots'):
    """
    Run the backtest on SPY data
    
    Parameters:
        data_path: Path to the SPY minute data CSV file
        initial_capital: Initial capital for the backtest
        lookback_days: Number of days to use for calculating the Noise Area
        start_date: Start date for the backtest (datetime.date)
        end_date: End date for the backtest (datetime.date)
        debug_days: List of dates to print detailed debug information for
        plot_days: List of specific dates to plot
        random_plots: Number of random trading days to plot (0 for none)
        plots_dir: Directory to save plots in
        
    Returns:
        DataFrame with daily results
        DataFrame with monthly results
        DataFrame with trades
    """
    # Load and process data
    print(f"Loading data from {data_path}...")
    spy_df = pd.read_csv(data_path, parse_dates=['DateTime'])
    spy_df.sort_values('DateTime', inplace=True)
    
    # Extract date and time components
    spy_df['Date'] = spy_df['DateTime'].dt.date
    spy_df['Time'] = spy_df['DateTime'].dt.strftime('%H:%M')
    
    # Filter data by date range if specified
    if start_date is not None:
        spy_df = spy_df[spy_df['Date'] >= start_date]
        print(f"Filtered data to start from {start_date}")
    
    if end_date is not None:
        spy_df = spy_df[spy_df['Date'] <= end_date]
        print(f"Filtered data to end at {end_date}")
    
    # Use DayOpen and DayClose from the filtered data
    # These represent the 9:30 AM open price and 4:00 PM close price
    spy_df['prev_close'] = spy_df.groupby('Date')['DayClose'].transform('first').shift(1)
    
    # Use the 9:30 AM price as the opening price for the day
    spy_df['day_open'] = spy_df.groupby('Date')['DayOpen'].transform('first')
    
    # 为每个交易日计算一次参考价格，并将其应用于该日的所有时间点
    # 这确保了整个交易日使用相同的参考价格
    unique_dates = spy_df['Date'].unique()
    
    # 创建临时DataFrame来存储每个日期的参考价格
    date_refs = []
    for d in unique_dates:
        day_data = spy_df[spy_df['Date'] == d].iloc[0]  # 获取该日第一行数据
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
    spy_df = spy_df.drop(columns=['upper_ref', 'lower_ref'], errors='ignore')
    spy_df = pd.merge(spy_df, date_refs_df, on='Date', how='left')
    
    print("已为每个交易日计算固定的参考价格")
    
    # Calculate return from open for each minute (using day_open for consistency)
    spy_df['ret'] = spy_df['Close'] / spy_df['day_open'] - 1
    
    # Calculate the Noise Area boundaries
    print("Calculating Noise Area boundaries...")
    # Pivot to get time-of-day in columns
    pivot = spy_df.pivot(index='Date', columns='Time', values='ret').abs()
    # Calculate rolling average of absolute returns for each time-of-day
    # This ensures we're using the previous 14 days for each time point
    sigma = pivot.rolling(window=lookback_days, min_periods=1).mean().shift(1)
    # Convert back to long format
    sigma = sigma.stack().reset_index(name='sigma')
    
    # Merge sigma back to the main dataframe
    spy_df = pd.merge(spy_df, sigma, on=['Date', 'Time'], how='left')
    
    # 检查每个交易日是否有足够的sigma数据
    # 创建一个标记，记录哪些日期的sigma数据不完整
    incomplete_sigma_dates = set()
    for date in spy_df['Date'].unique():
        day_data = spy_df[spy_df['Date'] == date]
        if day_data['sigma'].isna().any():
            incomplete_sigma_dates.add(date)
            print(f"警告: {date} 的sigma数据不完整，将跳过该交易日")
    
    # 移除sigma数据不完整的日期
    spy_df = spy_df[~spy_df['Date'].isin(incomplete_sigma_dates)]
    
    # 确保所有剩余的sigma值都有有效数据（不应该有NaN值了）
    if spy_df['sigma'].isna().any():
        print(f"警告: 仍有{spy_df['sigma'].isna().sum()}个缺失的sigma值")
    
    # Calculate upper and lower boundaries of the Noise Area using the correct reference prices
    spy_df['UpperBound'] = spy_df['upper_ref'] * (1 + spy_df['sigma'])
    spy_df['LowerBound'] = spy_df['lower_ref'] * (1 - spy_df['sigma'])
    
    # Define allowed trading times (semi-hourly intervals starting from 10:00)
    allowed_times = [
        '10:00', '10:30', '11:00', '11:30', '12:00', '12:30',
        '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00'
    ]
    
    # Initialize backtest variables
    capital = initial_capital
    daily_results = []
    all_trades = []
    
    # Run the backtest day by day
    print("Running backtest...")
    unique_dates = sorted(spy_df['Date'].unique())
    
    # 如果指定了随机生成图表的数量，随机选择交易日
    days_with_trades = []
    if random_plots > 0:
        # 先运行回测，记录有交易的日期
        for trade_date in unique_dates:
            day_spy = spy_df[spy_df['Date'] == trade_date].copy()
            if len(day_spy) < 10:  # 跳过数据不足的日期
                continue
                
            prev_close = day_spy['prev_close'].iloc[0] if not pd.isna(day_spy['prev_close'].iloc[0]) else None
            if prev_close is None:
                continue
                
            # 模拟当天交易
            trades = simulate_day(day_spy, prev_close, allowed_times, 100, debug=False)
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
    
    for i, trade_date in enumerate(unique_dates):
        # Get data for the current day
        day_spy = spy_df[spy_df['Date'] == trade_date].copy()
        day_spy = day_spy.sort_values('DateTime').reset_index(drop=True)
        
        # Skip days with insufficient data
        if len(day_spy) < 10:  # Arbitrary threshold
            print(f"{trade_date}: Insufficient data, skipping")
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
        
        # Get the previous day's close
        prev_close = day_spy['prev_close'].iloc[0] if not pd.isna(day_spy['prev_close'].iloc[0]) else None
        
        # Get the opening price for the day
        open_price = day_spy['day_open'].iloc[0]
        
        # Calculate position size (fixed, no VIX adjustment)
        position_size = floor(capital / open_price)
        
        # Skip days with insufficient capital
        if position_size <= 0:
            print(f"{trade_date}: Insufficient capital, skipping")
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
        
        # Check if this is a debug day
        debug = debug_days is not None and trade_date in debug_days
        
        # Simulate trading for the day
        trades = simulate_day(day_spy, prev_close, allowed_times, position_size, debug=debug)
        
        # 检查是否需要为这一天生成图表
        if trade_date in all_plot_days:
            # 打印出计算时的参数
            print(f"\nPlotting day {trade_date}:")
            print(f"  Previous close: {prev_close:.2f}")
            print(f"  Day open: {open_price:.2f}")
            print(f"  Upper reference: {day_spy['upper_ref'].iloc[0]:.2f}")
            print(f"  Lower reference: {day_spy['lower_ref'].iloc[0]:.2f}")
            
            # 打印前五分钟的sigma和各条线的值
            print("\n前五分钟的数据:")
            print("  时间    |   价格   |   sigma  |  上界线  |  下界线  |   VWAP   ")
            print("---------|----------|----------|----------|----------|----------")
            
            # 获取前五分钟的数据（或者所有数据，如果不足五分钟）
            first_minutes = min(5, len(day_spy))
            
            # 计算VWAP
            prices = []
            volumes = []
            vwaps = []
            
            for i in range(first_minutes):
                row = day_spy.iloc[i]
                time_str = row['Time']
                price = row['Close']
                sigma = row['sigma']
                upper = row['UpperBound']
                lower = row['LowerBound']
                
                # 更新VWAP计算
                prices.append(price)
                volumes.append(row['Volume'])
                vwap = calculate_vwap_incrementally(prices, volumes)[-1]
                vwaps.append(vwap)
                
                print(f"  {time_str} | {price:8.2f} | {sigma:8.6f} | {upper:8.2f} | {lower:8.2f} | {vwap:8.2f}")
            
            # 为当天的交易生成图表
            plot_path = os.path.join(plots_dir, f"trade_visualization_{trade_date}")
            
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
            plot_trading_day(day_spy, trades, save_path=plot_path)
        
        # Calculate daily P&L
        day_pnl = sum(trade['pnl'] for trade in trades)
        
        # Update capital and calculate daily return
        capital_start = capital
        capital += day_pnl
        daily_return = day_pnl / capital_start
        
        # Store daily results
        daily_results.append({
            'Date': trade_date,
            'capital': capital,
            'daily_return': daily_return
        })
        
        # Store trades
        for trade in trades:
            trade['Date'] = trade_date
            all_trades.append(trade)
        
        # Print detailed trade information
        trade_details = []
        for trade in trades:
            entry_time = trade['entry_time'].strftime('%H:%M')
            exit_time = trade['exit_time'].strftime('%H:%M')
            side = trade['side']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            pnl = trade['pnl']
            exit_reason = trade['exit_reason']
            
            trade_detail = f"{entry_time}-{exit_time} {side} {entry_price:.2f}->{exit_price:.2f} ${pnl:.2f} ({exit_reason})"
            trade_details.append(trade_detail)
        
        trade_details_str = ', '.join(trade_details) if trade_details else "No trades"
        print(f"{trade_date}: {trade_details_str}, P&L=${day_pnl:.2f}, Return={daily_return*100:.2f}%")
    
    # Create daily results DataFrame
    daily_df = pd.DataFrame(daily_results)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df.set_index('Date', inplace=True)
    
    # Calculate monthly returns
    monthly = daily_df.resample('ME').first()[['capital']].rename(columns={'capital': 'month_start'})
    monthly['month_end'] = daily_df.resample('ME').last()['capital']
    monthly['monthly_return'] = monthly['month_end'] / monthly['month_start'] - 1
    
    # Print monthly returns
    print("\nMonthly Returns:")
    print(monthly[['month_start', 'month_end', 'monthly_return']])
    
    # Calculate overall performance
    total_return = capital / initial_capital - 1
    print(f"\nTotal Return: {total_return*100:.2f}%")
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(all_trades)
    
    return daily_df, monthly, trades_df

def plot_specific_days(data_path, dates_to_plot, lookback_days=14, plots_dir='trading_plots'):
    """
    为指定的日期生成交易图表
    
    参数:
        data_path: SPY分钟数据CSV文件的路径
        dates_to_plot: 要绘制的日期列表 (datetime.date 对象列表)
        lookback_days: 用于计算Noise Area的天数
        plots_dir: 保存图表的目录
    """
    # 运行回测，指定要绘制的日期
    _, _, _ = run_backtest(
        data_path=data_path,
        lookback_days=lookback_days,
        plot_days=dates_to_plot,
        plots_dir=plots_dir
    )
    
    print(f"\n已为以下日期生成图表:")
    for d in dates_to_plot:
        print(f"- {d}")
    print(f"图表保存在 '{plots_dir}' 目录中")

# 示例用法
if __name__ == "__main__":
    # 运行回测，随机生成5个交易日的图表
    daily_results, monthly_results, trades = run_backtest(
        'spy_market_hours.csv', 
        initial_capital=100000, 
        lookback_days=14, 
        start_date=date(2007, 5, 1), 
        end_date=date(2024, 4, 1),
        # random_plots=5,  # 随机生成5个交易日的图表
        plot_days=[date(2022, 1, 20), date(2022, 1, 31), date(2022, 4, 29)],  # 指定要绘制的日期
        plots_dir='trading_plots'  # 图表保存目录
    )
