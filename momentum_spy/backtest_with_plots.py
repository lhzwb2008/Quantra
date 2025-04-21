import pandas as pd
import numpy as np
from math import floor, sqrt
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
        
        # Entry signals at allowed times
        if position == 0 and current_time in allowed_times:
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
    
    # Find the 15:40 data point if it exists
    close_time_rows = day_df[day_df['Time'] == '15:40']
    
    # If we have a 15:40 data point and still have an open position, close it
    if not close_time_rows.empty and position != 0:
        close_row = close_time_rows.iloc[0]
        exit_time = close_row['DateTime']
        close_price = close_row['Close']
        
        if position == 1:  # Long position
            pnl = position_size * (close_price - entry_price)
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Long',
                'entry_price': entry_price,
                'exit_price': close_price,
                'pnl': pnl,
                'exit_reason': 'Intraday Close'
            })
            
            if debug:
                print(f"LONG EXIT at 15:40 (INTRADAY CLOSE): Price={close_price:.2f}, P&L=${pnl:.2f}")
            
            position = 0
            trailing_stop = np.nan
                
        else:  # Short position
            pnl = position_size * (entry_price - close_price)
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Short',
                'entry_price': entry_price,
                'exit_price': close_price,
                'pnl': pnl,
                'exit_reason': 'Intraday Close'
            })
            
            if debug:
                print(f"SHORT EXIT at 15:40 (INTRADAY CLOSE): Price={close_price:.2f}, P&L=${pnl:.2f}")
            
            position = 0
            trailing_stop = np.nan
    
    # If we still have an open position at the end of the day (no 15:40 data point), close it
    elif position != 0:
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

def run_backtest(data_path, initial_capital=100000, lookback_days=90, start_date=None, end_date=None, 
                debug_days=None, plot_days=None, random_plots=0, plots_dir='trading_plots',
                vix_path='vix_all.csv', use_dynamic_leverage=True, use_volatility_sizing=False,
                volatility_target=0.02, check_interval_minutes=30):
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
        vix_path: Path to the VIX data CSV file
        use_dynamic_leverage: Whether to use dynamic leverage based on VIX levels
        use_volatility_sizing: Whether to use volatility-based position sizing
        volatility_target: Target daily volatility for position sizing
        check_interval_minutes: Interval in minutes between trading checks (default: 30)
        
    Returns:
        DataFrame with daily results
        DataFrame with monthly results
        DataFrame with trades
        Dictionary with performance metrics
    """
    # Load and process SPY data
    print(f"Loading SPY data from {data_path}...")
    spy_df = pd.read_csv(data_path, parse_dates=['DateTime'])
    spy_df.sort_values('DateTime', inplace=True)
    
    # Extract date and time components
    spy_df['Date'] = spy_df['DateTime'].dt.date
    spy_df['Time'] = spy_df['DateTime'].dt.strftime('%H:%M')
    
    # Load and process VIX data for position sizing
    if use_dynamic_leverage:
        print("Loading VIX data for dynamic position sizing...")
        try:
            # Load VIX data
            vix_df = pd.read_csv(vix_path)
            
            # Process VIX data - handle concatenated header if needed
            if 'DateTimeOpenHighLowClose' in vix_df.columns:
                print("Processing VIX data with concatenated header...")
                # Extract column names
                vix_columns = ['DateTime', 'Open', 'High', 'Low', 'Close']
                
                # Create a new DataFrame with proper columns
                new_vix_df = pd.DataFrame(columns=vix_columns)
                
                # Process each row
                for i, row in vix_df.iterrows():
                    # First row might be the header
                    if i == 0 and not row[0].startswith('2'):  # Skip if it's a header
                        continue
                    
                    # Split the concatenated data
                    values = row[0].split()
                    if len(values) >= 5:  # Ensure we have enough values
                        date_str = values[0]
                        time_str = values[1]
                        datetime_str = f"{date_str} {time_str}"
                        
                        # Add to new DataFrame
                        new_vix_df.loc[len(new_vix_df)] = [
                            datetime_str,
                            float(values[2]),
                            float(values[3]),
                            float(values[4]),
                            float(values[5]) if len(values) > 5 else float(values[4])
                        ]
                
                vix_df = new_vix_df
            
            # Ensure DateTime is parsed correctly
            vix_df['DateTime'] = pd.to_datetime(vix_df['DateTime'])
            
            # Extract date from DateTime
            vix_df['Date'] = vix_df['DateTime'].dt.date
            
            # Get daily opening VIX value (first value of each day)
            vix_daily = vix_df.groupby('Date')['Open'].first().reset_index()
            vix_daily['Date'] = pd.to_datetime(vix_daily['Date'])
            
            # Print some sample VIX data for debugging
            print("\nSample VIX data (first 5 days):")
            print(vix_daily.head())
            
            # Use simplified VIX thresholds as requested
            print(f"\nUsing simplified VIX thresholds for position sizing:")
            print(f"Lower threshold: 15")
            print(f"Middle threshold: 20")
            print(f"Upper threshold: 30")
            
            # Create a rule-based position sizing with 4 levels based on VIX thresholds
            # VIX > 30: 50% position (very high volatility)
            # VIX between 20 and 30: 100% position (high volatility)
            # VIX between 15 and 20: 200% position (moderate volatility)
            # VIX < 15: 400% position (low volatility)
            
            # First set all to default 100% (between middle and upper threshold)
            vix_daily['position_factor'] = 1.0
            
            # Then apply the rules in order
            vix_daily.loc[vix_daily['Open'] > 30, 'position_factor'] = 0.5  # 50% when VIX is very high
            vix_daily.loc[(vix_daily['Open'] <= 20) & (vix_daily['Open'] > 15), 'position_factor'] = 2.0  # 200% when VIX is moderate
            vix_daily.loc[vix_daily['Open'] <= 15, 'position_factor'] = 4.0  # 400% when VIX is low
            
            # Print some statistics about VIX levels and position factors
            vix_counts = {
                'VIX > 30 (50% position)': len(vix_daily[vix_daily['Open'] > 30]),
                'VIX 20-30 (100% position)': len(vix_daily[(vix_daily['Open'] <= 30) & (vix_daily['Open'] > 20)]),
                'VIX 15-20 (200% position)': len(vix_daily[(vix_daily['Open'] <= 20) & (vix_daily['Open'] > 15)]),
                'VIX < 15 (400% position)': len(vix_daily[vix_daily['Open'] <= 15])
            }
            
            print("\nVIX level distribution:")
            for label, count in vix_counts.items():
                print(f"  {label}: {count} days ({count/len(vix_daily)*100:.1f}%)")
            
            # Create a date-indexed series for easy lookup
            # Convert index to date (not datetime) for consistent lookup
            position_factors = vix_daily.set_index('Date')['position_factor']
            
            # Print the first few entries in position_factors for debugging
            print("\nFirst 5 entries in position_factors:")
            print(position_factors.head())
            print(f"position_factors index type: {type(position_factors.index[0])}")
            
            print(f"VIX-based position sizing completed. Found {len(vix_daily)} daily VIX records.")
            
        except Exception as e:
            print(f"Error loading or processing VIX data: {e}")
            print("Falling back to fixed position sizing (100%)...")
            use_dynamic_leverage = False
    
    # Filter data by date range if specified
    if start_date is not None:
        spy_df = spy_df[spy_df['Date'] >= start_date]
        print(f"Filtered data to start from {start_date}")
    
    if end_date is not None:
        spy_df = spy_df[spy_df['Date'] <= end_date]
        print(f"Filtered data to end at {end_date}")
        
    # Calculate daily returns for volatility-based position sizing
    if use_volatility_sizing:
        print("\nUsing volatility-based position sizing with target volatility of 2%")
        # Calculate daily returns based on close-to-close
        spy_daily = spy_df.groupby('Date')['Close'].last().reset_index()
        spy_daily['prev_close'] = spy_daily['Close'].shift(1)
        spy_daily['daily_return'] = spy_daily['Close'] / spy_daily['prev_close'] - 1
        
        # Calculate rolling 14-day volatility
        spy_daily['rolling_mean'] = spy_daily['daily_return'].rolling(window=14).mean()
        spy_daily['rolling_std'] = spy_daily['daily_return'].rolling(window=14).std()
        
        # Calculate position factor based on target volatility / actual volatility
        # Formula: min(4, σtarget/σSPY,t)
        spy_daily['position_factor'] = volatility_target / (spy_daily['rolling_std'] * np.sqrt(252))
        spy_daily['position_factor'] = spy_daily['position_factor'].clip(upper=4.0)  # Cap at 4x leverage
        
        # Fill NaN values with 1.0 (default position size)
        spy_daily['position_factor'] = spy_daily['position_factor'].fillna(1.0)
        
        # Create a date-indexed series for easy lookup
        position_factors = spy_daily.set_index('Date')['position_factor']
        
        # Print some statistics about position factors
        print(f"\nVolatility-based position sizing statistics:")
        print(f"Mean position factor: {position_factors.mean():.2f}x")
        print(f"Median position factor: {position_factors.median():.2f}x")
        print(f"Min position factor: {position_factors.min():.2f}x")
        print(f"Max position factor: {position_factors.max():.2f}x")
        
        # Print distribution of position factors
        bins = [0, 1, 2, 3, 4]
        bin_labels = ['0-1x', '1-2x', '2-3x', '3-4x']
        position_factor_counts = pd.cut(position_factors, bins=bins, labels=bin_labels).value_counts()
        print("\nDistribution of position factors:")
        for label, count in position_factor_counts.items():
            print(f"  {label}: {count} days ({count/len(position_factors)*100:.1f}%)")
        
        # If both use_dynamic_leverage and use_volatility_sizing are enabled, use_volatility_sizing takes precedence
        if use_dynamic_leverage:
            print("\nNote: Both VIX-based and volatility-based position sizing are enabled.")
            print("Volatility-based position sizing will be used.")
            use_dynamic_leverage = False
    
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
    
    # Generate allowed trading times based on the check interval
    allowed_times = []
    start_hour, start_minute = 10, 0  # Start at 10:00
    end_hour, end_minute = 15, 40     # End at 15:40 (new closing time)
    
    current_hour, current_minute = start_hour, start_minute
    while current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute):
        # Add current time to allowed_times
        allowed_times.append(f"{current_hour:02d}:{current_minute:02d}")
        
        # Increment by check_interval_minutes
        current_minute += check_interval_minutes
        if current_minute >= 60:
            current_hour += current_minute // 60
            current_minute = current_minute % 60
    
    # Always ensure 15:40 is included for position closing
    if '15:40' not in allowed_times:
        allowed_times.append('15:40')
        allowed_times.sort()
    
    print(f"Using check interval of {check_interval_minutes} minutes")
    print(f"Allowed trading times: {allowed_times}")
    
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
        
        # Check if this is a debug day
        debug = debug_days is not None and trade_date in debug_days
        
        # Convert trade_date to string format for consistent display
        date_str = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        
        # Calculate position size with dynamic position sizing if enabled
        if use_dynamic_leverage or use_volatility_sizing:
            # Debug output to check if the date exists in position_factors
            if i < 5 or i % 100 == 0:  # Print for first 5 days and then every 100 days
                print(f"DEBUG: Looking up position factor for date {date_str}")
                # Convert position_factors index to strings for comparison
                str_index = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in position_factors.index[:5]]
                if date_str in str_index:
                    idx_pos = str_index.index(date_str)
                    print(f"DEBUG: Found position factor: {position_factors.iloc[idx_pos]:.2f}x")
                else:
                    print(f"DEBUG: Date not found in position_factors index")
                    print(f"DEBUG: First 5 dates in position_factors: {str_index}")
            
            # Try to find the position factor using string comparison
            position_factor = 1.0  # Default
            for idx, idx_date in enumerate(position_factors.index):
                if hasattr(idx_date, 'strftime') and idx_date.strftime('%Y-%m-%d') == date_str:
                    position_factor = position_factors.iloc[idx]
                    break
            
            # Calculate position size with position factor
            if use_volatility_sizing:
                # For volatility-based sizing, allow up to 4x leverage
                position_size = floor(capital * position_factor * 4 / open_price)
                actual_leverage = position_factor * 4
            else:
                # For VIX-based sizing, use the position factor directly (already includes desired leverage)
                position_size = floor(capital * position_factor / open_price)
                actual_leverage = position_factor
            
            # Always print position factor for debugging
            sizing_method = "volatility-based" if use_volatility_sizing else "VIX-based"
            print(f"  Date: {date_str}, Position factor: {position_factor:.2f}x, Actual leverage: {actual_leverage:.2f}x ({sizing_method})")
        else:
            # Calculate position size (fixed, with 1x leverage)
            position_size = floor(capital / open_price)
            print(f"  Date: {date_str}, Using fixed 1x leverage")
        
        # Skip days with insufficient capital
        if position_size <= 0:
            print(f"{trade_date}: Insufficient capital, skipping")
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
        
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
            
            # Calculate position percentage relative to capital
            position_percentage = (position_size * open_price / capital_start) * 100
            leverage_text = f"{position_percentage:.0f}%" if position_percentage <= 100 else f"{position_percentage/100:.1f}x"
            
            trade_detail = f"{entry_time}-{exit_time} {side} {entry_price:.2f}->{exit_price:.2f} ${pnl:.2f} ({exit_reason}, 仓位: {leverage_text})"
            trade_details.append(trade_detail)
        
        trade_details_str = ', '.join(trade_details) if trade_details else "No trades"
        
        # Add position factor information if using dynamic leverage
        if use_dynamic_leverage and trades:
            position_info = f", 仓位系数: {position_factor:.2f}x"
        else:
            position_info = ""
            
        print(f"{trade_date}: {trade_details_str}, P&L=${day_pnl:.2f}, Return={daily_return*100:.2f}%{position_info}")
    
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
    
    # 准备SPY数据用于计算买入持有回报率
    # 创建每日SPY收盘价数据
    spy_daily_data = spy_df.groupby('Date')['Close'].last().reset_index()
    
    # 计算策略性能指标
    print("\n计算策略性能指标...")
    metrics = calculate_performance_metrics(daily_df, trades_df, initial_capital, spy_data=spy_daily_data)
    
    # 打印简化的性能指标
    print("\n策略性能指标:")
    if use_dynamic_leverage:
        print(f"Strategy: Momentum Curr.Band + VWAP with VIX-based Position Sizing")
        # Calculate average position factor
        if 'position_factors' in locals():
            avg_position = position_factors.mean()
            max_position = position_factors.max()
            min_position = position_factors.min()
            print(f"Average Position: {avg_position:.2f}x, Max Position: {max_position:.2f}x, Min Position: {min_position:.2f}x")
    else:
        print(f"Strategy: Momentum Curr.Band + VWAP")
        print(f"Size: 100%")
    
    print(f"Total Return: {metrics['total_return']*100:.1f}%")
    print(f"IRR: {metrics['irr']*100:.1f}%")
    print(f"Vol: {metrics['volatility']*100:.1f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Hit Ratio: {metrics['hit_ratio']*100:.1f}%")
    print(f"MDD: {metrics['mdd']*100:.1f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Avg Daily Trades: {metrics['avg_daily_trades']:.2f}")
    print(f"Max Daily Loss: ${metrics['max_daily_loss']:.2f}")
    print(f"Max Daily Gain: ${metrics['max_daily_gain']:.2f}")
    
    # 打印SPY买入持有的对比
    print(f"\nSPY Buy & Hold Return: {metrics['spy_buy_hold_return']*100:.1f}%")
    print(f"Strategy vs SPY: {metrics['relative_performance']*100:.1f}%")
    
    return daily_df, monthly, trades_df, metrics

def calculate_performance_metrics(daily_df, trades_df, initial_capital, 
                                 risk_free_rate=0.02, spy_return=0.072, 
                                 spy_volatility=0.202, correlation=-0.03,
                                 trading_days_per_year=252, **kwargs):
    """
    计算策略的性能指标
    
    参数:
        daily_df: 包含每日回测结果的DataFrame
        trades_df: 包含所有交易的DataFrame
        initial_capital: 初始资金
        risk_free_rate: 无风险利率，默认为2%
        spy_return: SPY基准年化收益率，默认为7.2%
        spy_volatility: SPY基准年化波动率，默认为20.2%
        correlation: 策略与市场的相关性，默认为-0.03
        trading_days_per_year: 一年的交易日数量，默认为252
        
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
            'hit_ratio': 0, 'mdd': 0, 'alpha': 0, 'beta': 0
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
    
    # 7 & 8. Alpha和Beta
    # 这需要市场基准数据，这里我们使用提供的SPY参数
    
    # 计算Beta
    metrics['beta'] = correlation * (metrics['volatility'] / spy_volatility)
    
    # 计算Alpha (相对于SPY买入持有策略)
    # 计算SPY买入持有的回报率
    years = (end_date - start_date).days / 365.25
    if years > 0:
        spy_total_return = (1 + spy_return) ** years - 1
        spy_irr = spy_total_return  # 对于买入持有策略，总回报率等于年化收益率
    else:
        spy_total_return = 0
        spy_irr = 0
    
    # Alpha是策略年化收益率与SPY买入持有年化收益率的差值
    metrics['alpha'] = metrics['irr'] - spy_irr
    
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
        
    # 计算SPY买入持有策略的表现
    # 获取回测的开始和结束日期
    start_date = daily_df.index[0]
    end_date = daily_df.index[-1]
    
    # 使用实际SPY数据计算买入持有回报率
    # 注意：这个参数需要从run_backtest函数传入
    if 'spy_data' in kwargs and not kwargs['spy_data'].empty:
        spy_data = kwargs['spy_data']
        # 获取回测期间的第一个和最后一个交易日的收盘价
        spy_start_price = spy_data[spy_data['Date'] == start_date.date()]['Close'].iloc[-1] if not spy_data[spy_data['Date'] == start_date.date()].empty else None
        spy_end_price = spy_data[spy_data['Date'] == end_date.date()]['Close'].iloc[-1] if not spy_data[spy_data['Date'] == end_date.date()].empty else None
        
        if spy_start_price is not None and spy_end_price is not None:
            spy_total_return = spy_end_price / spy_start_price - 1
        else:
            # 如果找不到精确的日期，使用最接近的日期
            spy_data_sorted = spy_data.sort_values('Date')
            spy_data_filtered = spy_data_sorted[(spy_data_sorted['Date'] >= start_date.date()) & 
                                               (spy_data_sorted['Date'] <= end_date.date())]
            
            if not spy_data_filtered.empty:
                spy_start_price = spy_data_filtered.iloc[0]['Close']
                spy_end_price = spy_data_filtered.iloc[-1]['Close']
                spy_total_return = spy_end_price / spy_start_price - 1
            else:
                # 如果仍然找不到数据，使用默认的年化回报率
                years = (end_date - start_date).days / 365.25
                if years > 0:
                    spy_total_return = (1 + spy_return) ** years - 1
                else:
                    spy_total_return = 0
    else:
        # 如果没有提供SPY数据，使用默认的年化回报率
        years = (end_date - start_date).days / 365.25
        if years > 0:
            spy_total_return = (1 + spy_return) ** years - 1
        else:
            spy_total_return = 0
    
    # 添加SPY买入持有的指标
    metrics['spy_buy_hold_return'] = spy_total_return
    
    # 计算超额收益（策略回报率 - SPY回报率）
    metrics['relative_performance'] = metrics['total_return'] - spy_total_return
    
    return metrics

def plot_specific_days(data_path, dates_to_plot, lookback_days=90, plots_dir='trading_plots', 
                      use_dynamic_leverage=True, use_volatility_sizing=False, volatility_target=0.02,
                      check_interval_minutes=30):
    """
    为指定的日期生成交易图表
    
    参数:
        data_path: SPY分钟数据CSV文件的路径
        dates_to_plot: 要绘制的日期列表 (datetime.date 对象列表)
        lookback_days: 用于计算Noise Area的天数
        plots_dir: 保存图表的目录
        use_dynamic_leverage: 是否使用VIX动态杠杆
        use_volatility_sizing: 是否使用波动率动态杠杆
        volatility_target: 目标日波动率
        check_interval_minutes: 交易检查间隔（分钟）
    """
    # 运行回测，指定要绘制的日期
    _, _, _, _ = run_backtest(
        data_path=data_path,
        lookback_days=lookback_days,
        plot_days=dates_to_plot,
        plots_dir=plots_dir,
        use_dynamic_leverage=use_dynamic_leverage,
        use_volatility_sizing=use_volatility_sizing,
        volatility_target=volatility_target,
        check_interval_minutes=check_interval_minutes
    )
    
    print(f"\n已为以下日期生成图表:")
    for d in dates_to_plot:
        print(f"- {d}")
    print(f"图表保存在 '{plots_dir}' 目录中")

# 示例用法
if __name__ == "__main__":
    # 运行回测，随机生成5个交易日的图表
    daily_results, monthly_results, trades, metrics = run_backtest(
        'spy_market_hours.csv', 
        initial_capital=100000, 
        lookback_days=14,  
        # start_date=date(2023, 1, 1), 
        # end_date=date(2025, 1, 1),
        start_date=date(2010, 6, 25), 
        end_date=date(2025, 1, 8),
        random_plots=5,  # 随机生成5个交易日的图表
        # plot_days=[date(2022, 1, 20), date(2022, 1, 31), date(2022, 4, 29)],  # 指定要绘制的日期
        plots_dir='trading_plots',  # 图表保存目录
        use_dynamic_leverage=False,  # 使用VIX动态杠杆
        use_volatility_sizing=False,  # 使用波动率动态杠杆
        volatility_target=0.02,  # 目标日波动率为2%
        check_interval_minutes=30  # 交易检查间隔（分钟）
    )
    
    # 如果想使用VIX动态杠杆，可以使用以下参数：
    # use_dynamic_leverage=True, use_volatility_sizing=False
    
    # 如果想使用固定杠杆，可以使用以下参数：
    # use_dynamic_leverage=False, use_volatility_sizing=False
