import pandas as pd
import numpy as np
from math import floor, sqrt
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta, date
import random
import os
from plot_trading_day import plot_trading_day
from calculate_indicators import calculate_macd

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

def simulate_day(day_df, prev_close, allowed_times, position_size, transaction_fee_per_share=0.01, trading_end_time=(15, 50), max_positions_per_day=float('inf'), use_macd=True, print_details=False):
    """
    Simulate trading for a single day using curr.band + VWAP strategy
    
    Parameters:
        day_df: DataFrame with intraday data
        prev_close: Previous day's closing price
        allowed_times: List of times when trading is allowed
        position_size: Position size for trades
        transaction_fee_per_share: Fee per share for each transaction
        max_positions_per_day: Maximum number of positions allowed to open per day (default: infinity)
        use_macd: Whether to use MACD histogram as an additional condition for entry (default: True)
        print_details: Whether to print boundary calculation details for trades (default: False)
    """
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = np.nan
    trailing_stop = np.nan
    trade_entry_time = None
    trades = []
    positions_opened_today = 0  # Counter for positions opened today
    
    # Calculate VWAP incrementally during simulation
    prices = []
    volumes = []
    vwaps = []
    
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
        if position == 0 and current_time in allowed_times and positions_opened_today < max_positions_per_day:
            # Get MACD histogram value if available
            macd_histogram = row.get('MACD_histogram', 0)
            
            # Check for potential long entry
            long_macd_condition = macd_histogram > 0 if use_macd else True
            if price > upper and price > vwap and long_macd_condition:
                # Print boundary calculation details if requested
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
                    if use_macd:
                        print(f"    - MACD直方图: {macd_histogram:.6f} (>0)")
                
                # Long entry allowed
                position = 1
                entry_price = price
                trade_entry_time = row['DateTime']
                positions_opened_today += 1  # Increment positions counter
                # Trailing stop: max(UpperBound, VWAP)
                trailing_stop = max(upper, vwap)
                    
            # Check for potential short entry
            short_macd_condition = macd_histogram < 0 if use_macd else True
            if price < lower and price < vwap and short_macd_condition:
                # Print boundary calculation details if requested
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
                    if use_macd:
                        print(f"    - MACD直方图: {macd_histogram:.6f} (<0)")
                
                # Short entry allowed
                position = -1
                entry_price = price
                trade_entry_time = row['DateTime']
                positions_opened_today += 1  # Increment positions counter
                # Trailing stop: min(LowerBound, VWAP)
                trailing_stop = min(lower, vwap)
        
        # Update trailing stop and check for exit signals
        if position != 0:
            if position == 1:  # Long position
                # Calculate stop levels
                # OR relationship: exit if price < max(upper, vwap)
                new_stop = max(upper, vwap)
                # Only update in favorable direction (raise the stop)
                trailing_stop = max(trailing_stop, new_stop)
                
                # Exit if price crosses below the trailing stop
                exit_condition = price < trailing_stop
                
                # Check for exit
                if exit_condition and current_time in allowed_times:
                    # Print exit details if requested
                    if print_details:
                        date_str = row['DateTime'].strftime('%Y-%m-%d')
                        print(f"\n交易点位详情 [{date_str} {current_time}] - 多头出场:")
                        print(f"  价格: {price:.2f} < 追踪止损: {trailing_stop:.2f}")
                        print(f"  止损计算: max(上边界={upper:.2f}, VWAP={vwap:.2f}) = {new_stop:.2f}")
                    
                    # Exit long position
                    exit_time = row['DateTime']
                    # Calculate transaction fees (entry and exit)
                    transaction_fees = position_size * transaction_fee_per_share * 2  # Buy and sell fees
                    pnl = position_size * (price - entry_price) - transaction_fees
                    
                    exit_reason = 'Stop Loss'
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
                    
            elif position == -1:  # Short position
                # Calculate stop levels
                # OR relationship: exit if price > min(lower, vwap)
                new_stop = min(lower, vwap)
                # Only update in favorable direction (lower the stop)
                trailing_stop = min(trailing_stop, new_stop)
                
                # Exit if price crosses above the trailing stop
                exit_condition = price > trailing_stop
                
                # Check for exit
                if exit_condition and current_time in allowed_times:
                    # Print exit details if requested
                    if print_details:
                        date_str = row['DateTime'].strftime('%Y-%m-%d')
                        print(f"\n交易点位详情 [{date_str} {current_time}] - 空头出场:")
                        print(f"  价格: {price:.2f} > 追踪止损: {trailing_stop:.2f}")
                        print(f"  止损计算: min(下边界={lower:.2f}, VWAP={vwap:.2f}) = {new_stop:.2f}")
                    
                    # Exit short position
                    exit_time = row['DateTime']
                    # Calculate transaction fees (entry and exit)
                    transaction_fees = position_size * transaction_fee_per_share * 2  # Buy and sell fees
                    pnl = position_size * (entry_price - price) - transaction_fees
                    
                    exit_reason = 'Stop Loss'
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
    
    # Get the end time string in HH:MM format
    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    
    # Find the end time data point if it exists
    close_time_rows = day_df[day_df['Time'] == end_time_str]
    
    # If we have an end time data point and still have an open position, close it
    if not close_time_rows.empty and position != 0:
        close_row = close_time_rows.iloc[0]
        exit_time = close_row['DateTime']
        close_price = close_row['Close']
        
        if position == 1:  # Long position
            # Print exit details if requested
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {end_time_str}] - 多头收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {close_price:.2f}")
            
            # Calculate transaction fees (entry and exit)
            transaction_fees = position_size * transaction_fee_per_share * 2  # Buy and sell fees
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
                
        else:  # Short position
            # Print exit details if requested
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {end_time_str}] - 空头收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {close_price:.2f}")
            
            # Calculate transaction fees (entry and exit)
            transaction_fees = position_size * transaction_fee_per_share * 2  # Buy and sell fees
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
    
    # If we still have an open position at the end of the day (no end time data point), close it
    elif position != 0:
        exit_time = day_df.iloc[-1]['DateTime']
        last_price = day_df.iloc[-1]['Close']
        last_time = day_df.iloc[-1]['Time']
        
        if position == 1:  # Long position
            # Print exit details if requested
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {last_time}] - 多头市场收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {last_price:.2f}")
            
            # Calculate transaction fees (entry and exit)
            transaction_fees = position_size * transaction_fee_per_share * 2  # Buy and sell fees
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
                
        else:  # Short position
            # Print exit details if requested
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {last_time}] - 空头市场收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {last_price:.2f}")
            
            # Calculate transaction fees (entry and exit)
            transaction_fees = position_size * transaction_fee_per_share * 2  # Buy and sell fees
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
                vix_path='vix_all.csv', use_dynamic_leverage=True, use_volatility_sizing=False,
                volatility_target=0.02, check_interval_minutes=30, 
                transaction_fee_per_share=0.01,
                trading_start_time=(10, 00), trading_end_time=(15, 40), max_positions_per_day=float('inf'),
                use_macd=True, print_daily_trades=True, print_trade_details=False):
    """
    Run the backtest on any 1-minute k-line data
    
    Parameters:
        data_path: Path to the minute data CSV file
        ticker: Ticker symbol for the data (if None, will be extracted from the file name)
        initial_capital: Initial capital for the backtest
        lookback_days: Number of days to use for calculating the Noise Area
        start_date: Start date for the backtest (datetime.date)
        end_date: End date for the backtest (datetime.date)
        plot_days: List of specific dates to plot
        random_plots: Number of random trading days to plot (0 for none)
        plots_dir: Directory to save plots in
        vix_path: Path to the VIX data CSV file
        use_dynamic_leverage: Whether to use dynamic leverage based on VIX levels
        use_volatility_sizing: Whether to use volatility-based position sizing
        volatility_target: Target daily volatility for position sizing
        check_interval_minutes: Interval in minutes between trading checks (default: 30)
        transaction_fee_per_share: Fee per share for each transaction
        max_positions_per_day: Maximum number of positions allowed to open per day (default: infinity)
        use_macd: Whether to use MACD as an entry condition (default: True)
        print_daily_trades: Whether to print details of each day's trades (default: True)
        print_trade_details: Whether to print detailed boundary calculation for each trade (default: False)
        
    Returns:
        DataFrame with daily results
        DataFrame with monthly results
        DataFrame with trades
        Dictionary with performance metrics
    """
    # Determine ticker from file name if not provided
    if ticker is None:
        # Extract ticker from file name
        file_name = os.path.basename(data_path)
        # Remove _market_hours.csv if present
        ticker = file_name.replace('_market_hours.csv', '')
    
    # Load and process data
    print(f"Loading {ticker} data from {data_path}...")
    price_df = pd.read_csv(data_path, parse_dates=['DateTime'])
    price_df.sort_values('DateTime', inplace=True)
    
    # Extract date and time components
    price_df['Date'] = price_df['DateTime'].dt.date
    price_df['Time'] = price_df['DateTime'].dt.strftime('%H:%M')
    
    # Check if MACD indicators are present, calculate them if not
    if 'MACD_histogram' not in price_df.columns:
        print("MACD indicators not found in data. Calculating MACD...")
        
        # Calculate MACD for each day separately
        grouped = price_df.groupby(price_df['Date'])
        result_dfs = []
        
        for date, group in grouped:
            # Calculate MACD for this day
            group_with_macd = calculate_macd(group)
            result_dfs.append(group_with_macd)
        
        # Combine all days back together
        price_df = pd.concat(result_dfs)
        print("MACD calculation completed.")
    
    # Load and process VIX data for position sizing
    if use_dynamic_leverage:
        print(f"Loading VIX data for {ticker} dynamic position sizing...")
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
                        # Remove leading digit if present (e.g., "02008-01-02" -> "2008-01-02")
                        date_str = values[0]
                        if date_str[0].isdigit() and date_str[1].isdigit():
                            date_str = date_str[1:]
                        time_str = values[1]
                        datetime_str = f"{date_str} {time_str}"
                        print(f"DEBUG: Parsed VIX date: {date_str}, time: {time_str}")
                        
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
            
            # Create a rule-based position sizing with 5 levels based on VIX thresholds
            # VIX > 30: 12.5% position (extreme volatility)
            # VIX between 25 and 30: 25% position (very high volatility)
            # VIX between 20 and 25: 50% position (high volatility)
            # VIX between 15 and 20: 100% position (moderate volatility)
            # VIX < 15: 200% position (low volatility)
            
            # First set all to default 50% (between 20 and 25 threshold)
            vix_daily['position_factor'] = 0.5
            
            # Then apply the rules in order
            vix_daily.loc[vix_daily['Open'] > 30, 'position_factor'] = 0.5  # 12.5% when VIX is extreme
            vix_daily.loc[(vix_daily['Open'] <= 30) & (vix_daily['Open'] > 20), 'position_factor'] = 1.0  # 100% when VIX is moderate
            vix_daily.loc[vix_daily['Open'] <= 20, 'position_factor'] = 2.0  # 200% when VIX is low
            
            # # 统计VIX在各个区间的分布比例
            # total_days = len(vix_daily)
            # vix_extreme = len(vix_daily[vix_daily['Open'] > 30])
            # vix_high = len(vix_daily[(vix_daily['Open'] <= 30) & (vix_daily['Open'] > 20)])
            # vix_low = len(vix_daily[vix_daily['Open'] <= 20])
            
            # print("\nVIX区间分布统计:")
            # print(f"VIX > 30 (极端波动): {vix_extreme}天 ({vix_extreme/total_days*100:.2f}%)")
            # print(f"20 < VIX <= 30 (高波动): {vix_high}天 ({vix_high/total_days*100:.2f}%)")
            # print(f"VIX <= 20 (低波动): {vix_low}天 ({vix_low/total_days*100:.2f}%)")
            # print(f"总计: {total_days}天")
            
            # Create a date-indexed series for easy lookup
            # Convert index to date (not datetime) for consistent lookup
            position_factors = vix_daily.set_index('Date')['position_factor']
            
            print(f"VIX-based position sizing completed. Found {len(vix_daily)} daily VIX records.")
            
        except Exception as e:
            print(f"Error loading or processing VIX data: {e}")
            print("Falling back to fixed position sizing (100%)...")
            use_dynamic_leverage = False
    
    # Filter data by date range if specified
    if start_date is not None:
        price_df = price_df[price_df['Date'] >= start_date]
        print(f"Filtered data to start from {start_date}")
    
    if end_date is not None:
        price_df = price_df[price_df['Date'] <= end_date]
        print(f"Filtered data to end at {end_date}")
        
    # Calculate daily returns for volatility-based position sizing
    if use_volatility_sizing:
        print(f"\nUsing volatility-based position sizing with target volatility of {volatility_target*100}%")
        # Calculate daily returns based on close-to-close
        daily_prices = price_df.groupby('Date')['Close'].last().reset_index()
        daily_prices['prev_close'] = daily_prices['Close'].shift(1)
        daily_prices['daily_return'] = daily_prices['Close'] / daily_prices['prev_close'] - 1
        
        # Calculate rolling 14-day volatility
        daily_prices['rolling_mean'] = daily_prices['daily_return'].rolling(window=14).mean()
        daily_prices['rolling_std'] = daily_prices['daily_return'].rolling(window=14).std()
        
        # Calculate position factor based on target volatility / actual volatility
        # Formula: min(2, σtarget/σ_ticker,t)
        daily_prices['position_factor'] = volatility_target / (daily_prices['rolling_std'] * np.sqrt(252))
        daily_prices['position_factor'] = daily_prices['position_factor'].clip(upper=2.0)  # Cap at 2x leverage
        
        # Fill NaN values with 1.0 (default position size)
        daily_prices['position_factor'] = daily_prices['position_factor'].fillna(1.0)
        
        # Create a date-indexed series for easy lookup
        position_factors = daily_prices.set_index('Date')['position_factor']
        
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
    
    # Check if DayOpen and DayClose columns exist, create them if they don't
    if 'DayOpen' not in price_df.columns or 'DayClose' not in price_df.columns:
        print(f"Creating DayOpen and DayClose columns for {ticker} data...")
        # For each day, get the first row (9:30 AM opening price)
        opening_prices = price_df.groupby('Date').first().reset_index()
        opening_prices = opening_prices[['Date', 'Open']].rename(columns={'Open': 'DayOpen'})

        # For each day, get the last row (4:00 PM closing price)
        closing_prices = price_df.groupby('Date').last().reset_index()
        closing_prices = closing_prices[['Date', 'Close']].rename(columns={'Close': 'DayClose'})

        # Merge the opening and closing prices back to the main dataframe
        price_df = pd.merge(price_df, opening_prices, on='Date', how='left')
        price_df = pd.merge(price_df, closing_prices, on='Date', how='left')
        
        print(f"Added DayOpen and DayClose columns to {ticker} data")
    
    # Use DayOpen and DayClose from the filtered data
    # These represent the 9:30 AM open price and 4:00 PM close price
    price_df['prev_close'] = price_df.groupby('Date')['DayClose'].transform('first').shift(1)
    
    # Use the 9:30 AM price as the opening price for the day
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
    
    print("已为每个交易日计算固定的参考价格")
    
    # Calculate return from open for each minute (using day_open for consistency)
    price_df['ret'] = price_df['Close'] / price_df['day_open'] - 1
    
    # Calculate the Noise Area boundaries
    print(f"Calculating Noise Area boundaries for {ticker}...")
    # Pivot to get time-of-day in columns
    pivot = price_df.pivot(index='Date', columns='Time', values='ret').abs()
    # Calculate rolling average of absolute returns for each time-of-day
    # This ensures we're using the previous 14 days for each time point
    sigma = pivot.rolling(window=lookback_days, min_periods=1).mean().shift(1)
    # Convert back to long format
    sigma = sigma.stack().reset_index(name='sigma')
    
    # Merge sigma back to the main dataframe
    price_df = pd.merge(price_df, sigma, on=['Date', 'Time'], how='left')
    
    # 检查每个交易日是否有足够的sigma数据
    # 创建一个标记，记录哪些日期的sigma数据不完整
    incomplete_sigma_dates = set()
    for date in price_df['Date'].unique():
        day_data = price_df[price_df['Date'] == date]
        if day_data['sigma'].isna().any():
            incomplete_sigma_dates.add(date)
            print(f"警告: {date} 的sigma数据不完整，将跳过该交易日")
    
    # 移除sigma数据不完整的日期
    price_df = price_df[~price_df['Date'].isin(incomplete_sigma_dates)]
    
    # 确保所有剩余的sigma值都有有效数据（不应该有NaN值了）
    if price_df['sigma'].isna().any():
        print(f"警告: 仍有{price_df['sigma'].isna().sum()}个缺失的sigma值")
    
    # Calculate upper and lower boundaries of the Noise Area using the correct reference prices
    price_df['UpperBound'] = price_df['upper_ref'] * (1 + price_df['sigma'])
    price_df['LowerBound'] = price_df['lower_ref'] * (1 - price_df['sigma'])
    
    # Generate allowed trading times based on the check interval
    allowed_times = []
    start_hour, start_minute = trading_start_time  # Use configurable start time
    end_hour, end_minute = trading_end_time        # Use configurable end time
    
    current_hour, current_minute = start_hour, start_minute
    while current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute):
        # Add current time to allowed_times
        allowed_times.append(f"{current_hour:02d}:{current_minute:02d}")
        
        # Increment by check_interval_minutes
        current_minute += check_interval_minutes
        if current_minute >= 60:
            current_hour += current_minute // 60
            current_minute = current_minute % 60
    
    # Always ensure trading_end_time is included for position closing
    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    if end_time_str not in allowed_times:
        allowed_times.append(end_time_str)
        allowed_times.sort()
    
    print(f"Using check interval of {check_interval_minutes} minutes")
    print(f"Allowed trading times: {allowed_times}")
    
    # Initialize backtest variables
    capital = initial_capital
    daily_results = []
    all_trades = []
    total_transaction_fees = 0  # Track total transaction fees
    
    # Run the backtest day by day
    print(f"Running {ticker} backtest...")
    unique_dates = sorted(price_df['Date'].unique())
    
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
                                           print_details=print_trade_details)
            
            # Extract trades from the result
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
    
    for i, trade_date in enumerate(unique_dates):
        # Get data for the current day
        day_data = price_df[price_df['Date'] == trade_date].copy()
        day_data = day_data.sort_values('DateTime').reset_index(drop=True)
        
        # Skip days with insufficient data
        if len(day_data) < 10:  # Arbitrary threshold
            print(f"{trade_date}: Insufficient data, skipping")
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
        
        # Get the previous day's close
        prev_close = day_data['prev_close'].iloc[0] if not pd.isna(day_data['prev_close'].iloc[0]) else None
        
        # Get the opening price for the day
        open_price = day_data['day_open'].iloc[0]
        
        # Convert trade_date to string format for consistent display
        date_str = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        
        # Calculate position size with dynamic position sizing if enabled
        if use_dynamic_leverage or use_volatility_sizing:
            # Find position factor without debug output
            
            # Try to find the position factor using string comparison
            position_factor = 1.0  # Default
            for idx, idx_date in enumerate(position_factors.index):
                if hasattr(idx_date, 'strftime') and idx_date.strftime('%Y-%m-%d') == date_str:
                    position_factor = position_factors.iloc[idx]
                    break
            
            # Calculate position size with position factor
            if use_volatility_sizing:
                # For volatility-based sizing, allow up to 2x leverage
                position_size = floor(capital * position_factor * 2 / open_price)
                actual_leverage = position_factor * 2
            else:
                # For VIX-based sizing, use the position factor directly (already includes desired leverage)
                position_size = floor(capital * position_factor / open_price)
                actual_leverage = position_factor
            
            # Calculate position factor and leverage (no logging)
            sizing_method = "volatility-based" if use_volatility_sizing else "VIX-based"
        else:
            # Calculate position size (fixed, with 1x leverage)
            position_size = floor(capital / open_price)
        
        # Skip days with insufficient capital
        if position_size <= 0:
            # Skip day with insufficient capital (no logging)
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
                
        # Simulate trading for the day
        simulation_result = simulate_day(day_data, prev_close, allowed_times, position_size,
                           transaction_fee_per_share=transaction_fee_per_share,
                           trading_end_time=trading_end_time, max_positions_per_day=max_positions_per_day,
                           use_macd=use_macd, print_details=print_trade_details)
        
        # Extract trades from the result
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
        
        # Calculate daily P&L and track transaction fees
        day_pnl = 0
        day_transaction_fees = 0
        for trade in trades:
            day_pnl += trade['pnl']
            # Extract transaction fees from each trade
            if 'transaction_fees' not in trade:
                # Calculate transaction fees if not already in trade data
                trade['transaction_fees'] = position_size * transaction_fee_per_share * 2  # Buy and sell fees
            day_transaction_fees += trade['transaction_fees']
        
        # Add to total transaction fees
        total_transaction_fees += day_transaction_fees
        
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
    
    # 准备数据用于计算买入持有回报率
    # 创建每日收盘价数据
    daily_price_data = price_df.groupby('Date')['Close'].last().reset_index()
    
    # 计算策略性能指标
    print(f"\n计算{ticker}策略性能指标...")
    metrics = calculate_performance_metrics(daily_df, trades_df, initial_capital, spy_data=daily_price_data)
    
    # 打印交易费用统计
    print(f"\n{ticker}交易费用统计:")
    print(f"总交易费用: ${total_transaction_fees:.2f}")
    if len(trades_df) > 0:
        print(f"平均每笔交易费用: ${total_transaction_fees / len(trades_df):.2f}")
    if len(daily_df) > 0:
        print(f"平均每日交易费用: ${total_transaction_fees / len(daily_df):.2f}")
    print(f"交易费用占初始资金比例: {total_transaction_fees / initial_capital * 100:.2f}%")
    print(f"交易费用占总收益比例: {total_transaction_fees / (capital - initial_capital) * 100:.2f}%" if capital > initial_capital else "交易费用占总收益比例: N/A (无盈利)")
    
    # 打印简化的性能指标
    print(f"\n{ticker}策略性能指标:")
    strategy_name = f"{ticker} Momentum Curr.Band + VWAP"
    if use_macd:
        strategy_name += " + MACD"
    
    if use_dynamic_leverage:
        print(f"Strategy: {strategy_name} with VIX-based Position Sizing")
        # Calculate average position factor
        if 'position_factors' in locals():
            avg_position = position_factors.mean()
            max_position = position_factors.max()
            min_position = position_factors.min()
            print(f"Average Position: {avg_position:.2f}x, Max Position: {max_position:.2f}x, Min Position: {min_position:.2f}x")
    else:
        print(f"Strategy: {strategy_name}")
        print(f"Size: 100%")
    
    # 创建一个表格格式来对比策略和买入持有的指标
    print("\n性能指标对比:")
    print(f"{'指标':<20} | {'策略':<15} | {f'{ticker} Buy & Hold':<15}")
    print("-" * 55)
    
    # 总回报率
    print(f"{'Total Return':<20} | {metrics['total_return']*100:>14.1f}% | {metrics['spy_buy_hold_return']*100:>14.1f}%")
    
    # 年化收益率
    if 'spy_irr' in metrics:
        spy_irr = metrics['spy_irr']
    else:
        # 如果没有计算过，使用总回报率计算
        years = (daily_df.index[-1] - daily_df.index[0]).days / 365.25
        if years > 0:
            spy_irr = (1 + metrics['spy_buy_hold_return']) ** (1 / years) - 1
        else:
            spy_irr = 0
    print(f"{'IRR':<20} | {metrics['irr']*100:>14.1f}% | {spy_irr*100:>14.1f}%")
    
    # 波动率
    if 'spy_volatility' in metrics:
        print(f"{'Volatility':<20} | {metrics['volatility']*100:>14.1f}% | {metrics['spy_volatility']*100:>14.1f}%")
    else:
        print(f"{'Volatility':<20} | {metrics['volatility']*100:>14.1f}% | {'N/A':>14}")
    
    # 夏普比率
    if 'spy_sharpe_ratio' in metrics:
        print(f"{'Sharpe Ratio':<20} | {metrics['sharpe_ratio']:>14.2f} | {metrics['spy_sharpe_ratio']:>14.2f}")
    else:
        print(f"{'Sharpe Ratio':<20} | {metrics['sharpe_ratio']:>14.2f} | {'N/A':>14}")
    
    # 最大回撤
    if 'spy_mdd' in metrics:
        print(f"{'Max Drawdown':<20} | {metrics['mdd']*100:>14.1f}% | {metrics['spy_mdd']*100:>14.1f}%")
    else:
        print(f"{'Max Drawdown':<20} | {metrics['mdd']*100:>14.1f}% | {'N/A':>14}")
    
    # Calmar比率
    if 'spy_calmar_ratio' in metrics and 'calmar_ratio' in metrics:
        print(f"{'Calmar Ratio':<20} | {metrics['calmar_ratio']:>14.2f} | {metrics['spy_calmar_ratio']:>14.2f}")
    
    # 胜率
    if 'spy_hit_ratio' in metrics:
        print(f"{'Hit Ratio':<20} | {metrics['hit_ratio']*100:>14.1f}% | {metrics['spy_hit_ratio']*100:>14.1f}%")
    else:
        print(f"{'Hit Ratio':<20} | {metrics['hit_ratio']*100:>14.1f}% | {'N/A':>14}")
    
    # 其他策略特有指标
    print(f"\n策略特有指标:")
    print(f"Hit Ratio: {metrics['hit_ratio']*100:.1f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Avg Daily Trades: {metrics['avg_daily_trades']:.2f}")
    print(f"Max Daily Loss: ${metrics['max_daily_loss']:.2f}")
    print(f"Max Daily Gain: ${metrics['max_daily_gain']:.2f}")
    
    # 打印超额收益
    print(f"\n策略超额收益: {metrics['relative_performance']*100:.1f}%")
    
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
        
    # 计算买入持有策略的表现
    # 获取回测的开始和结束日期
    start_date = daily_df.index[0]
    end_date = daily_df.index[-1]
    
    # 使用实际数据计算买入持有回报率
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
    
    # 添加买入持有的指标
    metrics['spy_buy_hold_return'] = spy_total_return
    
    # 计算超额收益（策略回报率 - 买入持有回报率）
    metrics['relative_performance'] = metrics['total_return'] - spy_total_return
    
    # 计算买入持有策略的其他技术指标
    # 创建买入持有策略的每日回报率序列
    if 'spy_data' in kwargs and not kwargs['spy_data'].empty:
        spy_data = kwargs['spy_data']
        spy_data_sorted = spy_data.sort_values('Date')
        spy_data_filtered = spy_data_sorted[(spy_data_sorted['Date'] >= start_date.date()) & 
                                           (spy_data_sorted['Date'] <= end_date.date())]
        
        if not spy_data_filtered.empty:
            # 计算每日收益率
            spy_data_filtered['prev_close'] = spy_data_filtered['Close'].shift(1)
            spy_data_filtered['daily_return'] = spy_data_filtered['Close'] / spy_data_filtered['prev_close'] - 1
            
            # 计算买入持有策略的波动率
            spy_daily_returns = spy_data_filtered['daily_return'].dropna()
            if len(spy_daily_returns) > 0:
                # 移除异常值
                spy_daily_returns = spy_daily_returns[spy_daily_returns.between(
                    spy_daily_returns.quantile(0.001), spy_daily_returns.quantile(0.999))]
                
                # 计算年化波动率
                metrics['spy_volatility'] = spy_daily_returns.std() * np.sqrt(trading_days_per_year)
                
                # 计算夏普比率
                years = (end_date - start_date).days / 365.25
                if years > 0:
                    spy_irr = (1 + spy_total_return) ** (1 / years) - 1
                    metrics['spy_sharpe_ratio'] = (spy_irr - risk_free_rate) / metrics['spy_volatility'] if metrics['spy_volatility'] > 0 else 0
                else:
                    metrics['spy_sharpe_ratio'] = 0
                
                # 计算最大回撤
                spy_data_filtered['peak'] = spy_data_filtered['Close'].cummax()
                spy_data_filtered['drawdown'] = (spy_data_filtered['Close'] - spy_data_filtered['peak']) / spy_data_filtered['peak']
                metrics['spy_mdd'] = spy_data_filtered['drawdown'].min() * -1
                
                # 计算Calmar比率
                if metrics['spy_mdd'] > 0 and years > 0:
                    metrics['spy_calmar_ratio'] = spy_irr / metrics['spy_mdd']
                else:
                    metrics['spy_calmar_ratio'] = float('inf')  # 如果没有回撤，设为无穷大
                
                # 计算买入持有策略的胜率（正收益天数比例）
                positive_days = (spy_daily_returns > 0).sum()
                total_days = len(spy_daily_returns)
                metrics['spy_hit_ratio'] = positive_days / total_days if total_days > 0 else 0
    
    return metrics

def plot_specific_days(data_path, dates_to_plot, lookback_days=90, plots_dir='trading_plots', 
                      use_dynamic_leverage=True, use_volatility_sizing=False, volatility_target=0.02,
                      check_interval_minutes=30, transaction_fee_per_share=0.01,
                      trading_start_time=(9, 40), trading_end_time=(15, 50), max_positions_per_day=float('inf')):
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
        check_interval_minutes=check_interval_minutes,
        transaction_fee_per_share=transaction_fee_per_share,
        trading_start_time=trading_start_time,
        trading_end_time=trading_end_time,
        max_positions_per_day=max_positions_per_day
    )
    
    print(f"\n已为以下日期生成图表:")
    for d in dates_to_plot:
        print(f"- {d}")
    print(f"图表保存在 '{plots_dir}' 目录中")

# 示例用法
if __name__ == "__main__":  
    # 运行回测
    daily_results, monthly_results, trades, metrics = run_backtest(
        'tqqq_market_hours_with_indicators.csv',  # 使用带有MACD指标的TQQQ数据
        # 'tqqq_longport.csv',  # 使用带有MACD指标的TQQQ数据
        ticker='TQQQ',                     # 指定ticker
        # 'qqq_market_hours_with_indicators.csv',  # 使用带有MACD指标的TQQQ数据
        # ticker='QQQ',                     # 指定ticker 
        initial_capital=10000, 
        lookback_days=10,
        start_date=date(2020, 1, 20), 
        end_date=date(2025, 1, 20),
        use_dynamic_leverage=False,
        check_interval_minutes=10,
        transaction_fee_per_share=0.005,  # 每股交易费用
        # 交易时间配置
        trading_start_time=(9, 40),  # 交易开始时间
        trading_end_time=(15, 40),      # 交易结束时间
        max_positions_per_day=3,  # 每天最多开仓3次
        use_macd=False,  # 使用MACD作为入场条件，设为False可以禁用MACD条件
        # random_plots=3,  # 随机选择3天生成图表
        # plots_dir='trading_plots',  # 图表保存目录
        print_daily_trades=False,  # 是否打印每日交易详情
        print_trade_details=False  # 是否打印交易细节
    )
