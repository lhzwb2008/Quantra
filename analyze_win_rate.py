import pandas as pd
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta, date

# Copy the helper functions from the notebook
def calculate_vwap_incrementally(prices, volumes):
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

def run_backtest(data_path, initial_capital=100000, lookback_days=14, start_date=None, end_date=None, debug_days=None):
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
    
    # Get previous day's close for each day
    spy_df['prev_close'] = spy_df.groupby('Date')['Close'].transform('last').shift(1)
    
    # Calculate daily open price (first price of the day)
    spy_df['day_open'] = spy_df.groupby('Date')['Open'].transform('first')
    
    # Calculate Open_day as per the paper: max/min of yesterday's close and today's open
    # For upper bound: max(Open, prev_close)
    # For lower bound: min(Open, prev_close)
    spy_df['upper_ref'] = spy_df.apply(lambda row: max(row['day_open'], row['prev_close']) 
                                      if not pd.isna(row['prev_close']) else row['day_open'], axis=1)
    spy_df['lower_ref'] = spy_df.apply(lambda row: min(row['day_open'], row['prev_close']) 
                                      if not pd.isna(row['prev_close']) else row['day_open'], axis=1)
    
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
    
    # Fill NaN values in sigma with a reasonable default
    spy_df['sigma'] = spy_df['sigma'].fillna(0.001)  # 0.1% default
    
    # Calculate upper and lower boundaries of the Noise Area using the correct reference prices
    spy_df['UpperBound'] = spy_df['upper_ref'] * (1 + spy_df['sigma'])
    spy_df['LowerBound'] = spy_df['lower_ref'] * (1 - spy_df['sigma'])
    
    # Define allowed trading times (semi-hourly intervals)
    allowed_times = [
        '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30',
        '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00'
    ]
    
    # Initialize backtest variables
    capital = initial_capital
    daily_results = []
    all_trades = []
    
    # Run the backtest day by day
    print("Running backtest...")
    unique_dates = sorted(spy_df['Date'].unique())
    
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

# Main execution
if __name__ == "__main__":
    # Focus on January 20, 2022 for detailed debugging
    debug_date = date(2022, 1, 20)
    
    print(f"Running detailed debug for {debug_date}...")
    debug_results, debug_monthly, debug_trades = run_backtest(
        "spy_all.csv", 
        initial_capital=100000, 
        lookback_days=14, 
        start_date=debug_date, 
        end_date=debug_date,
        debug_days=[debug_date]
    )
    
    # Also run for the full 2022 year
    start_date_full = date(2022, 1, 1)
    end_date_full = date(2022, 12, 31)
    
    print("\nRunning backtest for 2022 to debug win rate...")
    daily_results_full, monthly_results_full, trades_full = run_backtest(
        "spy_all.csv", 
        initial_capital=100000, 
        lookback_days=14, 
        start_date=start_date_full, 
        end_date=end_date_full
    )
    
    # Calculate more detailed trade statistics
    if len(trades_full) > 0:
        # Calculate win rate
        trades_full["is_win"] = trades_full["pnl"] > 0
        win_rate = trades_full["is_win"].mean() * 100
        
        # Calculate average win and loss
        avg_win = trades_full[trades_full["pnl"] > 0]["pnl"].mean()
        avg_loss = trades_full[trades_full["pnl"] < 0]["pnl"].mean()
        
        # Calculate win/loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        
        # Calculate profit factor
        total_wins = trades_full[trades_full["pnl"] > 0]["pnl"].sum()
        total_losses = abs(trades_full[trades_full["pnl"] < 0]["pnl"].sum())
        profit_factor = total_wins / total_losses if total_losses != 0 else float("inf")
        
        # Calculate by side (long vs short)
        long_trades = trades_full[trades_full["side"] == "Long"]
        short_trades = trades_full[trades_full["side"] == "Short"]
        
        long_win_rate = long_trades["is_win"].mean() * 100 if len(long_trades) > 0 else 0
        short_win_rate = short_trades["is_win"].mean() * 100 if len(short_trades) > 0 else 0
        
        # Calculate by exit reason
        stop_loss_trades = trades_full[trades_full["exit_reason"] == "Stop Loss"]
        market_close_trades = trades_full[trades_full["exit_reason"] == "Market Close"]
        
        stop_loss_win_rate = stop_loss_trades["is_win"].mean() * 100 if len(stop_loss_trades) > 0 else 0
        market_close_win_rate = market_close_trades["is_win"].mean() * 100 if len(market_close_trades) > 0 else 0
        
        # Calculate by year
        trades_full["year"] = trades_full["entry_time"].dt.year
        yearly_stats = trades_full.groupby("year").apply(
            lambda x: pd.Series({
                "trades": len(x),
                "win_rate": x["is_win"].mean() * 100,
                "avg_pnl": x["pnl"].mean(),
                "total_pnl": x["pnl"].sum()
            })
        )
        
        # Print detailed statistics
        print("\nDetailed Trade Statistics (Full Period 2007-2024):\n")
        print(f"Total Trades: {len(trades_full)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average P&L per Trade: ${trades_full['pnl'].mean():.2f}")
        
        print("\nBy Side:")
        print(f"Long Trades: {len(long_trades)} (Win Rate: {long_win_rate:.1f}%)")
        print(f"Short Trades: {len(short_trades)} (Win Rate: {short_win_rate:.1f}%)")
        
        print("\nBy Exit Reason:")
        print(f"Stop Loss Exits: {len(stop_loss_trades)} (Win Rate: {stop_loss_win_rate:.1f}%)")
        print(f"Market Close Exits: {len(market_close_trades)} (Win Rate: {market_close_win_rate:.1f}%)")
        
        print("\nYearly Statistics:")
        print(yearly_stats)
        
        # Save results to CSV files
        trades_full.to_csv("trades_full_period.csv")
        daily_results_full.to_csv("daily_results_full_period.csv")
        monthly_results_full.to_csv("monthly_results_full_period.csv")
        yearly_stats.to_csv("yearly_stats.csv")
        
        print("\nResults saved to CSV files.")
