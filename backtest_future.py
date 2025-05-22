import pandas as pd
import numpy as np
from datetime import time, timedelta, datetime

# --- Configuration ---
FILE_PATH = 'ss2401.csv'  # Path to your 1-minute K-line data
LOOKBACK_DAYS = 2  # Lookback period for sigma calculation
K1 = 1.2  # Upper boundary multiplier
K2 = 1.2  # Lower boundary multiplier
CHECK_INTERVAL_MINUTES = 10  # How often to check for opening conditions
SESSION_WARMUP_MINUTES = 10  # Wait time after session start
SESSION_COOLDOWN_MINUTES = 10  # Time before session end to force close
INITIAL_CAPITAL = 100000  # 初始资金（人民币）
# 期货交易相关参数
CONTRACT_MULTIPLIER = 5  # 合约乘数，实际使用时需要根据具体期货品种设置，如螺纹钢为10吨/手
# 保证金比例，简化处理，这里假设为100%
MARGIN_RATIO = 0.08  # 实际交易中通常为5%-15%
# 滑点设置
SLIPPAGE = 2.5  # 单边滑点（元），期权交易最小变动单位是5元，设置为2.5元的滑点

# SHFE Stainless Steel (ss) Trading Sessions
# (start_time_str, end_time_str, is_overnight_end (True if end_time is on next calendar day))
# Standard ss sessions: 21:00-23:00, 09:00-10:15, 10:30-11:30, 13:30-15:00
PREDEFINED_SESSIONS_SPEC = [
    ("21:00:00", "23:00:00", False), # Night session
    ("09:00:00", "10:15:00", False), # Morning session 1
    ("10:30:00", "11:30:00", False), # Morning session 2
    ("13:30:00", "15:00:00", False)  # Afternoon session
]

def generate_dummy_data():
    """Generates sample data for testing if actual CSV is not available/valid."""
    print("Generating dummy data for backtesting structure demonstration.")
    base_time = datetime(2024, 1, 8, 0, 0, 0)
    rng = pd.date_range(start=base_time, periods=5000, freq='1min')
    data = []
    price = 15000
    for ts in rng:
        # Simulate SHFE ss trading hours
        t = ts.time()
        is_trading_hour = False
        if (time(21,0) <= t < time(23,0)) or \
           (time(9,0) <= t < time(10,15)) or \
           (time(10,30) <= t < time(11,30)) or \
           (time(13,30) <= t < time(15,0)):
            is_trading_hour = True

        if is_trading_hour:
            open_p = price + np.random.randint(-5, 6)
            close_p = open_p + np.random.randint(-5, 6)
            high_p = max(open_p, close_p) + np.random.randint(0, 3)
            low_p = min(open_p, close_p) - np.random.randint(0, 3)
            volume = np.random.randint(100, 1000)
            data.append([ts, open_p, high_p, low_p, close_p, volume])
            price = close_p
        
    df = pd.DataFrame(data, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    if df.empty: # Fallback if loop generated no data
        dummy_dates = pd.to_datetime(['2024-01-08 09:00:00', '2024-01-08 09:01:00', '2024-01-08 21:00:00', '2024-01-08 21:01:00',
                                      '2024-01-09 09:00:00', '2024-01-09 09:01:00'])
        dummy_prices = {'Open': [100, 101, 105, 106, 110, 111], 'Close': [101, 100, 106, 105, 111, 110],
                        'High': [102,102,107,107,112,112], 'Low': [99,99,104,104,109,109], 'Volume':[10,10,10,10,10,10]}
        df = pd.DataFrame(dummy_prices, index=dummy_dates)
        df.index.name = 'DateTime'
        df.reset_index(inplace=True)

    df.set_index('DateTime', inplace=True)
    return df

# --- Helper Functions ---
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
        if df.columns[0].startswith('Unnamed:'): # Handles cases like ",DateTime,Open,..."
            df = df.iloc[:, 1:]

        if 'DateTime' not in df.columns:
            raise ValueError("CSV must contain a 'DateTime' column.")
        
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        df.sort_index(inplace=True)
        
        required_cols = ['Open', 'Close'] # Volume is optional for this strategy logic
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain {required_cols} columns. Found: {df.columns.tolist()}")
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return generate_dummy_data()
    except Exception as e:
        print(f"Error loading or processing data from {file_path}: {e}")
        return generate_dummy_data()

def get_all_trading_sessions(unique_dates, sessions_spec):
    all_sessions = []
    for date in unique_dates:
        for start_time_str, end_time_str, is_overnight in sessions_spec:
            start_t = time.fromisoformat(start_time_str)
            end_t = time.fromisoformat(end_time_str)
            
            session_start_dt = datetime.combine(date, start_t)
            
            if is_overnight: # End time is on the next calendar day
                session_end_dt = datetime.combine(date + timedelta(days=1), end_t)
            else: # End time is on the same calendar day
                session_end_dt = datetime.combine(date, end_t)
            
            # Ensure chronological order for sessions spanning across midnight but defined on one day
            if session_end_dt < session_start_dt : # e.g. 21:00 to 01:00 implies end is next day
                 session_end_dt += timedelta(days=1)

            all_sessions.append({'start': session_start_dt, 'end': session_end_dt})
    
    # Sort sessions chronologically
    all_sessions.sort(key=lambda x: x['start'])
    return all_sessions

def precompute_historical_first_opens(df, all_sessions):
    """ Precomputes the first opening price for each historical date. """
    historical_opens = {}
    
    # Find the earliest data point for each unique date
    # A simpler approach: group by date and take first open
    # This assumes 'Open' prices are available at the start of the day's trading data
    
    unique_dates_in_data = sorted(list(df.index.normalize().unique()))

    for date_val in unique_dates_in_data:
        day_data = df[df.index.date == date_val.date()]
        if not day_data.empty:
            historical_opens[date_val.date()] = day_data['Open'].iloc[0]
            
    # Fallback if a date in all_sessions isn't in historical_opens (e.g. holiday, missing data)
    # This might not be strictly necessary if sigma calc handles missing historical_opens gracefully
    
    return historical_opens


def calculate_sigma(current_dt, df_full_history, lookback_days, historical_first_opens_map):
    target_time_of_day = current_dt.time()
    abs_changes = []

    for i in range(1, lookback_days + 1):
        hist_date_obj = (current_dt - timedelta(days=i)).date()
        
        # Construct the historical datetime timestamp
        # Need to handle timezones if df_full_history.index is timezone-aware
        # Assuming naive datetimes for now
        try:
            hist_dt_at_target_time = datetime.combine(hist_date_obj, target_time_of_day)
        except TypeError: # current_dt might be Timestamp, hist_date_obj is date
            hist_dt_at_target_time = pd.Timestamp.combine(hist_date_obj, target_time_of_day)


        if hist_dt_at_target_time in df_full_history.index:
            close_at_target = df_full_history.loc[hist_dt_at_target_time, 'Close']
            daily_open_for_hist_date = historical_first_opens_map.get(hist_date_obj)

            if daily_open_for_hist_date is not None and daily_open_for_hist_date != 0:
                change = abs(close_at_target / daily_open_for_hist_date - 1)
                abs_changes.append(change)
            # else:
            #     print(f"Debug sigma: Missing daily_open_for_hist_date for {hist_date_obj} or it's zero.")
        # else:
        #     print(f"Debug sigma: Missing data for {hist_dt_at_target_time} in df_full_history.")


    if not abs_changes:
        return np.nan  # Not enough data to calculate sigma
    return np.mean(abs_changes)

# --- Main Backtest Logic ---
def run_backtest(df, config):
    trades = []
    position = None  # None, 'LONG', 'SHORT'
    entry_price = 0
    stop_loss_price = 0  # For ratcheting stop
    
    # 用于资金管理的变量
    current_capital = config['INITIAL_CAPITAL']  # 当前资金
    position_size = 0  # 持仓数量（手数，整数）
    current_equity = current_capital  # 当前净值（现金+持仓价值）
    
    # 滑点设置
    slippage = config.get('SLIPPAGE', 0)  # 获取滑点设置，默认为0
    
    # 用于记录每日净值，计算回撤等指标
    equity_curve = {}  # 格式为 {日期: 净值}
    
    # 记录每笔交易前的资金，用于计算实际收益率
    trade_capital_history = []

    unique_dates = sorted(df.index.normalize().unique())
    
    all_defined_sessions = get_all_trading_sessions(unique_dates, PREDEFINED_SESSIONS_SPEC)
    
    historical_first_opens = precompute_historical_first_opens(df, all_defined_sessions)

    global_prev_session_close_price = np.nan
    
    # 遍历每个交易时段前，先记录初始净值
    if unique_dates:
        first_date = unique_dates[0].date() if hasattr(unique_dates[0], 'date') else unique_dates[0]
        equity_curve[first_date] = current_capital

    for session_info in all_defined_sessions:
        session_start_dt = session_info['start']
        session_end_dt = session_info['end']

        # Filter data for the current session
        # Ensure we only select data within the main df's range
        actual_session_start = max(session_start_dt, df.index.min())
        actual_session_end = min(session_end_dt, df.index.max())

        if actual_session_start >= actual_session_end: # Session is outside data range or invalid
            continue
            
        session_df = df.loc[actual_session_start:actual_session_end]

        if session_df.empty:
            # Update global_prev_session_close_price if this "empty" session had a theoretical end within data
            if session_end_dt <= df.index.max() and session_end_dt > (df.index.min() if global_prev_session_close_price is np.nan else pd.Timestamp(global_prev_session_close_price_dt)) :
                 # Attempt to find a close price if the session was supposed to happen
                 potential_closes = df.loc[:session_end_dt]
                 if not potential_closes.empty:
                     global_prev_session_close_price = potential_closes['Close'].iloc[-1]
                     global_prev_session_close_price_dt = potential_closes.index[-1]

            continue

        current_session_open_price = session_df['Open'].iloc[0]

        if np.isnan(global_prev_session_close_price):
            print(f"交易时段 {session_start_dt} - {session_end_dt}: 无上一交易时段收盘价。跳过此时段交易。")
            global_prev_session_close_price = session_df['Close'].iloc[-1]
            global_prev_session_close_price_dt = session_df.index[-1]
            continue
        
        upper_ref = max(current_session_open_price, global_prev_session_close_price)
        lower_ref = min(current_session_open_price, global_prev_session_close_price)

        trade_start_offset = timedelta(minutes=config['SESSION_WARMUP_MINUTES'])
        force_close_offset = timedelta(minutes=config['SESSION_COOLDOWN_MINUTES'])
        
        actual_trade_start_time = session_start_dt + trade_start_offset
        force_close_deadline_time = session_end_dt - force_close_offset
        
        last_check_time = None

        # Iterate through 1-minute bars in the trading window of the session
        tradable_session_df = session_df.loc[actual_trade_start_time : force_close_deadline_time]

        for current_dt, row in tradable_session_df.iterrows():
            current_price = row['Close'] # Assume decisions based on close of the current 1-min bar
            
            # 更新当日净值
            current_date = current_dt.date()
            if current_date not in equity_curve:
                # 计算当前净值
                if position == 'LONG':
                    position_value = position_size * current_price * CONTRACT_MULTIPLIER
                    current_equity = current_capital + position_value  # 多头持仓价值 + 剩余现金
                elif position == 'SHORT':
                    # 空头净值 = 剩余现金 + 未实现盈亏
                    unrealized_pnl = position_size * (entry_price - current_price) * CONTRACT_MULTIPLIER
                    current_equity = current_capital + unrealized_pnl  # 空头持仓的未实现盈亏
                else:
                    current_equity = current_capital
                
                equity_curve[current_date] = current_equity

            # --- Force Close Condition ---
            if position and current_dt >= force_close_deadline_time:
                exit_reason = "强制平仓（交易时段结束）"
                
                exit_capital_before = current_capital
                
                if position == 'LONG':
                    # 多头平仓（考虑滑点）
                    exit_price_with_slippage = current_price - slippage  # 卖出时价格降低
                    pnl_points = exit_price_with_slippage - entry_price
                    pnl_amount = position_size * pnl_points * CONTRACT_MULTIPLIER
                    position_value = position_size * exit_price_with_slippage * CONTRACT_MULTIPLIER
                    current_capital = current_capital + position_value  # 平仓后加上持仓价值
                else:  # SHORT
                    # 空头平仓（考虑滑点）
                    exit_price_with_slippage = current_price + slippage  # 买入时价格提高
                    pnl_points = entry_price - exit_price_with_slippage
                    pnl_amount = position_size * pnl_points * CONTRACT_MULTIPLIER
                    current_capital = current_capital + pnl_amount  # 空头平仓后的资金
                
                # 计算此笔交易的收益率
                trade_return_pct = (pnl_amount / (position_size * entry_price * CONTRACT_MULTIPLIER)) * 100
                
                # 记录交易
                trade_record = {
                    'entry_time': entry_time, 'exit_time': current_dt,
                    'position': position, 'entry_price': entry_price,
                    'exit_price': exit_price_with_slippage,
                    'exit_price_market': current_price,  # 记录市场价格（不含滑点）
                    'pnl_points': pnl_points,
                    'pnl_amount': pnl_amount, 'position_size': position_size,
                    'exit_capital': current_capital, 'exit_reason': exit_reason,
                    'trade_return_pct': trade_return_pct,
                    'slippage': slippage  # 记录滑点
                }
                
                trades.append(trade_record)
                
                # 打印当前交易详情
                print(f"交易: {exit_reason} {position} | 开仓时间: {entry_time} 平仓时间: {current_dt}")
                print(f"  开仓价: {entry_price:.2f} 平仓价: {exit_price_with_slippage:.2f} (市场价: {current_price:.2f}, 滑点: {slippage:.2f}) 手数: {position_size}手")
                print(f"  盈亏: {pnl_amount:.2f}元 ({trade_return_pct:.2f}%) | 资金: {exit_capital_before:.2f} -> {current_capital:.2f}元")
                
                position = None
                position_size = 0
                continue # Done with this bar

            if position == 'LONG':
                sigma = calculate_sigma(current_dt, df, config['LOOKBACK_DAYS'], historical_first_opens)
                if np.isnan(sigma):
                    pass 
                else:
                    current_upper_bound = upper_ref * (1 + config['K1'] * sigma)
                    stop_loss_price = max(stop_loss_price, current_upper_bound) # Ratchet stop
                
                if current_price < stop_loss_price:
                    exit_reason = "止损（多头）"
                    
                    exit_capital_before = current_capital
                    
                    # 多头平仓（考虑滑点）
                    exit_price_with_slippage = current_price - slippage  # 卖出时价格降低
                    pnl_points = exit_price_with_slippage - entry_price
                    pnl_amount = position_size * pnl_points * CONTRACT_MULTIPLIER
                    position_value = position_size * exit_price_with_slippage * CONTRACT_MULTIPLIER
                    current_capital = current_capital + position_value  # 平仓后加上持仓价值
                    
                    # 计算此笔交易的收益率
                    trade_return_pct = (pnl_amount / (position_size * entry_price * CONTRACT_MULTIPLIER)) * 100
                    
                    trade_record = {
                        'entry_time': entry_time, 'exit_time': current_dt,
                        'position': 'LONG', 'entry_price': entry_price,
                        'exit_price': exit_price_with_slippage,
                        'exit_price_market': current_price,  # 记录市场价格（不含滑点）
                        'pnl_points': pnl_points,
                        'pnl_amount': pnl_amount, 'position_size': position_size,
                        'exit_capital': current_capital, 'exit_reason': exit_reason,
                        'trade_return_pct': trade_return_pct,
                        'slippage': slippage  # 记录滑点
                    }
                    
                    trades.append(trade_record)
                    
                    # 打印当前交易详情
                    print(f"交易: {exit_reason} | 开仓时间: {entry_time} 平仓时间: {current_dt}")
                    print(f"  开仓价: {entry_price:.2f} 平仓价: {exit_price_with_slippage:.2f} (市场价: {current_price:.2f}, 滑点: {slippage:.2f}) 手数: {position_size}手")
                    print(f"  盈亏: {pnl_amount:.2f}元 ({trade_return_pct:.2f}%) | 资金: {exit_capital_before:.2f} -> {current_capital:.2f}元")
                    
                    position = None
                    position_size = 0
                    continue

            elif position == 'SHORT':
                sigma = calculate_sigma(current_dt, df, config['LOOKBACK_DAYS'], historical_first_opens)
                if np.isnan(sigma):
                    pass
                else:
                    current_lower_bound = lower_ref * (1 - config['K2'] * sigma)
                    stop_loss_price = min(stop_loss_price, current_lower_bound) # Ratchet stop

                if current_price > stop_loss_price:
                    exit_reason = "止损（空头）"
                    
                    exit_capital_before = current_capital
                    
                    # 空头平仓（考虑滑点）
                    exit_price_with_slippage = current_price + slippage  # 买入时价格提高
                    pnl_points = entry_price - exit_price_with_slippage
                    pnl_amount = position_size * pnl_points * CONTRACT_MULTIPLIER
                    current_capital = current_capital + pnl_amount  # 空头平仓后的资金
                    
                    # 计算此笔交易的收益率
                    trade_return_pct = (pnl_amount / (position_size * entry_price * CONTRACT_MULTIPLIER)) * 100
                    
                    trade_record = {
                        'entry_time': entry_time, 'exit_time': current_dt,
                        'position': 'SHORT', 'entry_price': entry_price,
                        'exit_price': exit_price_with_slippage,
                        'exit_price_market': current_price,  # 记录市场价格（不含滑点）
                        'pnl_points': pnl_points,
                        'pnl_amount': pnl_amount, 'position_size': position_size,
                        'exit_capital': current_capital, 'exit_reason': exit_reason,
                        'trade_return_pct': trade_return_pct,
                        'slippage': slippage  # 记录滑点
                    }
                    
                    trades.append(trade_record)
                    
                    # 打印当前交易详情
                    print(f"交易: {exit_reason} | 开仓时间: {entry_time} 平仓时间: {current_dt}")
                    print(f"  开仓价: {entry_price:.2f} 平仓价: {exit_price_with_slippage:.2f} (市场价: {current_price:.2f}, 滑点: {slippage:.2f}) 手数: {position_size}手")
                    print(f"  盈亏: {pnl_amount:.2f}元 ({trade_return_pct:.2f}%) | 资金: {exit_capital_before:.2f} -> {current_capital:.2f}元")
                    
                    position = None
                    position_size = 0
                    continue
            
            # --- Check Open Condition (every N minutes) ---
            if not position and (last_check_time is None or (current_dt - last_check_time) >= timedelta(minutes=config['CHECK_INTERVAL_MINUTES'])):
                last_check_time = current_dt
                sigma = calculate_sigma(current_dt, df, config['LOOKBACK_DAYS'], historical_first_opens)

                if np.isnan(sigma):
                    continue

                upper_bound = upper_ref * (1 + config['K1'] * sigma)
                lower_bound = lower_ref * (1 - config['K2'] * sigma)

                if current_price > upper_bound:
                    # 多头开仓（全仓买入，考虑滑点）
                    position = 'LONG'
                    entry_price_with_slippage = current_price + slippage  # 买入时价格提高
                    entry_price = entry_price_with_slippage
                    entry_time = current_dt
                    stop_loss_price = upper_bound # Initial stop for long
                    
                    # 计算可买入的整数手数 - 期货只能买卖整数手
                    max_position_size_float = current_capital / (entry_price * CONTRACT_MULTIPLIER)
                    position_size = int(max_position_size_float)  # 向下取整，确保是整数手
                    
                    # 计算实际使用的资金
                    used_capital = position_size * entry_price * CONTRACT_MULTIPLIER
                    
                    # 更新当前资金（扣除购买所需资金）
                    if position_size > 0:
                        current_capital = current_capital - used_capital  # 多头开仓后，资金减少，但持有股票价值
                        
                        print(f"开仓: 多头 | 时间: {entry_time} | 价格: {entry_price:.2f} (市场价: {current_price:.2f}, 滑点: {slippage:.2f}) | 手数: {position_size}手")
                        print(f"  资金使用: {used_capital:.2f}元 | 资金剩余: {current_capital:.2f}元")
                    else:
                        # 如果资金不足以买一手，取消此次交易
                        print(f"资金不足 ({current_capital:.2f}元) 无法以 {entry_price:.2f}元/手 买入期货")
                        position = None
                        stop_loss_price = 0
                    
                elif current_price < lower_bound:
                    # 空头开仓（全仓卖空，考虑滑点）
                    position = 'SHORT'
                    entry_price_with_slippage = current_price - slippage  # 卖出时价格降低
                    entry_price = entry_price_with_slippage
                    entry_time = current_dt
                    stop_loss_price = lower_bound # Initial stop for short
                    
                    # 计算可卖空的整数手数 - 期货只能买卖整数手
                    # 使用保证金比例
                    max_position_size_float = current_capital / (entry_price * CONTRACT_MULTIPLIER * MARGIN_RATIO)
                    position_size = int(max_position_size_float)  # 向下取整，确保是整数手
                    
                    # 空头开仓不减少资金，只是冻结保证金
                    if position_size > 0:
                        # 资金不变
                        print(f"开仓: 空头 | 时间: {entry_time} | 价格: {entry_price:.2f} (市场价: {current_price:.2f}, 滑点: {slippage:.2f}) | 手数: {position_size}手")
                        print(f"  资金: {current_capital:.2f}元 (保证金)")
                    else:
                        # 如果资金不足以卖一手，取消此次交易
                        print(f"资金不足 ({current_capital:.2f}元) 无法以 {entry_price:.2f}元/手 卖空期货")
                        position = None
                        stop_loss_price = 0
        
        # --- End of session processing ---
        # Force close any open position at the very end of the tradable window if not caught by deadline check
        if position and not tradable_session_df.empty and tradable_session_df.index[-1] >= force_close_deadline_time:
             # This check might be redundant if loop handles deadline correctly, but as a safeguard
             last_bar_price = tradable_session_df['Close'].iloc[-1]
             exit_reason = "强制平仓（交易时段结束）"
             
             exit_capital_before = current_capital
             
             # 计算盈亏（考虑滑点）
             if position == 'LONG':
                 # 多头平仓
                 exit_price_with_slippage = last_bar_price - slippage  # 卖出时价格降低
                 pnl_points = exit_price_with_slippage - entry_price
                 pnl_amount = position_size * pnl_points * CONTRACT_MULTIPLIER
                 position_value = position_size * exit_price_with_slippage * CONTRACT_MULTIPLIER
                 current_capital = current_capital + position_value  # 平仓后加上持仓价值
             else:  # SHORT
                 # 空头平仓
                 exit_price_with_slippage = last_bar_price + slippage  # 买入时价格提高
                 pnl_points = entry_price - exit_price_with_slippage
                 pnl_amount = position_size * pnl_points * CONTRACT_MULTIPLIER
                 current_capital = current_capital + pnl_amount  # 空头平仓后的资金
             
             # 计算此笔交易的收益率
             trade_return_pct = (pnl_amount / (position_size * entry_price * CONTRACT_MULTIPLIER)) * 100
             
             trade_record = {
                 'entry_time': entry_time, 'exit_time': tradable_session_df.index[-1],
                 'position': position, 'entry_price': entry_price,
                 'exit_price': exit_price_with_slippage,
                 'exit_price_market': last_bar_price,  # 记录市场价格（不含滑点）
                 'pnl_points': pnl_points,
                 'pnl_amount': pnl_amount, 'position_size': position_size,
                 'exit_capital': current_capital, 'exit_reason': exit_reason,
                 'trade_return_pct': trade_return_pct,
                 'slippage': slippage  # 记录滑点
             }
             
             trades.append(trade_record)
             
             # 打印当前交易详情
             print(f"交易: {exit_reason} {position} | 开仓时间: {entry_time} 平仓时间: {tradable_session_df.index[-1]}")
             print(f"  开仓价: {entry_price:.2f} 平仓价: {exit_price_with_slippage:.2f} (市场价: {last_bar_price:.2f}, 滑点: {slippage:.2f}) 手数: {position_size}手")
             print(f"  盈亏: {pnl_amount:.2f}元 ({trade_return_pct:.2f}%) | 资金: {exit_capital_before:.2f} -> {current_capital:.2f}元")
             
             position = None
             position_size = 0

        # Update global_prev_session_close_price with the actual close of this session
        if not session_df.empty: # Ensure session_df had data
            global_prev_session_close_price = session_df['Close'].iloc[-1]
            global_prev_session_close_price_dt = session_df.index[-1]
    
    # 添加策略信息到返回值中
    strategy_info = {
        'final_capital': current_capital,
        'initial_capital': config['INITIAL_CAPITAL'],
        'equity_curve': equity_curve,
    }
    
    return trades, strategy_info

# --- Reporting ---
def print_results(trades_log, strategy_info, df_data, risk_free_rate_annual=0.0):
    if not trades_log:
        print("无交易记录。")
        # Calculate Buy and Hold for comparison even if no strategy trades
        if df_data is not None and not df_data.empty:
            initial_price = df_data['Open'].iloc[0]
            final_price = df_data['Close'].iloc[-1]
            
            # 计算买入持有策略的收益
            buy_hold_units = strategy_info['initial_capital'] / initial_price
            buy_hold_final_value = buy_hold_units * final_price
            buy_hold_pnl_amount = buy_hold_final_value - strategy_info['initial_capital']
            buy_hold_return_pct = (buy_hold_pnl_amount / strategy_info['initial_capital']) * 100
            
            print("\n--- 策略绩效统计 ---")
            print("策略: 未执行任何交易。")
            print("\n--- 买入持有对比 ---")
            print(f"初始价格: {initial_price:.2f}")
            print(f"最终价格: {final_price:.2f}")
            print(f"买入持有收益: {buy_hold_pnl_amount:.2f}元 ({buy_hold_return_pct:.2f}%)")
        return

    df_trades = pd.DataFrame(trades_log)
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    
    # 计算总收益和收益率
    initial_capital = strategy_info['initial_capital']
    final_capital = strategy_info['final_capital']
    total_profit = final_capital - initial_capital
    total_return_pct = (total_profit / initial_capital) * 100

    total_pnl_amount = df_trades['pnl_amount'].sum() if 'pnl_amount' in df_trades.columns else 0
    num_trades = len(df_trades)
    
    # --- 计算净值曲线和回撤 ---
    equity_curve = strategy_info.get('equity_curve', {})
    equity_dates = sorted(equity_curve.keys())
    equity_values = [equity_curve[date] for date in equity_dates]
    
    # 转换为pandas Series以便计算
    if equity_values:
        equity_series = pd.Series(equity_values, index=equity_dates)
        
        # 计算回撤
        max_equity = equity_series.cummax()
        drawdown_series = (equity_series - max_equity) / max_equity * 100
        max_drawdown = drawdown_series.min()
        max_drawdown_date = drawdown_series.idxmin() if not drawdown_series.empty else None
        
        # 计算每日收益率
        daily_returns = equity_series.pct_change().dropna()
        
        # 计算波动率 (年化)
        volatility_annual = daily_returns.std() * np.sqrt(252) * 100 if not daily_returns.empty else 0  # 转换为百分比
    else:
        max_drawdown = 0
        max_drawdown_date = None
        volatility_annual = 0
        daily_returns = pd.Series()
    
    # --- 首先输出每日交易摘要 ---
    print("\n--- 每日交易摘要 ---")
    print("日期, 总盈亏(元), 交易时间点")
    
    # 将交易按退出日期分组
    df_trades['exit_date'] = df_trades['exit_time'].dt.date
    df_trades['entry_date'] = df_trades['entry_time'].dt.date
    
    # 每个日期的总盈亏
    daily_pnl = df_trades.groupby('exit_date')['pnl_amount'].sum() if 'pnl_amount' in df_trades.columns else pd.Series()
    
    # 遍历每个交易日
    for date in sorted(daily_pnl.index):
        # 获取当日盈亏
        day_pnl = daily_pnl.loc[date]
        
        # 获取当日的所有交易
        day_trades = df_trades[
            (df_trades['exit_date'] == date) | 
            (df_trades['entry_date'] == date)
        ]
        
        # 整理交易时间点信息
        trade_times = []
        for _, trade in day_trades.iterrows():
            entry_time_str = trade['entry_time'].strftime('%H:%M')
            exit_time_str = trade['exit_time'].strftime('%H:%M') if pd.notna(trade['exit_time']) and trade['exit_date'] == date else "持仓中"
            position_type = "多" if trade['position'] == 'LONG' else "空"
            
            if trade['entry_date'] == date and trade['exit_date'] == date:
                # 当日开仓且平仓
                trade_times.append(f"{entry_time_str}->{exit_time_str}({position_type})")
            elif trade['entry_date'] == date:
                # 当日只开仓
                trade_times.append(f"{entry_time_str}->持仓({position_type})")
            elif trade['exit_date'] == date:
                # 当日只平仓
                trade_times.append(f"前日->{exit_time_str}({position_type})")
        
        # 打印当日交易摘要
        print(f"{date}, {day_pnl:.2f}元, {', '.join(trade_times)}")

    # --- 策略绩效统计 ---
    print("\n--- 策略绩效统计 ---")
    print(f"总交易次数: {num_trades}")
    print(f"初始资金: {initial_capital:.2f}元")
    print(f"最终资金: {final_capital:.2f}元")
    print(f"总收益: {total_profit:.2f}元 ({total_return_pct:.2f}%)")
    
    # 波动率和回撤统计
    print(f"年化波动率: {volatility_annual:.2f}%")
    print(f"最大回撤: {abs(max_drawdown):.2f}%") # 使用绝对值，保证为正值显示
    if max_drawdown_date:
        print(f"最大回撤日期: {max_drawdown_date}")

    if num_trades > 0:
        wins = df_trades[df_trades['pnl_amount'] > 0]
        losses = df_trades[df_trades['pnl_amount'] <= 0]
        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        print(f"盈利交易: {len(wins)}笔")
        print(f"亏损交易: {len(losses)}笔")
        print(f"胜率: {win_rate:.2%}")
        if not wins.empty:
            print(f"平均盈利: {wins['pnl_amount'].mean():.2f}元")
        if not losses.empty:
            print(f"平均亏损: {losses['pnl_amount'].mean():.2f}元")
        
        # 计算夏普比率
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            # 使用年化的数据计算
            mean_daily_return = daily_returns.mean()
            annual_return = ((1 + mean_daily_return) ** 252) - 1
            
            # 日度无风险利率
            daily_risk_free_rate = (1 + risk_free_rate_annual) ** (1/252) - 1
            
            # 夏普比率 = (年化收益率 - 年化无风险利率) / 年化波动率
            sharpe_ratio = (annual_return - risk_free_rate_annual) / (daily_returns.std() * np.sqrt(252))
            print(f"夏普比率(年化): {sharpe_ratio:.2f}")
        else:
            print("夏普比率: 无法计算 (收益率数据不足)")

    # --- Buy and Hold Benchmark ---
    if df_data is not None and not df_data.empty:
        initial_price = df_data['Open'].iloc[0]
        final_price = df_data['Close'].iloc[-1]
        
        # 计算买入持有策略的收益
        buy_hold_units = initial_capital / initial_price
        buy_hold_final_value = buy_hold_units * final_price
        buy_hold_pnl_amount = buy_hold_final_value - initial_capital
        buy_hold_return_pct = (buy_hold_pnl_amount / initial_capital) * 100
        
        print("\n--- 买入持有策略对比 ---")
        print(f"初始价格: {initial_price:.2f}")
        print(f"最终价格: {final_price:.2f}")
        print(f"买入持有收益: {buy_hold_pnl_amount:.2f}元 ({buy_hold_return_pct:.2f}%)")
        
        # 策略与买入持有的对比
        if total_profit != 0 and buy_hold_pnl_amount != 0:
            outperform_pct = ((total_profit / abs(buy_hold_pnl_amount)) - 1) * 100 if buy_hold_pnl_amount != 0 else float('inf')
            outperform_sign = '+' if total_profit > buy_hold_pnl_amount else ''
            print(f"策略相对买入持有超额收益: {outperform_sign}{outperform_pct:.2f}%")

    # 计算滑点的影响
    if trades_log and 'slippage' in trades_log[0]:
        total_slippage_cost = 0
        for trade in trades_log:
            # 每笔交易有开仓和平仓两次滑点
            slippage_points = trade['slippage'] * 2  # 开仓和平仓的总滑点点数
            slippage_cost = slippage_points * trade['position_size'] * CONTRACT_MULTIPLIER
            total_slippage_cost += slippage_cost
        
        print(f"滑点总成本: {total_slippage_cost:.2f}元")
        if num_trades > 0:
            print(f"平均每笔交易滑点成本: {total_slippage_cost/num_trades:.2f}元")
        else:
            print("平均每笔交易滑点成本: 0.00元")

# --- Execution ---
if __name__ == "__main__":
    config_params = {
        'LOOKBACK_DAYS': LOOKBACK_DAYS,
        'K1': K1,
        'K2': K2,
        'CHECK_INTERVAL_MINUTES': CHECK_INTERVAL_MINUTES,
        'SESSION_WARMUP_MINUTES': SESSION_WARMUP_MINUTES,
        'SESSION_COOLDOWN_MINUTES': SESSION_COOLDOWN_MINUTES,
        'INITIAL_CAPITAL': INITIAL_CAPITAL,
        'CONTRACT_MULTIPLIER': CONTRACT_MULTIPLIER,  # 添加合约乘数到配置
        'SLIPPAGE': SLIPPAGE,  # 添加滑点到配置
    }

    df_data = load_data(FILE_PATH)

    if df_data is not None and not df_data.empty:
        print(f"数据时间范围: {df_data.index.min()} 至 {df_data.index.max()}")
        trades_log, strategy_info = run_backtest(df_data, config_params)
        print_results(trades_log, strategy_info, df_data)
    else:
        print(f"无法从 {FILE_PATH} 加载数据，也无法生成模拟数据。回测未运行。") 