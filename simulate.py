import pandas as pd
from datetime import datetime, time, timedelta, date as date_type
import time as time_module
import os
import sys
import pytz
from math import floor
from decimal import Decimal
from dotenv import load_dotenv
import numpy as np

from longport.openapi import Config, TradeContext, QuoteContext, Period, OrderSide, OrderType, TimeInForceType, AdjustType, OutsideRTH

load_dotenv()

# 固定配置参数
CHECK_INTERVAL_MINUTES = 10
TRADING_START_TIME = (9, 40)  # 交易开始时间：9点40分
TRADING_END_TIME = (15, 40)   # 交易结束时间：15点40分
MAX_POSITIONS_PER_DAY = 3
LOOKBACK_DAYS = 10

# 默认交易品种
SYMBOL = os.environ.get('SYMBOL', 'TQQQ.US')

# 调试模式配置
DEBUG_MODE = True  # 设置为True开启调试模式
DEBUG_TIME = "2025-05-02 10:20:00"  # 调试使用的时间，格式: "YYYY-MM-DD HH:MM:SS"
DEBUG_ONCE = True  # 是否只运行一次就退出

def get_us_eastern_time():
    if DEBUG_MODE and DEBUG_TIME:
        # 如果处于调试模式且指定了时间，返回指定的时间
        try:
            dt = datetime.strptime(DEBUG_TIME, "%Y-%m-%d %H:%M:%S")
            eastern = pytz.timezone('US/Eastern')
            return eastern.localize(dt)
        except ValueError:
            print(f"错误的调试时间格式: {DEBUG_TIME}，应为 'YYYY-MM-DD HH:MM:SS'")
    
    # 正常模式或调试时间格式错误时返回当前时间
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

def create_contexts():
    config = Config.from_env()
    quote_ctx = QuoteContext(config)
    trade_ctx = TradeContext(config)
    return quote_ctx, trade_ctx

QUOTE_CTX, TRADE_CTX = create_contexts()

def get_account_balance():
    balance_list = TRADE_CTX.account_balance(currency="USD")
    available_cash = float(balance_list[0].net_assets)
    return available_cash

def get_current_positions():
    stock_positions_response = TRADE_CTX.stock_positions()
    positions = {}
    for channel in stock_positions_response.channels:
        for position in channel.positions:
            symbol = position.symbol
            quantity = int(position.quantity)
            cost_price = float(position.cost_price)
            positions[symbol] = {
                "quantity": quantity,
                "cost_price": cost_price
            }
    return positions

def get_historical_data(symbol, days_back=None):

    # 简化天数计算逻辑
    if days_back is None:
        days_back = LOOKBACK_DAYS + 10  # 简化为固定天数
        
    # 直接使用1分钟K线
    sdk_period = Period.Min_1
    adjust_type = AdjustType.ForwardAdjust
    eastern = pytz.timezone('US/Eastern')
    now_et = get_us_eastern_time()
    current_date = now_et.date()
    
    # 计算起始日期
    start_date = current_date - timedelta(days=days_back)
    
    # 对于1分钟数据使用按日获取的方式
    all_candles = []
    
    # 尝试从今天开始向前获取足够的数据
    date_to_check = current_date
    while date_to_check >= start_date:
        day_start_time = datetime.combine(date_to_check, time(9, 30))
        day_start_time_et = eastern.localize(day_start_time)
        
        # 每天最多获取390分钟数据（6.5小时交易时间）
        day_candles = QUOTE_CTX.history_candlesticks_by_offset(
            symbol, sdk_period, adjust_type, True, 390,
            day_start_time_et
        )
        
        if day_candles:
            all_candles.extend(day_candles)
            
        date_to_check -= timedelta(days=1)
    
    # 处理数据并去重
    data = []
    processed_timestamps = set()
    
    for candle in all_candles:
        timestamp = candle.timestamp
        if isinstance(timestamp, datetime):
            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            ts_str = str(timestamp)
            
        # 去重处理
        if ts_str in processed_timestamps:
            continue
        processed_timestamps.add(ts_str)
        
        # 标准化时区
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                hour = timestamp.hour
                if symbol.endswith(".US") and 9 <= hour < 17:
                    dt = eastern.localize(timestamp)
                elif symbol.endswith(".US") and (hour >= 21 or hour < 5):
                    beijing = pytz.timezone('Asia/Shanghai')
                    dt = beijing.localize(timestamp).astimezone(eastern)
                else:
                    utc = pytz.utc
                    dt = utc.localize(timestamp).astimezone(eastern)
            else:
                dt = timestamp.astimezone(eastern)
        else:
            dt = datetime.fromtimestamp(timestamp, eastern)
        
        # 过滤未来日期
        if dt.date() > current_date:
            continue
            
        # 添加到数据列表
        data.append({
            "Close": float(candle.close),
            "Open": float(candle.open),
            "High": float(candle.high),
            "Low": float(candle.low),
            "Volume": float(candle.volume),
            "Turnover": float(candle.turnover),
            "DateTime": dt
        })
    
    # 转换为DataFrame并进行后处理
    df = pd.DataFrame(data)
    if df.empty:
        return df
        
    df["Date"] = df["DateTime"].dt.date
    df["Time"] = df["DateTime"].dt.strftime('%H:%M')
    
    # 过滤交易时间
    if symbol.endswith(".US"):
        df = df[df["Time"].between("09:30", "16:00")]
        
    # 去除重复数据
    df = df.drop_duplicates(subset=['Date', 'Time'])
    
    # 过滤掉未来日期的数据（双重保险）
    df = df[df["Date"] <= current_date]
    
    # 添加日志，打印历史数据信息
    trading_days_count = len(df["Date"].unique())
    total_rows = len(df)
    print(f"获取到的历史数据: 共{trading_days_count}个交易日, {total_rows}行数据")
    
    # 保存到本地临时文件
    df.to_csv('temp_historical_data.csv', index=False)
    return df

def get_quote(symbol):
    quotes = QUOTE_CTX.quote([symbol])
    quote_data = {
        "symbol": quotes[0].symbol,
        "last_done": str(quotes[0].last_done),
        "open": str(quotes[0].open),
        "high": str(quotes[0].high),
        "low": str(quotes[0].low),
        "volume": str(quotes[0].volume),
        "turnover": str(quotes[0].turnover),
        "timestamp": quotes[0].timestamp.isoformat()
    }
    return quote_data

def calculate_vwap(df):
    # 直接使用成交额除以成交量
    vwap = df.apply(
        lambda x: x['Turnover'] / x['Volume'] if x['Volume'] > 0 else x['Close'], 
        axis=1
    )    
    return vwap

def calculate_noise_area(df, lookback_days=14):
    # 创建数据副本
    df_copy = df.copy()
    
    # 获取唯一日期并排序
    unique_dates = sorted(df_copy["Date"].unique())
    now_et = get_us_eastern_time()
    current_date = now_et.date()
    
    if DEBUG_MODE:
        print(f"\n调试 - 噪声区域计算:")
        print(f"当前日期: {current_date}")
        print(f"唯一日期数量: {len(unique_dates)}")
        if len(unique_dates) > 0:
            print(f"日期范围: {unique_dates[0]} 到 {unique_dates[-1]}")
        print(f"回溯天数: {lookback_days}")
    
    # 过滤未来日期
    if unique_dates and isinstance(unique_dates[0], date_type):
        unique_dates = [d for d in unique_dates if d <= current_date]
        df_copy = df_copy[df_copy["Date"].isin(unique_dates)]
        if DEBUG_MODE:
            print(f"过滤后日期数量: {len(unique_dates)}")
    
    # 检查数据是否足够
    if len(unique_dates) <= lookback_days:
        print(f"错误: 历史数据不足，至少需要{lookback_days+1}个交易日，当前只有{len(unique_dates)}个交易日")
        if DEBUG_MODE:
            print(f"可用交易日: {unique_dates}")
        sys.exit(1)
    
    # 为每个日期计算当日开盘价
    day_opens = {}
    for date in unique_dates:
        day_data = df_copy[df_copy["Date"] == date]
        if day_data.empty:
            print(f"错误: {date} 日期数据为空")
            sys.exit(1)
        day_opens[date] = day_data["Open"].iloc[0]
    
    # 计算前一日收盘价
    prev_closes = {}
    for i in range(1, len(unique_dates)):
        prev_date = unique_dates[i-1]
        curr_date = unique_dates[i]
        prev_day_data = df_copy[df_copy["Date"] == prev_date]
        if prev_day_data.empty:
            print(f"错误: {prev_date} 前一交易日数据为空")
            sys.exit(1)
        prev_closes[curr_date] = prev_day_data["Close"].iloc[-1]
    
    # 为每个时间点计算相对于开盘价的绝对变动率
    df_copy["move"] = 0.0
    for date in unique_dates:
        day_open = day_opens[date]
        df_copy.loc[df_copy["Date"] == date, "move"] = abs(df_copy.loc[df_copy["Date"] == date, "Close"] / day_open - 1)
    
    # 按日期和时间对数据进行分组
    df_copy["DateTime"] = df_copy["Date"].astype(str) + " " + df_copy["Time"]
    
    # 计算每个时间点的sigma (过去lookback_days天的平均变动率)
    time_sigma = {}
    for i in range(lookback_days, len(unique_dates)):
        curr_date = unique_dates[i]
        curr_day_data = df_copy[df_copy["Date"] == curr_date]
        
        # 获取当前日期的所有时间点
        times = curr_day_data["Time"].unique()
        
        # 对每个时间点计算sigma
        for tm in times:
            # 获取历史lookback_days天中相同时间点的数据
            historical_moves = []
            for j in range(i-lookback_days, i):
                hist_date = unique_dates[j]
                hist_data = df_copy[(df_copy["Date"] == hist_date) & (df_copy["Time"] == tm)]
                if not hist_data.empty:
                    historical_moves.append(hist_data["move"].iloc[0])
            
            # 确保有足够的历史数据计算sigma
            if len(historical_moves) == 0:
                if DEBUG_MODE and curr_date == current_date:
                    print(f"时间点 {tm} 没有足够的历史数据")
                continue
                
            # 计算平均变动率作为sigma
            sigma = sum(historical_moves) / len(historical_moves)
            time_sigma[(curr_date, tm)] = sigma
    
    if DEBUG_MODE:
        print(f"计算的时间点sigma数量: {len(time_sigma)}")
        if len(time_sigma) == 0:
            print("警告: 没有计算出任何sigma值!")
        
        # 检查当前日期是否有sigma值
        curr_date_sigmas = [(date, tm) for (date, tm) in time_sigma.keys() if date == current_date]
        print(f"当前日期 {current_date} 的sigma值数量: {len(curr_date_sigmas)}")
    
    # 计算上下边界
    df_copy["UpperBound"] = None
    df_copy["LowerBound"] = None
    
    bounds_count = 0
    for i in range(lookback_days, len(unique_dates)):
        curr_date = unique_dates[i]
        curr_day_data = df_copy[df_copy["Date"] == curr_date]
        day_open = day_opens[curr_date]
        prev_close = prev_closes.get(curr_date)
        
        if DEBUG_MODE and curr_date == current_date:
            print(f"\n当前日期 {curr_date} 的边界计算:")
            print(f"当日开盘价: {day_open}")
            print(f"前一日收盘价: {prev_close}")
            times_with_sigma = [tm for (dt, tm) in time_sigma.keys() if dt == curr_date]
            print(f"有sigma值的时间点数量: {len(times_with_sigma)}")
            if len(times_with_sigma) > 0:
                print(f"时间点示例: {times_with_sigma[:5] if len(times_with_sigma) >= 5 else times_with_sigma}")
        
        if prev_close is None:
            if DEBUG_MODE:
                print(f"日期 {curr_date} 没有前一日收盘价")
            continue
        
        # 根据算法计算参考价格
        upper_ref = max(day_open, prev_close)
        lower_ref = min(day_open, prev_close)
        
        # 对当日每个时间点计算上下边界
        for _, row in curr_day_data.iterrows():
            tm = row["Time"]
            sigma = time_sigma.get((curr_date, tm))
            
            if sigma is not None:
                # 使用时间点特定的sigma计算上下边界
                df_copy.loc[(df_copy["Date"] == curr_date) & (df_copy["Time"] == tm), "UpperBound"] = upper_ref * (1 + sigma)
                df_copy.loc[(df_copy["Date"] == curr_date) & (df_copy["Time"] == tm), "LowerBound"] = lower_ref * (1 - sigma)
                bounds_count += 1
    
    # 在调试模式下检查最后一天的边界数据
    if DEBUG_MODE:
        print(f"\n边界数据统计:")
        print(f"总计算的边界点数: {bounds_count}")
        
        if unique_dates:
            latest_date = max(unique_dates)
            latest_data = df_copy[df_copy["Date"] == latest_date]
            has_upper = latest_data["UpperBound"].notna().sum()
            has_lower = latest_data["LowerBound"].notna().sum()
            total_rows = len(latest_data)
            
            print(f"最新日期 {latest_date} 数据行数: {total_rows}")
            if total_rows > 0:
                print(f"有上边界值的行数: {has_upper}/{total_rows} ({has_upper/total_rows*100:.2f}%)")
                print(f"有下边界值的行数: {has_lower}/{total_rows} ({has_lower/total_rows*100:.2f}%)")
            
            # 检查当前测试时间点是否有边界值
            test_time = now_et.strftime('%H:%M')
            test_data = df_copy[(df_copy["Date"] == current_date) & (df_copy["Time"] == test_time)]
            if not test_data.empty:
                has_test_upper = test_data["UpperBound"].notna().all()
                has_test_lower = test_data["LowerBound"].notna().all()
                print(f"当前测试时间点 {test_time} 是否有上边界: {has_test_upper}")
                print(f"当前测试时间点 {test_time} 是否有下边界: {has_test_lower}")
            else:
                print(f"当前测试时间点 {test_time} 没有找到数据")
    
    return df_copy

def submit_order(symbol, side, quantity, order_type="MO", price=None, outside_rth=None):
    sdk_side = OrderSide.Buy if side == "Buy" else OrderSide.Sell
    if isinstance(order_type, str):
        order_type_map = {
            "MO": OrderType.MO, "LO": OrderType.LO, "ELO": OrderType.ELO,
            "AO": OrderType.AO, "ALO": OrderType.ALO
        }
        sdk_order_type = order_type_map.get(order_type, OrderType.MO)
    else:
        sdk_order_type = order_type
    time_in_force = TimeInForceType.Day
    if outside_rth is None:
        outside_rth = OutsideRTH.AnyTime
    elif isinstance(outside_rth, str):
        outside_rth_map = {
            "RTH_ONLY": OutsideRTH.RTHOnly,
            "ANY_TIME": OutsideRTH.AnyTime,
            "OVERNIGHT": OutsideRTH.Overnight
        }
        outside_rth = outside_rth_map.get(outside_rth, OutsideRTH.AnyTime)
    dec_quantity = Decimal(str(quantity)) if not isinstance(quantity, Decimal) else quantity
    if sdk_order_type == OrderType.LO and price is not None:
        dec_price = Decimal(str(price)) if not isinstance(price, Decimal) else price
        response = TRADE_CTX.submit_order(
            symbol=symbol,
            order_type=sdk_order_type,
            side=sdk_side,
            submitted_price=dec_price,
            submitted_quantity=dec_quantity,
            time_in_force=time_in_force,
            outside_rth=outside_rth
        )
    else:
        response = TRADE_CTX.submit_order(
            symbol=symbol,
            order_type=OrderType.MO,
            side=sdk_side,
            submitted_quantity=dec_quantity,
            time_in_force=time_in_force,
            outside_rth=outside_rth
        )
    return response.order_id

def get_order_status(order_id):
    order_detail = TRADE_CTX.order_detail(order_id)
    status_str = str(order_detail.status)
    order_info = {
        "order_id": order_detail.order_id,
        "status": status_str,
        "stock_name": order_detail.stock_name,
        "quantity": order_detail.quantity,
        "executed_quantity": order_detail.executed_quantity,
        "price": str(order_detail.price),
        "executed_price": str(order_detail.executed_price),
        "submitted_at": order_detail.submitted_at.isoformat(),
        "side": str(order_detail.side)
    }
    return order_info

def check_exit_conditions(df, position_quantity, trailing_stop):
    latest = df.iloc[-1]
    price = latest["Close"]
    vwap = latest["VWAP"]
    upper = latest["UpperBound"]
    lower = latest["LowerBound"]
    if position_quantity > 0:
        new_stop = max(upper, vwap)
        if trailing_stop is not None:
            new_stop = max(trailing_stop, new_stop)
        exit_signal = price < new_stop
        return exit_signal, new_stop
    elif position_quantity < 0:
        new_stop = min(lower, vwap)
        if trailing_stop is not None:
            new_stop = min(trailing_stop, new_stop)
        exit_signal = price > new_stop
        return exit_signal, new_stop
    return False, None

def is_trading_day(symbol=None):
    market = None
    if symbol:
        if symbol.endswith(".US"):
            market = "US"
        elif symbol.endswith(".HK"):
            market = "HK"
        elif symbol.endswith(".SH") or symbol.endswith(".SZ"):
            market = "CN"
        elif symbol.endswith(".SG"):
            market = "SG"
    if not market:
        market = "US"
    now_et = get_us_eastern_time()
    current_date = now_et.date()
    from longport.openapi import Market
    market_mapping = {
        "US": Market.US, "HK": Market.HK, "CN": Market.CN, "SG": Market.SG
    }
    sdk_market = market_mapping.get(market, Market.US)
    calendar_resp = QUOTE_CTX.trading_days(
        sdk_market, current_date, current_date
    )
    trading_dates = calendar_resp.trading_days
    half_trading_dates = calendar_resp.half_trading_days
    is_trade_day = current_date in trading_dates
    is_half_trade_day = current_date in half_trading_dates
    return is_trade_day or is_half_trade_day

def run_trading_strategy(symbol=SYMBOL, check_interval_minutes=CHECK_INTERVAL_MINUTES,
                        trading_start_time=TRADING_START_TIME, trading_end_time=TRADING_END_TIME,
                        max_positions_per_day=MAX_POSITIONS_PER_DAY, lookback_days=LOOKBACK_DAYS):
    now_et = get_us_eastern_time()
    print("\n" + "="*50)
    print(f"启动交易策略 - 交易品种: {symbol}")
    print(f"当前美东时间: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
    if DEBUG_MODE:
        print(f"调试模式已开启! 使用时间: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
        if DEBUG_ONCE:
            print("单次运行模式已开启，策略将只运行一次")
    print(f"交易时间: {trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d} (美东时间)")
    print(f"检查间隔: {check_interval_minutes} 分钟")
    print(f"每日最大持仓数: {max_positions_per_day}")
    print(f"回溯天数: {lookback_days}")
    print("="*50 + "\n")
    initial_capital = get_account_balance()
    if initial_capital <= 0:
        print("Error: Could not get account balance or balance is zero")
        sys.exit(1)
    print(f"使用 {check_interval_minutes} 分钟间隔进行定时检查")
    position_quantity = 0
    entry_price = None
    trailing_stop = None
    positions_opened_today = 0
    last_date = None
    is_us_market = symbol.endswith(".US")
    outside_rth_setting = OutsideRTH.AnyTime
    loop_count = 0
    
    while True:
        loop_count += 1
        print(f"\n----- 交易检查循环 #{loop_count} -----")
        now = get_us_eastern_time()
        current_date = now.date()
        print(f"\n当前美东时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查是否是交易日（调试模式下保持原有逻辑）
        print(f"准备检查今天 {current_date} 是否是交易日...")
        is_today_trading_day = is_trading_day(symbol)
        print(f"交易日检查结果: {'是交易日' if is_today_trading_day else '不是交易日'}")
            
        if not is_today_trading_day:
            print(f"今天不是交易日，跳过交易")
            if position_quantity != 0:
                print("非交易日，执行平仓")
                side = "Sell" if position_quantity > 0 else "Buy"
                close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
                print(f"平仓订单已提交，ID: {close_order_id}")
                position_quantity = 0
                print("持仓状态已重置")
            next_check_time = now + timedelta(hours=12)
            wait_seconds = (next_check_time - now).total_seconds()
            time_module.sleep(wait_seconds)
            continue
            
        if last_date is not None and current_date != last_date:
            print(f"新的交易日开始，重置今日开仓计数")
            positions_opened_today = 0
        last_date = current_date
        
        # 保持原有交易时间检查逻辑
        current_hour, current_minute = now.hour, now.minute
        start_hour, start_minute = trading_start_time
        end_hour, end_minute = trading_end_time
        is_trading_hours = (
            (current_hour > start_hour or (current_hour == start_hour and current_minute >= start_minute)) and
            (current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute))
        )
        print(f"检查是否在交易时间内...")
        print(f"当前时间: {current_hour:02d}:{current_minute:02d}")
        print(f"交易时间: {start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}")
        print(f"是否在交易时间内: {'是' if is_trading_hours else '否'}")
            
        print("正在获取历史数据...")
        df = get_historical_data(symbol)
        if df.empty:
            print("Error: Could not get historical data")
            sys.exit(1)
            
        # 调试模式下，根据指定时间截断数据
        if DEBUG_MODE:
            # 截断到调试时间之前的数据
            df = df[df["DateTime"] <= now]
            print(f"调试模式: 数据已截断至 {now.strftime('%Y-%m-%d %H:%M:%S')}")
            
        # 打印历史数据详细信息
        latest_date = df["Date"].max()
        earliest_date = df["Date"].min()
        print(f"历史数据日期范围: {earliest_date} 到 {latest_date}")
        
        if not is_trading_hours:
            print(f"当前不在交易时间内 ({trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d})")
            print("已获取历史数据，但不在交易时间内，暂不进行交易")
            if position_quantity != 0:
                print("交易日结束，执行平仓")
                side = "Sell" if position_quantity > 0 else "Buy"
                close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
                print(f"平仓订单已提交，ID: {close_order_id}")
                position_quantity = 0
                print("持仓状态已重置")
            now = get_us_eastern_time()
            today = now.date()
            today_start = datetime.combine(today, time(trading_start_time[0], trading_start_time[1]), tzinfo=now.tzinfo)
            if now < today_start:
                next_check_time = today_start
            else:
                tomorrow = today + timedelta(days=1)
                tomorrow_start = datetime.combine(tomorrow, time(trading_start_time[0], trading_start_time[1]), tzinfo=now.tzinfo)
                next_check_time = tomorrow_start
            wait_seconds = min(1800, (next_check_time - now).total_seconds())
            time_module.sleep(wait_seconds)
            continue
            
        print("在交易时间内，继续处理数据和交易逻辑...")
        print("计算VWAP指标...")
        # 使用新的VWAP计算方法
        df["VWAP"] = calculate_vwap(df)
        
        # 在调试模式下打印当前时间点的数据
        if DEBUG_MODE:
            current_time = now.strftime('%H:%M')
            latest_date = df["Date"].max()
            debug_data = df[(df["Date"] == latest_date) & (df["Time"] == current_time)]
            
            if not debug_data.empty:
                print("\n调试信息 - 当前时间点1分钟K线数据:")
                pd.set_option('display.float_format', '{:.6f}'.format)
                debug_info = debug_data[['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover', 'VWAP']].copy()
                print(debug_info.to_string(index=False))
                
                # 验证VWAP计算
                if debug_data["Volume"].iloc[0] > 0:
                    calc_vwap = debug_data["Turnover"].iloc[0] / debug_data["Volume"].iloc[0]
                    print(f"手动计算VWAP: Turnover({debug_data['Turnover'].iloc[0]}) / Volume({debug_data['Volume'].iloc[0]}) = {calc_vwap:.6f}")
                    print(f"计算的VWAP值: {debug_data['VWAP'].iloc[0]:.6f}")
                    print(f"差异: {abs(calc_vwap - debug_data['VWAP'].iloc[0]):.6f}")
            else:
                print(f"\n调试信息 - 未找到时间点 {current_time} 的数据")
        
        # 直接计算噪声区域，不需要中间复制
        print("计算噪声区域边界...")
        df = calculate_noise_area(df, lookback_days)
        
        if position_quantity != 0:
            print(f"检查退出条件 (当前持仓方向: {'多' if position_quantity > 0 else '空'}, 持仓数量: {abs(position_quantity)}, 追踪止损: {trailing_stop})...")
            exit_signal, new_stop = check_exit_conditions(df, position_quantity, trailing_stop)
            trailing_stop = new_stop
            print(f"更新追踪止损价格: {trailing_stop}")
            if exit_signal:
                print("触发退出信号!")
                side = "Sell" if position_quantity > 0 else "Buy"
                close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
                print(f"平仓订单已提交，ID: {close_order_id}")
                exit_price = df.iloc[-1]["Close"]
                pnl = (exit_price - entry_price) * (1 if position_quantity > 0 else -1) * abs(position_quantity)
                pnl_pct = (exit_price / entry_price - 1) * 100 * (1 if position_quantity > 0 else -1)
                print(f"平仓成功: {side} {abs(position_quantity)} {symbol} 价格: {exit_price}")
                print(f"交易结果: {'盈利' if pnl > 0 else '亏损'} ${abs(pnl):.2f} ({pnl_pct:.2f}%)")
                position_quantity = 0
                print("持仓状态已重置")
        else:
            print(f"检查入场条件 (今日已开仓: {positions_opened_today}/{max_positions_per_day})...")
            
            # 获取价格
            if DEBUG_MODE:
                # 调试模式：直接使用当前时间点的历史价格
                current_time = now.strftime('%H:%M')
                latest_date = df["Date"].max()
                debug_data = df[(df["Date"] == latest_date) & (df["Time"] == current_time)]
                
                if not debug_data.empty:
                    latest_price = float(debug_data["Close"].iloc[0])
                else:
                    latest_price = float(df.iloc[-1]["Close"])
                    
                print(f"调试模式: 使用历史价格 {latest_price}")
            else:
                # 正常模式: 使用API获取实时价格
                quote = get_quote(symbol)
                latest_price = float(quote.get("last_done", df.iloc[-1]["Close"]))
                print(f"获取实时价格: {latest_price}")
            
            latest_date = df["Date"].max()
            latest_data = df[df["Date"] == latest_date].copy()
            if not latest_data.empty:
                latest_row = latest_data.iloc[-1].copy()
                latest_row["Close"] = latest_price
                print("\n开仓条件判断:")
                print(f"当前价格: {latest_price:.2f}")
                print(f"VWAP: {latest_row['VWAP']:.2f}")
                print(f"上边界: {latest_row['UpperBound']:.2f}")
                print(f"下边界: {latest_row['LowerBound']:.2f}")
                long_price_above_upper = latest_price > latest_row["UpperBound"]
                long_price_above_vwap = latest_price > latest_row["VWAP"]
                print("\n多头入场条件:")
                print(f"价格 > 上边界: {long_price_above_upper} ({latest_price:.2f} > {latest_row['UpperBound']:.2f})")
                print(f"价格 > VWAP: {long_price_above_vwap} ({latest_price:.2f} > {latest_row['VWAP']:.2f})")
                signal = 0
                price = latest_price
                stop = None
                if long_price_above_upper and long_price_above_vwap:
                    print("满足多头入场条件!")
                    signal = 1
                    stop = max(latest_row["UpperBound"], latest_row["VWAP"])
                else:
                    short_price_below_lower = latest_price < latest_row["LowerBound"]
                    short_price_below_vwap = latest_price < latest_row["VWAP"]
                    print("\n空头入场条件:")
                    print(f"价格 < 下边界: {short_price_below_lower} ({latest_price:.2f} < {latest_row['LowerBound']:.2f})")
                    print(f"价格 < VWAP: {short_price_below_vwap} ({latest_price:.2f} < {latest_row['VWAP']:.2f})")
                    if short_price_below_lower and short_price_below_vwap:
                        print("满足空头入场条件!")
                        signal = -1
                        stop = min(latest_row["LowerBound"], latest_row["VWAP"])
                    else:
                        print("不满足任何入场条件")
                if signal != 0:
                    print(f"触发{'多' if signal == 1 else '空'}头入场信号! 价格: {price}, 止损: {stop}")
                    available_capital = get_account_balance()
                    position_size = floor(available_capital / latest_price)
                    if position_size <= 0:
                        print("Warning: Insufficient capital for position")
                        sys.exit(1)
                    side = "Buy" if signal > 0 else "Sell"
                    order_id = submit_order(symbol, side, position_size, outside_rth=outside_rth_setting)
                    print(f"订单已提交，ID: {order_id}")
                    order_info = get_order_status(order_id)
                    status = order_info.get("status", "")
                    if "FilledStatus" in status or "PartialFilledStatus" in status:
                        executed_quantity = int(float(order_info.get("executed_quantity", 0)))
                        if executed_quantity > 0:
                            position_quantity = executed_quantity if signal > 0 else -executed_quantity
                            if order_info.get("executed_price") and float(order_info.get("executed_price")) > 0:
                                entry_price = float(order_info.get("executed_price"))
                            else:
                                entry_price = latest_price
                            positions_opened_today += 1
                            print(f"开仓成功: {side} {executed_quantity} {symbol} 价格: {entry_price}")
                    elif "RejectedStatus" in status or "CanceledStatus" in status or "ExpiredStatus" in status or "FailedStatus" in status:
                        print(f"订单未成功: {status}")
        
        # 调试模式且单次运行模式，完成一次循环后退出
        if DEBUG_MODE and DEBUG_ONCE:
            print("\n调试模式单次运行完成，程序退出")
            break
            
        next_check_time = now + timedelta(minutes=check_interval_minutes)
        sleep_seconds = (next_check_time - now).total_seconds()
        if sleep_seconds > 0:
            time_module.sleep(sleep_seconds)
            print(f"等待结束，开始下一次检查 - {get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    print("\n" + "*"*70)
    print("* 长桥API交易策略启动")
    print("* 版本: 1.0.0")
    print("* 时间:", get_us_eastern_time().strftime("%Y-%m-%d %H:%M:%S"), "(美东时间)")
    if DEBUG_MODE:
        print("* 调试模式已开启")
        if DEBUG_TIME:
            print(f"* 调试时间: {DEBUG_TIME}")
        if DEBUG_ONCE:
            print("* 单次运行模式已开启")
    print("*"*70 + "\n")
    
    if QUOTE_CTX is None or TRADE_CTX is None:
        print("错误: 无法创建API上下文")
        sys.exit(1)
        
    run_trading_strategy(
        symbol=SYMBOL,
        check_interval_minutes=CHECK_INTERVAL_MINUTES,
        trading_start_time=TRADING_START_TIME,
        trading_end_time=TRADING_END_TIME,
        max_positions_per_day=MAX_POSITIONS_PER_DAY,
        lookback_days=LOOKBACK_DAYS
    )
    