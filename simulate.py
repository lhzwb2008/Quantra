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
LOOKBACK_DAYS = 2
LEVERAGE = 1.5 # 杠杆倍数，默认为1倍
K1 = 1.2 # 上边界sigma乘数
K2 = 1.2 # 下边界sigma乘数

# 默认交易品种
SYMBOL = os.environ.get('SYMBOL', 'TQQQ.US')

# 调试模式配置
DEBUG_MODE = False   # 设置为True开启调试模式
DEBUG_TIME = "2025-05-15 12:36:00"  # 调试使用的时间，格式: "YYYY-MM-DD HH:MM:SS"
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
    print(f"\n=== 开始获取历史数据 ===")

    # 简化天数计算逻辑
    if days_back is None:
        days_back = LOOKBACK_DAYS + 10  # 简化为固定天数
    print(f"回溯天数: {days_back}")
        
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
    
    # print(f"总共获取到 {len(all_candles)} 条原始K线数据")
    
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
    
    # print(f"处理后数据行数: {len(data)}")
    
    # 转换为DataFrame并进行后处理
    df = pd.DataFrame(data)
    if df.empty:
        print("警告: 转换后的DataFrame为空")
        return df
        
    df["Date"] = df["DateTime"].dt.date
    df["Time"] = df["DateTime"].dt.strftime('%H:%M')
    
    # 过滤交易时间
    rows_before = len(df)
    if symbol.endswith(".US"):
        df = df[df["Time"].between("09:30", "16:00")]
    # print(f"过滤交易时间后行数: {len(df)} (过滤掉 {rows_before - len(df)} 行)")
        
    # 去除重复数据
    rows_before = len(df)
    df = df.drop_duplicates(subset=['Date', 'Time'])
    # print(f"去重后行数: {len(df)} (过滤掉 {rows_before - len(df)} 行)")
    
    # 过滤掉未来日期的数据（双重保险）
    rows_before = len(df)
    df = df[df["Date"] <= current_date]
    # print(f"过滤未来日期后行数: {len(df)} (过滤掉 {rows_before - len(df)} 行)")
    
    # 添加日志，打印历史数据信息
    trading_days_count = len(df["Date"].unique())
    total_rows = len(df)
    date_time_counts = df.groupby(['Date', 'Time']).size()
    min_time = df["Time"].min() if not df.empty else "N/A"
    max_time = df["Time"].max() if not df.empty else "N/A"
    
    print(f"获取到的历史数据: 共{trading_days_count}个交易日, {total_rows}行数据")
    print(f"当前日期数据行数: {len(df[df['Date'] == current_date])}")
    print("=== 历史数据获取完成 ===\n")
    
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
    # 创建一个结果DataFrame的副本
    result_df = df.copy()
    
    # 按照日期分组
    for date in result_df['Date'].unique():
        # 获取当日数据
        day_data = result_df[result_df['Date'] == date]
        
        # 按时间排序确保正确累计
        day_data = day_data.sort_values('Time')
        
        # 计算累计成交量和成交额
        cumulative_volume = day_data['Volume'].cumsum()
        cumulative_turnover = day_data['Turnover'].cumsum()
        
        # 计算VWAP: 累计成交额 / 累计成交量
        vwap = cumulative_turnover / cumulative_volume
        # 处理成交量为0的情况
        vwap = vwap.fillna(day_data['Close'])
        
        # 更新结果DataFrame中的对应行
        result_df.loc[result_df['Date'] == date, 'VWAP'] = vwap.values
    
    return result_df['VWAP']

def calculate_noise_area(df, lookback_days=LOOKBACK_DAYS, K1=1, K2=1):
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
    
    # 假设最后一天是当前交易日，直接排除
    if len(unique_dates) > 1:
        target_date = unique_dates[-1]  # 保存目标日期（当前交易日）
        history_dates = unique_dates[:-1]  # 排除最后一天
        
        # 从剩余日期中选择最近的lookback_days天
        history_dates = history_dates[-lookback_days:] if len(history_dates) >= lookback_days else history_dates
        
        if DEBUG_MODE:
            print(f"目标日期（当前交易日）: {target_date}")
            print(f"历史数据日期数量: {len(history_dates)}")
            if len(history_dates) > 0:
                print(f"历史数据日期范围: {history_dates[0]} 到 {history_dates[-1]}")
    else:
        print(f"错误: 数据中只有一天或没有数据，无法计算噪声空间")
        sys.exit(1)
    
    # 检查数据是否足够
    if len(history_dates) < lookback_days:
        print(f"错误: 历史数据不足，至少需要{lookback_days}个交易日，当前只有{len(history_dates)}个交易日")
        if DEBUG_MODE:
            print(f"可用历史交易日: {history_dates}")
        sys.exit(1)
    
    # 为历史日期计算当日开盘价和相对变动率
    history_df = df_copy[df_copy["Date"].isin(history_dates)].copy()
    
    # 为每个历史日期计算当日开盘价
    day_opens = {}
    for date in history_dates:
        day_data = history_df[history_df["Date"] == date]
        if day_data.empty:
            print(f"错误: {date} 日期数据为空")
            sys.exit(1)
        day_opens[date] = day_data["Open"].iloc[0]
    
    # 为每个时间点计算相对于开盘价的绝对变动率
    history_df["move"] = 0.0
    for date in history_dates:
        day_open = day_opens[date]
        history_df.loc[history_df["Date"] == date, "move"] = abs(history_df.loc[history_df["Date"] == date, "Close"] / day_open - 1)
    
    # 计算每个时间点的sigma (使用历史数据)
    time_sigma = {}
    
    # 获取目标日期的所有时间点
    target_day_data = df[df["Date"] == target_date]
    times = target_day_data["Time"].unique()
    
    # 对每个时间点计算sigma
    for tm in times:
        # 获取历史数据中相同时间点的数据
        historical_moves = []
        for date in history_dates:
            hist_data = history_df[(history_df["Date"] == date) & (history_df["Time"] == tm)]
            if not hist_data.empty:
                historical_moves.append(hist_data["move"].iloc[0])
        
        # 确保有足够的历史数据计算sigma
        if len(historical_moves) == 0:
            if DEBUG_MODE:
                print(f"时间点 {tm} 没有足够的历史数据")
            continue
        
        # 计算平均变动率作为sigma
        sigma = sum(historical_moves) / len(historical_moves)
        time_sigma[(target_date, tm)] = sigma
    
    # 计算上下边界
    # 获取目标日期的开盘价
    target_day_data = df[df["Date"] == target_date]
    if target_day_data.empty:
        print(f"错误: 目标日期 {target_date} 数据为空")
        sys.exit(1)
    
    day_open = target_day_data["Open"].iloc[0]
    
    # 获取目标日期的前一日收盘价
    if target_date in unique_dates and unique_dates.index(target_date) > 0:
        prev_date = unique_dates[unique_dates.index(target_date) - 1]
        prev_day_data = df[df["Date"] == prev_date]
        if not prev_day_data.empty:
            prev_close = prev_day_data["Close"].iloc[-1]
        else:
            prev_close = None
    else:
        prev_close = None
    
    if DEBUG_MODE:
        print(f"\n目标日期 {target_date} 的边界计算:")
        print(f"当日开盘价: {day_open}")
        print(f"前一日收盘价: {prev_close}")
        times_with_sigma = [tm for (dt, tm) in time_sigma.keys() if dt == target_date]
        print(f"有sigma值的时间点数量: {len(times_with_sigma)}")
        if len(times_with_sigma) > 0:
            print(f"时间点示例: {times_with_sigma[:5] if len(times_with_sigma) >= 5 else times_with_sigma}")
    
    if prev_close is None:
        if DEBUG_MODE:
            print(f"日期 {target_date} 没有前一日收盘价")
        return df
    
    # 根据算法计算参考价格
    upper_ref = max(day_open, prev_close)
    lower_ref = min(day_open, prev_close)
    
    if DEBUG_MODE:
        print(f"上界参考价格选择: max({day_open}, {prev_close}) = {upper_ref}")
        print(f"下界参考价格选择: min({day_open}, {prev_close}) = {lower_ref}")
    
    # 对目标日期的每个时间点计算上下边界
    bounds_count = 0
    
    # 使用目标日期的数据
    for _, row in target_day_data.iterrows():
        tm = row["Time"]
        sigma = time_sigma.get((target_date, tm))
        
        if sigma is not None:
            # 使用时间点特定的sigma计算上下边界，应用K1和K2乘数
            upper_bound = upper_ref * (1 + K1 * sigma)
            lower_bound = lower_ref * (1 - K2 * sigma)
            
            # 更新df中的边界值
            df.loc[(df["Date"] == target_date) & (df["Time"] == tm), "UpperBound"] = upper_bound
            df.loc[(df["Date"] == target_date) & (df["Time"] == tm), "LowerBound"] = lower_bound
            bounds_count += 1
            
            if DEBUG_MODE and target_date == current_date and tm == now_et.strftime('%H:%M'):
                print(f"\n当前时间点 {tm} 的边界计算详情:")
                print(f"Sigma值: {sigma:.6f}")
                print(f"K1(上边界乘数): {K1}")
                print(f"K2(下边界乘数): {K2}")
                print(f"上界计算: {upper_ref} * (1 + {K1} * {sigma:.6f}) = {upper_bound:.6f}")
                print(f"下界计算: {lower_ref} * (1 - {K2} * {sigma:.6f}) = {lower_bound:.6f}")
    
    # 在调试模式下检查目标日期的边界数据
    if DEBUG_MODE:
        print(f"\n边界数据统计:")
        print(f"总计算的边界点数: {bounds_count}")
        
        target_data = df[df["Date"] == target_date]
        has_upper = target_data["UpperBound"].notna().sum()
        has_lower = target_data["LowerBound"].notna().sum()
        total_rows = len(target_data)
        
        print(f"目标日期 {target_date} 数据行数: {total_rows}")
        if total_rows > 0:
            print(f"有上边界值的行数: {has_upper}/{total_rows} ({has_upper/total_rows*100:.2f}%)")
            print(f"有下边界值的行数: {has_lower}/{total_rows} ({has_lower/total_rows*100:.2f}%)")
        
        # 检查当前测试时间点是否有边界值
        if target_date == current_date:
            test_time = now_et.strftime('%H:%M')
            test_data = df[(df["Date"] == current_date) & (df["Time"] == test_time)]
            if not test_data.empty:
                has_test_upper = test_data["UpperBound"].notna().all()
                has_test_lower = test_data["LowerBound"].notna().all()
                print(f"当前测试时间点 {test_time} 是否有上边界: {has_test_upper}")
                print(f"当前测试时间点 {test_time} 是否有下边界: {has_test_lower}")
                if has_test_upper and has_test_lower:
                    print(f"当前测试时间点 {test_time} 的上边界值: {test_data['UpperBound'].iloc[0]:.6f}")
                    print(f"当前测试时间点 {test_time} 的下边界值: {test_data['LowerBound'].iloc[0]:.6f}")
            else:
                print(f"当前测试时间点 {test_time} 没有找到数据")
    
    return df

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
    # 获取当前时间点
    now = get_us_eastern_time()
    current_time = now.strftime('%H:%M')
    current_date = now.date()
    
    print(f"\n=== 检查退出条件 ===")
    print(f"当前日期时间: {current_date} {current_time}")
    print(f"数据集行数: {len(df)}")
    print(f"数据集日期范围: {df['Date'].min()} 到 {df['Date'].max()}")
    
    # 精简日志，直接获取当前时间点数据
    current_data = df[(df["Date"] == current_date) & (df["Time"] == current_time)]
    
    # 如果当前时间点没有数据，使用最新数据
    if current_data.empty:
        print(f"警告: 当前时间点 {current_time} 没有数据，使用最新数据")
        # 按日期和时间排序，获取最新的数据
        df_sorted = df.sort_values(by=["Date", "Time"], ascending=True)
        latest = df_sorted.iloc[-1]
        print(f"最新数据日期时间: {latest['Date']} {latest['Time']}")
    else:
        latest = current_data.iloc[0]
        print(f"使用当前时间点数据: {latest['Date']} {latest['Time']}")
        
    price = latest["Close"]
    vwap = latest["VWAP"]
    upper = latest["UpperBound"]
    lower = latest["LowerBound"]
    
    # 检查数据是否为空值
    if price is None:
        print("警告: 价格数据为空，无法检查退出条件")
        return False, trailing_stop
    
    # 打印数据情况帮助调试
    print(f"退出条件检查数据: 价格={price}, VWAP={vwap}, 上边界={upper}, 下边界={lower}")
    print(f"=== 检查退出条件结束 ===\n")
    
    if position_quantity > 0:
        # 检查上边界或VWAP是否为None
        if upper is None or vwap is None:
            print("错误: 上边界或VWAP数据为空")
            sys.exit(1)  # 直接退出程序
        else:
            new_stop = max(upper, vwap)
            
        if trailing_stop is not None:
            new_stop = max(trailing_stop, new_stop)
        exit_signal = price < new_stop
        return exit_signal, new_stop
    elif position_quantity < 0:
        # 检查下边界或VWAP是否为None
        if lower is None or vwap is None:
            print("错误: 下边界或VWAP数据为空")
            sys.exit(1)  # 直接退出程序
        else:
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
    
    # 获取当前实际持仓
    current_positions = get_current_positions()
    symbol_position = current_positions.get(symbol, {"quantity": 0, "cost_price": 0})
    position_quantity = symbol_position["quantity"]
    
    # 初始化入场价格为None，后续由交易操作更新
    entry_price = None
    
    print(f"当前持仓: {symbol} 数量: {position_quantity}, 成本价: {entry_price}")
    
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
        
        # 每次循环都更新当前持仓状态
        current_positions = get_current_positions()
        symbol_position = current_positions.get(symbol, {"quantity": 0, "cost_price": 0})
        position_quantity = symbol_position["quantity"]
        
        # 如果持仓量变为0，重置入场价格
        if position_quantity == 0:
            entry_price = None
            
        print(f"当前持仓: {symbol} 数量: {position_quantity}, 成本价: {entry_price}")
        
        # 检查是否是交易时间结束点，如果是且有持仓，则强制平仓
        current_hour, current_minute = now.hour, now.minute
        is_trading_end = current_hour == trading_end_time[0] and current_minute == trading_end_time[1]
        if is_trading_end and position_quantity != 0:
            print(f"当前时间为交易结束时间 {trading_end_time[0]}:{trading_end_time[1]}，执行平仓")
            
            # 获取历史数据
            print("\n=== 开始获取平仓价格数据 ===")
            df = get_historical_data(symbol)
            if df.empty:
                print("错误: 获取历史数据为空")
                sys.exit(1)
                
            print(f"获取到的历史数据行数: {len(df)}")
            print(f"历史数据日期范围: {df['Date'].min()} 到 {df['Date'].max()}")
            print(f"历史数据时间范围: {df['Time'].min()} 到 {df['Time'].max()}")
                
            if DEBUG_MODE:
                df = df[df["DateTime"] <= now]
                print(f"调试模式: 数据已截断至 {now.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"截断后数据行数: {len(df)}")
                
            # 获取当前时间点的价格数据
            current_time = now.strftime('%H:%M')
            print(f"当前日期: {current_date}, 当前时间点: {current_time}")
            
            # 简化日志，直接检查当前时间点数据
            current_data = df[(df["Date"] == current_date) & (df["Time"] == current_time)]
            
            if current_data.empty:
                print(f"错误: 当前时间点 {current_time} 没有数据，无法获取正确价格进行平仓")
                # 打印最后几行数据用于调试
                if not df.empty:
                    print("最后5行数据:")
                    print(df.tail(5)[["Date", "Time", "Close", "VWAP"]])
                sys.exit(1)
                
            # 使用当前时间点的价格
            current_price = float(current_data["Close"].iloc[0])
            print(f"使用当前时间点 {current_time} 的价格: {current_price}")
            print("=== 价格数据获取完成 ===\n")
            
            # 执行平仓
            side = "Sell" if position_quantity > 0 else "Buy"
            close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
            print(f"平仓订单已提交，ID: {close_order_id}")
            
            # 计算盈亏
            if entry_price:
                pnl = (current_price - entry_price) * (1 if position_quantity > 0 else -1) * abs(position_quantity)
                pnl_pct = (current_price / entry_price - 1) * 100 * (1 if position_quantity > 0 else -1)
                print(f"平仓成功: {side} {abs(position_quantity)} {symbol} 价格: {current_price}")
                print(f"交易结果: {'盈利' if pnl > 0 else '亏损'} ${abs(pnl):.2f} ({pnl_pct:.2f}%)")
            else:
                print(f"平仓成功: {side} {abs(position_quantity)} {symbol} 价格: {current_price}")
                
            position_quantity = 0
            entry_price = None
            print("持仓状态已重置")
            if DEBUG_MODE and DEBUG_ONCE:
                print("\n调试模式单次运行完成，程序退出")
                break
            continue
        
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
                entry_price = None
                print("持仓状态已重置")
            next_check_time = now + timedelta(hours=12)
            wait_seconds = (next_check_time - now).total_seconds()
            time_module.sleep(wait_seconds)
            continue
            
        # 检查是否是新交易日，如果是则重置今日开仓计数
        if last_date is not None and current_date != last_date:
            print(f"新的交易日开始，重置今日开仓计数")
            positions_opened_today = 0
        last_date = current_date
        
        # 保持原有交易时间检查逻辑
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
                entry_price = None
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
            else:
                print(f"\n调试信息 - 未找到时间点 {current_time} 的数据")
        
        # 直接计算噪声区域，不需要中间复制
        print("计算噪声区域边界...")
        print(f"使用上边界乘数K1={K1}，下边界乘数K2={K2}")
        df = calculate_noise_area(df, lookback_days, K1, K2)
        
        if position_quantity != 0:
            print(f"检查退出条件 (当前持仓方向: {'多' if position_quantity > 0 else '空'}, 持仓数量: {abs(position_quantity)}, 追踪止损: {trailing_stop})...")
            print(f"当前价格: {latest_price:.2f}")
            print(f"VWAP: {latest_row['VWAP']:.2f}")
            print(f"上边界: {latest_row['UpperBound']:.2f}")
            print(f"下边界: {latest_row['LowerBound']:.2f}")
            exit_signal, new_stop = check_exit_conditions(df, position_quantity, trailing_stop)
            trailing_stop = new_stop
            print(f"更新追踪止损价格: {trailing_stop}")
            if exit_signal:
                print("触发退出信号!")
                
                # 确保使用当前时间点的价格数据
                current_time = now.strftime('%H:%M')
                current_data = df[(df["Date"] == current_date) & (df["Time"] == current_time)]
                
                if current_data.empty:
                    print(f"错误: 当前时间点 {current_time} 没有数据，无法获取正确价格进行平仓")
                    sys.exit(1)
                    
                # 使用当前时间点的价格
                exit_price = float(current_data["Close"].iloc[0])
                print(f"使用当前时间点 {current_time} 的价格: {exit_price}")
                
                # 执行平仓
                side = "Sell" if position_quantity > 0 else "Buy"
                close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
                print(f"平仓订单已提交，ID: {close_order_id}")
                
                # 计算盈亏
                if entry_price:
                    pnl = (exit_price - entry_price) * (1 if position_quantity > 0 else -1) * abs(position_quantity)
                    pnl_pct = (exit_price / entry_price - 1) * 100 * (1 if position_quantity > 0 else -1)
                    print(f"平仓成功: {side} {abs(position_quantity)} {symbol} 价格: {exit_price}")
                    print(f"交易结果: {'盈利' if pnl > 0 else '亏损'} ${abs(pnl):.2f} ({pnl_pct:.2f}%)")
                
                # 平仓后增加交易次数计数器
                positions_opened_today += 1
                print(f"更新今日交易次数: {positions_opened_today}/{max_positions_per_day}")
                
                position_quantity = 0
                entry_price = None
                print("持仓状态已重置")
        else:
            print(f"检查入场条件 (今日已交易: {positions_opened_today}/{max_positions_per_day})...")
            
            # 检查是否已有持仓，如果有则不再开仓
            if position_quantity != 0:
                print(f"已有持仓 {abs(position_quantity)} 份，跳过开仓检查")
                continue
                
            # 检查今日是否达到最大持仓数
            if positions_opened_today >= max_positions_per_day:
                print(f"今日已达到最大持仓数 {max_positions_per_day}，跳过开仓检查")
                continue
            
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
                print(f"当前价格: {latest_price:.2f}")
                print(f"VWAP: {latest_row['VWAP']:.2f}")
                print(f"上边界: {latest_row['UpperBound']:.2f}")
                print(f"下边界: {latest_row['LowerBound']:.2f}")
                long_price_above_upper = latest_price > latest_row["UpperBound"]
                long_price_above_vwap = latest_price > latest_row["VWAP"]
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
                    if short_price_below_lower and short_price_below_vwap:
                        print("满足空头入场条件!")
                        signal = -1
                        stop = min(latest_row["LowerBound"], latest_row["VWAP"])
                    else:
                        print("不满足任何入场条件")
                if signal != 0:
                    print(f"触发{'多' if signal == 1 else '空'}头入场信号! 价格: {price}, 止损: {stop}")
                    available_capital = get_account_balance()
                    # 应用杠杆比例
                    adjusted_capital = available_capital * LEVERAGE
                    position_size = floor(adjusted_capital / latest_price)
                    if position_size <= 0:
                        print("Warning: Insufficient capital for position")
                        sys.exit(1)
                    print(f"可用资金: ${available_capital:.2f}, 杠杆比例: {LEVERAGE}倍, 调整后资金: ${adjusted_capital:.2f}")
                    print(f"开仓数量: {position_size} 股")
                    side = "Buy" if signal > 0 else "Sell"
                    order_id = submit_order(symbol, side, position_size, outside_rth=outside_rth_setting)
                    print(f"订单已提交，ID: {order_id}")
                    
                    # 删除订单状态检查代码，直接更新持仓状态
                    position_quantity = position_size if signal > 0 else -position_size
                    entry_price = latest_price
                    print(f"记录成本价: {entry_price}")
                    print(f"开仓成功: {side} {position_size} {symbol} 价格: {entry_price}")
        
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
    print(f"* 杠杆倍数: {LEVERAGE}倍")
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
