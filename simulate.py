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
from functools import wraps

from longport.openapi import Config, TradeContext, QuoteContext, Period, OrderSide, OrderType, TimeInForceType, AdjustType, OutsideRTH

load_dotenv(override=True)

# 固定配置参数
CHECK_INTERVAL_MINUTES = 15
TRADING_START_TIME = (9, 40)  # 交易开始时间：9点40分
TRADING_END_TIME = (15, 40)   # 交易结束时间：15点40分
MAX_POSITIONS_PER_DAY = 10
LOOKBACK_DAYS = 1
LEVERAGE = 2.8 # 杠杆倍数，默认为1倍
K1 = 1 # 上边界sigma乘数
K2 = 1 # 下边界sigma乘数

# VWAP开关：False=不使用VWAP作为入场/止损条件，True=使用VWAP
USE_VWAP = False

# 🎯 动态追踪止盈配置
ENABLE_TRAILING_TAKE_PROFIT = True   # 是否启用动态追踪止盈
TRAILING_TP_ACTIVATION_PCT = 0.01    # 激活追踪止盈的最低浮盈百分比（1%）
TRAILING_TP_CALLBACK_PCT = 0.7       # 保护的利润比例（70%），即从最大浮盈回撤30%时触发止盈

# 默认交易品种
SYMBOL = os.environ.get('SYMBOL', 'QQQ.US')

# 调试模式配置
DEBUG_MODE = True   # 设置为True开启调试模式

# 收益统计全局变量
TOTAL_PNL = 0.0  # 总收益
DAILY_PNL = 0.0  # 当日收益
LAST_STATS_DATE = None  # 上次统计日期
DAILY_TRADES = []  # 当日交易记录

def api_retry(max_retries=3, retry_delay=2):
    """API调用重试装饰器
    
    Args:
        max_retries: 最大重试次数，默认3次
        retry_delay: 重试间隔（秒），默认2秒
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    if attempt < max_retries - 1:
                        if DEBUG_MODE:
                            print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] {func.__name__} 调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                            print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] {retry_delay}秒后重试...")
                        time_module.sleep(retry_delay)
                    else:
                        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] {func.__name__} 调用失败，已达最大重试次数: {error_msg}")
                        raise
        return wrapper
    return decorator

def get_us_eastern_time():
    # 正常模式返回当前时间
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

def create_contexts():
    max_retries = 5
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        try:
            config = Config.from_env()
            quote_ctx = QuoteContext(config)
            trade_ctx = TradeContext(config)
            if DEBUG_MODE:
                print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] API连接成功")
            return quote_ctx, trade_ctx
        except Exception as e:
            if attempt < max_retries - 1:
                if DEBUG_MODE:
                    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] API连接失败 ({attempt + 1}/{max_retries}): {str(e)}")
                    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] {retry_delay}秒后重试...")
                time_module.sleep(retry_delay)
            else:
                print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] API连接失败，已达最大重试次数")
                raise

QUOTE_CTX, TRADE_CTX = create_contexts()

@api_retry(max_retries=10, retry_delay=2)
def get_account_balance():
    if DEBUG_MODE:
        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 获取美元账户余额")
    balance_list = TRADE_CTX.account_balance()  # 不需要指定currency参数
    
    # 从cash_infos中找到USD的可用现金
    usd_available_cash = 0.0
    for balance_info in balance_list:
        for cash_info in balance_info.cash_infos:
            if cash_info.currency == "USD":
                usd_available_cash = float(cash_info.available_cash)
                if DEBUG_MODE:
                    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 美元可用现金: ${usd_available_cash:.2f}")
                return usd_available_cash
    
    # 如果没有找到USD账户，返回0
    if DEBUG_MODE:
        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 警告: 未找到美元账户，返回余额为0")
    return 0.0

@api_retry(max_retries=10, retry_delay=2)
def get_current_positions():
    if DEBUG_MODE:
        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 获取当前持仓")
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

@api_retry(max_retries=10, retry_delay=2)
def get_historical_data(symbol, days_back=None):
    # 简化天数计算逻辑
    if days_back is None:
        days_back = LOOKBACK_DAYS + 5  # 简化为固定天数
        
    # 直接使用1分钟K线
    sdk_period = Period.Min_1
    adjust_type = AdjustType.ForwardAdjust
    eastern = pytz.timezone('US/Eastern')
    now_et = get_us_eastern_time()
    current_date = now_et.date()
    
    if DEBUG_MODE:
        print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] 开始获取历史数据: {symbol}")
    
    # 计算起始日期
    start_date = current_date - timedelta(days=days_back)
    
    # 对于1分钟数据使用按日获取的方式
    all_candles = []
    
    # 尝试从今天开始向前获取足够的数据
    date_to_check = current_date
    api_call_count = 0
    while date_to_check >= start_date:
        # 跳过周末（周六=5, 周日=6）
        if date_to_check.weekday() >= 5:
            if DEBUG_MODE:
                print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] 跳过周末: {date_to_check}")
            date_to_check -= timedelta(days=1)
            continue
            
        day_start_time = datetime.combine(date_to_check, time(9, 30))
        day_start_time_et = eastern.localize(day_start_time)
        
        # 添加API调用间隔控制
        if api_call_count > 0:
            time_module.sleep(0.2)  # 200毫秒延迟，避免触发限流
        
        # 重试机制
        max_retries = 3
        retry_delay = 1
        day_candles = None
        
        for attempt in range(max_retries):
            try:
                # 使用history_candlesticks_by_date方法（与backtest数据源一致）
                # 这个方法返回的是完整交易日的数据，避免了by_offset方法可能的日期错误
                day_candles = QUOTE_CTX.history_candlesticks_by_date(
                    symbol, sdk_period, adjust_type,
                    date_to_check,  # 开始日期
                    date_to_check   # 结束日期（同一天）
                )
                api_call_count += 1
                break  # 成功则跳出重试循环
            except Exception as e:
                if "rate limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        if DEBUG_MODE:
                            print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] API限流，等待 {retry_delay} 秒后重试 ({attempt + 1}/{max_retries})")
                        time_module.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        if DEBUG_MODE:
                            print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] API限流，已达最大重试次数")
                        raise
                else:
                    raise  # 其他错误直接抛出
        
        if day_candles:
            all_candles.extend(day_candles)
            if DEBUG_MODE:
                print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] 获取 {date_to_check} 数据: {len(day_candles)} 条")
            
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
    
    # 过滤周末数据（双重保险）
    weekday_mask = df["Date"].apply(lambda x: x.weekday() < 5 if isinstance(x, date_type) else True)
    df = df[weekday_mask]
    
    if DEBUG_MODE and not df.empty:
        unique_dates = sorted(df["Date"].unique())
        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 最终数据包含的日期: {unique_dates}")
    
    return df

@api_retry(max_retries=10, retry_delay=2)
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

def get_daily_klines(symbol, start_date, end_date):
    """
    获取指定日期范围的日K线数据
    返回: {日期字符串: {Open, High, Low, Close, Volume, Turnover}}
    """
    try:
        resp = QUOTE_CTX.history_candlesticks_by_date(
            symbol,
            Period.Day,
            AdjustType.NoAdjust,
            start_date,
            end_date
        )
        
        # 转换为字典，key为日期字符串
        daily_data = {}
        eastern = pytz.timezone('US/Eastern')
        hk_tz = pytz.timezone('Asia/Hong_Kong')
        
        for candle in resp:
            # API返回的timestamp是香港本地的naive时间
            dt_hk = hk_tz.localize(candle.timestamp)
            # 转到美东
            dt_et = dt_hk.astimezone(eastern)
            date_str = dt_et.strftime('%Y-%m-%d')
            
            daily_data[date_str] = {
                'Open': candle.open,
                'High': candle.high,
                'Low': candle.low,
                'Close': candle.close,
                'Volume': candle.volume,
                'Turnover': candle.turnover
            }
        
        return daily_data
    except Exception as e:
        print(f"获取日K线数据失败: {e}")
        return {}


def calculate_noise_area(df, lookback_days=LOOKBACK_DAYS, K1=1, K2=1):
    # 创建数据副本
    df_copy = df.copy()
    
    # 获取唯一日期并排序
    unique_dates = sorted(df_copy["Date"].unique())
    now_et = get_us_eastern_time()
    current_date = now_et.date()
    
    # 过滤未来日期
    if unique_dates and isinstance(unique_dates[0], date_type):
        unique_dates = [d for d in unique_dates if d <= current_date]
        df_copy = df_copy[df_copy["Date"].isin(unique_dates)]
    
    # 过滤周末数据：只保留周一到周五的数据
    weekday_dates = []
    for d in unique_dates:
        if isinstance(d, date_type):
            # weekday(): 0=Monday, 1=Tuesday, ..., 6=Sunday
            if d.weekday() < 5:  # 0-4 表示周一到周五
                weekday_dates.append(d)
        else:
            weekday_dates.append(d)  # 如果不是date类型，保留原样
    
    unique_dates = weekday_dates
    df_copy = df_copy[df_copy["Date"].isin(unique_dates)]
    
    if DEBUG_MODE:
        print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] 过滤周末后的日期数量: {len(unique_dates)}")
        if len(unique_dates) > 0:
            print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] 最近的交易日: {unique_dates[-5:]}")
    
    # 假设最后一天是当前交易日，直接排除
    if len(unique_dates) > 1:
        target_date = unique_dates[-1]  # 保存目标日期（当前交易日）
        history_dates = unique_dates[:-1]  # 排除最后一天
        
        # 从剩余日期中选择最近的lookback_days天
        history_dates = history_dates[-lookback_days:] if len(history_dates) >= lookback_days else history_dates
    else:
        print(f"错误: 数据中只有一天或没有数据，无法计算噪声空间")
        sys.exit(1)
    
    # 检查数据是否足够
    if len(history_dates) < lookback_days:
        print(f"错误: 历史数据不足，至少需要{lookback_days}个交易日，当前只有{len(history_dates)}个交易日")
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
    
    # 获取API层面的日K线数据进行对比
    # 获取当天和前一天的日K线
    if len(unique_dates) >= 2:
        # 获取最近两个交易日的日K线数据
        query_start_date = unique_dates[-2]
        query_end_date = unique_dates[-1]
        daily_klines = get_daily_klines(SYMBOL, query_start_date, query_end_date)
    else:
        daily_klines = {}
    
    # 使用指定时间点的K线数据
    # 获取当日09:30的开盘价
    day_0930_data = target_day_data[target_day_data["Time"] == "09:30"]
    
    # 获取分钟K线拼接的"日K"（第一根分钟K线的Open）
    minute_daily_open = target_day_data.iloc[0]["Open"]
    minute_first_time = target_day_data.iloc[0]["Time"]
    
    # 获取真实日K线的Open
    target_date_str = target_date.strftime('%Y-%m-%d') if isinstance(target_date, date_type) else str(target_date)
    real_daily_open = daily_klines.get(target_date_str, {}).get('Open', None)
    
    if not day_0930_data.empty:
        day_open = day_0930_data["Open"].iloc[0]
        day_0930_close = day_0930_data["Close"].iloc[0]
        if real_daily_open is not None:
            print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 09:30分钟K Open={day_open:.2f}, Close={day_0930_close:.2f} | 拼接日K Open({minute_first_time})={minute_daily_open:.2f} | API日K Open={real_daily_open:.2f}")
        else:
            print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 09:30分钟K Open={day_open:.2f}, Close={day_0930_close:.2f} | 拼接日K Open({minute_first_time})={minute_daily_open:.2f} | API日K Open=未获取")
    else:
        # 如果没有09:30数据，回退到第一根K线
        day_open = target_day_data["Open"].iloc[0]
        first_time = target_day_data.iloc[0]["Time"]
        if real_daily_open is not None:
            print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 未找到09:30数据，使用{first_time}开盘价: {day_open:.2f} | 拼接日K Open={minute_daily_open:.2f} | API日K Open={real_daily_open:.2f}")
        else:
            print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 未找到09:30数据，使用{first_time}开盘价: {day_open:.2f} | 拼接日K Open={minute_daily_open:.2f} | API日K Open=未获取")
    
    # 获取前一日15:59的收盘价
    if target_date in unique_dates and unique_dates.index(target_date) > 0:
        prev_date = unique_dates[unique_dates.index(target_date) - 1]
        prev_day_data = df[df["Date"] == prev_date]
        if not prev_day_data.empty:
            # 获取分钟K线拼接的"日K"收盘价（最后一根分钟K线的Close）
            minute_daily_close = prev_day_data.iloc[-1]["Close"]
            minute_last_time = prev_day_data.iloc[-1]["Time"]
            
            # 获取真实日K线的Close
            prev_date_str = prev_date.strftime('%Y-%m-%d') if isinstance(prev_date, date_type) else str(prev_date)
            real_daily_close = daily_klines.get(prev_date_str, {}).get('Close', None)
            
            # 尝试获取15:59的收盘价
            prev_1559_data = prev_day_data[prev_day_data["Time"] == "15:59"]
            if not prev_1559_data.empty:
                prev_close = prev_1559_data["Close"].iloc[0]
                if real_daily_close is not None:
                    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 前日({prev_date}) 15:59分钟K={prev_close:.2f} | 拼接日K Close({minute_last_time})={minute_daily_close:.2f} | API日K Close={real_daily_close:.2f}")
                else:
                    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 前日({prev_date}) 15:59分钟K={prev_close:.2f} | 拼接日K Close({minute_last_time})={minute_daily_close:.2f} | API日K Close=未获取")
            else:
                # 如果没有15:59数据，回退到最后一根K线
                prev_close = prev_day_data["Close"].iloc[-1]
                last_time = prev_day_data.iloc[-1]["Time"]
                if real_daily_close is not None:
                    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 前日({prev_date})收盘(使用{last_time}): {prev_close:.2f} | 拼接日K Close={minute_daily_close:.2f} | API日K Close={real_daily_close:.2f}")
                else:
                    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 前日({prev_date})收盘(使用{last_time}): {prev_close:.2f} | 拼接日K Close={minute_daily_close:.2f} | API日K Close=未获取")
        else:
            prev_close = None
    else:
        prev_close = None
    
    if prev_close is None:
        return df
    
    # 根据算法计算参考价格
    upper_ref = max(day_open, prev_close)
    lower_ref = min(day_open, prev_close)
    
    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 边界参考: 当日开盘={day_open:.2f}, 前日收盘={prev_close:.2f}, 上边界参考={upper_ref:.2f}, 下边界参考={lower_ref:.2f}")
    
    # 对目标日期的每个时间点计算上下边界
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
    
    return df

@api_retry(max_retries=10, retry_delay=2)
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

def check_exit_conditions(df, position_quantity, current_stop):
    # 获取当前时间点
    now = get_us_eastern_time()
    current_time = now.strftime('%H:%M')
    current_date = now.date()
    
    # 精简日志，直接获取当前时间点数据
    current_data = df[(df["Date"] == current_date) & (df["Time"] == current_time)]
    
    # 如果当前时间点没有数据，使用最新数据
    if current_data.empty:
        # 按日期和时间排序，获取最新的数据
        df_sorted = df.sort_values(by=["Date", "Time"], ascending=True)
        latest = df_sorted.iloc[-1]
    else:
        latest = current_data.iloc[0]
        
    price = latest["Close"]
    vwap = latest["VWAP"]
    upper = latest["UpperBound"]
    lower = latest["LowerBound"]
    
    # 检查数据是否为空值
    if price is None:
        return False, current_stop
    
    if position_quantity > 0:
        # 检查上边界是否为None
        if upper is None or (USE_VWAP and vwap is None):
            # 如果已有止损，继续使用
            if current_stop is not None:
                new_stop = current_stop
                exit_signal = price < new_stop
                return exit_signal, new_stop
            else:
                return False, current_stop
        else:
            # 直接使用当前时刻的止损水平，不考虑历史止损
            new_stop = max(upper, vwap) if USE_VWAP else upper
            
        exit_signal = price < new_stop
        return exit_signal, new_stop
    elif position_quantity < 0:
        # 检查下边界是否为None
        if lower is None or (USE_VWAP and vwap is None):
            # 如果已有止损，继续使用
            if current_stop is not None:
                new_stop = current_stop
                exit_signal = price > new_stop
                return exit_signal, new_stop
            else:
                return False, current_stop
        else:
            # 直接使用当前时刻的止损水平，不考虑历史止损
            new_stop = min(lower, vwap) if USE_VWAP else lower
            
        exit_signal = price > new_stop
        return exit_signal, new_stop
    return False, None

@api_retry(max_retries=10, retry_delay=2)
def is_trading_day(symbol=None):
    """
    检查是否是交易日
    返回: (is_trading_day, is_half_day)
        is_trading_day: 是否是交易日（包括半交易日）
        is_half_day: 是否是半交易日
    """
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
    return (is_trade_day or is_half_trade_day, is_half_trade_day)

def run_trading_strategy(symbol=SYMBOL, check_interval_minutes=CHECK_INTERVAL_MINUTES,
                        trading_start_time=TRADING_START_TIME, trading_end_time=TRADING_END_TIME,
                        max_positions_per_day=MAX_POSITIONS_PER_DAY, lookback_days=LOOKBACK_DAYS):
    global TOTAL_PNL, DAILY_PNL, LAST_STATS_DATE, DAILY_TRADES
    
    now_et = get_us_eastern_time()
    print(f"启动交易策略 - 交易品种: {symbol}")
    print(f"当前美东时间: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"交易时间: {trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d}")
    print(f"每日最大开仓次数: {max_positions_per_day}")
    if DEBUG_MODE:
        print(f"调试模式已开启! 使用时间: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
    
    initial_capital = get_account_balance()
    if initial_capital <= 0:
        print("Error: Could not get account balance or balance is zero")
        sys.exit(1)
    
    # 获取当前实际持仓
    current_positions = get_current_positions()
    symbol_position = current_positions.get(symbol, {"quantity": 0, "cost_price": 0})
    position_quantity = symbol_position["quantity"]
    
    # 初始化入场价格为None，后续由交易操作更新
    entry_price = None
    
    current_stop = None
    positions_opened_today = 0
    last_date = None
    outside_rth_setting = OutsideRTH.AnyTime
    
    # 🎯 动态追踪止盈状态变量
    max_profit_price = None         # 持仓期间的最优价格（多头：最高价，空头：最低价）
    trailing_tp_activated = False   # 追踪止盈是否已激活
    
    while True:
        now = get_us_eastern_time()
        current_date = now.date()
        if DEBUG_MODE:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S.%f')}] 主循环开始 (精确时间)")
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 时间精度: 秒={now.second}, 微秒={now.microsecond}")
        
        # 每次循环都更新当前持仓状态和账户余额
        current_positions = get_current_positions()
        symbol_position = current_positions.get(symbol, {"quantity": 0, "cost_price": 0})
        position_quantity = symbol_position["quantity"]
        
        # 获取当前美元账户余额
        current_balance = get_account_balance()
        
        # 如果持仓量变为0，重置入场价格和追踪止盈状态
        if position_quantity == 0:
            entry_price = None
            # 🎯 重置动态追踪止盈状态
            max_profit_price = None
            trailing_tp_activated = False
        
        # 检查是否到达检查时间点
        current_hour, current_minute = now.hour, now.minute
        current_second = now.second
        
        # 生成今天所有的检查时间点（这些是K线时间，不是触发时间）
        k_line_check_times = []
        h, m = trading_start_time
        while h < trading_end_time[0] or (h == trading_end_time[0] and m <= trading_end_time[1]):
            k_line_check_times.append((h, m))
            m += check_interval_minutes
            if m >= 60:
                h += 1
                m = m % 60
        
        # 始终添加结束时间
        if (trading_end_time[0], trading_end_time[1]) not in k_line_check_times:
            k_line_check_times.append((trading_end_time[0], trading_end_time[1]))
        
        # 生成实际的触发时间点（K线时间的下一分钟）
        trigger_times = []
        for k_h, k_m in k_line_check_times:
            # 计算下一分钟作为触发时间
            trigger_m = k_m + 1
            trigger_h = k_h
            if trigger_m >= 60:
                trigger_h += 1
                trigger_m = 0
            # 跳过超出交易时间的触发点
            if trigger_h < 16:  # 假设市场在16:00关闭
                trigger_times.append((trigger_h, trigger_m))
        
        # 判断当前是否是触发时间点（允许前后30秒的误差）
        is_trigger_time = False
        for trigger_h, trigger_m in trigger_times:
            trigger_time = now.replace(hour=trigger_h, minute=trigger_m, second=1, microsecond=0)
            time_diff = abs((now - trigger_time).total_seconds())
            if time_diff <= 30:  # 30秒误差范围内都认为是触发时间
                is_trigger_time = True
                break
        
        if is_trigger_time:
            # 找到最接近的触发时间对应的K线时间
            closest_trigger_idx = None
            min_diff = float('inf')
            for i, (trigger_h, trigger_m) in enumerate(trigger_times):
                trigger_time = now.replace(hour=trigger_h, minute=trigger_m, second=1, microsecond=0)
                time_diff = abs((now - trigger_time).total_seconds())
                if time_diff < min_diff:
                    min_diff = time_diff
                    closest_trigger_idx = i
            
            if closest_trigger_idx is not None:
                k_h, k_m = k_line_check_times[closest_trigger_idx]
                check_time_str = f"{k_h:02d}:{k_m:02d}"
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 触发检查，使用 {check_time_str} 的K线数据")
            else:
                # 如果没有找到合适的触发时间，跳过本次检查
                continue
        else:
            # 如果不是触发时间点，计算下一个触发时间
            next_trigger_time = None
            for trigger_h, trigger_m in trigger_times:
                if trigger_h > current_hour or (trigger_h == current_hour and trigger_m > current_minute):
                    next_trigger_time = datetime.combine(current_date, time(trigger_h, trigger_m), tzinfo=now.tzinfo)
                    break
            
            if next_trigger_time is None:
                # 今天没有更多触发时间，等到明天
                tomorrow = current_date + timedelta(days=1)
                if trigger_times:
                    next_trigger_time = datetime.combine(tomorrow, time(trigger_times[0][0], trigger_times[0][1]), tzinfo=now.tzinfo)
                else:
                    # 如果没有触发时间，使用默认的开始时间
                    next_trigger_time = datetime.combine(tomorrow, time(trading_start_time[0], trading_start_time[1] + 1), tzinfo=now.tzinfo)
            
            wait_seconds = (next_trigger_time - now).total_seconds()
            if wait_seconds > 0:
                wait_seconds = min(wait_seconds, 300)  # 最多等待5分钟
                if DEBUG_MODE:
                    # 找到下一个K线检查时间用于显示
                    next_trigger_idx = None
                    for i, (t_h, t_m) in enumerate(trigger_times):
                        if t_h > current_hour or (t_h == current_hour and t_m > current_minute):
                            next_trigger_idx = i
                            break
                    if next_trigger_idx is not None:
                        next_k_h, next_k_m = k_line_check_times[next_trigger_idx]
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 等待 {wait_seconds:.0f} 秒到下一个检查时间 {next_k_h:02d}:{next_k_m:02d} (触发时间: {next_trigger_time.strftime('%H:%M:%S')})")
                    else:
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 等待 {wait_seconds:.0f} 秒到下一个检查时间 (触发时间: {next_trigger_time.strftime('%H:%M:%S')})")
                time_module.sleep(wait_seconds)
                continue
        
        # 更新当前时间信息
        now = get_us_eastern_time()
        current_date = now.date()
        
        # 只在触发时间点进行交易检查
        if not is_trigger_time:
            # 如果不是触发时间，跳过本次循环
            continue
            
        # 检查是否是交易时间结束点，如果是且有持仓，则强制平仓
        is_trading_end = (current_hour, current_minute) == (trading_end_time[0], trading_end_time[1])
        if is_trading_end and position_quantity != 0:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 当前时间为交易结束时间 {trading_end_time[0]}:{trading_end_time[1]}，执行平仓")
            
            # 获取历史数据
            if DEBUG_MODE:
                print("获取历史数据")
            df = get_historical_data(symbol)
            if df.empty:
                print("错误: 获取历史数据为空")
                sys.exit(1)
                
            if DEBUG_MODE:
                df = df[df["DateTime"] <= now]
            
            # 获取当前时间点的价格数据
            current_time = now.strftime('%H:%M')
            
            # 尝试获取当前时间点数据，如果没有则等待重试
            retry_count = 0
            max_retries = 10
            retry_interval = 5
            current_price = None
            
            while retry_count < max_retries:
                current_data = df[(df["Date"] == current_date) & (df["Time"] == current_time)]
                
                if not current_data.empty:
                    # 使用当前时间点的价格
                    current_price = float(current_data["Close"].iloc[0])
                    break
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        if DEBUG_MODE:
                            print(f"警告: 当前时间点 {current_time} 没有数据，等待{retry_interval}秒后重试 ({retry_count}/{max_retries})")
                        time_module.sleep(retry_interval)
                        # 重新获取数据
                        df = get_historical_data(symbol)
                        if DEBUG_MODE:
                            df = df[df["DateTime"] <= now]
            
            if current_price is None:
                print(f"错误: 尝试{max_retries}次后仍无法获取当前时间点 {current_time} 的数据")
                sys.exit(1)
            
            # 执行平仓
            side = "Sell" if position_quantity > 0 else "Buy"
            close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓订单已提交，ID: {close_order_id}")
            
            # 计算盈亏
            if entry_price:
                pnl = (current_price - entry_price) * (1 if position_quantity > 0 else -1) * abs(position_quantity)
                pnl_pct = (current_price / entry_price - 1) * 100 * (1 if position_quantity > 0 else -1)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓成功: {side} {abs(position_quantity)} {symbol} 价格: {current_price}")
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 交易结果: {'盈利' if pnl > 0 else '亏损'} ${abs(pnl):.2f} ({pnl_pct:.2f}%)")
                # 更新收益统计
                DAILY_PNL += pnl
                TOTAL_PNL += pnl
                # 记录平仓交易
                DAILY_TRADES.append({
                    "time": now.strftime('%Y-%m-%d %H:%M:%S'),
                    "action": "平仓",
                    "side": side,
                    "quantity": abs(position_quantity),
                    "price": current_price,
                    "pnl": pnl
                })
            else:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓成功: {side} {abs(position_quantity)} {symbol} 价格: {current_price}")
                
            position_quantity = 0
            entry_price = None
            # 🎯 重置动态追踪止盈状态
            max_profit_price = None
            trailing_tp_activated = False
            
            # 在交易日结束时打印当日所有交易记录
            if DAILY_TRADES:
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] ===== 当日交易记录 =====")
                for i, trade in enumerate(DAILY_TRADES, 1):
                    print(f"交易 #{i}:")
                    print(f"  时间: {trade['time']}")
                    print(f"  操作: {trade['action']} {trade['side']} {trade['quantity']} 股")
                    print(f"  价格: ${trade['price']:.2f}")
                    if trade['pnl'] is not None:
                        print(f"  盈亏: ${trade['pnl']:+.2f}")
                
                # 计算当日统计
                total_trades = len([t for t in DAILY_TRADES if t['action'] == '平仓'])
                winning_trades = len([t for t in DAILY_TRADES if t['action'] == '平仓' and t['pnl'] > 0])
                losing_trades = len([t for t in DAILY_TRADES if t['action'] == '平仓' and t['pnl'] < 0])
                
                print(f"\n当日交易统计:")
                print(f"  总交易次数: {total_trades}")
                print(f"  盈利次数: {winning_trades}")
                print(f"  亏损次数: {losing_trades}")
                if total_trades > 0:
                    print(f"  胜率: {winning_trades/total_trades*100:.1f}%")
                print(f"  当日盈亏: ${DAILY_PNL:+.2f}")
                print(f"  累计盈亏: ${TOTAL_PNL:+.2f}")
                print("=" * 50)
                
                # 清空当日交易记录，为下一个交易日准备
                DAILY_TRADES.clear()
            

            continue
        
        # 检查是否是交易日（调试模式下保持原有逻辑）
        is_today_trading_day, is_half_day = is_trading_day(symbol)
        if DEBUG_MODE:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 是否交易日: {is_today_trading_day}, 是否半交易日: {is_half_day}")
            
        if not is_today_trading_day:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 今天不是交易日，跳过交易")
            # 如果有持仓，执行平仓
            if position_quantity != 0:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 非交易日，执行平仓")
                
                # 获取当前价格用于计算盈亏
                quote = get_quote(symbol)
                current_price = float(quote.get("last_done", 0))
                
                side = "Sell" if position_quantity > 0 else "Buy"
                close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓订单已提交，ID: {close_order_id}")
                
                # 计算盈亏
                if entry_price and current_price > 0:
                    pnl = (current_price - entry_price) * (1 if position_quantity > 0 else -1) * abs(position_quantity)
                    DAILY_PNL += pnl
                    TOTAL_PNL += pnl
                    # 记录平仓交易
                    DAILY_TRADES.append({
                        "time": now.strftime('%Y-%m-%d %H:%M:%S'),
                        "action": "平仓",
                        "side": side,
                        "quantity": abs(position_quantity),
                        "price": current_price,
                        "pnl": pnl
                    })
                    
                position_quantity = 0
                entry_price = None
                # 🎯 重置动态追踪止盈状态
                max_profit_price = None
                trailing_tp_activated = False
            next_check_time = now + timedelta(hours=12)
            wait_seconds = (next_check_time - now).total_seconds()
            time_module.sleep(wait_seconds)
            continue
        elif is_half_day:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 今天是半交易日，不开新仓")
            if position_quantity != 0:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 非交易日，执行平仓")
                
                # 获取当前价格用于计算盈亏
                quote = get_quote(symbol)
                current_price = float(quote.get("last_done", 0))
                
                side = "Sell" if position_quantity > 0 else "Buy"
                close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓订单已提交，ID: {close_order_id}")
                
                # 计算盈亏
                if entry_price and current_price > 0:
                    pnl = (current_price - entry_price) * (1 if position_quantity > 0 else -1) * abs(position_quantity)
                    DAILY_PNL += pnl
                    TOTAL_PNL += pnl
                    # 记录平仓交易
                    DAILY_TRADES.append({
                        "time": now.strftime('%Y-%m-%d %H:%M:%S'),
                        "action": "平仓",
                        "side": side,
                        "quantity": abs(position_quantity),
                        "price": current_price,
                        "pnl": pnl
                    })
                    
                position_quantity = 0
                entry_price = None
                # 🎯 重置动态追踪止盈状态
                max_profit_price = None
                trailing_tp_activated = False
            next_check_time = now + timedelta(hours=12)
            wait_seconds = (next_check_time - now).total_seconds()
            time_module.sleep(wait_seconds)
            continue
            
        # 检查是否是新交易日，如果是则重置今日开仓计数
        if last_date is not None and current_date != last_date:
            positions_opened_today = 0
            
            # 打印前一日交易记录
            if DAILY_TRADES:
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] ===== 前一日交易记录 ({last_date}) =====")
                for i, trade in enumerate(DAILY_TRADES, 1):
                    print(f"交易 #{i}:")
                    print(f"  时间: {trade['time']}")
                    print(f"  操作: {trade['action']} {trade['side']} {trade['quantity']} 股")
                    print(f"  价格: ${trade['price']:.2f}")
                    if trade['pnl'] is not None:
                        print(f"  盈亏: ${trade['pnl']:+.2f}")
                
                # 计算前一日统计
                total_trades = len([t for t in DAILY_TRADES if t['action'] == '平仓'])
                winning_trades = len([t for t in DAILY_TRADES if t['action'] == '平仓' and t['pnl'] > 0])
                losing_trades = len([t for t in DAILY_TRADES if t['action'] == '平仓' and t['pnl'] < 0])
                
                print(f"\n前一日交易统计:")
                print(f"  总交易次数: {total_trades}")
                print(f"  盈利次数: {winning_trades}")
                print(f"  亏损次数: {losing_trades}")
                if total_trades > 0:
                    print(f"  胜率: {winning_trades/total_trades*100:.1f}%")
                    
                # 清空交易记录，为新交易日准备
                DAILY_TRADES.clear()
            
            # 输出前一日收益统计
            if LAST_STATS_DATE is not None and DAILY_PNL != 0:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] === 收益统计 ===")
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 昨日盈亏: ${DAILY_PNL:+.2f}")
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 累计盈亏: ${TOTAL_PNL:+.2f}")
                print("=" * 50)
                
            DAILY_PNL = 0.0  # 重置当日收益
        last_date = current_date
        LAST_STATS_DATE = current_date
        
        # 保持原有交易时间检查逻辑
        start_hour, start_minute = trading_start_time
        end_hour, end_minute = trading_end_time
        is_trading_hours = (
            (current_hour > start_hour or (current_hour == start_hour and current_minute >= start_minute)) and
            (current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute))
        )
            
        df = get_historical_data(symbol)
        if df.empty:
            print("Error: Could not get historical data")
            sys.exit(1)
        if DEBUG_MODE:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 历史数据获取完成: {len(df)} 条")
            
        # 调试模式下，根据指定时间截断数据
        if DEBUG_MODE:
            # 截断到调试时间之前的数据
            df = df[df["DateTime"] <= now]
            
        if not is_trading_hours:
            if DEBUG_MODE:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 当前不在交易时间内 ({trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d})")
            if position_quantity != 0:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 交易日结束，执行平仓")
                
                # 获取当前价格用于计算盈亏
                quote = get_quote(symbol)
                current_price = float(quote.get("last_done", 0))
                
                side = "Sell" if position_quantity > 0 else "Buy"
                close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓订单已提交，ID: {close_order_id}")
                
                # 计算盈亏
                if entry_price and current_price > 0:
                    pnl = (current_price - entry_price) * (1 if position_quantity > 0 else -1) * abs(position_quantity)
                    DAILY_PNL += pnl
                    TOTAL_PNL += pnl
                    # 记录平仓交易
                    DAILY_TRADES.append({
                        "time": now.strftime('%Y-%m-%d %H:%M:%S'),
                        "action": "平仓",
                        "side": side,
                        "quantity": abs(position_quantity),
                        "price": current_price,
                        "pnl": pnl
                    })
                    
                position_quantity = 0
                entry_price = None
                # 🎯 重置动态追踪止盈状态
                max_profit_price = None
                trailing_tp_activated = False
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
            
        # 使用新的VWAP计算方法
        df["VWAP"] = calculate_vwap(df)
        
        # 直接计算噪声区域，不需要中间复制
        df = calculate_noise_area(df, lookback_days, K1, K2)
        
        if position_quantity != 0:
            # 使用检查时间点的数据进行止损检查
            if 'check_time_str' not in locals():
                # 如果没有设置check_time_str，使用当前时间的前一分钟
                if current_minute > 0:
                    check_time_str = f"{current_hour:02d}:{current_minute-1:02d}"
                else:
                    check_time_str = f"{current_hour-1:02d}:59"
            
            # 获取检查时间点的数据
            latest_date = df["Date"].max()
            check_data = df[(df["Date"] == latest_date) & (df["Time"] == check_time_str)]
            
            if not check_data.empty:
                check_row = check_data.iloc[0]
                check_price = float(check_row["Close"])
                check_high = float(check_row["High"])
                check_low = float(check_row["Low"])
                check_upper = check_row["UpperBound"]
                check_lower = check_row["LowerBound"]
                check_vwap = check_row["VWAP"]
                
                # 根据持仓方向检查退出条件
                exit_signal = False
                trailing_tp_exit = False  # 🎯 追踪止盈触发标志
                exit_reason = "Stop Loss"  # 默认退出原因
                
                if position_quantity > 0:  # 多头持仓
                    # 🎯 动态追踪止盈逻辑 - 多头
                    if ENABLE_TRAILING_TAKE_PROFIT and entry_price is not None:
                        # 更新最优价格（使用K线的最高价）
                        if max_profit_price is None or check_high > max_profit_price:
                            max_profit_price = check_high
                        
                        # 计算当前浮盈百分比（使用最高价）
                        current_profit_pct = (max_profit_price - entry_price) / entry_price
                        
                        # 检查是否激活追踪止盈
                        if not trailing_tp_activated and current_profit_pct >= TRAILING_TP_ACTIVATION_PCT:
                            trailing_tp_activated = True
                            if DEBUG_MODE:
                                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 🎯 追踪止盈激活！浮盈: {current_profit_pct*100:.2f}%, 最高价: {max_profit_price:.2f}")
                        
                        # 如果追踪止盈已激活，计算动态止盈线
                        if trailing_tp_activated:
                            # 动态止盈线 = 入场价 + (最大浮盈 * 保护比例)
                            protected_profit = (max_profit_price - entry_price) * TRAILING_TP_CALLBACK_PCT
                            dynamic_take_profit_level = entry_price + protected_profit
                            
                            # 检查是否触发动态止盈
                            if check_price <= dynamic_take_profit_level:
                                trailing_tp_exit = True
                                exit_reason = "Trailing Take Profit"
                                if DEBUG_MODE:
                                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 🎯 动态止盈触发！价格: {check_price:.2f} <= 止盈线: {dynamic_take_profit_level:.2f}")
                    
                    # 使用检查时间点的上边界（和VWAP）作为止损
                    new_stop = max(check_upper, check_vwap) if USE_VWAP else check_upper
                    stop_loss_exit = check_price < new_stop
                    current_stop = new_stop
                    
                    # 综合判断退出信号
                    exit_signal = trailing_tp_exit or stop_loss_exit
                    if stop_loss_exit and not trailing_tp_exit:
                        exit_reason = "Stop Loss"
                        
                elif position_quantity < 0:  # 空头持仓
                    # 🎯 动态追踪止盈逻辑 - 空头
                    if ENABLE_TRAILING_TAKE_PROFIT and entry_price is not None:
                        # 更新最优价格（使用K线的最低价，空头时低价是有利的）
                        if max_profit_price is None or check_low < max_profit_price:
                            max_profit_price = check_low
                        
                        # 计算当前浮盈百分比（空头：入场价 - 最低价）
                        current_profit_pct = (entry_price - max_profit_price) / entry_price
                        
                        # 检查是否激活追踪止盈
                        if not trailing_tp_activated and current_profit_pct >= TRAILING_TP_ACTIVATION_PCT:
                            trailing_tp_activated = True
                            if DEBUG_MODE:
                                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 🎯 追踪止盈激活！浮盈: {current_profit_pct*100:.2f}%, 最低价: {max_profit_price:.2f}")
                        
                        # 如果追踪止盈已激活，计算动态止盈线
                        if trailing_tp_activated:
                            # 动态止盈线 = 入场价 - (最大浮盈 * 保护比例)
                            protected_profit = (entry_price - max_profit_price) * TRAILING_TP_CALLBACK_PCT
                            dynamic_take_profit_level = entry_price - protected_profit
                            
                            # 检查是否触发动态止盈
                            if check_price >= dynamic_take_profit_level:
                                trailing_tp_exit = True
                                exit_reason = "Trailing Take Profit"
                                if DEBUG_MODE:
                                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 🎯 动态止盈触发！价格: {check_price:.2f} >= 止盈线: {dynamic_take_profit_level:.2f}")
                    
                    # 使用检查时间点的下边界（和VWAP）作为止损
                    new_stop = min(check_lower, check_vwap) if USE_VWAP else check_lower
                    stop_loss_exit = check_price > new_stop
                    current_stop = new_stop
                    
                    # 综合判断退出信号
                    exit_signal = trailing_tp_exit or stop_loss_exit
                    if stop_loss_exit and not trailing_tp_exit:
                        exit_reason = "Stop Loss"
                
                if DEBUG_MODE:
                    tp_info = f", 追踪止盈={'激活' if trailing_tp_activated else '未激活'}" if ENABLE_TRAILING_TAKE_PROFIT else ""
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 持仓检查 {check_time_str}: 数量={position_quantity}, 价格={check_price:.2f}, 止损={current_stop:.2f}, 退出信号={exit_signal}{tp_info}")
            else:
                # 如果没有检查时间点的数据，使用原有逻辑
                exit_signal, new_stop = check_exit_conditions(df, position_quantity, current_stop)
                current_stop = new_stop
                exit_reason = "Stop Loss"  # 回退逻辑默认为止损
                if DEBUG_MODE:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 持仓检查: 数量={position_quantity}, 退出信号={exit_signal}, 当前止损={current_stop}")
            if exit_signal:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 触发退出信号!")
                
                # 确保使用当前时间点的价格数据
                current_time = now.strftime('%H:%M')
                
                # 尝试获取当前时间点数据，如果没有则等待重试
                retry_count = 0
                max_retries = 10
                retry_interval = 5
                exit_price = None
                
                while retry_count < max_retries:
                    current_data = df[(df["Date"] == current_date) & (df["Time"] == current_time)]
                    
                    if not current_data.empty:
                        # 使用当前时间点的价格
                        exit_price = float(current_data["Close"].iloc[0])
                        break
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            if DEBUG_MODE:
                                print(f"警告: 当前时间点 {current_time} 没有数据，等待{retry_interval}秒后重试 ({retry_count}/{max_retries})")
                            time_module.sleep(retry_interval)
                            # 重新获取数据
                            df = get_historical_data(symbol)
                            if DEBUG_MODE:
                                df = df[df["DateTime"] <= now]
                            # 重新计算VWAP和噪声区域
                            df["VWAP"] = calculate_vwap(df)
                            df = calculate_noise_area(df, lookback_days, K1, K2)
                
                if exit_price is None:
                    print(f"错误: 尝试{max_retries}次后仍无法获取当前时间点 {current_time} 的数据")
                    continue  # 继续下一次循环，而不是退出
                
                # 执行平仓
                side = "Sell" if position_quantity > 0 else "Buy"
                close_order_id = submit_order(symbol, side, abs(position_quantity), outside_rth=outside_rth_setting)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓订单已提交 ({exit_reason})，ID: {close_order_id}")
                
                # 计算盈亏
                if entry_price:
                    pnl = (exit_price - entry_price) * (1 if position_quantity > 0 else -1) * abs(position_quantity)
                    pnl_pct = (exit_price / entry_price - 1) * 100 * (1 if position_quantity > 0 else -1)
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓成功 ({exit_reason}): {side} {abs(position_quantity)} {symbol} 价格: {exit_price}")
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 交易结果: {'盈利' if pnl > 0 else '亏损'} ${abs(pnl):.2f} ({pnl_pct:.2f}%)")
                    # 更新收益统计
                    DAILY_PNL += pnl
                    TOTAL_PNL += pnl
                    # 记录平仓交易
                    DAILY_TRADES.append({
                        "time": now.strftime('%Y-%m-%d %H:%M:%S'),
                        "action": "平仓",
                        "side": side,
                        "quantity": abs(position_quantity),
                        "price": exit_price,
                        "pnl": pnl,
                        "exit_reason": exit_reason  # 🎯 记录退出原因
                    })
                
                # 平仓后增加交易次数计数器
                positions_opened_today += 1
                
                position_quantity = 0
                entry_price = None
                # 🎯 重置动态追踪止盈状态
                max_profit_price = None
                trailing_tp_activated = False
        else:
            # 检查是否已有持仓，如果有则不再开仓
            if position_quantity != 0:
                if DEBUG_MODE:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 已有持仓，跳过开仓检查")
                continue
            
            # 检查是否是半交易日，如果是则不开新仓
            if is_half_day:
                if DEBUG_MODE:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 半交易日，跳过开仓")
                continue
                
            # 检查今日是否达到最大持仓数
            if positions_opened_today >= max_positions_per_day:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 今日已开仓 {positions_opened_today} 次，达到上限")
                continue
            
            # 使用检查时间点的完整K线数据
            # check_time_str 在前面已经设置为要检查的时间（如 "09:40"）
            if 'check_time_str' not in locals():
                # 如果没有设置check_time_str，使用当前时间的前一分钟
                if current_minute > 0:
                    check_time_str = f"{current_hour:02d}:{current_minute-1:02d}"
                else:
                    check_time_str = f"{current_hour-1:02d}:59"
            
            # 获取检查时间点的数据
            latest_date = df["Date"].max()
            check_data = df[(df["Date"] == latest_date) & (df["Time"] == check_time_str)]
            
            if check_data.empty:
                if DEBUG_MODE:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 警告: 没有找到 {check_time_str} 的数据，跳过本次检查")
                continue
            
            # 使用检查时间点的完整K线数据
            latest_row = check_data.iloc[0].copy()
            latest_price = float(latest_row["Close"])
            long_price_above_upper = latest_price > latest_row["UpperBound"]
            long_price_above_vwap = latest_price > latest_row["VWAP"] if USE_VWAP else True
            
            if DEBUG_MODE:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 检查 {check_time_str} 的数据:")
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 价格={latest_price:.2f}, 上界={latest_row['UpperBound']:.2f}, VWAP={latest_row['VWAP']:.2f}, 下界={latest_row['LowerBound']:.2f}")
            
            signal = 0
            price = latest_price
            stop = None
            
            if long_price_above_upper and long_price_above_vwap:
                if DEBUG_MODE:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 满足多头入场条件!")
                signal = 1
                stop = max(latest_row["UpperBound"], latest_row["VWAP"]) if USE_VWAP else latest_row["UpperBound"]
            else:
                short_price_below_lower = latest_price < latest_row["LowerBound"]
                short_price_below_vwap = latest_price < latest_row["VWAP"] if USE_VWAP else True
                if short_price_below_lower and short_price_below_vwap:
                    if DEBUG_MODE:
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 满足空头入场条件!")
                    signal = -1
                    stop = min(latest_row["LowerBound"], latest_row["VWAP"]) if USE_VWAP else latest_row["LowerBound"]
                else:
                    if DEBUG_MODE:
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 不满足入场条件: 多头({long_price_above_upper} & {long_price_above_vwap}), 空头({short_price_below_lower} & {short_price_below_vwap})")
            if signal != 0:
                # 保留交易信号日志，并添加VWAP和上下界信息
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 触发{'多' if signal == 1 else '空'}头入场信号!")
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 当前价格: {price}, VWAP: {latest_row['VWAP']:.4f}, 上界: {latest_row['UpperBound']:.4f}, 下界: {latest_row['LowerBound']:.4f}, 止损: {stop}")
                
                available_capital = get_account_balance()
                # 应用杠杆比例
                adjusted_capital = available_capital * LEVERAGE
                position_size = floor(adjusted_capital / latest_price)
                if position_size <= 0:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Insufficient capital for position")
                    sys.exit(1)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 可用资金: ${available_capital:.2f}, 杠杆比例: {LEVERAGE}倍, 调整后资金: ${adjusted_capital:.2f}")
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 开仓数量: {position_size} 股")
                side = "Buy" if signal > 0 else "Sell"
                order_id = submit_order(symbol, side, position_size, outside_rth=outside_rth_setting)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 订单已提交，ID: {order_id}")
                
                # 删除订单状态检查代码，直接更新持仓状态
                position_quantity = position_size if signal > 0 else -position_size
                entry_price = latest_price
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 开仓成功: {side} {position_size} {symbol} 价格: {entry_price}")
                
                # 记录开仓交易
                DAILY_TRADES.append({
                    "time": now.strftime('%Y-%m-%d %H:%M:%S'),
                    "action": "开仓",
                    "side": side,
                    "quantity": position_size,
                    "price": entry_price,
                    "pnl": None  # 开仓时还没有盈亏
                })
        

            
        # 计算下一个精确的检查时间点（避免累积误差）
        current_time = now.time()
        current_hour, current_minute = current_time.hour, current_time.minute
        
        # 计算下一个检查时间点
        next_check_minute = ((current_minute // check_interval_minutes) + 1) * check_interval_minutes
        next_check_hour = current_hour
        
        if next_check_minute >= 60:
            next_check_hour += next_check_minute // 60
            next_check_minute = next_check_minute % 60
        
        # 创建下一个检查时间的datetime对象
        next_check_time = now.replace(hour=next_check_hour, minute=next_check_minute, second=1, microsecond=0)
        
        # 如果计算的时间已经过了，则加一天
        if next_check_time <= now:
            next_check_time += timedelta(days=1)
        
        sleep_seconds = (next_check_time - now).total_seconds()
        if sleep_seconds > 0:
            if DEBUG_MODE:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 等待 {sleep_seconds:.1f} 秒到下一个精确检查时间 {next_check_time.strftime('%H:%M:%S')}")
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 当前时间精度检查: 秒={now.second}, 微秒={now.microsecond}")
            time_module.sleep(sleep_seconds)

if __name__ == "__main__":
    print("\n长桥API交易策略启动")
    print("版本: 1.1.0 (带动态追踪止盈)")
    print("时间:", get_us_eastern_time().strftime("%Y-%m-%d %H:%M:%S"), "(美东时间)")
    if DEBUG_MODE:
        print("调试模式已开启")
    print(f"杠杆倍数: {LEVERAGE}倍")
    # 🎯 显示动态追踪止盈配置
    if ENABLE_TRAILING_TAKE_PROFIT:
        print(f"🎯 动态追踪止盈: 已启用")
        print(f"   激活阈值: {TRAILING_TP_ACTIVATION_PCT*100:.1f}%")
        print(f"   保护比例: {TRAILING_TP_CALLBACK_PCT*100:.0f}%")
    else:
        print(f"🎯 动态追踪止盈: 未启用")
    
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
