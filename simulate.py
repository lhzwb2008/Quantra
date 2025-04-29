import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date
import time as time_module
import json
import os
import sys
import pytz
from math import floor
from decimal import Decimal  # 添加Decimal支持
from dotenv import load_dotenv  # 添加dotenv支持

# 加载.env文件中的环境变量
load_dotenv()

# Import calculation functions from existing files
from calculate_indicators import calculate_macd

# Import Longport SDK
from longport.openapi import Config, TradeContext, QuoteContext, Period, OrderSide, OrderType, TimeInForceType, AdjustType, OutsideRTH, OrderStatus  # 添加OutsideRTH和OrderStatus

# Longport API credentials will be loaded from environment variables
# 需要设置以下环境变量 (可通过.env文件配置):
# LONGPORT_APP_KEY
# LONGPORT_APP_SECRET
# LONGPORT_ACCESS_TOKEN

# Trading parameters - 可以从环境变量加载或使用默认值
CHECK_INTERVAL_MINUTES = int(os.environ.get('CHECK_INTERVAL_MINUTES', 10))  # 时间间隔，默认10分钟
TRADING_START_HOUR = int(os.environ.get('TRADING_START_HOUR', 9))  # 交易开始小时，默认9点
TRADING_START_MINUTE = int(os.environ.get('TRADING_START_MINUTE', 40))  # 交易开始分钟，默认40分
TRADING_END_HOUR = int(os.environ.get('TRADING_END_HOUR', 15))  # 交易结束小时，默认15点
TRADING_END_MINUTE = int(os.environ.get('TRADING_END_MINUTE', 40))  # 交易结束分钟，默认40分
MAX_POSITIONS_PER_DAY = int(os.environ.get('MAX_POSITIONS_PER_DAY', 3))  # 每日最大持仓数，默认3
USE_MACD = os.environ.get('USE_MACD', 'true').lower() == 'true'  # 是否使用MACD，默认True
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', 10))  # 回溯天数，默认10天

# 交易开始和结束时间元组
TRADING_START_TIME = (TRADING_START_HOUR, TRADING_START_MINUTE)
TRADING_END_TIME = (TRADING_END_HOUR, TRADING_END_MINUTE)

# 其他参数
SYMBOL = os.environ.get('SYMBOL', 'TQQQ.US')  # 默认交易品种，可从环境变量配置

# print(os.environ)

# 获取美东时间
def get_us_eastern_time():
    """
    获取当前的美东时间
    
    Returns:
        datetime: 当前的美东时间
    """
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

# 创建API配置和上下文
def create_contexts():
    """
    创建长桥API的行情和交易上下文
    
    Returns:
        tuple: (QuoteContext, TradeContext)
    """
    try:
        print("正在初始化长桥API连接...")
        # 注意: 通过python-dotenv已经将.env文件中的配置加载到环境变量中
        # 所以这里使用Config.from_env()会读取到.env中设置的值
        config = Config.from_env()
        print("成功从环境变量加载API配置 (已通过.env文件设置)")
        
        # 检查是否成功加载API凭证
        app_key = os.environ.get("LONGPORT_APP_KEY", "")
        if app_key == "" or app_key == "your_app_key_here":
            print("警告: API凭证可能未正确设置。请检查.env文件中的配置。")
        
        # 创建行情上下文
        quote_ctx = QuoteContext(config)
        print("成功创建行情上下文")
        
        # 创建交易上下文
        trade_ctx = TradeContext(config)
        print("成功创建交易上下文")
        
        return quote_ctx, trade_ctx
    except Exception as e:
        print(f"Error creating API contexts: {e}")
        return None, None

# 全局上下文对象
QUOTE_CTX, TRADE_CTX = create_contexts()

def get_account_balance():
    """
    Get the account balance from Longport API (USD only)
    
    Returns:
        float: Available cash balance in USD
    """
    try:
        if TRADE_CTX is None:
            print("Trade context is not initialized")
            return 0
            
        print("正在获取美元账户余额...")
        # 直接获取美元账户余额
        balance_list = TRADE_CTX.account_balance(currency="USD")
        
        # 处理返回结果
        if balance_list and len(balance_list) > 0:
            balance = balance_list[0]
            available_cash = float(balance.net_assets)
            print(f"美元账户余额: ${available_cash:.2f}")
            return available_cash
        else:
            print("未找到美元账户，请检查您的账户设置")
            return 0
    except Exception as e:
        print(f"Error getting account balance: {e}")
        return 0

def get_current_positions():
    """
    Get current stock positions from Longport API
    
    Returns:
        dict: Dictionary of positions with symbol as key
    """
    try:
        if TRADE_CTX is None:
            print("Trade context is not initialized")
            return {}
            
        # 获取股票持仓
        stock_positions_response = TRADE_CTX.stock_positions()
        
        # 打印原始响应对象，帮助调试
        # print(f"股票持仓响应对象类型: {type(stock_positions_response)}")
        # print(f"股票持仓响应对象内容: {stock_positions_response}")
        
        # 提取持仓信息
        positions = {}
        
        # 正确处理StockPositionsResponse对象
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
    except Exception as e:
        print(f"Error getting positions: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        return {}

def get_historical_data(symbol, period="1m", count=390, trade_sessions="normal", days_back=None):
    """
    Get historical candlestick data from Longport API
    
    Parameters:
        symbol: Stock symbol
        period: Candlestick period (e.g., "1m", "5m", "day")
        count: Number of candlesticks to retrieve per request (max 1000)
        trade_sessions: Trading sessions to include
        days_back: Number of days to look back for historical data. 
                  If None, defaults to LOOKBACK_DAYS + 3 for safety margin
        
    Returns:
        DataFrame: Historical data
    """
    try:
        if QUOTE_CTX is None:
            print("Quote context is not initialized")
            return pd.DataFrame()
        
        # 如果days_back为None，则使用LOOKBACK_DAYS + 3作为默认值
        if days_back is None:
            days_back = LOOKBACK_DAYS + 3  # 额外的3天作为安全边际
        
        print(f"正在获取 {symbol} 的历史数据 (周期: {period}, 回溯天数: {days_back}, 目标天数: {LOOKBACK_DAYS})...")
            
        # 转换period字符串为SDK的Period枚举
        period_map = {
            "1m": Period.Min_1,
            "5m": Period.Min_5,
            "15m": Period.Min_15,
            "30m": Period.Min_30,
            "60m": Period.Min_60,
            "day": Period.Day,
            "week": Period.Week,
            "month": Period.Month,
            "year": Period.Year
        }
        sdk_period = period_map.get(period, Period.Min_1)
        
        # 设置调整类型
        adjust_type = AdjustType.ForwardAdjust  # forward_adjust
        
        # 确定需要获取的数据
        target_days = max(days_back, LOOKBACK_DAYS + 2)  # 至少需要LOOKBACK_DAYS天数据+额外余量
        all_candles = []
        
        if period == "1m":
            print(f"将获取约 {target_days} 个交易日的分钟级数据...")
            
            # 获取最近一次数据以确定最新交易日
            recent_candles = QUOTE_CTX.history_candlesticks_by_offset(
                symbol,                # 股票代码
                sdk_period,            # K线周期
                adjust_type,           # 复权类型
                False,                 # forward=False，向历史方向查找
                1,                     # count参数：只获取1条记录 
                datetime.now()         # time参数：基准时间，当前时间
            )
            
            if not recent_candles:
                print(f"警告: 无法获取 {symbol} 的最新数据")
                return pd.DataFrame()
                
            latest_time = recent_candles[0].timestamp
            # 修正：如果timestamp已经是datetime对象，直接获取date
            if isinstance(latest_time, datetime):
                latest_date = latest_time.date()
            else:
                # 如果是时间戳数字，则转换
                latest_date = datetime.fromtimestamp(latest_time).date()
            print(f"最新交易日期: {latest_date}")
            
            # 计算交易日起始日期（近似值）
            # 由于不确定具体的交易日历，我们假设每7天有5个交易日
            calendar_days_needed = int(target_days * 7 / 5) + 5  # 添加额外天数作为缓冲
            start_date = latest_date - timedelta(days=calendar_days_needed)
            
            # 对每个可能的交易日进行数据获取
            current_date = latest_date
            trading_days_fetched = 0
            
            # 设置进度显示
            print(f"开始获取历史数据，预计需要处理 {calendar_days_needed} 个日历日...")
            
            while current_date >= start_date and trading_days_fetched < target_days:
                try:
                    date_str = current_date.strftime("%Y%m%d")
                    minute_str = "09:30"  # 美股/港股交易开始时间
                    
                    print(f"获取 {date_str} 的分钟K线数据...")
                    
                    # 获取该日的分钟K线
                    day_candles = QUOTE_CTX.history_candlesticks_by_offset(
                        symbol,                                      # 股票代码
                        sdk_period,                                  # K线周期
                        adjust_type,                                 # 复权类型
                        True,                                        # forward=True，向最新方向查找
                        390,                                         # count参数：一天的分钟数
                        datetime.combine(current_date, time(9, 30))  # time参数：该天09:30的datetime对象
                    )
                    
                    if day_candles:
                        all_candles.extend(day_candles)
                        trading_days_fetched += 1
                        print(f"成功获取 {date_str} 的数据: {len(day_candles)} 条记录")
                    else:
                        print(f"{date_str} 可能不是交易日，无数据")
                    
                    # 检查是否达到目标
                    if trading_days_fetched >= target_days:
                        print(f"已获取 {trading_days_fetched} 个交易日的数据，达到目标")
                        break
                        
                    # 避免API请求过于频繁
                    time_module.sleep(0.5)
                    
                except Exception as e:
                    print(f"获取 {date_str} 数据时出错: {e}")
                
                # 移至前一天
                current_date -= timedelta(days=1)
            
            print(f"数据获取完成，共获取 {trading_days_fetched} 个交易日，{len(all_candles)} 条记录")
            
        else:
            # 对于非分钟级数据，采用一次性获取策略
            print(f"获取{period}周期的历史数据...")
            max_request_count = 1000  # API单次请求上限
            
            # 获取当前日期和过去的日期
            now = get_us_eastern_time()
            past_date = now - timedelta(days=days_back)
            
            all_candles = QUOTE_CTX.history_candlesticks_by_offset(
                symbol,                # 股票代码
                sdk_period,            # K线周期
                adjust_type,           # 复权类型
                False,                 # forward=False，向历史方向查找
                max_request_count,     # count参数：请求数量
                past_date              # time参数：基准时间，过去的日期
            )
        
        # 转换为DataFrame
        data = []
        eastern = pytz.timezone('US/Eastern')
        
        # 调试信息：检查第一条数据的时间戳类型
        if all_candles:
            first_timestamp = all_candles[0].timestamp
            print(f"调试信息 - 首条数据时间戳类型: {type(first_timestamp)}")
            print(f"调试信息 - 首条数据时间戳值: {first_timestamp}")
            if isinstance(first_timestamp, datetime):
                print(f"调试信息 - 时区信息: {first_timestamp.tzinfo}")
        
        for candle in all_candles:
            timestamp = candle.timestamp
            
            # 首先确定原始时间戳的时区
            original_dt = None
            
            if isinstance(timestamp, datetime):
                original_dt = timestamp
                print(f"原始时间戳: {timestamp}, 时区: {timestamp.tzinfo}")
            else:
                # 假设时间戳是UTC的Unix时间戳
                original_dt = datetime.fromtimestamp(timestamp, pytz.utc)
                print(f"Unix时间戳 {timestamp} 转换为UTC时间: {original_dt}")
            
            # 确保交易时间在美股交易时段 (9:30-16:00 ET)
            # 先转换为美东时间
            if original_dt.tzinfo is None:
                # 无时区信息，根据数值判断
                if original_dt.hour >= 9 and original_dt.hour < 17:
                    # 合理的交易时间，假设是美东时间
                    dt = eastern.localize(original_dt)
                else:
                    # 不合理的交易时间，假设是UTC时间
                    dt = pytz.utc.localize(original_dt).astimezone(eastern)
            else:
                # 已有时区信息，直接转换
                dt = original_dt.astimezone(eastern)
            
            # 根据证券类型调整时间
            # 美股的交易时间是9:30-16:00 ET
            if symbol.endswith(".US"):
                # 验证时间是否在合理范围内
                hour = dt.hour
                if not (9 <= hour < 17):
                    # 不在美股交易时间内，需要重新猜测时区
                    print(f"警告: 时间 {dt} 不在美股交易时间内，尝试调整...")
                    
                    # 尝试当作UTC+8(北京时间)转换为美东时间
                    if isinstance(timestamp, datetime):
                        if timestamp.tzinfo is None:
                            dt_china = pytz.timezone('Asia/Shanghai').localize(timestamp)
                            dt = dt_china.astimezone(eastern)
                    else:
                        # 对于Unix时间戳，假设是UTC+0，转为美东
                        dt = datetime.fromtimestamp(timestamp, pytz.utc).astimezone(eastern)
                    
                    print(f"调整后的时间: {dt}")
            
            data.append({
                "Close": float(candle.close),
                "Open": float(candle.open),
                "High": float(candle.high),
                "Low": float(candle.low),
                "Volume": float(candle.volume),
                "Turnover": float(candle.turnover),
                "DateTime": dt
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            print(f"警告: 未获取到 {symbol} 的历史数据")
            return df
        
        print(f"成功获取 {symbol} 的历史数据: {len(df)} 条记录")
        
        # 显示一些样本数据，帮助调试时区问题
        sample_times = df["DateTime"].head(5).tolist()
        print(f"样本时间数据 (前5条):")
        for sample_time in sample_times:
            print(f"  - {sample_time} ({sample_time.tzinfo})")
        
        # 提取日期和时间组件
        df["Date"] = df["DateTime"].dt.date
        df["Time"] = df["DateTime"].dt.strftime('%H:%M')
        
        # 打印获取的日期范围
        if not df.empty:
            earliest_date = df["Date"].min()
            latest_date = df["Date"].max()
            unique_dates = df["Date"].nunique()
            print(f"获取的数据日期范围: {earliest_date} 至 {latest_date}, 共 {unique_dates} 个交易日")
            
            # 检查是否获取了足够的历史数据用于噪声区域计算
            if unique_dates < LOOKBACK_DAYS:
                print(f"警告: 获取的历史数据不足 {LOOKBACK_DAYS} 个交易日(只有 {unique_dates} 天)，可能会影响噪声区域计算")
        
        # 检查时间范围是否合理
        time_range = df["Time"].unique()
        time_min = min(time_range) if len(time_range) > 0 else "N/A"
        time_max = max(time_range) if len(time_range) > 0 else "N/A"
        print(f"时间范围: {time_min} 到 {time_max}")
        
        # 检查是否有在美股交易时间(9:30-16:00)之外的数据
        if symbol.endswith(".US"):
            outside_hours = df[~df["Time"].between("09:30", "16:00")]
            if not outside_hours.empty:
                outside_count = len(outside_hours)
                outside_percent = outside_count / len(df) * 100
                print(f"警告: 发现 {outside_count} 条数据 ({outside_percent:.1f}%) 在美股交易时间之外")
                
                # 显示部分非交易时间的数据作为示例
                print("非交易时间数据示例:")
                for _, row in outside_hours.head(3).iterrows():
                    print(f"  - 日期: {row['Date']}, 时间: {row['Time']}, 价格: {row['Close']}")
                
                # 过滤掉非交易时间的数据
                df = df[df["Time"].between("09:30", "16:00")]
                print(f"已过滤非交易时间数据，剩余 {len(df)} 条记录")
        
        return df
    except Exception as e:
        print(f"Error getting historical data: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误堆栈
        return pd.DataFrame()

def get_quote(symbol):
    """
    Get real-time quote for a symbol from Longport API
    
    Parameters:
        symbol: Stock symbol
        
    Returns:
        dict: Quote data
    """
    try:
        if QUOTE_CTX is None:
            print("Quote context is not initialized")
            return {}
            
        # 获取实时行情
        quotes = QUOTE_CTX.quote([symbol])
        
        if not quotes:
            return {}
            
        # 转换为字典格式
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
    except Exception as e:
        print(f"Error getting quote: {e}")
        return {}

def calculate_vwap_incrementally(prices, volumes):
    """
    Calculate VWAP incrementally for a series of prices and volumes
    
    Parameters:
        prices: List of prices
        volumes: List of volumes
        
    Returns:
        list: VWAP values
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

def calculate_noise_area(df, lookback_days=10):
    """
    Calculate noise area boundaries for a DataFrame
    
    Parameters:
        df: DataFrame with price data
        lookback_days: Number of days to look back
        
    Returns:
        DataFrame: DataFrame with upper and lower bounds
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Get day open and previous close
    df_copy["day_open"] = df_copy.groupby("Date")["Open"].transform("first")
    
    # Calculate previous day's close
    unique_dates = sorted(df_copy["Date"].unique())
    prev_close_map = {}
    
    for i in range(1, len(unique_dates)):
        prev_date = unique_dates[i-1]
        curr_date = unique_dates[i]
        prev_close = df_copy[df_copy["Date"] == prev_date]["Close"].iloc[-1]
        prev_close_map[curr_date] = prev_close
    
    # Apply previous close
    df_copy["prev_close"] = df_copy["Date"].map(prev_close_map)
    
    # Calculate reference prices for each day
    date_refs = []
    for d in unique_dates:
        day_data = df_copy[df_copy["Date"] == d].iloc[0]
        day_open = day_data["day_open"]
        prev_close = day_data.get("prev_close")
        
        if not pd.isna(prev_close):
            upper_ref = max(day_open, prev_close)
            lower_ref = min(day_open, prev_close)
        else:
            upper_ref = day_open
            lower_ref = day_open
            
        date_refs.append({
            "Date": d,
            "upper_ref": upper_ref,
            "lower_ref": lower_ref
        })
    
    # Create date reference DataFrame
    date_refs_df = pd.DataFrame(date_refs)
    
    # Merge reference prices back to main DataFrame
    df_copy = df_copy.drop(columns=["upper_ref", "lower_ref"], errors="ignore")
    df_copy = pd.merge(df_copy, date_refs_df, on="Date", how="left")
    
    # Calculate return from open for each minute
    df_copy["ret"] = df_copy["Close"] / df_copy["day_open"] - 1
    
    # 检查是否有重复的日期和时间组合
    date_time_counts = df_copy.groupby(['Date', 'Time']).size()
    duplicate_combinations = date_time_counts[date_time_counts > 1].reset_index()[['Date', 'Time']]
    
    if not duplicate_combinations.empty:
        print(f"警告: 发现{len(duplicate_combinations)}个重复的日期和时间组合，将保留每个组合的最后一条记录")
        # 为每个日期和时间组合保留最后一条记录
        df_copy = df_copy.drop_duplicates(subset=['Date', 'Time'], keep='last')
    
    # Calculate the Noise Area boundaries
    # Pivot to get time-of-day in columns
    pivot = df_copy.pivot(index="Date", columns="Time", values="ret").abs()
    
    # Calculate rolling average of absolute returns for each time-of-day
    # Use min_periods=1 to allow calculation with fewer days of data
    sigma = pivot.rolling(window=lookback_days, min_periods=1).mean()
    
    # 如果没有足够的历史数据，使用所有可用数据的平均值
    if sigma.isna().all().all():
        print("警告: 没有足够的历史数据来计算噪声区域边界，使用所有可用数据的平均值")
        # 使用所有可用数据的平均值
        mean_ret = df_copy["ret"].abs().mean()
        # 创建一个与pivot相同形状的DataFrame，填充平均值
        sigma = pd.DataFrame(mean_ret, index=pivot.index, columns=pivot.columns)
    
    # Convert back to long format
    sigma = sigma.stack().reset_index(name="sigma")
    
    # Merge sigma back to the main dataframe
    df_copy = pd.merge(df_copy, sigma, on=["Date", "Time"], how="left")
    
    # Calculate upper and lower boundaries of the Noise Area
    df_copy["UpperBound"] = df_copy["upper_ref"] * (1 + df_copy["sigma"])
    df_copy["LowerBound"] = df_copy["lower_ref"] * (1 - df_copy["sigma"])
    
    # 打印噪声区域边界计算结果
    print("\n噪声区域边界计算结果:")
    print(f"唯一日期数量: {len(unique_dates)}")
    print(f"最新日期: {unique_dates[-1] if unique_dates else 'N/A'}")
    
    # 打印最新的几个时间点的边界值
    latest_date = unique_dates[-1] if unique_dates else None
    if latest_date:
        latest_data = df_copy[df_copy["Date"] == latest_date].tail(5)
        print("\n最新几个时间点的边界值:")
        for _, row in latest_data.iterrows():
            print(f"时间: {row['Time']}, 价格: {row['Close']:.2f}, VWAP: {row['VWAP']:.2f}, 上边界: {row['UpperBound']:.2f}, 下边界: {row['LowerBound']:.2f}")
    
    # 打印sigma的统计信息
    print(f"\nSigma统计信息:")
    print(f"平均值: {df_copy['sigma'].mean():.4f}")
    print(f"最大值: {df_copy['sigma'].max():.4f}")
    print(f"最小值: {df_copy['sigma'].min():.4f}")
    print(f"NaN值数量: {df_copy['sigma'].isna().sum()}")
    
    return df_copy

def submit_order(symbol, side, quantity, order_type="MO", price=None, outside_rth=None):
    """
    Submit an order using Longport API
    
    Parameters:
        symbol: Stock symbol
        side: "Buy" or "Sell"
        quantity: Order quantity
        order_type: Order type (MO for Market Order, LO for Limit Order)
        price: Limit price (required for LO)
        outside_rth: Whether to allow trading outside regular trading hours (OutsideRTH.RTHOnly, OutsideRTH.AnyTime, OutsideRTH.Overnight),
                     If None, defaults to OutsideRTH.AnyTime
        
    Returns:
        str: Order ID if successful, None otherwise
    """
    try:
        if TRADE_CTX is None:
            print("Trade context is not initialized")
            return None
        
        # 打印参数类型以便调试    
        print(f"订单参数: symbol={symbol}, side={side}, quantity={quantity}, order_type={order_type}, price={price}, outside_rth={outside_rth}")
            
        # 转换side为SDK的OrderSide枚举
        sdk_side = OrderSide.Buy if side == "Buy" else OrderSide.Sell
        
        # 转换order_type为SDK的OrderType枚举
        if isinstance(order_type, str):
            order_type_map = {
                "MO": OrderType.MO,  # 市价单
                "LO": OrderType.LO,  # 限价单
                "ELO": OrderType.ELO,  # 增强限价单
                "AO": OrderType.AO,  # 竞价单
                "ALO": OrderType.ALO  # 竞价限价单
            }
            sdk_order_type = order_type_map.get(order_type, OrderType.MO)
        else:
            sdk_order_type = order_type
        
        # 设置time_in_force
        time_in_force = TimeInForceType.Day
        
        # 处理outside_rth参数
        if outside_rth is None:
            outside_rth = OutsideRTH.AnyTime
        elif isinstance(outside_rth, str):
            outside_rth_map = {
                "RTH_ONLY": OutsideRTH.RTHOnly,
                "ANY_TIME": OutsideRTH.AnyTime,
                "OVERNIGHT": OutsideRTH.Overnight
            }
            outside_rth = outside_rth_map.get(outside_rth, OutsideRTH.AnyTime)
            
        # 将数量和价格转换为Decimal类型
        dec_quantity = Decimal(str(quantity))
        
        print(f"提交订单: {symbol}, {side}, 数量: {quantity}, 订单类型: {sdk_order_type}, outside_rth: {outside_rth}")
        
        # 提交订单
        if sdk_order_type == OrderType.LO and price is not None:
            # 限价单需要价格 - 转换为Decimal类型
            dec_price = Decimal(str(price))
            print(f"提交限价单，价格: {dec_price}")
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
            # 市价单不需要价格
            print(f"提交市价单")
            response = TRADE_CTX.submit_order(
                symbol=symbol,
                order_type=OrderType.MO,  # 确保使用市价单
                side=sdk_side,
                submitted_quantity=dec_quantity,
                time_in_force=time_in_force,
                outside_rth=outside_rth
            )
        
        # 从SubmitOrderResponse对象中提取order_id
        print(f"订单提交成功: {response}")
        return response.order_id
    except Exception as e:
        print(f"提交订单时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_order_status(order_id):
    """
    Get order status from Longport API
    
    Parameters:
        order_id: Order ID
        
    Returns:
        dict: Order details
    """
    try:
        if TRADE_CTX is None:
            print("Trade context is not initialized")
            return {}
        
        # 确保order_id是字符串类型
        if not isinstance(order_id, str):
            order_id = str(order_id)
            
        print(f"正在获取订单 {order_id} 的状态...")
            
        # 获取订单详情
        order_detail = TRADE_CTX.order_detail(order_id)
        
        # 转换OrderStatus为字符串
        status_str = str(order_detail.status)
        
        # 转换为字典格式
        order_info = {
            "order_id": order_detail.order_id,
            "status": status_str,
            "stock_name": order_detail.stock_name,
            "quantity": order_detail.quantity,
            "executed_quantity": order_detail.executed_quantity,
            "price": str(order_detail.price),
            "executed_price": str(order_detail.executed_price),
            "submitted_at": order_detail.submitted_at.isoformat(),
            "side": str(order_detail.side)  # 同样转换OrderSide为字符串
        }
        
        print(f"订单状态获取成功: {status_str}")
        return order_info
    except Exception as e:
        print(f"获取订单状态时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}

def cancel_order(order_id):
    """
    Cancel an order using Longport API
    
    Parameters:
        order_id: Order ID to cancel
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if TRADE_CTX is None:
            print("Trade context is not initialized")
            return False
            
        # 确保order_id是字符串类型
        if not isinstance(order_id, str):
            order_id = str(order_id)
            
        print(f"取消订单 {order_id}")
            
        # Cancel the order
        TRADE_CTX.cancel_order(order_id)
        
        return True
    except Exception as e:
        print(f"取消订单时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_trading_conditions(df, current_time, allowed_times, positions_opened_today, max_positions_per_day):
    """
    Check if trading conditions are met for entry
    
    Parameters:
        df: DataFrame with price data and indicators
        current_time: Current time string (HH:MM)
        allowed_times: List of allowed trading times
        positions_opened_today: Number of positions opened today
        max_positions_per_day: Maximum number of positions allowed per day
        
    Returns:
        tuple: (signal, entry_price, stop_price)
            signal: 1 for long, -1 for short, 0 for no signal
            entry_price: Entry price
            stop_price: Initial stop price
    """
    # Check if we've reached the maximum positions for the day
    if positions_opened_today >= max_positions_per_day:
        return 0, None, None
    
    # Check if current time is in allowed times
    if current_time not in allowed_times:
        return 0, None, None
    
    # Get the latest data point
    latest = df.iloc[-1]
    
    # Get current price, VWAP, and bounds
    price = latest["Close"]
    vwap = latest["VWAP"]
    upper = latest["UpperBound"]
    lower = latest["LowerBound"]
    
    # Get MACD histogram value if available and enabled
    macd_histogram = latest.get("MACD_histogram", 0)
    
    # Check for long entry
    long_macd_condition = macd_histogram > 0 if USE_MACD else True
    if price > upper and price > vwap and long_macd_condition:
        # Long entry allowed
        # Initial stop: max(UpperBound, VWAP)
        stop_price = max(upper, vwap)
        return 1, price, stop_price
    
    # Check for short entry
    short_macd_condition = macd_histogram < 0 if USE_MACD else True
    if price < lower and price < vwap and short_macd_condition:
        # Short entry allowed
        # Initial stop: min(LowerBound, VWAP)
        stop_price = min(lower, vwap)
        return -1, price, stop_price
    
    # No entry signal
    return 0, None, None

def check_exit_conditions(df, position, trailing_stop):
    """
    Check if exit conditions are met
    
    Parameters:
        df: DataFrame with price data and indicators
        position: Current position (1 for long, -1 for short)
        trailing_stop: Current trailing stop price
        
    Returns:
        tuple: (exit_signal, new_stop)
            exit_signal: True if exit conditions are met, False otherwise
            new_stop: Updated trailing stop price
    """
    # Get the latest data point
    latest = df.iloc[-1]
    
    # Get current price, VWAP, and bounds
    price = latest["Close"]
    vwap = latest["VWAP"]
    upper = latest["UpperBound"]
    lower = latest["LowerBound"]
    
    if position == 1:  # Long position
        # Calculate new stop level
        new_stop = max(upper, vwap)
        
        # Only update in favorable direction (raise the stop)
        if trailing_stop is not None:
            new_stop = max(trailing_stop, new_stop)
        
        # Exit if price crosses below the trailing stop
        exit_signal = price < new_stop
        
        return exit_signal, new_stop
    
    elif position == -1:  # Short position
        # Calculate new stop level
        new_stop = min(lower, vwap)
        
        # Only update in favorable direction (lower the stop)
        if trailing_stop is not None:
            new_stop = min(trailing_stop, new_stop)
        
        # Exit if price crosses above the trailing stop
        exit_signal = price > new_stop
        
        return exit_signal, new_stop
    
    # No position
    return False, None

def generate_allowed_times(start_time, end_time, check_interval_minutes):
    """
    Generate allowed trading times based on the check interval
    
    Parameters:
        start_time: Trading start time (hour, minute)
        end_time: Trading end time (hour, minute)
        check_interval_minutes: Time between checks in minutes
        
    Returns:
        list: List of allowed trading times in HH:MM format
    """
    allowed_times = []
    start_hour, start_minute = start_time
    end_hour, end_minute = end_time
    
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
    end_time_str = f"{end_time[0]:02d}:{end_time[1]:02d}"
    if end_time_str not in allowed_times:
        allowed_times.append(end_time_str)
        allowed_times.sort()
    
    return allowed_times

def run_trading_strategy(symbol=SYMBOL, check_interval_minutes=CHECK_INTERVAL_MINUTES,
                        trading_start_time=TRADING_START_TIME, trading_end_time=TRADING_END_TIME,
                        max_positions_per_day=MAX_POSITIONS_PER_DAY, lookback_days=LOOKBACK_DAYS):
    """
    Run the trading strategy using Longport API
    
    Parameters:
        symbol: Stock symbol to trade
        check_interval_minutes: Time between trading checks in minutes
        trading_start_time: Trading start time (hour, minute) in US Eastern Time
        trading_end_time: Trading end time (hour, minute) in US Eastern Time
        max_positions_per_day: Maximum number of positions to open per day
        lookback_days: Days to look back for calculating noise area
    """
    # 获取当前美东时间
    now_et = get_us_eastern_time()
    
    print("\n" + "="*50)
    print(f"启动交易策略 - 交易品种: {symbol}")
    print(f"当前美东时间: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"交易时间: {trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d} (美东时间)")
    print(f"检查间隔: {check_interval_minutes} 分钟")
    print(f"每日最大持仓数: {max_positions_per_day}")
    print(f"回溯天数: {lookback_days}")
    print(f"使用MACD: {'是' if USE_MACD else '否'}")
    print("="*50 + "\n")
    
    # Get account balance
    initial_capital = get_account_balance()
    if initial_capital <= 0:
        print("Error: Could not get account balance or balance is zero")
        return
    
    # Generate allowed trading times
    allowed_times = generate_allowed_times(trading_start_time, trading_end_time, check_interval_minutes)
    print(f"允许交易的时间点 (美东时间): {allowed_times}")
    
    # Initialize trading variables
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = None
    trailing_stop = None
    entry_time = None
    order_id = None
    positions_opened_today = 0
    last_date = None
    
    # 止损止盈比例设置
    stop_loss_pct = 0.005  # 0.5%的止损比例
    take_profit_pct = 0.01  # 1.0%的止盈比例
    
    # 判断是否是美股交易
    is_us_market = symbol.endswith(".US")
    outside_rth_setting = "ANY_TIME" if is_us_market else "RTH_ONLY"
    
    # Main trading loop
    while True:
        try:
            # 获取当前美东时间
            now = get_us_eastern_time()
            current_time = now.strftime("%H:%M")
            current_date = now.date()
            
            print(f"\n当前美东时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Reset positions_opened_today if it's a new day
            if last_date is not None and current_date != last_date:
                print(f"新的交易日开始，重置今日开仓计数")
                positions_opened_today = 0
            
            last_date = current_date
            
            # Check if it's within trading hours
            current_hour, current_minute = now.hour, now.minute
            start_hour, start_minute = trading_start_time
            end_hour, end_minute = trading_end_time
            
            is_trading_hours = (
                (current_hour > start_hour or (current_hour == start_hour and current_minute >= start_minute)) and
                (current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute))
            )
            
            if not is_trading_hours:
                print(f"当前不在交易时间内 ({trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d})")
                
                # 交易日结束，强制平仓
                try:
                    if position != 0:
                        print("交易日结束，执行平仓")
                        side = "Sell" if position > 0 else "Buy"
                        quantity = abs(position)
                        
                        # 平仓时使用AnyTime设置
                        outside_rth_setting = OutsideRTH.AnyTime
                        
                        close_order_id = submit_order(symbol, side, quantity, outside_rth=outside_rth_setting)
                        if close_order_id:
                            print(f"交易日结束，平仓: {side} {quantity} {symbol}")
                        
                        # Reset position variables
                        position = 0
                        entry_price = 0
                        entry_time = None
                except Exception as e:
                    print(f"平仓出错: {e}")
                
                # Sleep until next check
                print(f"等待 60 秒后再次检查...")
                time_module.sleep(60)
                continue
            
            # Get historical data for the day
            df = get_historical_data(symbol, period="1m")
            if df.empty:
                print("Error: Could not get historical data")
                time_module.sleep(60)
                continue
            
            # Calculate VWAP
            print("计算VWAP指标...")
            prices = df["Close"].values
            volumes = df["Volume"].values
            df["VWAP"] = calculate_vwap_incrementally(prices, volumes)
            
            # Calculate MACD if needed
            if USE_MACD and "MACD_histogram" not in df.columns:
                print("计算MACD指标...")
                df = calculate_macd(df)
            
            # Calculate noise area boundaries
            print("计算噪声区域边界...")
            df = calculate_noise_area(df, lookback_days)
            
            # Check for missing data
            if df["UpperBound"].isna().any() or df["LowerBound"].isna().any():
                print("警告: 边界数据缺失，跳过本次检查")
                time_module.sleep(60)
                continue
            
            # Get current positions
            print("获取当前持仓...")
            positions = get_current_positions()
            if positions:
                print(f"当前持仓: {positions}")
            else:
                print("当前无持仓")
            
            # Update position status based on actual positions
            if position != 0:
                # Check if we still have the position
                if symbol not in positions or positions[symbol]["quantity"] == 0:
                    # Position was closed externally
                    position = 0
                    entry_price = None
                    trailing_stop = None
                    entry_time = None
            
            # Check for exit if we have a position
            if position != 0:
                print(f"检查退出条件 (当前持仓方向: {'多' if position == 1 else '空'}, 追踪止损: {trailing_stop})...")
                exit_signal, new_stop = check_exit_conditions(df, position, trailing_stop)
                
                # Update trailing stop
                trailing_stop = new_stop
                print(f"更新追踪止损价格: {trailing_stop}")
                
                if exit_signal and current_time in allowed_times:
                    print("触发退出信号!")
                    # Exit position
                    side = "Sell" if position == 1 else "Buy"
                    quantity = positions[symbol]["quantity"]
                    
                    # 执行平仓
                    try:
                        close_order_id = submit_order(symbol, side, quantity, outside_rth=outside_rth_setting)
                        if close_order_id:
                            exit_time = now
                            exit_price = df.iloc[-1]["Close"]
                            
                            print(f"平仓: {side} {quantity} {symbol} 价格: {exit_price}")
                            
                            # Calculate PnL
                            pnl = (exit_price - entry_price) * position
                            pnl_pct = pnl / (entry_price * abs(position)) * 100
                            print(f"交易结果: {'盈利' if pnl > 0 else '亏损'} ${abs(pnl):.2f} ({pnl_pct:.2f}%)")
                            
                            # Reset position variables
                            position = 0
                            entry_price = 0
                            entry_time = None
                    except Exception as e:
                        print(f"平仓出错: {e}")
            
            # Check for entry if we don't have a position
            elif position == 0:
                print(f"检查入场条件 (今日已开仓: {positions_opened_today}/{max_positions_per_day})...")
                signal, price, stop = check_trading_conditions(df, current_time, allowed_times, positions_opened_today, max_positions_per_day)
                
                if signal != 0:
                    print(f"触发{'多' if signal == 1 else '空'}头入场信号! 价格: {price}, 止损: {stop}")
                    # Calculate position size
                    quote = get_quote(symbol)
                    if not quote:
                        print("Error: Could not get quote")
                        time_module.sleep(60)
                        continue
                    
                    # Get latest price
                    latest_price = float(quote.get("last_done", price))
                    
                    # Calculate position size (use 90% of available capital)
                    available_capital = get_account_balance() * 0.9
                    position_size = floor(available_capital / latest_price)
                    
                    if position_size <= 0:
                        print("Warning: Insufficient capital for position")
                        time_module.sleep(60)
                        continue
                    
                    # 开仓时使用AnyTime设置
                    outside_rth_setting = OutsideRTH.AnyTime
                    
                    order_id = submit_order(symbol, side, position_size, outside_rth=outside_rth_setting)
                    
                    if order_id:
                        # Update position variables
                        position = signal
                        entry_price = latest_price
                        entry_time = now
                        
                        # Set stop loss and take profit levels
                        stop_loss = entry_price * (1 - stop_loss_pct) if signal > 0 else entry_price * (1 + stop_loss_pct)
                        take_profit = entry_price * (1 + take_profit_pct) if signal > 0 else entry_price * (1 - take_profit_pct)
                        
                        print(f"开仓: {side} {position_size} {symbol} 价格: {entry_price}")
                        print(f"止损: {stop_loss}, 止盈: {take_profit}")
            
            # Sleep until next check
            next_check_time = now + timedelta(minutes=check_interval_minutes)
            sleep_seconds = (next_check_time - now).total_seconds()
            if sleep_seconds > 0:
                print(f"等待 {sleep_seconds:.0f} 秒后进行下一次检查...")
                time_module.sleep(sleep_seconds)
            
        except Exception as e:
            print(f"交易循环中出现错误: {e}")
            time_module.sleep(60)

if __name__ == "__main__":
    import argparse
    
    print("\n" + "*"*70)
    print("* 长桥API交易策略启动")
    print("* 版本: 1.0.0")
    print("* 时间:", get_us_eastern_time().strftime("%Y-%m-%d %H:%M:%S"), "(美东时间)")
    print("*"*70 + "\n")
    
    parser = argparse.ArgumentParser(description="Run trading strategy using Longport API")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="Symbol to trade")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL_MINUTES, help="Check interval in minutes")
    parser.add_argument("--start-hour", type=int, default=TRADING_START_TIME[0], help="Trading start hour")
    parser.add_argument("--start-minute", type=int, default=TRADING_START_TIME[1], help="Trading start minute")
    parser.add_argument("--end-hour", type=int, default=TRADING_END_TIME[0], help="Trading end hour")
    parser.add_argument("--end-minute", type=int, default=TRADING_END_TIME[1], help="Trading end minute")
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS_PER_DAY, help="Maximum positions per day")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS, help="Lookback days for noise area")
    parser.add_argument("--no-macd", action="store_true", help="Disable MACD condition")
    
    args = parser.parse_args()
    
    # Update global variables from arguments
    SYMBOL = args.symbol
    CHECK_INTERVAL_MINUTES = args.interval
    TRADING_START_TIME = (args.start_hour, args.start_minute)
    TRADING_END_TIME = (args.end_hour, args.end_minute)
    MAX_POSITIONS_PER_DAY = args.max_positions
    LOOKBACK_DAYS = args.lookback
    USE_MACD = not args.no_macd
    
    # 检查API上下文是否成功创建
    if QUOTE_CTX is None or TRADE_CTX is None:
        print("错误: 无法创建API上下文。请检查API凭证是否正确设置。")
        print("请确保以下环境变量已正确设置:")
        print("- LONGPORT_APP_KEY")
        print("- LONGPORT_APP_SECRET")
        print("- LONGPORT_ACCESS_TOKEN")
        sys.exit(1)
    
    try:
        # Run trading strategy
        run_trading_strategy(
            symbol=SYMBOL,
            check_interval_minutes=CHECK_INTERVAL_MINUTES,
            trading_start_time=TRADING_START_TIME,
            trading_end_time=TRADING_END_TIME,
            max_positions_per_day=MAX_POSITIONS_PER_DAY,
            lookback_days=LOOKBACK_DAYS
        )
    finally:
        # 关闭API连接
        try:
            if TRADE_CTX is not None:
                TRADE_CTX.close()
        except Exception as e:
            print(f"Error closing trade context: {e}")
