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

# 全局变量，用于跟踪最后一次获取历史数据的日期
LAST_HISTORICAL_DATA_DATE = None
HISTORICAL_DATA_CACHE = None

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
    global LAST_HISTORICAL_DATA_DATE, HISTORICAL_DATA_CACHE
    
    try:
        if QUOTE_CTX is None:
            print("Quote context is not initialized")
            return pd.DataFrame()
        
        # 获取当前美东时间
        now_et = get_us_eastern_time()
        current_date = now_et.date()
        
        # 检查是否已经获取了今天的数据
        if LAST_HISTORICAL_DATA_DATE is not None and HISTORICAL_DATA_CACHE is not None:
            if LAST_HISTORICAL_DATA_DATE == current_date:
                print(f"使用缓存的历史数据，最后更新日期: {LAST_HISTORICAL_DATA_DATE}")
                return HISTORICAL_DATA_CACHE
        
        # 设置目标天数
        # 如果当前是交易时间前，今天的数据可能不完整，所以目标天数+1
        now_hour = now_et.hour
        now_minute = now_et.minute
        is_before_market_open = (now_hour < 9 or (now_hour == 9 and now_minute < 30))
        
        # 如果当前时间是交易日但还未开盘，目标天数+1
        target_days = LOOKBACK_DAYS + 1 if is_before_market_open else LOOKBACK_DAYS
        print(f"当前时间是否在开盘前: {is_before_market_open}, 目标天数: {target_days}")
        
        # 如果days_back为None，则使用目标天数+15作为安全边际
        if days_back is None:
            days_back = target_days + 15  # 额外的15天作为安全边际，确保能获取到足够的数据
        
        print(f"正在获取 {symbol} 的历史数据 (周期: {period}, 回溯天数: {days_back}, 目标天数: {target_days})...")
            
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
            # 使用days_back参数来确定需要处理的日历日数
            calendar_days_needed = days_back  # 直接使用days_back作为日历日数
            start_date = latest_date - timedelta(days=calendar_days_needed)
            
            print(f"将处理 {calendar_days_needed} 个日历日，从 {start_date} 到 {latest_date}")
            
            # 重置all_candles，确保每次调用函数时都是空的
            all_candles = []
            
            # 对每个可能的交易日进行数据获取
            current_date = latest_date
            trading_days_fetched = 0
            
            # 设置进度显示
            print(f"开始获取历史数据，预计需要处理 {calendar_days_needed} 个日历日...")
            
            # 创建一个集合来跟踪已经获取过的日期，避免重复获取
            fetched_dates = set()
            
            while current_date >= start_date and trading_days_fetched < target_days:
                try:
                    date_str = current_date.strftime("%Y%m%d")
                    
                    # 检查是否已经获取过这个日期的数据
                    if date_str in fetched_dates:
                        print(f"日期 {date_str} 的数据已经获取过，跳过")
                        current_date -= timedelta(days=1)
                        continue
                    
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
                    
                    # 打印每个K线的日期，用于调试
                    if day_candles and len(day_candles) > 0:
                        sample_candle = day_candles[0]
                        sample_time = sample_candle.timestamp
                        if isinstance(sample_time, datetime):
                            sample_date = sample_time.date()
                        else:
                            sample_date = datetime.fromtimestamp(sample_time).date()
                        print(f"样本K线日期: {sample_date}, 请求日期: {current_date}")
                        
                        # 如果样本K线日期与请求日期不匹配，则跳过
                        if sample_date != current_date:
                            print(f"警告: K线日期 {sample_date} 与请求日期 {current_date} 不匹配，跳过")
                            current_date -= timedelta(days=1)
                            continue
                    
                    if day_candles:
                        all_candles.extend(day_candles)
                        # 只有当数据点数量足够多时才计入交易日计数
                        # 美股一天通常有390个分钟K线
                        if len(day_candles) >= 300:  # 至少需要300个数据点才算完整交易日
                            trading_days_fetched += 1
                            print(f"成功获取 {date_str} 的数据: {len(day_candles)} 条记录 (计入交易日计数)")
                        else:
                            print(f"成功获取 {date_str} 的数据: {len(day_candles)} 条记录 (数据不完整，不计入交易日计数)")
                        
                        # 记录已获取的日期，无论是否计入交易日计数
                        fetched_dates.add(date_str)
                    else:
                        print(f"{date_str} 可能不是交易日，无数据")
                    
                    # 检查是否达到目标
                    if trading_days_fetched >= target_days:
                        print(f"已获取 {trading_days_fetched} 个完整交易日的数据，达到目标")
                        # 不要立即退出循环，继续获取更多数据，直到处理完所有日历日
                        # 这样可以确保获取到足够的历史数据，即使有些日期不是交易日
                        # 注释掉break语句，确保继续获取更多数据
                        # break
                        
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
        eastern = pytz.timezone('US/Eastern')  # 美东时区

        # 调试: 输出时区和第一条数据信息
        print("开始处理时间戳数据...")
        
        # 打印一些调试信息，帮助理解数据结构
        print(f"获取到的K线数据总数: {len(all_candles)}")
        
        # 检查是否有重复的时间戳
        timestamp_counts = {}
        for candle in all_candles:
            timestamp = candle.timestamp
            if isinstance(timestamp, datetime):
                ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                ts_str = str(timestamp)
            
            timestamp_counts[ts_str] = timestamp_counts.get(ts_str, 0) + 1
        
        # 打印重复的时间戳
        duplicates = {ts: count for ts, count in timestamp_counts.items() if count > 1}
        if duplicates:
            print(f"发现 {len(duplicates)} 个重复的时间戳:")
            for ts, count in list(duplicates.items())[:5]:  # 只显示前5个
                print(f"  - {ts}: 出现 {count} 次")
            if len(duplicates) > 5:
                print(f"  ... 还有 {len(duplicates) - 5} 个重复时间戳")
        else:
            print("未发现重复的时间戳")
        
        for candle in all_candles:
            timestamp = candle.timestamp
            
            # 简化时区处理逻辑
            if isinstance(timestamp, datetime):
                # 如果是datetime对象
                if timestamp.tzinfo is None:
                    # 无时区信息，根据数值判断时间是否合理
                    hour = timestamp.hour
                    
                    # 美股交易时间为9:30-16:00，如果时间在这个范围，可能已经是美东时间
                    if symbol.endswith(".US") and 9 <= hour < 17:
                        # 直接作为美东时间处理
                        dt = eastern.localize(timestamp)
                    else:
                        # 尝试判断是否是标准时间+8小时(亚洲时区)
                        if symbol.endswith(".US") and (hour >= 21 or hour < 5):
                            # 可能是北京时间晚上9点-凌晨5点 (对应美东时间上午9点-下午5点)
                            # 转换为美东时间
                            beijing = pytz.timezone('Asia/Shanghai')
                            dt = beijing.localize(timestamp).astimezone(eastern)
                        else:
                            # 其他情况当作UTC处理
                            utc = pytz.utc
                            dt = utc.localize(timestamp).astimezone(eastern)
                else:
                    # 已有时区信息，直接转换到美东时间
                    dt = timestamp.astimezone(eastern)
            else:
                # 如果是时间戳数字，先转为UTC，再转为美东时间
                dt = datetime.fromtimestamp(timestamp, pytz.utc).astimezone(eastern)

            # 只有在时间在合理范围内才添加数据
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
            # 检查每个日期的数据点数量
            daily_counts = df.groupby("Date").size()
            
            # 打印每个日期的数据点数量，用于调试
            print("\n各日期数据点数量:")
            for date, count in daily_counts.items():
                print(f"  - {date}: {count} 条记录")
            
            # 筛选出完整的交易日（数据点数量 >= 300）
            complete_dates = daily_counts[daily_counts >= 300].index.tolist()
            complete_trading_days = len(complete_dates)
            
            # 按日期排序
            complete_dates.sort()
            
            if complete_dates:
                earliest_complete_date = complete_dates[0]
                latest_complete_date = complete_dates[-1]
                print(f"完整交易日日期范围: {earliest_complete_date} 至 {latest_complete_date}, 共 {complete_trading_days} 个完整交易日")
            
            earliest_date = df["Date"].min()
            latest_date = df["Date"].max()
            unique_dates = df["Date"].nunique()
            print(f"获取的数据日期范围: {earliest_date} 至 {latest_date}, 共 {unique_dates} 个交易日")
            
            # 检查是否获取了足够的历史数据用于噪声区域计算
            if complete_trading_days < LOOKBACK_DAYS:
                print(f"警告: 获取的完整历史数据不足 {LOOKBACK_DAYS} 个交易日(只有 {complete_trading_days} 天)，可能会影响噪声区域计算")
                
            # 检查是否有日期被跳过
            all_dates = pd.date_range(start=earliest_date, end=latest_date)
            missing_dates = [d.date() for d in all_dates if d.date() not in df["Date"].unique()]
            if missing_dates:
                print(f"\n警告: 数据中有 {len(missing_dates)} 个日期被跳过:")
                for date in missing_dates[:5]:  # 只显示前5个
                    print(f"  - {date}")
                if len(missing_dates) > 5:
                    print(f"  ... 还有 {len(missing_dates) - 5} 个日期被跳过")
                    
            # 检查是否有周末日期被包含在数据中
            weekend_dates = [d for d in df["Date"].unique() if pd.Timestamp(d).dayofweek >= 5]
            if weekend_dates:
                print(f"\n警告: 数据中包含 {len(weekend_dates)} 个周末日期:")
                for date in weekend_dates[:5]:  # 只显示前5个
                    print(f"  - {date} ({pd.Timestamp(date).day_name()})")
                if len(weekend_dates) > 5:
                    print(f"  ... 还有 {len(weekend_dates) - 5} 个周末日期")
        
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
        
        # 检查历史数据的完整性
        print("\n检查历史数据的完整性...")
        
        # 1. 检查数据是否为空
        if df.empty:
            print("警告: 历史数据为空!")
            return df
        
        # 2. 检查数据的日期范围
        date_range = df["Date"].nunique()
        earliest_date = df["Date"].min()
        latest_date = df["Date"].max()
        print(f"数据日期范围: {earliest_date} 至 {latest_date}, 共 {date_range} 个交易日")
        
        # 筛选出完整的交易日（数据点数量 >= 300）
        daily_counts_for_check = df.groupby("Date").size()
        complete_dates_for_check = daily_counts_for_check[daily_counts_for_check >= 300].index.tolist()
        complete_trading_days_for_check = len(complete_dates_for_check)
        
        if complete_trading_days_for_check < LOOKBACK_DAYS:
            print(f"警告: 完整交易日数据不足 {LOOKBACK_DAYS} 天 (只有 {complete_trading_days_for_check} 天)")
        
        # 3. 检查每个交易日的数据点数量，并处理可能的重复数据
        print("\n各交易日数据点数量:")
        # 首先确保每个日期和时间组合只有一条记录
        df = df.drop_duplicates(subset=['Date', 'Time'])
        
        daily_counts = df.groupby("Date").size()
        for date, count in daily_counts.items():
            expected_count = 390 if symbol.endswith(".US") else 330  # 美股一天390分钟，港股一天330分钟
            completeness = count / expected_count * 100
            status = "完整" if completeness > 95 else "部分缺失" if completeness > 70 else "严重缺失"
            print(f"  - {date}: {count} 条记录 ({completeness:.1f}%, {status})")
        
        # 4. 检查是否有异常的价格跳跃
        df_sorted = df.sort_values(["Date", "Time"])
        df_sorted["price_change_pct"] = df_sorted["Close"].pct_change() * 100
        large_jumps = df_sorted[abs(df_sorted["price_change_pct"]) > 5]  # 超过5%的价格变化
        
        if not large_jumps.empty:
            print("\n检测到异常价格跳跃:")
            for _, row in large_jumps.head(5).iterrows():  # 只显示前5个异常
                print(f"  - {row['Date']} {row['Time']}: 价格变化 {row['price_change_pct']:.2f}% ({row['Close']:.2f})")
            
            if len(large_jumps) > 5:
                print(f"  ... 还有 {len(large_jumps) - 5} 个异常价格跳跃")
        else:
            print("\n未检测到异常价格跳跃")
        
        # 5. 检查VWAP和价格的合理性
        if "VWAP" in df.columns:
            vwap_price_diff = ((df["VWAP"] - df["Close"]) / df["Close"] * 100).abs()
            large_vwap_diff = df[vwap_price_diff > 10]  # VWAP和价格相差超过10%
            
            if not large_vwap_diff.empty:
                print("\n检测到VWAP和价格的异常差异:")
                for _, row in large_vwap_diff.head(5).iterrows():
                    diff_pct = ((row["VWAP"] - row["Close"]) / row["Close"] * 100)
                    print(f"  - {row['Date']} {row['Time']}: VWAP {row['VWAP']:.2f} vs 价格 {row['Close']:.2f} (差异: {diff_pct:.2f}%)")
                
                if len(large_vwap_diff) > 5:
                    print(f"  ... 还有 {len(large_vwap_diff) - 5} 个VWAP异常")
            else:
                print("\nVWAP和价格差异在正常范围内")
        
        # 数据完整性总结
        print("\n历史数据完整性总结:")
        total_expected = date_range * (390 if symbol.endswith(".US") else 330)
        total_actual = len(df)
        overall_completeness = total_actual / total_expected * 100 if total_expected > 0 else 0
        
        if overall_completeness > 95:
            print(f"数据完整性良好: {overall_completeness:.1f}% ({total_actual}/{total_expected})")
        elif overall_completeness > 80:
            print(f"数据完整性一般: {overall_completeness:.1f}% ({total_actual}/{total_expected})")
        else:
            print(f"数据完整性较差: {overall_completeness:.1f}% ({total_actual}/{total_expected})")
        
        # 更新全局缓存和日期
        # 只保留完整的交易日数据（数据点数量 >= 300）
        daily_counts_for_cache = df.groupby("Date").size()
        complete_dates_for_cache = daily_counts_for_cache[daily_counts_for_cache >= 300].index.tolist()
        
        # 如果有完整的交易日数据，则只保留这些数据
        if complete_dates_for_cache:
            df_complete = df[df["Date"].isin(complete_dates_for_cache)].copy()
            if not df_complete.empty:
                print(f"更新历史数据缓存，只保留完整交易日数据 ({len(complete_dates_for_cache)} 天)")
                HISTORICAL_DATA_CACHE = df_complete.copy()
            else:
                print("警告: 过滤后没有完整的交易日数据，使用原始数据")
                HISTORICAL_DATA_CACHE = df.copy()
        else:
            print("警告: 没有找到完整的交易日数据，使用原始数据")
            HISTORICAL_DATA_CACHE = df.copy()
            
        LAST_HISTORICAL_DATA_DATE = now_et.date()  # 使用当前日期而不是current_date
        print(f"更新历史数据缓存，日期: {LAST_HISTORICAL_DATA_DATE}")
        
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
        
        # 打印一些重复组合的示例，帮助调试
        print("重复组合示例:")
        for _, row in duplicate_combinations.head(5).iterrows():
            date = row['Date']
            time = row['Time']
            duplicates = df_copy[(df_copy['Date'] == date) & (df_copy['Time'] == time)]
            print(f"  - {date} {time}: 出现 {len(duplicates)} 次")
            for i, dup in duplicates.iterrows():
                print(f"    价格: {dup['Close']:.2f}, 成交量: {dup['Volume']}")
        
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

def check_trading_conditions(df, positions_opened_today, max_positions_per_day):
    """
    Check if trading conditions are met for entry
    
    Parameters:
        df: DataFrame with price data and indicators
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
        print(f"已达到每日最大持仓数: {positions_opened_today}/{max_positions_per_day}")
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
    
    print("\n开仓条件判断:")
    print(f"当前价格: {price:.2f}")
    print(f"VWAP: {vwap:.2f}")
    print(f"上边界: {upper:.2f}")
    print(f"下边界: {lower:.2f}")
    print(f"MACD柱状图值: {macd_histogram:.6f}")
    
    # Check for long entry
    long_macd_condition = macd_histogram > 0 if USE_MACD else True
    long_price_above_upper = price > upper
    long_price_above_vwap = price > vwap
    
    print("\n多头入场条件:")
    print(f"价格 > 上边界: {long_price_above_upper} ({price:.2f} > {upper:.2f})")
    print(f"价格 > VWAP: {long_price_above_vwap} ({price:.2f} > {vwap:.2f})")
    print(f"MACD柱状图 > 0: {long_macd_condition} ({macd_histogram:.6f} > 0)")
    
    if long_price_above_upper and long_price_above_vwap and long_macd_condition:
        print("满足多头入场条件!")
        # Long entry allowed
        # Initial stop: max(UpperBound, VWAP)
        stop_price = max(upper, vwap)
        return 1, price, stop_price
    
    # Check for short entry
    short_macd_condition = macd_histogram < 0 if USE_MACD else True
    short_price_below_lower = price < lower
    short_price_below_vwap = price < vwap
    
    print("\n空头入场条件:")
    print(f"价格 < 下边界: {short_price_below_lower} ({price:.2f} < {lower:.2f})")
    print(f"价格 < VWAP: {short_price_below_vwap} ({price:.2f} < {vwap:.2f})")
    print(f"MACD柱状图 < 0: {short_macd_condition} ({macd_histogram:.6f} < 0)")
    
    if short_price_below_lower and short_price_below_vwap and short_macd_condition:
        print("满足空头入场条件!")
        # Short entry allowed
        # Initial stop: min(LowerBound, VWAP)
        stop_price = min(lower, vwap)
        return -1, price, stop_price
    
    print("不满足任何入场条件")
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

# 此函数在模拟交易中不再需要，因为我们直接使用 CHECK_INTERVAL_MINUTES 进行定时检查
# 而不是预先生成允许交易的时间点列表

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
    
    # 不再需要生成允许交易的时间点，直接使用 CHECK_INTERVAL_MINUTES 进行定时检查
    print(f"使用 {check_interval_minutes} 分钟间隔进行定时检查")
    
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
    
    # 跟踪循环次数
    loop_count = 0
    
    # Main trading loop
    while True:
        loop_count += 1
        print(f"\n----- 交易检查循环 #{loop_count} -----")
        
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
                
                # 新的交易日，重置历史数据缓存
                global LAST_HISTORICAL_DATA_DATE, HISTORICAL_DATA_CACHE
                LAST_HISTORICAL_DATA_DATE = None
                HISTORICAL_DATA_CACHE = None
                print("新的交易日，重置历史数据缓存")
            
            last_date = current_date
            
            # Check if it's within trading hours
            current_hour, current_minute = now.hour, now.minute
            start_hour, start_minute = trading_start_time
            end_hour, end_minute = trading_end_time
            
            is_trading_hours = (
                (current_hour > start_hour or (current_hour == start_hour and current_minute >= start_minute)) and
                (current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute))
            )
            
            # 无论是否在交易时间内，都获取历史数据
            print("正在获取历史数据...")
            try:
                df = get_historical_data(symbol, period="1m")
                if df.empty:
                    print("Error: Could not get historical data")
                    time_module.sleep(60)
                    continue
            except Exception as e:
                print(f"获取历史数据时出错: {e}")
                import traceback
                traceback.print_exc()
                print("跳过本次检查")
                time_module.sleep(60)
                continue
                
            # 如果不在交易时间内，则只获取数据但不进行交易
            if not is_trading_hours:
                print(f"当前不在交易时间内 ({trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d})")
                print("已获取历史数据，但不在交易时间内，暂不进行交易")
                
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
                
                # 非交易时间，计算到下一个交易时间的等待时间
                now = get_us_eastern_time()
                next_check_time = None
                
                # 计算今天的交易开始时间
                today = now.date()
                today_start = datetime.combine(today, time(trading_start_time[0], trading_start_time[1]), tzinfo=now.tzinfo)
                
                # 如果当前时间在今天交易开始时间之前，等到交易开始
                if now < today_start:
                    next_check_time = today_start
                    wait_minutes = int((next_check_time - now).total_seconds() / 60)
                    print(f"等待今天交易时段开始，还有约 {wait_minutes} 分钟")
                else:
                    # 已经过了今天的交易时间，等到明天交易开始
                    tomorrow = today + timedelta(days=1)
                    tomorrow_start = datetime.combine(tomorrow, time(trading_start_time[0], trading_start_time[1]), tzinfo=now.tzinfo)
                    next_check_time = tomorrow_start
                    wait_minutes = int((next_check_time - now).total_seconds() / 60)
                    print(f"等待明天交易时段开始，还有约 {wait_minutes} 分钟")
                
                # 设置最长等待时间为30分钟，然后再次检查
                # 这样可以确保系统能定期更新，但不会太频繁
                wait_seconds = min(1800, (next_check_time - now).total_seconds())
                print(f"设置下次检查时间为 {wait_seconds/60:.1f} 分钟后 ({now + timedelta(seconds=wait_seconds)})")
                
                try:
                    # 对于较长时间等待，每30分钟输出一条状态消息
                    segments = max(1, int(wait_seconds / 1800))
                    segment_length = wait_seconds / segments
                    
                    for i in range(segments):
                        time_module.sleep(segment_length)
                        minutes_passed = int((i+1) * segment_length / 60)
                        minutes_total = int(wait_seconds / 60)
                        print(f"非交易时间等待中...已过 {minutes_passed} 分钟，共需 {minutes_total} 分钟")
                    
                    print("等待结束，准备下一次检查")
                except Exception as e:
                    print(f"等待过程中出错: {e}")
                continue
            
            # 在交易时间内，继续处理数据和交易逻辑
            print("在交易时间内，继续处理数据和交易逻辑...")
            
            try:
                # Calculate VWAP
                print("计算VWAP指标...")
                prices = df["Close"].values
                volumes = df["Volume"].values
                df["VWAP"] = calculate_vwap_incrementally(prices, volumes)
                
                # 创建一个副本用于计算指标，避免修改原始数据
                df_indicators = df.copy()
                
                # Calculate MACD if needed
                if USE_MACD and "MACD_histogram" not in df_indicators.columns:
                    print("计算MACD指标...")
                    df_indicators = calculate_macd(df_indicators)
                
                # Calculate noise area boundaries
                print("计算噪声区域边界...")
                df_indicators = calculate_noise_area(df_indicators, lookback_days)
                
                # 确保保留原始的上下边界值
                df = df_indicators.copy()
            except Exception as e:
                print(f"计算指标时出错: {e}")
                import traceback
                traceback.print_exc()
                print("跳过本次检查")
                time_module.sleep(60)
                continue
            
            # Check for missing data
            if df["UpperBound"].isna().any() or df["LowerBound"].isna().any():
                print("警告: 边界数据缺失，跳过本次检查")
                time_module.sleep(60)
                continue
            
            # Get current positions
            print("获取当前持仓...")
            try:
                all_positions = get_current_positions()
                # 只关注指定标的的持仓
                positions = {symbol: all_positions[symbol]} if symbol in all_positions else {}
                if positions:
                    print(f"当前持仓 ({symbol}): {positions[symbol]}")
                else:
                    print(f"当前无 {symbol} 持仓")
            except Exception as e:
                print(f"获取持仓时出错: {e}")
                import traceback
                traceback.print_exc()
                print("使用上次记录的持仓信息继续")
                positions = {}
            
            # Update position status based on actual positions
            if position != 0:
                # Check if we still have the position
                if symbol not in positions or positions[symbol]["quantity"] == 0:
                    # Position was closed externally
                    print(f"{symbol} 持仓已被外部平仓")
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
                
                if exit_signal:
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
                
                # 获取最新实时价格
                quote = get_quote(symbol)
                if not quote:
                    print("Error: Could not get quote")
                    time_module.sleep(60)
                    continue
                
                # 更新最新价格到DataFrame
                latest_price = float(quote.get("last_done", df.iloc[-1]["Close"]))
                print(f"获取实时价格: {latest_price}")
                
                # 获取最新的VWAP和上下边界值
                # 注意：这里我们需要确保使用的是计算噪声区域边界后的正确数据
                # 获取最新日期的数据
                latest_date = df["Date"].max()
                latest_data = df[df["Date"] == latest_date].copy()
                
                if not latest_data.empty:
                    # 获取最新时间点的数据
                    latest_row = latest_data.iloc[-1].copy()
                    latest_row["Close"] = latest_price
                else:
                    print("错误：无法获取最新日期的数据")
                    time_module.sleep(60)
                    continue
                
                # 使用最新的数据进行判断
                # 注意：这里我们直接使用之前计算好的VWAP和上下边界值
                print("\n开仓条件判断:")
                print(f"当前价格: {latest_price:.2f}")
                print(f"VWAP: {latest_row['VWAP']:.2f}")
                print(f"上边界: {latest_row['UpperBound']:.2f}")
                print(f"下边界: {latest_row['LowerBound']:.2f}")
                
                # 获取MACD柱状图值
                macd_histogram = latest_row.get("MACD_histogram", 0)
                print(f"MACD柱状图值: {macd_histogram:.6f}")
                
                # 检查多头入场条件
                long_macd_condition = macd_histogram > 0 if USE_MACD else True
                long_price_above_upper = latest_price > latest_row["UpperBound"]
                long_price_above_vwap = latest_price > latest_row["VWAP"]
                
                print("\n多头入场条件:")
                print(f"价格 > 上边界: {long_price_above_upper} ({latest_price:.2f} > {latest_row['UpperBound']:.2f})")
                print(f"价格 > VWAP: {long_price_above_vwap} ({latest_price:.2f} > {latest_row['VWAP']:.2f})")
                print(f"MACD柱状图 > 0: {long_macd_condition} ({macd_histogram:.6f} > 0)")
                
                signal = 0
                price = latest_price
                stop = None
                
                if long_price_above_upper and long_price_above_vwap and long_macd_condition:
                    print("满足多头入场条件!")
                    signal = 1
                    stop = max(latest_row["UpperBound"], latest_row["VWAP"])
                else:
                    # 检查空头入场条件
                    short_macd_condition = macd_histogram < 0 if USE_MACD else True
                    short_price_below_lower = latest_price < latest_row["LowerBound"]
                    short_price_below_vwap = latest_price < latest_row["VWAP"]
                    
                    print("\n空头入场条件:")
                    print(f"价格 < 下边界: {short_price_below_lower} ({latest_price:.2f} < {latest_row['LowerBound']:.2f})")
                    print(f"价格 < VWAP: {short_price_below_vwap} ({latest_price:.2f} < {latest_row['VWAP']:.2f})")
                    print(f"MACD柱状图 < 0: {short_macd_condition} ({macd_histogram:.6f} < 0)")
                    
                    if short_price_below_lower and short_price_below_vwap and short_macd_condition:
                        print("满足空头入场条件!")
                        signal = -1
                        stop = min(latest_row["LowerBound"], latest_row["VWAP"])
                    else:
                        print("不满足任何入场条件")
                
                if signal != 0:
                    print(f"触发{'多' if signal == 1 else '空'}头入场信号! 价格: {price}, 止损: {stop}")
                    
                    # Calculate position size (use 90% of available capital)
                    available_capital = get_account_balance() * 0.9
                    position_size = floor(available_capital / latest_price)
                    
                    if position_size <= 0:
                        print("Warning: Insufficient capital for position")
                        time_module.sleep(60)
                        continue
                    
                    # 开仓时使用AnyTime设置
                    outside_rth_setting = OutsideRTH.AnyTime
                    
                    side = "Buy" if signal > 0 else "Sell"  # 修复: 确保side变量被定义
                    
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
                
                # 分段sleep，每分钟输出一条状态消息
                segments = int(sleep_seconds / 60) + 1
                for i in range(segments):
                    segment_length = min(60, sleep_seconds - i*60)
                    if segment_length <= 0:
                        break
                        
                    time_module.sleep(segment_length)
                    minutes_passed = int((i*60 + segment_length) / 60)
                    print(f"等待中...已过 {minutes_passed} 分钟，还剩 {int(sleep_seconds - (i*60 + segment_length)) / 60:.1f} 分钟")
                
                print(f"等待结束，开始下一次检查 - {get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"交易循环中出现错误: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈
            print("尝试在60秒后恢复...")
            try:
                time_module.sleep(60)
                print("错误恢复，重新开始检查...")
            except Exception as e2:
                print(f"恢复过程中出现新错误: {e2}")

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
