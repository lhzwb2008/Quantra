import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date
import time as time_module
import os
import sys
import pytz
from math import floor
from decimal import Decimal
from dotenv import load_dotenv

from longport.openapi import Config, TradeContext, QuoteContext, Period, OrderSide, OrderType, TimeInForceType, AdjustType, OutsideRTH, OrderStatus

load_dotenv()

# 固定配置参数
CHECK_INTERVAL_MINUTES = 10
TRADING_START_TIME = (9, 40)  # 交易开始时间：9点40分
TRADING_END_TIME = (15, 40)   # 交易结束时间：15点40分
MAX_POSITIONS_PER_DAY = 3
LOOKBACK_DAYS = 10

# 默认交易品种
SYMBOL = os.environ.get('SYMBOL', 'TQQQ.US')

def get_us_eastern_time():
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

def create_contexts():
    try:
        print("正在初始化长桥API连接...")
        config = Config.from_env()
        print("成功从环境变量加载API配置 (已通过.env文件设置)")
        
        app_key = os.environ.get("LONGPORT_APP_KEY", "")
        if app_key == "" or app_key == "your_app_key_here":
            print("警告: API凭证可能未正确设置。请检查.env文件中的配置。")
        
        quote_ctx = QuoteContext(config)
        print("成功创建行情上下文")
        
        trade_ctx = TradeContext(config)
        print("成功创建交易上下文")
        
        return quote_ctx, trade_ctx
    except Exception as e:
        print(f"Error creating API contexts: {e}")
        return None, None

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
        balance_list = TRADE_CTX.account_balance(currency="USD")
        
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
    except Exception as e:
        print(f"Error getting positions: {e}")
        import traceback
        traceback.print_exc()
        return {}

# 全局变量，用于跟踪最后一次获取历史数据的日期
LAST_HISTORICAL_DATA_DATE = None
HISTORICAL_DATA_CACHE = None

def get_historical_data(symbol, period="1m", count=390, trade_sessions="normal", days_back=None):
    global LAST_HISTORICAL_DATA_DATE, HISTORICAL_DATA_CACHE
    
    try:
        if QUOTE_CTX is None:
            print("Quote context is not initialized")
            return pd.DataFrame()
        
        now_et = get_us_eastern_time()
        current_date = now_et.date()
        
        if LAST_HISTORICAL_DATA_DATE is not None and HISTORICAL_DATA_CACHE is not None:
            if LAST_HISTORICAL_DATA_DATE == current_date:
                print(f"使用缓存的历史数据，最后更新日期: {LAST_HISTORICAL_DATA_DATE}")
                return HISTORICAL_DATA_CACHE
        
        now_hour = now_et.hour
        now_minute = now_et.minute
        is_before_market_open = (now_hour < 9 or (now_hour == 9 and now_minute < 30))
        
        target_days = LOOKBACK_DAYS + 1 if is_before_market_open else LOOKBACK_DAYS
        
        if days_back is None:
            days_back = target_days + 15
        
        print(f"正在获取 {symbol} 的历史数据，目标天数: {target_days}...")
            
        period_map = {
            "1m": Period.Min_1, "5m": Period.Min_5, "15m": Period.Min_15,
            "30m": Period.Min_30, "60m": Period.Min_60, "day": Period.Day,
            "week": Period.Week, "month": Period.Month, "year": Period.Year
        }
        sdk_period = period_map.get(period, Period.Min_1)
        
        adjust_type = AdjustType.ForwardAdjust
        all_candles = []
        
        if period == "1m":
            recent_candles = QUOTE_CTX.history_candlesticks_by_offset(
                symbol, sdk_period, adjust_type, False, 1, now_et
            )
            
            if not recent_candles:
                print(f"警告: 无法获取 {symbol} 的最新数据")
                return pd.DataFrame()
                
            latest_time = recent_candles[0].timestamp
            if isinstance(latest_time, datetime):
                latest_date = latest_time.date()
            else:
                latest_date = datetime.fromtimestamp(latest_time).date()
            
            calendar_days_needed = days_back
            start_date = latest_date - timedelta(days=calendar_days_needed)
            
            all_candles = []
            current_date = latest_date
            trading_days_fetched = 0
            
            fetched_dates = set()
            
            while current_date >= start_date and trading_days_fetched < target_days:
                try:
                    date_str = current_date.strftime("%Y%m%d")
                    
                    if date_str in fetched_dates:
                        current_date -= timedelta(days=1)
                        continue
                    
                    eastern = pytz.timezone('US/Eastern')
                    day_start_time = datetime.combine(current_date, time(9, 30))
                    day_start_time_et = eastern.localize(day_start_time)
                    
                    day_candles = QUOTE_CTX.history_candlesticks_by_offset(
                        symbol, sdk_period, adjust_type, True, 390,
                        day_start_time_et
                    )
                    
                    if day_candles and len(day_candles) > 0:
                        sample_candle = day_candles[0]
                        sample_time = sample_candle.timestamp
                        if isinstance(sample_time, datetime):
                            sample_date = sample_time.date()
                        else:
                            sample_date = datetime.fromtimestamp(sample_time).date()
                        
                        if sample_date != current_date:
                            current_date -= timedelta(days=1)
                            continue
                    
                    if day_candles:
                        all_candles.extend(day_candles)
                        if len(day_candles) >= 300:
                            trading_days_fetched += 1
                        
                        fetched_dates.add(date_str)
                    
                    if trading_days_fetched >= target_days:
                        print(f"已获取 {trading_days_fetched} 个交易日的数据")
                        
                    time_module.sleep(0.5)
                    
                except Exception as e:
                    print(f"获取 {date_str} 数据时出错: {e}")
                
                current_date -= timedelta(days=1)
            
            print(f"数据获取完成，共获取 {trading_days_fetched} 个交易日，{len(all_candles)} 条记录")
            
        else:
            max_request_count = 1000
            now = get_us_eastern_time()
            past_date = now - timedelta(days=days_back)
            
            all_candles = QUOTE_CTX.history_candlesticks_by_offset(
                symbol, sdk_period, adjust_type, False, max_request_count, past_date
            )
        
        data = []
        eastern = pytz.timezone('US/Eastern')

        timestamp_counts = {}
        for candle in all_candles:
            timestamp = candle.timestamp
            if isinstance(timestamp, datetime):
                ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                ts_str = str(timestamp)
            
            timestamp_counts[ts_str] = timestamp_counts.get(ts_str, 0) + 1
        
        duplicates = {ts: count for ts, count in timestamp_counts.items() if count > 1}
        if duplicates:
            print(f"发现 {len(duplicates)} 个重复的时间戳")
            print("将对重复时间戳数据进行处理...")
        
        # 记录处理过的时间戳，用于跳过重复项
        processed_timestamps = set()
        
        for candle in all_candles:
            timestamp = candle.timestamp
            
            if isinstance(timestamp, datetime):
                ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                ts_str = str(timestamp)
                
            # 如果是重复的时间戳且已处理过，则跳过
            if ts_str in duplicates and ts_str in processed_timestamps:
                continue
                
            processed_timestamps.add(ts_str)
            
            if isinstance(timestamp, datetime):
                if timestamp.tzinfo is None:
                    hour = timestamp.hour
                    
                    if symbol.endswith(".US") and 9 <= hour < 17:
                        dt = eastern.localize(timestamp)
                    else:
                        if symbol.endswith(".US") and (hour >= 21 or hour < 5):
                            beijing = pytz.timezone('Asia/Shanghai')
                            dt = beijing.localize(timestamp).astimezone(eastern)
                        else:
                            utc = pytz.utc
                            dt = utc.localize(timestamp).astimezone(eastern)
                else:
                    dt = timestamp.astimezone(eastern)
            else:
                dt = datetime.fromtimestamp(timestamp, pytz.utc).astimezone(eastern)

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
        
        # 提取日期和时间组件
        df["Date"] = df["DateTime"].dt.date
        df["Time"] = df["DateTime"].dt.strftime('%H:%M')
        
        # 过滤非交易时间数据
        if symbol.endswith(".US"):
            df = df[df["Time"].between("09:30", "16:00")]
        
        # 去除重复数据
        df = df.drop_duplicates(subset=['Date', 'Time'])
        
        # 更新全局缓存
        HISTORICAL_DATA_CACHE = df.copy()
        LAST_HISTORICAL_DATA_DATE = now_et.date()
        
        return df
    except Exception as e:
        print(f"Error getting historical data: {e}")
        import traceback
        traceback.print_exc()
        # 修复缓存重置问题：出现异常时重置缓存日期，确保下次重新获取数据
        LAST_HISTORICAL_DATA_DATE = None
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
    cum_vol = 0
    cum_pv = 0
    vwaps = []
    
    for price, volume in zip(prices, volumes):
        cum_vol += volume
        cum_pv += price * volume
        
        if cum_vol > 0:
            vwap = cum_pv / cum_vol
        else:
            vwap = price
            
        vwaps.append(vwap)
        
    return vwaps

def calculate_noise_area(df, lookback_days=10):
    df_copy = df.copy()
    
    df_copy["day_open"] = df_copy.groupby("Date")["Open"].transform("first")
    
    unique_dates = sorted(df_copy["Date"].unique())
    prev_close_map = {}
    
    for i in range(1, len(unique_dates)):
        prev_date = unique_dates[i-1]
        curr_date = unique_dates[i]
        prev_close = df_copy[df_copy["Date"] == prev_date]["Close"].iloc[-1]
        prev_close_map[curr_date] = prev_close
    
    df_copy["prev_close"] = df_copy["Date"].map(prev_close_map)
    
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
    
    date_refs_df = pd.DataFrame(date_refs)
    
    df_copy = df_copy.drop(columns=["upper_ref", "lower_ref"], errors="ignore")
    df_copy = pd.merge(df_copy, date_refs_df, on="Date", how="left")
    
    df_copy["ret"] = df_copy["Close"] / df_copy["day_open"] - 1
    
    date_time_counts = df_copy.groupby(['Date', 'Time']).size()
    duplicate_combinations = date_time_counts[date_time_counts > 1].reset_index()[['Date', 'Time']]
    
    if not duplicate_combinations.empty:
        print(f"警告: 发现{len(duplicate_combinations)}个重复的日期和时间组合，将保留每个组合的最后一条记录")
        df_copy = df_copy.drop_duplicates(subset=['Date', 'Time'], keep='last')
    
    pivot = df_copy.pivot(index="Date", columns="Time", values="ret").abs()
    
    latest_date = max(unique_dates)
    lookback_start_idx = max(0, len(unique_dates) - lookback_days)
    lookback_dates = unique_dates[lookback_start_idx:]
    
    pivot = pivot.loc[pivot.index.isin(lookback_dates)]
    
    sigma = pivot.rolling(window=lookback_days, min_periods=1).mean()
    
    if sigma.isna().all().all():
        print("警告: 没有足够的历史数据来计算噪声区域边界，使用所有可用数据的平均值")
        mean_ret = df_copy["ret"].abs().mean()
        sigma = pd.DataFrame(mean_ret, index=pivot.index, columns=pivot.columns)
    
    sigma = sigma.stack().reset_index(name="sigma")
    
    df_copy = pd.merge(df_copy, sigma, on=["Date", "Time"], how="left")
    
    if df_copy["sigma"].isna().any():
        print("为当天缺失的sigma值填充历史平均值...")
        
        today_missing = df_copy[(df_copy["Date"] == latest_date) & df_copy["sigma"].isna()]
        
        if not today_missing.empty:
            print(f"当天有 {len(today_missing)} 个时间点缺失sigma值")
            
            for _, row in today_missing.iterrows():
                time_val = row["Time"]
                
                historical_sigmas = df_copy[(df_copy["Date"] < latest_date) & (df_copy["Time"] == time_val) & (~df_copy["sigma"].isna())]["sigma"]
                
                if not historical_sigmas.empty:
                    avg_sigma = historical_sigmas.mean()
                    df_copy.loc[(df_copy["Date"] == latest_date) & (df_copy["Time"] == time_val), "sigma"] = avg_sigma
                else:
                    all_avg_sigma = df_copy[~df_copy["sigma"].isna()]["sigma"].mean()
                    df_copy.loc[(df_copy["Date"] == latest_date) & (df_copy["Time"] == time_val), "sigma"] = all_avg_sigma
    
    df_copy["UpperBound"] = df_copy["upper_ref"] * (1 + df_copy["sigma"])
    df_copy["LowerBound"] = df_copy["lower_ref"] * (1 - df_copy["sigma"])
    
    print("\n噪声区域边界计算结果:")
    print(f"唯一日期数量: {len(unique_dates)}")
    print(f"最新日期: {latest_date}")
    
    return df_copy

def submit_order(symbol, side, quantity, order_type="MO", price=None, outside_rth=None):
    try:
        if TRADE_CTX is None:
            print("Trade context is not initialized")
            return None
        
        print(f"订单参数: symbol={symbol}, side={side}, quantity={quantity}, order_type={order_type}, price={price}, outside_rth={outside_rth}")
            
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
            
        # 直接使用Decimal处理数量，避免字符串转换导致的精度问题
        dec_quantity = Decimal(str(quantity)) if not isinstance(quantity, Decimal) else quantity
        
        print(f"提交订单: {symbol}, {side}, 数量: {quantity}, 订单类型: {sdk_order_type}, outside_rth: {outside_rth}")
        
        if sdk_order_type == OrderType.LO and price is not None:
            # 优化：使用Decimal处理价格，避免精度损失
            dec_price = Decimal(str(price)) if not isinstance(price, Decimal) else price
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
            print(f"提交市价单")
            response = TRADE_CTX.submit_order(
                symbol=symbol,
                order_type=OrderType.MO,
                side=sdk_side,
                submitted_quantity=dec_quantity,
                time_in_force=time_in_force,
                outside_rth=outside_rth
            )
        
        print(f"订单提交成功: {response}")
        return response.order_id
    except Exception as e:
        print(f"提交订单时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_order_status(order_id):
    try:
        if TRADE_CTX is None:
            print("Trade context is not initialized")
            return {}
        
        if not isinstance(order_id, str):
            order_id = str(order_id)
            
        print(f"正在获取订单 {order_id} 的状态...")
            
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
        
        print(f"订单状态获取成功: {status_str}")
        return order_info
    except Exception as e:
        print(f"获取订单状态时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}

def check_trading_conditions(df, positions_opened_today, max_positions_per_day):
    # 检查是否已达到每日最大持仓数
    if positions_opened_today >= max_positions_per_day:
        print(f"已达到每日最大持仓数: {positions_opened_today}/{max_positions_per_day}")
        return 0, None, None
    
    latest = df.iloc[-1]
    
    price = latest["Close"]
    vwap = latest["VWAP"]
    upper = latest["UpperBound"]
    lower = latest["LowerBound"]
    
    print("\n开仓条件判断:")
    print(f"当前价格: {price:.2f}")
    print(f"VWAP: {vwap:.2f}")
    print(f"上边界: {upper:.2f}")
    print(f"下边界: {lower:.2f}")
    
    long_price_above_upper = price > upper
    long_price_above_vwap = price > vwap
    
    print("\n多头入场条件:")
    print(f"价格 > 上边界: {long_price_above_upper} ({price:.2f} > {upper:.2f})")
    print(f"价格 > VWAP: {long_price_above_vwap} ({price:.2f} > {vwap:.2f})")
    
    if long_price_above_upper and long_price_above_vwap:
        print("满足多头入场条件!")
        stop_price = max(upper, vwap)
        return 1, price, stop_price
    
    short_price_below_lower = price < lower
    short_price_below_vwap = price < vwap
    
    print("\n空头入场条件:")
    print(f"价格 < 下边界: {short_price_below_lower} ({price:.2f} < {lower:.2f})")
    print(f"价格 < VWAP: {short_price_below_vwap} ({price:.2f} < {vwap:.2f})")
    
    if short_price_below_lower and short_price_below_vwap:
        print("满足空头入场条件!")
        stop_price = min(lower, vwap)
        return -1, price, stop_price
    
    print("不满足任何入场条件")
    return 0, None, None

def check_exit_conditions(df, position_direction, trailing_stop):
    latest = df.iloc[-1]
    
    price = latest["Close"]
    vwap = latest["VWAP"]
    upper = latest["UpperBound"]
    lower = latest["LowerBound"]
    
    if position_direction == 1:  # Long position
        new_stop = max(upper, vwap)
        
        if trailing_stop is not None:
            new_stop = max(trailing_stop, new_stop)
        
        exit_signal = price < new_stop
        return exit_signal, new_stop
    
    elif position_direction == -1:  # Short position
        new_stop = min(lower, vwap)
        
        if trailing_stop is not None:
            new_stop = min(trailing_stop, new_stop)
        
        exit_signal = price > new_stop
        return exit_signal, new_stop
    
    return False, None

def is_trading_day(symbol=None):
    try:
        if QUOTE_CTX is None:
            print("Quote context is not initialized")
            return False
            
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
        
        today_str = current_date.strftime("%Y%m%d")
        
        print(f"正在查询当天 {today_str} 是否是交易日 ({market} 市场)")
        
        from longport.openapi import Market
        market_mapping = {
            "US": Market.US, "HK": Market.HK, "CN": Market.CN, "SG": Market.SG
        }
        
        sdk_market = market_mapping.get(market, Market.US)
        
        calendar_resp = QUOTE_CTX.trading_days(
            sdk_market, current_date, current_date
        )
        
        if calendar_resp:
            trading_dates = calendar_resp.trading_days
            half_trading_dates = calendar_resp.half_trading_days
            
            is_trade_day = current_date in trading_dates
            is_half_trade_day = current_date in half_trading_dates
            
            print(f"当前日期: {current_date}")
            print(f"是否是交易日: {is_trade_day}")
            print(f"是否是半日交易: {is_half_trade_day}")
            
            return is_trade_day or is_half_trade_day
        else:
            print("无法获取交易日历数据")
            return False
    except Exception as e:
        print(f"检查交易日时出错: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            now_et = get_us_eastern_time()
            current_weekday = now_et.weekday()
            
            is_weekday = current_weekday < 5
            
            print(f"无法通过API确定交易日，使用传统检查: 当前是星期{current_weekday+1}, 是否是工作日: {is_weekday}")
            
            return is_weekday
        except:
            return True
        
def run_trading_strategy(symbol=SYMBOL, check_interval_minutes=CHECK_INTERVAL_MINUTES,
                        trading_start_time=TRADING_START_TIME, trading_end_time=TRADING_END_TIME,
                        max_positions_per_day=MAX_POSITIONS_PER_DAY, lookback_days=LOOKBACK_DAYS):
    now_et = get_us_eastern_time()
    
    print("\n" + "="*50)
    print(f"启动交易策略 - 交易品种: {symbol}")
    print(f"当前美东时间: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"交易时间: {trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d} (美东时间)")
    print(f"检查间隔: {check_interval_minutes} 分钟")
    print(f"每日最大持仓数: {max_positions_per_day}")
    print(f"回溯天数: {lookback_days}")
    print("="*50 + "\n")
    
    initial_capital = get_account_balance()
    if initial_capital <= 0:
        print("Error: Could not get account balance or balance is zero")
        return
    
    print(f"使用 {check_interval_minutes} 分钟间隔进行定时检查")
    
    # 修改：使用单独的变量跟踪持仓方向和数量
    position_direction = 0  # 0: 无持仓, 1: 多头, -1: 空头
    position_quantity = 0   # 实际持仓数量
    entry_price = None
    trailing_stop = None
    entry_time = None
    order_id = None
    positions_opened_today = 0
    last_date = None
    
    is_us_market = symbol.endswith(".US")
    outside_rth_setting = "ANY_TIME" if is_us_market else "RTH_ONLY"
    
    loop_count = 0
    
    while True:
        loop_count += 1
        print(f"\n----- 交易检查循环 #{loop_count} -----")
        
        try:
            now = get_us_eastern_time()
            current_time = now.strftime("%H:%M")
            current_date = now.date()
            
            print(f"\n当前美东时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            print(f"准备检查今天 {current_date} 是否是交易日...")
            is_today_trading_day = is_trading_day(symbol)
            print(f"交易日检查结果: {'是交易日' if is_today_trading_day else '不是交易日'}")
            
            if not is_today_trading_day:
                print(f"今天不是交易日，跳过交易")
                
                if position_direction != 0 and position_quantity > 0:
                    try:
                        print("非交易日，执行平仓")
                        side = "Sell" if position_direction > 0 else "Buy"
                        
                        outside_rth_setting = OutsideRTH.AnyTime
                        
                        # 提交平仓订单
                        close_order_id = submit_order(symbol, side, position_quantity, outside_rth=outside_rth_setting)
                        if close_order_id:
                            # 等待订单状态更新
                            print(f"平仓订单已提交，ID: {close_order_id}，等待确认...")
                            time_module.sleep(2)
                            
                            # 检查订单状态
                            max_retries = 5
                            for i in range(max_retries):
                                order_status = get_order_status(close_order_id)
                                status = order_status.get("status", "")
                                print(f"订单状态: {status} (检查 {i+1}/{max_retries})")
                                
                                if "FilledStatus" in status or "PartialFilledStatus" in status:
                                    executed_quantity = int(float(order_status.get("executed_quantity", 0)))
                                    print(f"平仓成功: {side} {executed_quantity} {symbol}")
                                    break
                                elif "RejectedStatus" in status or "CanceledStatus" in status:
                                    print(f"订单未成功执行: {status}")
                                    break
                                elif "NewStatus" in status or "PendingNewStatus" in status or "PendingCancelStatus" in status:
                                    if i < max_retries - 1:
                                        print("等待订单状态更新...")
                                        time_module.sleep(3)
                                    else:
                                        print(f"订单状态确认超时，状态: {status}")
                            
                            # 无论订单状态如何，都重置持仓状态
                            position_direction = 0
                            position_quantity = 0
                            entry_price = 0
                            entry_time = None
                            print("持仓状态已重置")
                    except Exception as e:
                        print(f"平仓出错: {e}")
                
                # 设置下次检查时间
                next_check_time = now + timedelta(hours=12)
                wait_seconds = (next_check_time - now).total_seconds()
                
                print(f"等待 {wait_seconds/3600:.1f} 小时后再次检查")
                
                segments = max(1, int(wait_seconds / 3600))
                segment_length = wait_seconds / segments
                
                for i in range(segments):
                    time_module.sleep(segment_length)
                    minutes_passed = int((i+1) * segment_length / 60)
                    minutes_total = int(wait_seconds / 60)
                    print(f"非交易时间等待中...已过 {minutes_passed} 分钟，共需 {minutes_total} 分钟")
                
                continue
            
            if last_date is not None and current_date != last_date:
                print(f"新的交易日开始，重置今日开仓计数")
                positions_opened_today = 0
                
                global LAST_HISTORICAL_DATA_DATE, HISTORICAL_DATA_CACHE
                LAST_HISTORICAL_DATA_DATE = None
                HISTORICAL_DATA_CACHE = None
                print("新的交易日，重置历史数据缓存")
            
            last_date = current_date
            
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
                
            if not is_trading_hours:
                print(f"当前不在交易时间内 ({trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d})")
                print("已获取历史数据，但不在交易时间内，暂不进行交易")
                
                try:
                    if position_direction != 0 and position_quantity > 0:
                        print("交易日结束，执行平仓")
                        side = "Sell" if position_direction > 0 else "Buy"
                        
                        outside_rth_setting = OutsideRTH.AnyTime
                        
                        # 提交平仓订单
                        close_order_id = submit_order(symbol, side, position_quantity, outside_rth=outside_rth_setting)
                        if close_order_id:
                            # 等待订单状态更新
                            print(f"平仓订单已提交，ID: {close_order_id}，等待确认...")
                            time_module.sleep(2)
                            
                            # 检查订单状态
                            max_retries = 5
                            for i in range(max_retries):
                                order_status = get_order_status(close_order_id)
                                status = order_status.get("status", "")
                                print(f"订单状态: {status} (检查 {i+1}/{max_retries})")
                                
                                if "FilledStatus" in status or "PartialFilledStatus" in status:
                                    executed_quantity = int(float(order_status.get("executed_quantity", 0)))
                                    print(f"平仓成功: {side} {executed_quantity} {symbol}")
                                    break
                                elif "RejectedStatus" in status or "CanceledStatus" in status:
                                    print(f"订单未成功执行: {status}")
                                    break
                                elif "NewStatus" in status or "PendingNewStatus" in status or "PendingCancelStatus" in status:
                                    if i < max_retries - 1:
                                        print("等待订单状态更新...")
                                        time_module.sleep(3)
                                    else:
                                        print(f"订单状态确认超时，状态: {status}")
                            
                            # 无论订单状态如何，都重置持仓状态
                            position_direction = 0
                            position_quantity = 0
                            entry_price = 0
                            entry_time = None
                            print("持仓状态已重置")
                except Exception as e:
                    print(f"平仓出错: {e}")
                
                now = get_us_eastern_time()
                
                today = now.date()
                today_start = datetime.combine(today, time(trading_start_time[0], trading_start_time[1]), tzinfo=now.tzinfo)
                
                if now < today_start:
                    next_check_time = today_start
                    wait_minutes = int((next_check_time - now).total_seconds() / 60)
                    print(f"等待今天交易时段开始，还有约 {wait_minutes} 分钟")
                else:
                    tomorrow = today + timedelta(days=1)
                    tomorrow_start = datetime.combine(tomorrow, time(trading_start_time[0], trading_start_time[1]), tzinfo=now.tzinfo)
                    next_check_time = tomorrow_start
                    wait_minutes = int((next_check_time - now).total_seconds() / 60)
                    print(f"等待明天交易时段开始，还有约 {wait_minutes} 分钟")
                
                wait_seconds = min(1800, (next_check_time - now).total_seconds())
                print(f"设置下次检查时间为 {wait_seconds/60:.1f} 分钟后 ({now + timedelta(seconds=wait_seconds)})")
                
                try:
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
            
            print("在交易时间内，继续处理数据和交易逻辑...")
            
            try:
                print("计算VWAP指标...")
                prices = df["Close"].values
                volumes = df["Volume"].values
                df["VWAP"] = calculate_vwap_incrementally(prices, volumes)
                
                df_indicators = df.copy()
                
                print("计算噪声区域边界...")
                df_indicators = calculate_noise_area(df_indicators, lookback_days)
                
                df = df_indicators.copy()
            except Exception as e:
                print(f"计算指标时出错: {e}")
                import traceback
                traceback.print_exc()
                print("跳过本次检查")
                time_module.sleep(60)
                continue
            
            if df["UpperBound"].isna().any() or df["LowerBound"].isna().any():
                print("警告: 边界数据缺失，跳过本次检查")
                time_module.sleep(60)
                continue
            
            print("获取当前持仓...")
            try:
                all_positions = get_current_positions()
                positions = {symbol: all_positions[symbol]} if symbol in all_positions else {}
                if positions:
                    print(f"当前持仓 ({symbol}): {positions[symbol]}")
                    # 如果从API获取到实际持仓，更新持仓数量
                    if symbol in positions and positions[symbol]["quantity"] > 0:
                        position_quantity = positions[symbol]["quantity"]
                        # 补充：根据持仓成本和当前价格推断持仓方向
                        current_price = df.iloc[-1]["Close"]
                        cost_price = positions[symbol]["cost_price"]
                        
                        # 如果之前没有设置方向，根据成本价与当前价格关系推断方向
                        if position_direction == 0:
                            if cost_price < current_price:
                                position_direction = 1  # 多头
                                print(f"根据成本价({cost_price})与当前价格({current_price})推断为多头持仓")
                            else:
                                position_direction = -1  # 空头
                                print(f"根据成本价({cost_price})与当前价格({current_price})推断为空头持仓")
                            
                            # 设置入场价格和时间
                            if entry_price is None:
                                entry_price = cost_price
                                entry_time = now - timedelta(minutes=10)  # 估计入场时间
                                print(f"设置入场价格: {entry_price}")
                else:
                    print(f"当前无 {symbol} 持仓")
                    if position_direction != 0:
                        # 如果持仓方向不为0但API返回无持仓，则重置持仓状态
                        position_direction = 0
                        position_quantity = 0
                        entry_price = None
                        trailing_stop = None
                        entry_time = None
            except Exception as e:
                print(f"获取持仓时出错: {e}")
                import traceback
                traceback.print_exc()
                print("使用上次记录的持仓信息继续")
                positions = {}
            
            if position_direction != 0:
                if symbol not in positions or positions[symbol]["quantity"] == 0:
                    print(f"{symbol} 持仓已被外部平仓")
                    position_direction = 0
                    position_quantity = 0
                    entry_price = None
                    trailing_stop = None
                    entry_time = None
            
            if position_direction != 0 and position_quantity > 0:
                print(f"检查退出条件 (当前持仓方向: {'多' if position_direction == 1 else '空'}, 持仓数量: {position_quantity}, 追踪止损: {trailing_stop})...")
                exit_signal, new_stop = check_exit_conditions(df, position_direction, trailing_stop)
                
                trailing_stop = new_stop
                print(f"更新追踪止损价格: {trailing_stop}")
                
                if exit_signal:
                    print("触发退出信号!")
                    side = "Sell" if position_direction == 1 else "Buy"
                    
                    try:
                        # 提交平仓订单
                        close_order_id = submit_order(symbol, side, position_quantity, outside_rth=outside_rth_setting)
                        if close_order_id:
                            # 等待订单状态更新
                            print(f"平仓订单已提交，ID: {close_order_id}，等待确认...")
                            time_module.sleep(2)
                            
                            # 获取当前价格用于计算盈亏
                            exit_price = df.iloc[-1]["Close"]
                            
                            # 检查订单状态
                            max_retries = 5
                            for i in range(max_retries):
                                order_status = get_order_status(close_order_id)
                                status = order_status.get("status", "")
                                print(f"订单状态: {status} (检查 {i+1}/{max_retries})")
                                
                                if "FilledStatus" in status or "PartialFilledStatus" in status:
                                    # 如果有成交价格，使用成交价格
                                    executed_quantity = int(float(order_status.get("executed_quantity", 0)))
                                    if order_status.get("executed_price") and float(order_status.get("executed_price")) > 0:
                                        exit_price = float(order_status.get("executed_price"))
                                    
                                    # 计算盈亏
                                    pnl = (exit_price - entry_price) * position_direction * position_quantity
                                    pnl_pct = (exit_price / entry_price - 1) * 100 * position_direction
                                    
                                    print(f"平仓成功: {side} {executed_quantity} {symbol} 价格: {exit_price}")
                                    print(f"交易结果: {'盈利' if pnl > 0 else '亏损'} ${abs(pnl):.2f} ({pnl_pct:.2f}%)")
                                    break
                                elif "RejectedStatus" in status or "CanceledStatus" in status:
                                    print(f"订单未成功执行: {status}")
                                    break
                                elif "NewStatus" in status or "PendingNewStatus" in status or "PendingCancelStatus" in status:
                                    if i < max_retries - 1:
                                        print("等待订单状态更新...")
                                        time_module.sleep(3)
                                    else:
                                        print(f"订单状态确认超时，状态: {status}")
                                else:
                                    # 未知状态，等待
                                    if i < max_retries - 1:
                                        print(f"订单状态未知: {status}，等待更新...")
                                        time_module.sleep(3)
                                    else:
                                        print(f"订单状态确认超时，状态未知: {status}")
                            
                            # 无论订单状态如何，都重置持仓状态
                            position_direction = 0
                            position_quantity = 0
                            entry_price = 0
                            entry_time = None
                            print("持仓状态已重置")
                    except Exception as e:
                        print(f"平仓出错: {e}")
            
            elif position_direction == 0 or position_quantity == 0:
                print(f"检查入场条件 (今日已开仓: {positions_opened_today}/{max_positions_per_day})...")
                
                quote = get_quote(symbol)
                if not quote:
                    print("Error: Could not get quote")
                    time_module.sleep(60)
                    continue
                
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
                            time_module.sleep(60)
                            continue
                        
                        outside_rth_setting = OutsideRTH.AnyTime
                        
                        side = "Buy" if signal > 0 else "Sell"
                        
                        order_id = submit_order(symbol, side, position_size, outside_rth=outside_rth_setting)
                        
                        if order_id:
                            # 等待订单状态更新
                            print(f"订单已提交，ID: {order_id}，等待确认订单状态...")
                            time_module.sleep(2)
                            
                            max_retries = 5
                            for retry in range(max_retries):
                                order_info = get_order_status(order_id)
                                status = order_info.get("status", "")
                                
                                print(f"订单状态: {status} (尝试 {retry+1}/{max_retries})")
                                
                                # 成交完成或部分成交
                                if "FilledStatus" in status or "PartialFilledStatus" in status:
                                    executed_quantity = int(float(order_info.get("executed_quantity", 0)))
                                    if executed_quantity > 0:
                                        position_direction = signal
                                        position_quantity = executed_quantity
                                        # 使用实际成交价格
                                        if order_info.get("executed_price") and float(order_info.get("executed_price")) > 0:
                                            entry_price = float(order_info.get("executed_price"))
                                        else:
                                            entry_price = latest_price
                                        entry_time = now
                                        positions_opened_today += 1
                                        
                                        print(f"开仓成功: {side} {executed_quantity} {symbol} 价格: {entry_price}")
                                        break
                                # 已拒绝、已取消或其他最终状态
                                elif "RejectedStatus" in status or "CanceledStatus" in status or "ExpiredStatus" in status or "FailedStatus" in status:
                                    print(f"订单未成功: {status}")
                                    break
                                # 仍在处理中，继续等待
                                elif "NewStatus" in status or "PendingNewStatus" in status or "PendingCancelStatus" in status:
                                    if retry < max_retries - 1:
                                        print(f"订单仍在处理中，等待更新...")
                                        time_module.sleep(3)
                                    else:
                                        print(f"订单状态确认超时，尝试取消订单")
                                        try:
                                            TRADE_CTX.cancel_order(order_id)
                                            print(f"已发送取消请求，订单ID: {order_id}")
                                        except Exception as e:
                                            print(f"取消订单时出错: {e}")
                                else:
                                    # 未知状态，等待
                                    if retry < max_retries - 1:
                                        print(f"订单状态未知: {status}，等待更新...")
                                        time_module.sleep(3)
                                    else:
                                        print(f"订单状态确认超时，状态未知: {status}")
            
            next_check_time = now + timedelta(minutes=check_interval_minutes)
            sleep_seconds = (next_check_time - now).total_seconds()
            if sleep_seconds > 0:
                print(f"等待 {sleep_seconds:.0f} 秒后进行下一次检查...")
                
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
            traceback.print_exc()
            print("尝试在60秒后恢复...")
            try:
                time_module.sleep(60)
                print("错误恢复，重新开始检查...")
            except Exception as e2:
                print(f"恢复过程中出现新错误: {e2}")

if __name__ == "__main__":
    print("\n" + "*"*70)
    print("* 长桥API交易策略启动")
    print("* 版本: 1.0.0")
    print("* 时间:", get_us_eastern_time().strftime("%Y-%m-%d %H:%M:%S"), "(美东时间)")
    print("*"*70 + "\n")
    
    if QUOTE_CTX is None or TRADE_CTX is None:
        print("错误: 无法创建API上下文。请检查API凭证是否正确设置。")
        print("请确保以下环境变量已正确设置:")
        print("- LONGPORT_APP_KEY")
        print("- LONGPORT_APP_SECRET")
        print("- LONGPORT_ACCESS_TOKEN")
        sys.exit(1)
    
    try:
        run_trading_strategy(
            symbol=SYMBOL,
            check_interval_minutes=CHECK_INTERVAL_MINUTES,
            trading_start_time=TRADING_START_TIME,
            trading_end_time=TRADING_END_TIME,
            max_positions_per_day=MAX_POSITIONS_PER_DAY,
            lookback_days=LOOKBACK_DAYS
        )
    finally:
        try:
            if TRADE_CTX is not None:
                TRADE_CTX.close()
        except Exception as e:
            print(f"Error closing trade context: {e}")
    