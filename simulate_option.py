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
CHECK_INTERVAL_MINUTES = 15
TRADING_START_TIME = (9, 40)  # 交易开始时间：9点40分
TRADING_END_TIME = (15, 45)   # 交易结束时间：15点45分
LOOKBACK_DAYS = 2
K1 = 1.2 # 上边界sigma乘数
K2 = 1.2 # 下边界sigma乘数
OPTION_CONTRACTS = int(os.environ.get('OPTION_CONTRACTS', '1'))  # 期权合约数，默认1手

# 默认交易品种
SYMBOL = os.environ.get('SYMBOL', 'SPY.US')

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
    max_retries = 5
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        try:
            config = Config.from_env()
            quote_ctx = QuoteContext(config)
            trade_ctx = TradeContext(config)
            print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] API连接成功")
            return quote_ctx, trade_ctx
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] API连接失败 ({attempt + 1}/{max_retries}): {str(e)}")
                print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] {retry_delay}秒后重试...")
                time_module.sleep(retry_delay)
            else:
                print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] API连接失败，已达最大重试次数")
                raise

QUOTE_CTX, TRADE_CTX = create_contexts()

def get_today_expiry_options(symbol_base="SPY"):
    """获取今日到期的期权链"""
    try:
        # 获取所有到期日
        expiry_dates = QUOTE_CTX.option_chain_expiry_date_list(f"{symbol_base}.US")
        
        # 获取今日日期（YYMMDD格式）
        today = get_us_eastern_time().strftime("%y%m%d")
        
        # 检查今日是否有到期期权
        if today not in expiry_dates:
            print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 今日没有到期的期权")
            return None
            
        # 获取今日到期的期权链
        from longport.openapi import OptionDirection
        option_chain_data = QUOTE_CTX.option_chain_info_by_date(f"{symbol_base}.US", today)
        
        # 转换为列表格式
        option_list = []
        if hasattr(option_chain_data, 'call_options'):
            for opt in option_chain_data.call_options:
                opt.option_type = "C"
                option_list.append(opt)
        if hasattr(option_chain_data, 'put_options'):
            for opt in option_chain_data.put_options:
                opt.option_type = "P"
                option_list.append(opt)
                
        return option_list
    except Exception as e:
        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 获取期权链失败: {str(e)}")
        return None

def select_option_contract(option_chain, current_price, option_type="C"):
    """选择最接近现价的虚值期权"""
    if not option_chain:
        return None
        
    suitable_options = []
    
    for option in option_chain:
        if option.option_type != option_type:
            continue
            
        strike = float(option.strike_price)
        
        if option_type == "C":  # Call期权
            if strike > current_price:
                suitable_options.append(option)
        else:  # Put期权
            if strike < current_price:
                suitable_options.append(option)
    
    if not suitable_options:
        return None
        
    # 选择最接近现价的
    if option_type == "C":
        return min(suitable_options, key=lambda x: float(x.strike_price))
    else:
        return max(suitable_options, key=lambda x: float(x.strike_price))

def get_option_quote(option_symbol):
    """获取期权实时报价"""
    try:
        quotes = QUOTE_CTX.option_quote([option_symbol])
        if quotes and len(quotes) > 0:
            quote = quotes[0]
            return {
                "symbol": quote.symbol,
                "last_price": float(quote.last_done) if quote.last_done else 0,
                "volume": quote.volume,
                "bid": float(quote.bid) if hasattr(quote, 'bid') and quote.bid else 0,
                "ask": float(quote.ask) if hasattr(quote, 'ask') and quote.ask else 0
            }
    except Exception as e:
        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 获取期权报价失败: {str(e)}")
    return None

def get_account_balance():
    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 获取美元账户余额")
    balance_list = TRADE_CTX.account_balance()  # 不需要指定currency参数
    
    # 从cash_infos中找到USD的可用现金
    usd_available_cash = 0.0
    for balance_info in balance_list:
        for cash_info in balance_info.cash_infos:
            if cash_info.currency == "USD":
                usd_available_cash = float(cash_info.available_cash)
                print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 美元可用现金: ${usd_available_cash:.2f}")
                return usd_available_cash
    
    # 如果没有找到USD账户，返回0
    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 警告: 未找到美元账户，返回余额为0")
    return 0.0

def get_current_positions():
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
    
    print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] 开始获取历史数据: {symbol}")
    
    # 计算起始日期
    start_date = current_date - timedelta(days=days_back)
    
    # 对于1分钟数据使用按日获取的方式
    all_candles = []
    
    # 尝试从今天开始向前获取足够的数据
    date_to_check = current_date
    api_call_count = 0
    while date_to_check >= start_date:
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
                # 每天最多获取390分钟数据（6.5小时交易时间）
                day_candles = QUOTE_CTX.history_candlesticks_by_offset(
                    symbol, sdk_period, adjust_type, True, 390,
                    day_start_time_et
                )
                api_call_count += 1
                break  # 成功则跳出重试循环
            except Exception as e:
                if "rate limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] API限流，等待 {retry_delay} 秒后重试 ({attempt + 1}/{max_retries})")
                        time_module.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] API限流，已达最大重试次数")
                        raise
                else:
                    raise  # 其他错误直接抛出
        
        if day_candles:
            all_candles.extend(day_candles)
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
    
    # 过滤未来日期
    if unique_dates and isinstance(unique_dates[0], date_type):
        unique_dates = [d for d in unique_dates if d <= current_date]
        df_copy = df_copy[df_copy["Date"].isin(unique_dates)]
    
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
    
    if prev_close is None:
        return df
    
    # 根据算法计算参考价格
    upper_ref = max(day_open, prev_close)
    lower_ref = min(day_open, prev_close)
    
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
                        lookback_days=LOOKBACK_DAYS):
    now_et = get_us_eastern_time()
    print(f"启动末日期权交易策略 - 标的品种: {symbol}")
    print(f"当前美东时间: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
    if DEBUG_MODE:
        print(f"调试模式已开启! 使用时间: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
        if DEBUG_ONCE:
            print("单次运行模式已开启，策略将只运行一次")
    
    initial_capital = get_account_balance()
    if initial_capital <= 0:
        print("Error: Could not get account balance or balance is zero")
        sys.exit(1)
    
    # 期权持仓追踪
    calls_traded_today = False
    puts_traded_today = False
    current_call_position = None  # 当前持有的Call期权代码
    current_put_position = None   # 当前持有的Put期权代码
    call_entry_price = None       # Call期权买入价格
    put_entry_price = None        # Put期权买入价格
    
    last_date = None
    outside_rth_setting = OutsideRTH.RTHOnly  # 期权只在正常交易时间交易
    
    # 从symbol中提取基础标的代码
    symbol_base = symbol.split('.')[0]  # 例如从SPY.US提取SPY
    
    while True:
        now = get_us_eastern_time()
        current_date = now.date()
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 主循环开始")
        
        # 检查是否是新交易日，如果是则重置今日交易标志
        if last_date is not None and current_date != last_date:
            calls_traded_today = False
            puts_traded_today = False
            current_call_position = None
            current_put_position = None
        last_date = current_date
        
        # 检查是否是交易时间结束点，如果是且有持仓，则强制平仓
        current_hour, current_minute = now.hour, now.minute
        is_trading_end = current_hour == trading_end_time[0] and current_minute == trading_end_time[1]
        
        if is_trading_end and (current_call_position or current_put_position):
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 交易时间结束，平仓所有期权持仓")
            
            # 平仓Call期权
            if current_call_position:
                try:
                    quote = get_option_quote(current_call_position)
                    if quote:
                        exit_price = quote["last_price"]
                        # 卖出期权
                        close_order_id = submit_order(current_call_position, "Sell", OPTION_CONTRACTS * 100, outside_rth=outside_rth_setting)
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Call期权平仓订单已提交，ID: {close_order_id}")
                        
                        # 计算盈亏
                        if call_entry_price:
                            pnl = (exit_price - call_entry_price) * OPTION_CONTRACTS * 100
                            pnl_pct = (exit_price / call_entry_price - 1) * 100 if call_entry_price > 0 else 0
                            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Call期权平仓: {current_call_position} 买入价: ${call_entry_price:.2f} 卖出价: ${exit_price:.2f}")
                            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 盈亏: ${pnl:.2f} ({pnl_pct:.2f}%)")
                except Exception as e:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Call期权平仓失败: {str(e)}")
                    
            # 平仓Put期权
            if current_put_position:
                try:
                    quote = get_option_quote(current_put_position)
                    if quote:
                        exit_price = quote["last_price"]
                        # 卖出期权
                        close_order_id = submit_order(current_put_position, "Sell", OPTION_CONTRACTS * 100, outside_rth=outside_rth_setting)
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Put期权平仓订单已提交，ID: {close_order_id}")
                        
                        # 计算盈亏
                        if put_entry_price:
                            pnl = (exit_price - put_entry_price) * OPTION_CONTRACTS * 100
                            pnl_pct = (exit_price / put_entry_price - 1) * 100 if put_entry_price > 0 else 0
                            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Put期权平仓: {current_put_position} 买入价: ${put_entry_price:.2f} 卖出价: ${exit_price:.2f}")
                            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 盈亏: ${pnl:.2f} ({pnl_pct:.2f}%)")
                except Exception as e:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Put期权平仓失败: {str(e)}")
                    
            # 重置持仓状态
            current_call_position = None
            current_put_position = None
            call_entry_price = None
            put_entry_price = None
            
            if DEBUG_MODE and DEBUG_ONCE:
                print("\n调试模式单次运行完成，程序退出")
                break
            continue
        
        # 检查是否是交易日（调试模式下保持原有逻辑）
        is_today_trading_day = is_trading_day(symbol)
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 是否交易日: {is_today_trading_day}")
            
        if not is_today_trading_day:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 今天不是交易日，跳过交易")
            if current_call_position or current_put_position:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 非交易日，执行平仓")
                side = "Sell" if current_call_position else "Buy"
                close_order_id = submit_order(symbol, side, OPTION_CONTRACTS * 100, outside_rth=outside_rth_setting)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓订单已提交，ID: {close_order_id}")
                current_call_position = None
                current_put_position = None
                call_entry_price = None
                put_entry_price = None
            next_check_time = now + timedelta(hours=12)
            wait_seconds = (next_check_time - now).total_seconds()
            time_module.sleep(wait_seconds)
            continue
            
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
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 历史数据获取完成: {len(df)} 条")
            
        # 调试模式下，根据指定时间截断数据
        if DEBUG_MODE:
            # 截断到调试时间之前的数据
            df = df[df["DateTime"] <= now]
            
        if not is_trading_hours:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 当前不在交易时间内 ({trading_start_time[0]:02d}:{trading_start_time[1]:02d} - {trading_end_time[0]:02d}:{trading_end_time[1]:02d})")
            if current_call_position or current_put_position:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 交易日结束，执行平仓")
                side = "Sell" if current_call_position else "Buy"
                close_order_id = submit_order(symbol, side, OPTION_CONTRACTS * 100, outside_rth=outside_rth_setting)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 平仓订单已提交，ID: {close_order_id}")
                current_call_position = None
                current_put_position = None
                call_entry_price = None
                put_entry_price = None
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
        
        if not current_call_position and not current_put_position:
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
            else:
                # 正常模式: 使用API获取实时价格
                quote = get_quote(symbol)
                latest_price = float(quote.get("last_done", df.iloc[-1]["Close"]))
            
            latest_date = df["Date"].max()
            latest_data = df[df["Date"] == latest_date].copy()
            if not latest_data.empty:
                latest_row = latest_data.iloc[-1].copy()
                latest_row["Close"] = latest_price
                long_price_above_upper = latest_price > latest_row["UpperBound"]
                long_price_above_vwap = latest_price > latest_row["VWAP"]
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 价格={latest_price:.2f}, 上界={latest_row['UpperBound']:.2f}, VWAP={latest_row['VWAP']:.2f}, 下界={latest_row['LowerBound']:.2f}")
                signal = 0
                price = latest_price
                stop = None
                if long_price_above_upper and long_price_above_vwap:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 满足多头入场条件!")
                    signal = 1
                    stop = max(latest_row["UpperBound"], latest_row["VWAP"])
                else:
                    short_price_below_lower = latest_price < latest_row["LowerBound"]
                    short_price_below_vwap = latest_price < latest_row["VWAP"]
                    if short_price_below_lower and short_price_below_vwap:
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 满足空头入场条件!")
                        signal = -1
                        stop = min(latest_row["LowerBound"], latest_row["VWAP"])
                    else:
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 不满足入场条件: 多头({long_price_above_upper} & {long_price_above_vwap}), 空头({short_price_below_lower} & {short_price_below_vwap})")
                
                # 检查是否满足开仓条件
                if signal != 0:
                    if (signal == 1 and calls_traded_today) or (signal == -1 and puts_traded_today):
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 今日已交易过{'Call' if signal == 1 else 'Put'}期权，跳过")
                        continue
                    
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 触发{'多' if signal == 1 else '空'}头入场信号! 价格: {price}, 止损: {stop}")
                    
                    # 获取今日到期的期权链
                option_chain = get_today_expiry_options(symbol_base)
                if not option_chain:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 今日没有到期的期权，跳过交易")
                    continue
                
                # 选择合适的期权合约
                option_type = "C" if signal > 0 else "P"
                selected_option = select_option_contract(option_chain, latest_price, option_type)
                
                if not selected_option:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 没有找到合适的{'Call' if signal > 0 else 'Put'}期权")
                    continue
                
                option_symbol = selected_option.symbol
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 选择期权: {option_symbol} 行权价: {selected_option.strike_price}")
                
                # 获取期权报价
                option_quote = get_option_quote(option_symbol)
                if not option_quote:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 无法获取期权报价")
                    continue
                
                # 提交期权买入订单
                try:
                    order_id = submit_order(option_symbol, "Buy", OPTION_CONTRACTS * 100, outside_rth=outside_rth_setting)
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 期权订单已提交，ID: {order_id}")
                    
                    # 更新持仓状态
                    if signal > 0:
                        current_call_position = option_symbol
                        call_entry_price = option_quote["last_price"]
                        calls_traded_today = True
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 买入Call期权: {option_symbol} 数量: {OPTION_CONTRACTS}手 价格: ${call_entry_price:.2f}")
                    else:
                        current_put_position = option_symbol
                        put_entry_price = option_quote["last_price"]
                        puts_traded_today = True
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 买入Put期权: {option_symbol} 数量: {OPTION_CONTRACTS}手 价格: ${put_entry_price:.2f}")
                    
                except Exception as e:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 期权订单提交失败: {str(e)}")
        
        # 调试模式且单次运行模式，完成一次循环后退出
        if DEBUG_MODE and DEBUG_ONCE:
            print("\n调试模式单次运行完成，程序退出")
            break
            
        next_check_time = now + timedelta(minutes=check_interval_minutes)
        sleep_seconds = (next_check_time - now).total_seconds()
        if sleep_seconds > 0:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 等待 {sleep_seconds:.0f} 秒")
            time_module.sleep(sleep_seconds)

if __name__ == "__main__":
    print("\n长桥API末日期权交易策略启动")
    print("版本: 2.0.0")
    print("时间:", get_us_eastern_time().strftime("%Y-%m-%d %H:%M:%S"), "(美东时间)")
    if DEBUG_MODE:
        print("调试模式已开启")
        if DEBUG_TIME:
            print(f"调试时间: {DEBUG_TIME}")
        if DEBUG_ONCE:
            print("单次运行模式已开启")
    print(f"期权合约数: {OPTION_CONTRACTS}手")
    
    if QUOTE_CTX is None or TRADE_CTX is None:
        print("错误: 无法创建API上下文")
        sys.exit(1)
        
    run_trading_strategy(
        symbol=SYMBOL,
        check_interval_minutes=CHECK_INTERVAL_MINUTES,
        trading_start_time=TRADING_START_TIME,
        trading_end_time=TRADING_END_TIME,
        lookback_days=LOOKBACK_DAYS
    )
