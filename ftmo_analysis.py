import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from backtest import run_backtest
import warnings
import random
import sys
import os
warnings.filterwarnings('ignore')

# 全局数据缓存
_data_cache = {}
_processed_data_cache = {}

def load_and_cache_data(data_path):
    """
    加载并缓存原始数据，避免重复读取文件
    
    参数:
        data_path: 数据文件路径
    
    返回:
        原始数据DataFrame
    """
    if data_path not in _data_cache:
        print(f"首次加载数据文件: {data_path}")
        try:
            price_df = pd.read_csv(data_path, parse_dates=['DateTime'])
            price_df.sort_values('DateTime', inplace=True)
            _data_cache[data_path] = price_df.copy()
            print(f"数据加载完成，共 {len(price_df)} 行数据")
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    else:
        print(f"使用缓存数据: {data_path}")
    
    return _data_cache[data_path].copy()

def get_processed_data(config):
    """
    获取处理后的数据，包括指标计算等
    使用配置的关键参数作为缓存键
    
    参数:
        config: 配置字典
    
    返回:
        处理后的数据DataFrame
    """
    # 创建缓存键，包含影响数据处理的关键参数
    cache_key = (
        config['data_path'],
        config.get('start_date'),
        config.get('end_date'), 
        config.get('lookback_days', 90),
        config.get('K1', 1),
        config.get('K2', 1)
    )
    
    if cache_key not in _processed_data_cache:
        print(f"首次处理数据，参数: lookback_days={config.get('lookback_days')}, K1={config.get('K1')}, K2={config.get('K2')}")
        
        # 加载原始数据
        price_df = load_and_cache_data(config['data_path'])
        
        # 提取日期和时间组件
        price_df['Date'] = price_df['DateTime'].dt.date
        price_df['Time'] = price_df['DateTime'].dt.strftime('%H:%M')
        
        # 按日期范围过滤数据
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        
        if start_date is not None:
            price_df = price_df[price_df['Date'] >= start_date]
        
        if end_date is not None:
            price_df = price_df[price_df['Date'] <= end_date]
        
        # 检查并创建DayOpen和DayClose列
        if 'DayOpen' not in price_df.columns or 'DayClose' not in price_df.columns:
            opening_prices = price_df.groupby('Date').first().reset_index()
            opening_prices = opening_prices[['Date', 'Open']].rename(columns={'Open': 'DayOpen'})

            closing_prices = price_df.groupby('Date').last().reset_index()
            closing_prices = closing_prices[['Date', 'Close']].rename(columns={'Close': 'DayClose'})

            price_df = pd.merge(price_df, opening_prices, on='Date', how='left')
            price_df = pd.merge(price_df, closing_prices, on='Date', how='left')
        
        # 计算前一日收盘价和当日开盘价
        price_df['prev_close'] = price_df.groupby('Date')['DayClose'].transform('first').shift(1)
        price_df['day_open'] = price_df.groupby('Date')['DayOpen'].transform('first')
        
        # 计算参考价格
        unique_dates = price_df['Date'].unique()
        date_refs = []
        for d in unique_dates:
            day_data = price_df[price_df['Date'] == d].iloc[0]
            day_open = day_data['day_open']
            prev_close = day_data['prev_close']
            
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
        
        date_refs_df = pd.DataFrame(date_refs)
        price_df = price_df.drop(columns=['upper_ref', 'lower_ref'], errors='ignore')
        price_df = pd.merge(price_df, date_refs_df, on='Date', how='left')
        
        # 计算回报
        price_df['ret'] = price_df['Close'] / price_df['day_open'] - 1 
        
        # 计算噪声区域边界
        print(f"计算噪声区域边界...")
        pivot = price_df.pivot(index='Date', columns='Time', values='ret').abs()
        lookback_days = config.get('lookback_days', 90)
        sigma = pivot.rolling(window=lookback_days, min_periods=lookback_days).mean().shift(1)
        sigma = sigma.stack().reset_index(name='sigma')
        
        # 合并sigma
        price_df = pd.merge(price_df, sigma, on=['Date', 'Time'], how='left')
        
        # 移除sigma数据不完整的日期
        incomplete_sigma_dates = set()
        for date in price_df['Date'].unique():
            day_data = price_df[price_df['Date'] == date]
            if day_data['sigma'].isna().any():
                incomplete_sigma_dates.add(date)
        
        price_df = price_df[~price_df['Date'].isin(incomplete_sigma_dates)]
        
        # 计算边界
        K1 = config.get('K1', 1)
        K2 = config.get('K2', 1)
        
        price_df['UpperBound'] = price_df['upper_ref'] * (1 + K1 * price_df['sigma'])
        price_df['LowerBound'] = price_df['lower_ref'] * (1 - K2 * price_df['sigma'])
        
        # 缓存处理后的数据
        _processed_data_cache[cache_key] = price_df.copy()
        print(f"数据处理完成，有效数据 {len(price_df)} 行")
        
    else:
        print(f"使用缓存的处理数据")
    
    return _processed_data_cache[cache_key].copy()

def clear_data_cache():
    """清空数据缓存"""
    global _data_cache, _processed_data_cache
    _data_cache.clear()
    _processed_data_cache.clear()
    print("数据缓存已清空")

def run_backtest_with_daily_stop_loss_old(config, daily_stop_loss=0.045):
    """
    运行带有日内止损的回测
    当日亏损达到4.5%时强制平仓并停止当日交易
    
    参数:
        config: 配置字典
        daily_stop_loss: 日内止损阈值（默认4.5%）
    
    返回:
        与run_backtest相同的返回值
    """
    # 先运行正常的回测获取所有交易数据
    daily_results, monthly_results, trades_df, metrics = run_backtest(config)
    
    if trades_df.empty:
        return daily_results, monthly_results, trades_df, metrics
    
    # 确保trades_df的Date列是datetime类型
    if not isinstance(trades_df['Date'].iloc[0], pd.Timestamp):
        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
    
    # 添加时间戳列（如果没有的话）
    if 'entry_time' not in trades_df.columns:
        trades_df['entry_time'] = trades_df['Date']
    if 'exit_time' not in trades_df.columns:
        trades_df['exit_time'] = trades_df['Date']
    
    # 按日期分组处理交易
    initial_capital = config['initial_capital']
    leverage = config.get('leverage', 1)
    
    # 创建新的交易列表，过滤掉触发日内止损后的交易
    filtered_trades = []
    daily_stopped = {}  # 记录已触发止损的日期和时间
    
    # 按日期和时间排序
    trades_df_sorted = trades_df.sort_values(['Date', 'entry_time'])
    
    # 按日期分组处理
    for date, day_trades in trades_df_sorted.groupby(trades_df_sorted['Date'].dt.date):
        # 获取前一天的收盘资金（如果有的话）
        date_idx = daily_results.index.get_loc(pd.Timestamp(date))
        if date_idx > 0:
            day_start_capital = daily_results['capital'].iloc[date_idx - 1]
        else:
            day_start_capital = initial_capital
            
        # 追踪当日的累计损益
        cumulative_pnl = 0
        day_filtered_trades = []
        stop_triggered = False
        
        for idx, trade in day_trades.iterrows():
            # 计算当前损益百分比（相对于当日开始资金）
            current_loss_pct = cumulative_pnl / day_start_capital
            
            # 如果已经触发止损，跳过后续交易
            if stop_triggered:
                continue
                
            # 检查这笔交易是否会触发止损
            # 注意：我们需要考虑交易过程中的损失
            trade_pnl = trade['pnl']
            
            # 模拟交易过程中的最大损失（假设最大损失可能是pnl的1.5倍）
            max_potential_loss = min(trade_pnl, trade_pnl * 1.5 if trade_pnl < 0 else 0)
            potential_loss_pct = (cumulative_pnl + max_potential_loss) / day_start_capital
            
            if potential_loss_pct <= -daily_stop_loss:
                # 触发止损
                stop_triggered = True
                daily_stopped[date] = trade.get('entry_time', trade['Date'])
                
                # 修改这笔交易，假设在触发止损时立即平仓
                # 计算止损时的损失
                stop_loss_pnl = -daily_stop_loss * day_start_capital - cumulative_pnl
                
                # 创建一个修改后的交易记录
                modified_trade = trade.copy()
                modified_trade['pnl'] = stop_loss_pnl
                modified_trade['exit_time'] = modified_trade.get('entry_time', modified_trade['Date'])
                modified_trade['stopped'] = True
                
                day_filtered_trades.append(modified_trade)
                cumulative_pnl += stop_loss_pnl
                break
            else:
                # 未触发止损，正常记录交易
                cumulative_pnl += trade_pnl
                day_filtered_trades.append(trade)
        
        # 添加当日的交易到总列表
        filtered_trades.extend(day_filtered_trades)
    
    # 如果没有任何交易被修改，返回原始结果
    if len(daily_stopped) == 0:
        return daily_results, monthly_results, trades_df, metrics
    
    # 创建新的交易DataFrame
    filtered_trades_df = pd.DataFrame(filtered_trades)
    
    # 重新计算每日资金
    new_daily_results = []
    current_capital = initial_capital
    
    # 获取所有交易日期
    all_dates = pd.date_range(start=daily_results.index[0], end=daily_results.index[-1], freq='D')
    
    for date in all_dates:
        date_only = date.date()
        
        # 计算当日损益
        if not filtered_trades_df.empty:
            # 获取当日的所有交易
            mask = filtered_trades_df['Date'].dt.date == date_only
            if mask.any():
                day_trades = filtered_trades_df[mask]
                day_pnl = day_trades['pnl'].sum()
            else:
                day_pnl = 0
        else:
            day_pnl = 0
            
        current_capital += day_pnl
        
        new_daily_results.append({
            'Date': date,
            'capital': current_capital,
            'daily_pnl': day_pnl
        })
    
    new_daily_df = pd.DataFrame(new_daily_results)
    new_daily_df.set_index('Date', inplace=True)
    
    # 重新计算月度结果
    new_monthly_results = new_daily_df.resample('M').agg({
        'capital': 'last',
        'daily_pnl': 'sum'
    })
    
    # 重新计算指标
    new_metrics = calculate_metrics(new_daily_df, filtered_trades_df, initial_capital)
    
    # 添加止损统计
    new_metrics['daily_stops_triggered'] = len(daily_stopped)
    new_metrics['stop_loss_days'] = list(daily_stopped.keys())
    
    return new_daily_df, new_monthly_results, filtered_trades_df, new_metrics

def calculate_metrics(daily_results, trades_df, initial_capital):
    """计算性能指标"""
    # 计算收益率
    final_capital = daily_results['capital'].iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    
    # 计算年化收益率
    num_days = len(daily_results)
    years = num_days / 252
    irr = (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    # 计算最大回撤
    running_max = daily_results['capital'].cummax()
    drawdown = (daily_results['capital'] - running_max) / running_max
    mdd = drawdown.min()
    
    # 计算夏普比率
    daily_returns = daily_results['capital'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    
    # 计算交易统计
    total_trades = len(trades_df) if not trades_df.empty else 0
    winning_trades = len(trades_df[trades_df['pnl'] > 0]) if not trades_df.empty else 0
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'irr': irr,
        'mdd': mdd,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate
    }

def analyze_ftmo_compliance(daily_results, trades_df, initial_capital, max_daily_loss=0.05, max_total_loss=0.10):
    """
    分析交易结果是否符合FTMO规则
    
    参数:
        daily_results: 每日结果DataFrame
        trades_df: 交易记录DataFrame
        initial_capital: 初始资金
        max_daily_loss: 最大日损失限制 (5%)
        max_total_loss: 最大总损失限制 (10%)
    
    返回:
        分析结果字典
    """
    results = {}
    
    # 1. 计算每日损失
    daily_results = daily_results.copy()
    daily_results['daily_pnl'] = daily_results['capital'].diff()
    daily_results['daily_pnl'].iloc[0] = daily_results['capital'].iloc[0] - initial_capital
    daily_results['daily_loss_pct'] = daily_results['daily_pnl'] / initial_capital
    
    # 2. 计算从初始资金开始的回撤
    daily_results['drawdown_from_initial'] = (daily_results['capital'] - initial_capital) / initial_capital
    daily_results['min_drawdown_from_initial'] = daily_results['drawdown_from_initial'].cummin()
    
    # 3. 统计违规情况
    daily_violations = (daily_results['daily_loss_pct'] < -max_daily_loss).sum()
    total_violation = (daily_results['min_drawdown_from_initial'] < -max_total_loss).any()
    
    # 4. 找出最大日损失和最大总回撤
    max_daily_loss_observed = daily_results['daily_loss_pct'].min()
    max_total_drawdown = daily_results['min_drawdown_from_initial'].min()
    
    # 5. 计算风险指标
    results['daily_violations'] = daily_violations
    results['total_violation'] = total_violation
    results['max_daily_loss_pct'] = max_daily_loss_observed
    results['max_total_drawdown_pct'] = max_total_drawdown
    results['days_to_violation'] = None
    
    # 如果有违规，找出第一次违规的日期
    if total_violation:
        violation_date = daily_results[daily_results['min_drawdown_from_initial'] < -max_total_loss].index[0]
        results['days_to_violation'] = len(daily_results[:violation_date])
    
    # 6. 计算每日最大浮亏（日内回撤）
    if not trades_df.empty:
        # 按日期分组计算每日的交易统计
        daily_trades = trades_df.groupby('Date').agg({
            'pnl': ['sum', 'min', 'max', 'count']
        })
        daily_trades.columns = ['total_pnl', 'worst_trade', 'best_trade', 'num_trades']
        
        results['avg_trades_per_day'] = daily_trades['num_trades'].mean()
        results['worst_single_trade_pct'] = daily_trades['worst_trade'].min() / initial_capital
    
    return results, daily_results

def run_backtest_ftmo_cached(config, daily_stop_loss=0.048):
    """
    使用缓存数据的优化版回测函数，基于最新的backtest.py
    
    参数:
        config: 配置字典
        daily_stop_loss: 日内止损阈值
    
    返回:
        与run_backtest相同的返回值
    """
    # 获取处理后的数据（使用缓存）
    price_df = get_processed_data(config)
    
    if len(price_df) == 0:
        print("警告: 没有有效数据")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    # 从这里开始复制backtest_ftmo.py中的交易逻辑
    # 但跳过数据加载和处理部分
    
    # 获取配置参数
    initial_capital = config.get('initial_capital', 100000)
    leverage = config.get('leverage', 1)
    check_interval_minutes = config.get('check_interval_minutes', 30)
    trading_start_time = config.get('trading_start_time', (10, 0))
    trading_end_time = config.get('trading_end_time', (15, 40))
    max_positions_per_day = config.get('max_positions_per_day', float('inf'))
    
    # 生成允许的交易时间
    allowed_times = []
    start_hour, start_minute = trading_start_time
    end_hour, end_minute = trading_end_time
    
    current_hour, current_minute = start_hour, start_minute
    while current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute):
        allowed_times.append(f"{current_hour:02d}:{current_minute:02d}")
        current_minute += check_interval_minutes
        if current_minute >= 60:
            current_hour += current_minute // 60
            current_minute = current_minute % 60
    
    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    if end_time_str not in allowed_times:
        allowed_times.append(end_time_str)
        allowed_times.sort()
    
    # 初始化回测变量
    capital = initial_capital
    daily_results = []
    all_trades = []
    
    # 获取唯一日期
    unique_dates = price_df['Date'].unique()
    
    # 导入simulate_day函数（从最新的backtest.py导入）
    from backtest import simulate_day
    
    # 处理每个交易日
    for trade_date in unique_dates:
        day_data = price_df[price_df['Date'] == trade_date].copy()
        day_data = day_data.sort_values('DateTime').reset_index(drop=True)
        
        if len(day_data) < 10:
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
        
        prev_close = day_data['prev_close'].iloc[0] if not pd.isna(day_data['prev_close'].iloc[0]) else None
        
        if prev_close is None:
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
        
        # 计算仓位大小
        day_open_price = day_data['day_open'].iloc[0]
        leveraged_capital = capital * leverage
        position_size = int(leveraged_capital / day_open_price)
        
        if position_size <= 0:
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
        
        # 模拟当天的交易（使用backtest.py中的标准simulate_day函数）
        trades = simulate_day(
            day_data, prev_close, allowed_times, position_size, config
        )
        
        # 计算每日盈亏
        day_pnl = sum(trade['pnl'] for trade in trades)
        capital_start = capital
        capital += day_pnl
        daily_return = day_pnl / capital_start
        
        # 存储每日结果
        daily_results.append({
            'Date': trade_date,
            'capital': capital,
            'daily_return': daily_return
        })
        
        # 存储交易
        for trade in trades:
            trade['Date'] = trade_date
            all_trades.append(trade)
    
    # 创建结果DataFrames
    daily_df = pd.DataFrame(daily_results)
    if len(daily_df) > 0:
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df.set_index('Date', inplace=True)
    
    trades_df = pd.DataFrame(all_trades)
    
    # 计算月度结果
    if len(daily_df) > 0:
        monthly = daily_df.resample('ME').first()[['capital']].rename(columns={'capital': 'month_start'})
        monthly['month_end'] = daily_df.resample('ME').last()['capital']
        monthly['monthly_return'] = monthly['month_end'] / monthly['month_start'] - 1
    else:
        monthly = pd.DataFrame()
    
    # 计算简化的指标
    if len(daily_df) > 0 and len(trades_df) > 0:
        total_return = (daily_df['capital'].iloc[-1] - initial_capital) / initial_capital
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        hit_ratio = winning_trades / len(trades_df) if len(trades_df) > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'total_trades': len(trades_df),
            'hit_ratio': hit_ratio
        }
    else:
        metrics = {
            'total_return': 0,
            'total_trades': 0,
            'hit_ratio': 0
        }
    
    return daily_df, monthly, trades_df, metrics

def simulate_ftmo_challenge(config, start_date, profit_target=0.10, max_daily_loss=0.05, max_total_loss=0.10, daily_stop_loss=0.048):
    """
    模拟单次FTMO挑战（无时间限制）
    
    重要改进：考虑日内实时违规情况，不仅仅是收盘后检查
    
    参数:
        config: 配置字典
        start_date: 挑战开始日期
        profit_target: 盈利目标 (10%)
        max_daily_loss: 最大日损失 (5%)
        max_total_loss: 最大总损失 (10%)
        daily_stop_loss: 日内止损阈值 (4.5%)
    
    返回:
        (是否通过, 结束原因, 持续天数, 最终收益率, 失败详情字典)
    """
    # 设置一个较长的结束日期，让策略自然运行
    end_date = config['end_date']  # 使用配置中的结束日期
    
    # 更新配置
    challenge_config = config.copy()
    challenge_config['start_date'] = start_date
    challenge_config['end_date'] = end_date
    challenge_config['print_daily_trades'] = False
    challenge_config['print_trade_details'] = False
    
    # 运行回测，重定向输出到null
    try:
        # 临时重定向stdout来隐藏backtest输出
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        # 直接使用标准回测函数（backtest.py不支持日内止损）
        daily_results, _, trades_df, _ = run_backtest(challenge_config)
        
        # 恢复stdout
        sys.stdout.close()
        sys.stdout = original_stdout
        
    except Exception as e:
        # 确保stdout被恢复
        if sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        
        # 打印详细的错误信息以便调试
        import traceback
        error_details = {
            'error_msg': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'N/A'
        }
        
        print(f"\n⚠️  FTMO挑战模拟出现错误:")
        print(f"  错误类型: {error_details['error_type']}")
        print(f"  错误信息: {error_details['error_msg']}")
        print(f"  开始日期: {error_details['start_date']}")
        print(f"  详细堆栈:")
        print(f"  {error_details['traceback']}")
        
        return False, 'error', 0, 0, error_details
    
    if len(daily_results) == 0:
        return False, 'no_data', 0, 0, {}
    
    # 确保trades_df['Date']是Timestamp类型
    if not trades_df.empty and not isinstance(trades_df['Date'].iloc[0], pd.Timestamp):
        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
    
    initial_capital = config['initial_capital']
    
    # 逐日检查是否达到目标或违反规则
    for i in range(len(daily_results)):
        current_day = i + 1
        current_date = daily_results.index[i]
        
        # 计算当前资金和收益率（收盘时）
        current_capital = daily_results['capital'].iloc[i]
        current_return = (current_capital - initial_capital) / initial_capital
        
        # 计算当日开始资金
        if i == 0:
            day_start_capital = initial_capital
        else:
            day_start_capital = daily_results['capital'].iloc[i-1]
        
        # 计算当日损失（基于初始资金，符合FTMO规则）
        daily_pnl = current_capital - day_start_capital
        daily_loss = daily_pnl / initial_capital
        
        # 重要改进：检查日内是否有违规
        # 获取当日的所有交易
        if not trades_df.empty:
            day_trades = trades_df[trades_df['Date'].dt.date == current_date.date()]
            
            if not day_trades.empty:
                # 模拟日内资金变化
                intraday_capital = day_start_capital
                cumulative_daily_pnl = 0
                
                for _, trade in day_trades.iterrows():
                    # 累计当日盈亏
                    cumulative_daily_pnl += trade['pnl']
                    intraday_capital += trade['pnl']
                    
                    # 检查日内是否违反最大日损失（基于初始资金）
                    intraday_daily_loss = cumulative_daily_pnl / initial_capital
                    if intraday_daily_loss < -max_daily_loss:
                        # 找到违规的具体时间
                        violation_time = trade.get('exit_time', trade.get('entry_time', current_date))
                        failure_details = {
                            'violation_date': current_date.strftime('%Y-%m-%d'),
                            'violation_time': str(violation_time),
                            'violation_type': '日内5%损失限制',
                            'daily_loss_pct': intraday_daily_loss * 100,
                            'capital_at_violation': intraday_capital,
                            'total_return_at_violation': (intraday_capital - initial_capital) / initial_capital * 100,
                            'trade_pnl': trade['pnl'],
                            'cumulative_daily_pnl': cumulative_daily_pnl
                        }
                        return False, 'daily_loss', current_day, (intraday_capital - initial_capital) / initial_capital, failure_details
                    
                    # 检查日内是否违反最大总损失（基于初始资金）
                    intraday_total_return = (intraday_capital - initial_capital) / initial_capital
                    if intraday_total_return < -max_total_loss:
                        violation_time = trade.get('exit_time', trade.get('entry_time', current_date))
                        failure_details = {
                            'violation_date': current_date.strftime('%Y-%m-%d'),
                            'violation_time': str(violation_time),
                            'violation_type': '日内10%总损失限制',
                            'total_return_pct': intraday_total_return * 100,
                            'capital_at_violation': intraday_capital,
                            'trade_pnl': trade['pnl'],
                            'cumulative_daily_pnl': cumulative_daily_pnl
                        }
                        return False, 'total_loss', current_day, intraday_total_return, failure_details
                    
                    # 🆕 检查日内是否违反最大回撤（基于历史最高资金）
                    # 计算到当前交易时刻的最大回撤
                    capital_series_so_far = daily_results['capital'][:i].tolist() + [intraday_capital]
                    if len(capital_series_so_far) > 1:
                        peak_capital_so_far = max(capital_series_so_far)
                        current_drawdown = (intraday_capital - peak_capital_so_far) / initial_capital
                        if current_drawdown < -max_total_loss:  # 回撤超过10%
                            violation_time = trade.get('exit_time', trade.get('entry_time', current_date))
                            failure_details = {
                                'violation_date': current_date.strftime('%Y-%m-%d'),
                                'violation_time': str(violation_time),
                                'violation_type': '日内10%最大回撤限制',
                                'drawdown_pct': abs(current_drawdown) * 100,
                                'peak_capital': peak_capital_so_far,
                                'current_capital': intraday_capital,
                                'trade_pnl': trade['pnl']
                            }
                            return False, 'total_loss', current_day, intraday_total_return, failure_details
        
        # 检查是否达到盈利目标
        if current_return >= profit_target:
            # 计算最大回撤
            capital_series = daily_results['capital'][:i+1]  # 到当前日期的资金序列
            peak_capital = capital_series.cummax()  # 累计最高资金
            drawdown_series = (capital_series - peak_capital) / initial_capital  # 基于初始资金的回撤
            max_drawdown = drawdown_series.min()  # 最大回撤（负值）
            max_drawdown_pct = abs(max_drawdown) * 100  # 转为正值百分比
            
            success_details = {
                'success_date': current_date.strftime('%Y-%m-%d'),
                'final_return_pct': current_return * 100,
                'final_capital': current_capital,
                'max_drawdown_pct': max_drawdown_pct  # 🆕 添加最大回撤
            }
            return True, 'profit_target', current_day, current_return, success_details
        
        # 检查收盘时是否违反日损失规则（基于初始资金）
        if daily_loss < -max_daily_loss:
            failure_details = {
                'violation_date': current_date.strftime('%Y-%m-%d'),
                'violation_time': '收盘时',
                'violation_type': '收盘5%日损失限制',
                'daily_loss_pct': daily_loss * 100,
                'capital_at_violation': current_capital,
                'total_return_at_violation': current_return * 100
            }
            return False, 'daily_loss', current_day, current_return, failure_details
        
        # 检查收盘时是否违反总损失规则（基于初始资金）
        if current_return < -max_total_loss:
            failure_details = {
                'violation_date': current_date.strftime('%Y-%m-%d'),
                'violation_time': '收盘时',
                'violation_type': '收盘10%总损失限制',
                'total_return_pct': current_return * 100,
                'capital_at_violation': current_capital
            }
            return False, 'total_loss', current_day, current_return, failure_details
    
    # 数据用完但未达到目标（这种情况下返回最终收益率）
    final_return = (daily_results['capital'].iloc[-1] - initial_capital) / initial_capital
    return False, 'data_exhausted', len(daily_results), final_return, {}

def save_intermediate_results(results_summary, filename='ftmo_intermediate_results.csv'):
    """保存中间结果"""
    # 不再保存文件，只返回DataFrame用于显示
    if results_summary:
        df = pd.DataFrame(results_summary)
        return df
    return None

def generate_fixed_test_dates(config, num_simulations=100):
    """
    预先生成固定的测试日期列表，确保所有杠杆率使用相同的测试数据
    
    参数:
        config: 基础配置字典
        num_simulations: 模拟次数
        
    返回:
        测试开始日期列表
    """
    start_date = config['start_date']
    end_date = config['end_date']
    total_days = (end_date - start_date).days
    
    # 确保有足够的数据进行可靠的蒙特卡洛分析
    if total_days < 60:
        print(f"警告: 数据时间范围太短，需要至少60天的数据进行可靠分析（当前只有{total_days}天）")
        print(f"提示: 请获取更长时间范围的数据后重新运行分析")
        return None
    
    # 预先生成所有的测试开始日期
    test_dates = []
    max_start_offset = max(0, total_days - 60)
    
    for sim in range(num_simulations):
        start_offset = random.randint(0, max_start_offset)
        sim_start_date = start_date + timedelta(days=start_offset)
        test_dates.append(sim_start_date)
    
    return test_dates

def monte_carlo_ftmo_analysis(config, num_simulations=100, leverage_range=None, use_daily_stop_loss=True, daily_stop_loss=0.048):
    """
    使用蒙特卡洛方法分析FTMO挑战通过率（使用固定的测试日期确保公平性）
    
    参数:
        config: 基础配置字典
        num_simulations: 每个杠杆率的模拟次数
        leverage_range: 杠杆率范围
    """
    if leverage_range is None:
        leverage_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # 🎯 预先生成固定的测试日期列表，确保所有杠杆率使用相同的测试数据
    print(f"🎲 生成固定测试数据集（{num_simulations}个随机日期）...")
    fixed_test_dates = generate_fixed_test_dates(config, num_simulations)
    
    if fixed_test_dates is None:
        return None
    
    print(f"✅ 测试日期生成完成，范围: {min(fixed_test_dates)} 至 {max(fixed_test_dates)}")
    print(f"📊 所有杠杆率将使用相同的{len(fixed_test_dates)}个测试日期，确保公平比较")
    
    results_summary = []
    
    for leverage_idx, leverage in enumerate(leverage_range):
        print(f"\n[{leverage_idx+1}/{len(leverage_range)}] 分析杠杆率: {leverage}x")
        
        # 更新配置
        test_config = config.copy()
        test_config['leverage'] = leverage
        
        # 运行多次模拟
        simulation_results = []
        failure_examples = []  # 存储失败案例的详细信息
        
        for sim in range(num_simulations):
            # 🎯 使用预设的固定测试日期，确保所有杠杆率测试条件相同
            sim_start_date = fixed_test_dates[sim]
            
            # 模拟挑战
            passed, reason, days, final_return, details = simulate_ftmo_challenge(
                test_config, 
                sim_start_date,
                daily_stop_loss=daily_stop_loss if use_daily_stop_loss else None
            )
            
            # 提取最大回撤信息（如果是成功的case）
            max_drawdown_pct = 0
            if passed and 'max_drawdown_pct' in details:
                max_drawdown_pct = details['max_drawdown_pct']
            
            simulation_results.append({
                'passed': passed,
                'reason': reason,
                'days': days,
                'final_return': final_return,
                'start_date': sim_start_date,
                'details': details,
                'max_drawdown_pct': max_drawdown_pct  # 🆕 添加最大回撤
            })
            
            # 收集失败案例（包括程序错误和FTMO规则违规）
            if not passed and reason in ['daily_loss_intraday', 'total_loss_intraday', 'daily_loss', 'total_loss', 'error'] and len(failure_examples) < 5:
                failure_examples.append({
                    'simulation_id': sim + 1,
                    'reason': reason,
                    'details': details,
                    'start_date': sim_start_date.strftime('%Y-%m-%d'),
                    'days': days,
                    'final_return': final_return
                })
            
            # 显示进度
            if (sim + 1) % 10 == 0:
                current_passed = sum(1 for r in simulation_results if r['passed'])
                current_rate = current_passed / (sim + 1) * 100
                print(f"  进度: {sim + 1}/{num_simulations} | 当前通过率: {current_rate:.1f}%")
                
                # 显示最近几次测试的详细信息
                if sim >= 4:  # 显示最近5次测试
                    print(f"  📊 最近5次测试详情:")
                    recent_results = simulation_results[-5:]
                    for j, result in enumerate(recent_results, 1):
                        start_date_str = result.get('start_date', 'N/A')
                        days = result.get('days', 0)
                        final_return = result.get('final_return', 0)
                        passed_status = "✅通过" if result['passed'] else "❌失败"
                        reason = result.get('reason', 'N/A')
                        
                        # 计算结束日期
                        if start_date_str != 'N/A' and days > 0:
                            try:
                                # 尝试解析字符串格式的日期
                                start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                                end_date_obj = start_date_obj + timedelta(days=days-1)
                                end_date_str = end_date_obj.strftime('%Y-%m-%d')
                            except (TypeError, ValueError):
                                # 如果已经是date对象或其他格式，直接使用
                                if hasattr(start_date_str, 'strftime'):
                                    end_date_obj = start_date_str + timedelta(days=days-1)
                                    end_date_str = end_date_obj.strftime('%Y-%m-%d')
                                    start_date_str = start_date_str.strftime('%Y-%m-%d')
                                else:
                                    end_date_str = start_date_str
                        else:
                            end_date_str = start_date_str
                        
                        print(f"    测试{sim-4+j}: {start_date_str} → {end_date_str} | {days}天 | {final_return:+.1f}% | {passed_status} ({reason})")
        
        # 统计结果
        passed_count = sum(1 for r in simulation_results if r['passed'])
        # 只计算有效的测试（排除数据用完的情况）
        valid_results = [r for r in simulation_results if r['reason'] != 'data_exhausted']
        valid_count = len(valid_results)
        
        if valid_count > 0:
            valid_passed_count = sum(1 for r in valid_results if r['passed'])
            pass_rate = valid_passed_count / valid_count
        else:
            pass_rate = 0
            valid_passed_count = 0
        
        # 按失败原因分类
        failure_reasons = {}
        for r in simulation_results:
            if not r['passed']:
                reason = r['reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        # 计算平均完成天数（仅成功的）
        successful_runs = [r for r in simulation_results if r['passed']]
        avg_days_to_success = np.mean([r['days'] for r in successful_runs]) if successful_runs else 0
        
        # 计算所有模拟的平均天数（包括失败的）
        all_days = [r['days'] for r in simulation_results]
        avg_days_all = np.mean(all_days) if all_days else 0
        
        # 计算有效测试的平均天数
        valid_days = [r['days'] for r in valid_results]
        avg_days_valid = np.mean(valid_days) if valid_days else 0
        
        # 计算收益统计
        all_returns = [r['final_return'] for r in simulation_results]
        avg_return = np.mean(all_returns)
        
        summary = {
            'leverage': leverage,
            'pass_rate': pass_rate,
            'num_simulations': num_simulations,
            'valid_count': valid_count,
            'passed_count': valid_passed_count,
            'data_exhausted_count': failure_reasons.get('data_exhausted', 0),
            'avg_days_to_success': avg_days_to_success,
            'avg_days_all': avg_days_all,
            'avg_days_valid': avg_days_valid,
            'avg_return': avg_return,
            'failure_daily_loss': failure_reasons.get('daily_loss', 0),
            'failure_daily_loss_intraday': failure_reasons.get('daily_loss_intraday', 0),
            'failure_total_loss': failure_reasons.get('total_loss', 0),
            'failure_total_loss_intraday': failure_reasons.get('total_loss_intraday', 0),
            'failure_data_exhausted': failure_reasons.get('data_exhausted', 0),
            'failure_error': failure_reasons.get('error', 0),
            'failure_no_data': failure_reasons.get('no_data', 0),
            'failure_examples': failure_examples  # 添加失败案例详情
        }
        
        results_summary.append(summary)
        
        # 打印当前杠杆率结果
        print(f"  ✓ 有效测试: {valid_count}/{num_simulations} (排除数据用完: {failure_reasons.get('data_exhausted', 0)}次)")
        print(f"  ✓ 通过率: {pass_rate*100:.1f}% ({valid_passed_count}/{valid_count})")
        if successful_runs:
            print(f"  ✓ 平均成功天数: {avg_days_to_success:.1f}天")
        print(f"  ✓ 平均有效测试天数: {avg_days_valid:.1f}天")
        # 合并日内和收盘的失败次数
        total_daily_loss_failures = failure_reasons.get('daily_loss', 0) + failure_reasons.get('daily_loss_intraday', 0)
        total_total_loss_failures = failure_reasons.get('total_loss', 0) + failure_reasons.get('total_loss_intraday', 0)
        other_failures = failure_reasons.get('data_exhausted', 0) + failure_reasons.get('error', 0) + failure_reasons.get('no_data', 0)
        
        print(f"  ✓ 失败原因: 日损失{total_daily_loss_failures}次 | 总损失{total_total_loss_failures}次 | 其他{other_failures}次")
        
        # 打印详细的失败原因统计
        if failure_reasons:
            failure_details = []
            for reason, count in failure_reasons.items():
                if count > 0:
                    reason_name = {
                        'daily_loss': '日损失超限',
                        'total_loss': '总损失超限', 
                        'data_exhausted': '数据用完未达目标',
                        'error': '程序错误',
                        'no_data': '无数据'
                    }.get(reason, reason)
                    failure_details.append(f"{reason_name}{count}次")
            print(f"  ✓ 失败详情: {' | '.join(failure_details)}")
        
        # 打印失败案例详情（只显示真正的FTMO规则违规案例）
        if failure_examples:
            print(f"  📋 典型FTMO规则违规案例:")
            for i, example in enumerate(failure_examples, 1):
                details = example['details']
                print(f"    案例{i}: 模拟#{example['simulation_id']} | 开始日期: {example['start_date']} | 持续{example['days']}天")
                print(f"           违规日期: {details.get('violation_date', 'N/A')} {details.get('violation_time', '')}")
                print(f"           违规类型: {details.get('violation_type', example['reason'])}")
                if 'daily_loss_pct' in details:
                    print(f"           当日损失: {details['daily_loss_pct']:.2f}%")
                if 'total_return_pct' in details:
                    print(f"           总收益率: {details['total_return_pct']:.2f}%")
                if 'capital_at_violation' in details:
                    print(f"           违规时资金: ${details['capital_at_violation']:.2f}")
                print()
        elif total_daily_loss_failures == 0 and total_total_loss_failures == 0:
            # 检查是否有程序错误
            error_count = failure_reasons.get('error', 0)
            if error_count > 0:
                print(f"  ⚠️  有{error_count}次程序错误，请检查上方的错误日志")
            else:
                print(f"  ✓ 无FTMO规则违规案例（所有失败都是温和原因）")
        
        # 🆕 显示所有成功案例的详细信息（包括最大回撤）
        success_cases = [r for r in simulation_results if r['passed']]
        if success_cases:
            print(f"\\n  🎉 所有成功案例详情 (共{len(success_cases)}个):")
            for i, case in enumerate(success_cases, 1):
                start_date_str = case['start_date'].strftime('%Y-%m-%d') if hasattr(case['start_date'], 'strftime') else str(case['start_date'])
                end_date_str = 'N/A'
                if case['days'] > 0:
                    try:
                        if hasattr(case['start_date'], 'strftime'):
                            end_date_obj = case['start_date'] + timedelta(days=case['days']-1)
                            end_date_str = end_date_obj.strftime('%Y-%m-%d')
                        else:
                            start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                            end_date_obj = start_date_obj + timedelta(days=case['days']-1)
                            end_date_str = end_date_obj.strftime('%Y-%m-%d')
                    except:
                        end_date_str = 'N/A'
                
                final_return_pct = case['final_return'] * 100
                max_drawdown_pct = case.get('max_drawdown_pct', 0)
                
                print(f"    案例{i:2d}: {start_date_str} → {end_date_str} | {case['days']:3d}天 | 收益: {final_return_pct:+5.1f}% | 最大回撤: {max_drawdown_pct:4.1f}%")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results_summary)
    
    # 找出最优杠杆率（80%通过率以上的最高杠杆）
    qualified = results_df[results_df['pass_rate'] >= 0.8]
    if not qualified.empty:
        optimal = qualified.loc[qualified['leverage'].idxmax()]
        print(f"\n🎯 最优杠杆率（通过率≥80%）:")
        print(f"  杠杆率: {optimal['leverage']}x")
        print(f"  通过率: {optimal['pass_rate']*100:.1f}%")
        print(f"  平均成功天数: {optimal['avg_days_to_success']:.1f}天")
    else:
        print(f"\n⚠️  没有杠杆率能达到80%的通过率")
    
    return results_df

def monte_carlo_multi_timing_analysis(config, num_simulations=100, leverage=4, use_daily_stop_loss=True, daily_stop_loss=0.048):
    """
    同时测试多个时间配置的FTMO挑战通过率，用于验证时间错配是否能分散风险
    
    参数:
        config: 基础配置字典
        num_simulations: 模拟次数
        leverage: 杠杆率
        use_daily_stop_loss: 是否使用日内止损
        daily_stop_loss: 日内止损阈值
    
    返回:
        包含分析结果的DataFrame
    """
    # 定义三种时间配置
    time_configs = [
        {'name': '账户A (9:40-15:40)', 'start': (9, 40), 'end': (15, 40)},
        {'name': '账户B (9:39-15:39)', 'start': (9, 39), 'end': (15, 39)},
        {'name': '账户C (9:41-15:41)', 'start': (9, 41), 'end': (15, 41)}
    ]
    
    # 获取数据的时间范围
    start_date = config['start_date']
    end_date = config['end_date']
    total_days = (end_date - start_date).days
    
    if total_days < 60:
        print(f"警告: 数据时间范围太短，需要至少60天的数据进行可靠分析（当前只有{total_days}天）")
        print(f"提示: 请获取更长时间范围的数据后重新运行分析")
        return None
    
    print(f"\n🔄 多账户时间错配分析")
    print(f"  杠杆率: {leverage}x")
    print(f"  模拟次数: {num_simulations}")
    print(f"  测试配置:")
    for tc in time_configs:
        print(f"    - {tc['name']}")
    print()
    
    # 存储所有模拟结果
    all_simulations = []
    
    # 运行模拟
    for sim in range(num_simulations):
        # 为这次模拟随机选择起始日期（三个账户使用相同的起始日期）
        max_start_offset = max(0, total_days - 60)
        start_offset = random.randint(0, max_start_offset)
        sim_start_date = start_date + timedelta(days=start_offset)
        
        # 存储这次模拟中三个账户的结果
        sim_results = {
            'simulation_id': sim + 1,
            'start_date': sim_start_date
        }
        
        # 打印每次模拟的详细信息
        print(f"\n模拟 #{sim + 1}/{num_simulations} | 开始日期: {sim_start_date.strftime('%Y-%m-%d')}")
        
        # 对每个时间配置运行回测
        for i, tc in enumerate(time_configs):
            # 创建特定时间配置
            test_config = config.copy()
            test_config['leverage'] = leverage
            test_config['trading_start_time'] = tc['start']
            test_config['trading_end_time'] = tc['end']
            
            # 运行模拟，如果数据不够则重试
            max_retries = 5
            for retry in range(max_retries):
                passed, reason, days, final_return, details = simulate_ftmo_challenge(
                    test_config, 
                    sim_start_date,
                    daily_stop_loss=daily_stop_loss if use_daily_stop_loss else None
                )
                
                # 如果数据不够，重新选择起始日期重试
                if reason == 'data_exhausted' and retry < max_retries - 1:
                    # 重新选择起始日期
                    new_start_offset = random.randint(0, max_start_offset)
                    sim_start_date = start_date + timedelta(days=new_start_offset)
                    test_config['start_date'] = sim_start_date
                    continue
                else:
                    break
            
            # 存储结果
            account_key = f'account_{i+1}'
            sim_results[f'{account_key}_passed'] = passed
            sim_results[f'{account_key}_reason'] = reason
            sim_results[f'{account_key}_days'] = days
            sim_results[f'{account_key}_return'] = final_return
            sim_results[f'{account_key}_details'] = details
            sim_results[f'{account_key}_name'] = tc['name']
            
            # 打印账户结果
            status_icon = "✅" if passed else "❌"
            print(f"  {status_icon} {tc['name']}: ", end="")
            
            if passed:
                print(f"成功 | {days}天达到10%收益")
            else:
                if reason in ['daily_loss', 'total_loss']:
                    violation_date = details.get('violation_date', 'N/A')
                    violation_time = details.get('violation_time', 'N/A')
                    
                    # 统一显示失败原因，不区分日内和收盘
                    if reason == 'daily_loss':
                        violation_type = '5%日损失限制'
                    elif reason == 'total_loss':
                        violation_type = '10%总损失限制'
                    else:
                        violation_type = reason
                    
                    # 计算测试区间
                    end_date_str = (sim_start_date + timedelta(days=days-1)).strftime('%Y-%m-%d') if days > 0 else sim_start_date.strftime('%Y-%m-%d')
                    
                    print(f"失败 | 测试区间: {sim_start_date.strftime('%Y-%m-%d')} 至 {end_date_str} | "
                          f"爆仓时间: {violation_date} {violation_time} | "
                          f"原因: {violation_type}")
                    
                    # 如果有更多详细信息，也打印出来
                    if 'daily_loss_pct' in details:
                        print(f"         当日损失: {details['daily_loss_pct']:.2f}%", end="")
                    if 'total_return_pct' in details:
                        print(f" | 总损失: {details['total_return_pct']:.2f}%", end="")
                    print()  # 换行
                else:
                    print(f"失败 | 原因: {reason} | 持续{days}天")
        
        # 计算组合结果
        accounts_passed = sum(1 for i in range(1, 4) if sim_results[f'account_{i}_passed'])
        sim_results['accounts_passed'] = accounts_passed
        sim_results['all_passed'] = accounts_passed == 3
        sim_results['all_failed'] = accounts_passed == 0
        sim_results['at_least_one_passed'] = accounts_passed >= 1
        sim_results['at_least_two_passed'] = accounts_passed >= 2
        
        # 打印组合结果摘要
        if sim_results['all_failed']:
            print(f"  ⚠️  三个账户全部失败!")
        elif sim_results['all_passed']:
            print(f"  🎉 三个账户全部成功!")
        else:
            print(f"  📊 {accounts_passed}/3 个账户成功")
        
        all_simulations.append(sim_results)
        
        # 显示进度（每10次显示一次汇总）
        if (sim + 1) % 10 == 0:
            at_least_one = sum(1 for s in all_simulations if s['at_least_one_passed'])
            all_failed = sum(1 for s in all_simulations if s['all_failed'])
            print(f"\n--- 进度汇总: {sim + 1}/{num_simulations} ---")
            print(f"  至少一个成功: {at_least_one/(sim+1)*100:.1f}% | 全部失败: {all_failed/(sim+1)*100:.1f}%")
            print("-" * 40)
    
    # 统计分析
    print(f"\n\n📊 统计分析结果:")
    print("="*80)
    
    # 单个账户成功率
    for i in range(1, 4):
        account_name = time_configs[i-1]['name']
        passed_count = sum(1 for s in all_simulations if s[f'account_{i}_passed'])
        print(f"{account_name} 成功率: {passed_count/num_simulations*100:.1f}% ({passed_count}/{num_simulations})")
    
    print("\n组合成功率:")
    all_passed = sum(1 for s in all_simulations if s['all_passed'])
    at_least_two = sum(1 for s in all_simulations if s['at_least_two_passed'])
    at_least_one = sum(1 for s in all_simulations if s['at_least_one_passed'])
    all_failed = sum(1 for s in all_simulations if s['all_failed'])
    
    print(f"  三个账户全部成功: {all_passed/num_simulations*100:.1f}% ({all_passed}/{num_simulations})")
    print(f"  至少两个账户成功: {at_least_two/num_simulations*100:.1f}% ({at_least_two}/{num_simulations})")
    print(f"  至少一个账户成功: {at_least_one/num_simulations*100:.1f}% ({at_least_one}/{num_simulations})")
    print(f"  三个账户全部失败: {all_failed/num_simulations*100:.1f}% ({all_failed}/{num_simulations})")
    
    # 分析同时失败的案例
    print(f"\n💥 同时失败案例分析:")
    simultaneous_failures = []
    
    for sim in all_simulations:
        if sim['all_failed']:
            # 检查是否在同一天失败
            failure_dates = []
            failure_info = []
            for i in range(1, 4):
                details = sim[f'account_{i}_details']
                if 'violation_date' in details:
                    failure_dates.append(details['violation_date'])
                    failure_info.append({
                        'account': time_configs[i-1]['name'],
                        'date': details['violation_date'],
                        'time': details.get('violation_time', 'N/A'),
                        'type': details.get('violation_type', sim[f'account_{i}_reason'])
                    })
            
            if len(set(failure_dates)) == 1:  # 同一天失败
                simultaneous_failures.append({
                    'sim_id': sim['simulation_id'],
                    'start_date': sim['start_date'].strftime('%Y-%m-%d'),
                    'failure_date': failure_dates[0],
                    'reasons': [sim[f'account_{i}_reason'] for i in range(1, 4)],
                    'failure_info': failure_info
                })
    
    if simultaneous_failures:
        print(f"  同一天失败的案例: {len(simultaneous_failures)}个")
        for i, case in enumerate(simultaneous_failures[:5], 1):  # 显示前5个
            print(f"\n    案例{i}: 模拟#{case['sim_id']} | 开始:{case['start_date']} | 失败日:{case['failure_date']}")
            for info in case['failure_info']:
                print(f"      - {info['account']}: {info['time']} | {info['type']}")
    else:
        print("  没有发现同一天失败的案例")
    
    # 分析失败时间分布
    print(f"\n📅 失败时间分布分析:")
    failure_day_gaps = []
    
    for sim in all_simulations:
        if sim['accounts_passed'] > 0 and sim['accounts_passed'] < 3:  # 部分成功部分失败
            failure_days = []
            for i in range(1, 4):
                if not sim[f'account_{i}_passed']:
                    failure_days.append(sim[f'account_{i}_days'])
            
            if len(failure_days) >= 2:
                failure_day_gaps.append(max(failure_days) - min(failure_days))
    
    if failure_day_gaps:
        avg_gap = np.mean(failure_day_gaps)
        print(f"  失败时间平均间隔: {avg_gap:.1f}天")
        print(f"  最大间隔: {max(failure_day_gaps)}天")
        print(f"  最小间隔: {min(failure_day_gaps)}天")
    
    # 创建详细结果DataFrame
    results_df = pd.DataFrame(all_simulations)
    
    # 添加摘要统计
    summary = {
        'leverage': leverage,
        'num_simulations': num_simulations,
        'account_1_pass_rate': sum(1 for s in all_simulations if s['account_1_passed']) / num_simulations,
        'account_2_pass_rate': sum(1 for s in all_simulations if s['account_2_passed']) / num_simulations,
        'account_3_pass_rate': sum(1 for s in all_simulations if s['account_3_passed']) / num_simulations,
        'all_pass_rate': all_passed / num_simulations,
        'at_least_two_pass_rate': at_least_two / num_simulations,
        'at_least_one_pass_rate': at_least_one / num_simulations,
        'all_fail_rate': all_failed / num_simulations,
        'simultaneous_failures': len(simultaneous_failures)
    }
    
    return results_df, summary

# 保留原有的函数以便兼容
def rolling_window_analysis(config, window_days=30, leverage_range=None):
    """
    保留原函数签名，但调用新的蒙特卡洛分析
    """
    return monte_carlo_ftmo_analysis(config, num_simulations=100, leverage_range=leverage_range)

def analyze_leverage_ftmo_performance(config, leverage_range=None, num_simulations=100, use_daily_stop_loss=True, daily_stop_loss=0.048):
    """
    分析不同杠杆倍数下FTMO挑战的通过率和爆仓率
    
    参数:
        config: 基础配置字典
        leverage_range: 杠杆倍数范围，默认[1,2,3,4,5,6,7,8,9,10]
        num_simulations: 每个杠杆倍数的模拟次数
        use_daily_stop_loss: 是否使用日内止损
        daily_stop_loss: 日内止损阈值
    
    返回:
        包含各杠杆倍数的通过率和爆仓率的DataFrame
    """
    if leverage_range is None:
        leverage_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print(f"\n🚀 FTMO挑战通过率分析（使用最新配置）")
    print(f"="*80)
    print(f"📈 数据文件: {config['data_path']}")
    print(f"📈 股票代码: {config['ticker']}")
    print(f"📅 数据范围: {config['start_date']} 至 {config['end_date']}")
    print(f"🔄 模拟次数: {num_simulations}次/杠杆率")
    print(f"⚡ 杠杆率范围: {leverage_range}")
    print(f"💰 手续费: {config.get('transaction_fee_per_share', 0.008166):.6f}/股")
    print(f"📦 滑点: {config.get('slippage_per_share', 0.01):.3f}/股")
    if use_daily_stop_loss:
        print(f"🛡️ 日内止损: 启用 ({daily_stop_loss*100:.1f}%)")
    else:
        print(f"🛡️ 日内止损: 禁用")
    print(f"🎯 目标: 达到10%收益即通过（无时间限制）")
    print(f"💥 爆仓标准: 每日最大损失>5%或总损失>10%")
    print(f"="*80)
    
    # 使用已有的蒙特卡洛分析函数
    results_df = monte_carlo_ftmo_analysis(
        config, 
        num_simulations=num_simulations,
        leverage_range=leverage_range,
        use_daily_stop_loss=use_daily_stop_loss,
        daily_stop_loss=daily_stop_loss
    )
    
    if results_df is not None:
        # 打印结果表格
        print(f"\n\n📋 杠杆倍数与通过率/爆仓率关系汇总:")
        print(f"="*120)
        print(f"{'杠杆率':>6} | {'通过率':>7} | {'爆仓率':>7} | {'有效测试':>8} | {'成功次数':>8} | {'爆仓次数':>8} | {'平均成功天数':>10} | {'日损失爆仓':>10} | {'总损失爆仓':>10}")
        print(f"="*120)
        
        for _, row in results_df.iterrows():
            # 计算爆仓率（爆仓 = 日损失 + 总损失）
            total_daily_failures = row['failure_daily_loss'] + row.get('failure_daily_loss_intraday', 0)
            total_total_failures = row['failure_total_loss'] + row.get('failure_total_loss_intraday', 0)
            total_failures = total_daily_failures + total_total_failures
            failure_rate = total_failures / row['valid_count'] if row['valid_count'] > 0 else 0
            
            print(f"{row['leverage']:>6}x | {row['pass_rate']*100:>6.1f}% | {failure_rate*100:>6.1f}% | {row['valid_count']:>8} | {row['passed_count']:>8} | {total_failures:>8} | "
                  f"{row['avg_days_to_success']:>9.1f}天 | {total_daily_failures:>10} | {total_total_failures:>10}")
        
        # 找出最优杠杆率
        print(f"\n\n🎆 最优杠杆率分析:")
        print(f"="*60)
        
        # 按通过率排序，找出最高通过率
        best_pass_rate = results_df.loc[results_df['pass_rate'].idxmax()]
        print(f"📈 最高通过率: {best_pass_rate['leverage']}x杠杆 ({best_pass_rate['pass_rate']*100:.1f}%)")
        
        # 找出80%以上通过率的最高杠杆
        high_success = results_df[results_df['pass_rate'] >= 0.8]
        if not high_success.empty:
            optimal = high_success.loc[high_success['leverage'].idxmax()]
            print(f"🎯 推荐杠杆率（通过率≥80%）: {optimal['leverage']}x杠杆 ({optimal['pass_rate']*100:.1f}%)")
        else:
            print(f"⚠️  没有杠杆率能达到80%的通过率")
        
        # 分析爆仓原因
        print(f"\n💥 爆仓原因分析:")
        print(f"-"*40)
        total_daily_failures_all = results_df['failure_daily_loss'].sum() + results_df.get('failure_daily_loss_intraday', 0).sum()
        total_total_failures_all = results_df['failure_total_loss'].sum() + results_df.get('failure_total_loss_intraday', 0).sum()
        total_all_failures = total_daily_failures_all + total_total_failures_all
        
        if total_all_failures > 0:
            daily_pct = total_daily_failures_all / total_all_failures * 100
            total_pct = total_total_failures_all / total_all_failures * 100
            print(f"📉 日损失爆仓 (>5%): {total_daily_failures_all}次 ({daily_pct:.1f}%)")
            print(f"📉 总损失爆仓 (>10%): {total_total_failures_all}次 ({total_pct:.1f}%)")
        
        print(f"\n📊 关键结论:")
        print(f"-"*40)
        print(f"• 随着杠杆倍数增加，通过率通常会下降")
        print(f"• 随着杠杆倍数增加，爆仓率通常会上升")
        print(f"• 需要在收益潜力和风险控制之间找到平衡")
        print(f"• 建议选择通过率≥80%的最高杠杆倍数")
        
    return results_df

def analyze_single_leverage(config, leverage, use_daily_stop_loss=True, daily_stop_loss=0.048):
    """
    详细分析单个杠杆率的表现
    
    参数:
        config: 基础配置字典
        leverage: 杠杆率
        use_daily_stop_loss: 是否使用日内止损
        daily_stop_loss: 日内止损阈值
    """
    print(f"\n详细分析杠杆率: {leverage}x")
    if use_daily_stop_loss:
        print(f"  使用日内止损: {daily_stop_loss*100:.1f}%")
    
    # 更新配置
    test_config = config.copy()
    test_config['leverage'] = leverage
    test_config['print_daily_trades'] = False
    
    # 运行回测
    daily_results, monthly_results, trades_df, metrics = run_backtest(test_config)
    
    # 全局分析
    analysis, daily_with_metrics = analyze_ftmo_compliance(
        daily_results, 
        trades_df, 
        config['initial_capital']
    )
    
    print(f"\n整体表现:")
    print(f"  年化收益率: {metrics.get('irr', 0)*100:.1f}%")
    print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {metrics.get('mdd', 0)*100:.1f}%")
    print(f"  总交易次数: {metrics.get('total_trades', 0)}")
    
    print(f"\nFTMO规则分析:")
    print(f"  最大日损失: {analysis['max_daily_loss_pct']*100:.2f}%")
    print(f"  最大总回撤: {analysis['max_total_drawdown_pct']*100:.2f}%")
    print(f"  违反5%日损失规则次数: {analysis['daily_violations']}")
    print(f"  是否违反10%总损失规则: {'是' if analysis['total_violation'] else '否'}")
    
    if analysis['days_to_violation']:
        print(f"  首次违规发生在第 {analysis['days_to_violation']} 天")
    
    # 找出最差的几天
    worst_days = daily_with_metrics.nsmallest(5, 'daily_loss_pct')[['daily_loss_pct', 'capital']]
    print(f"\n最差的5个交易日:")
    for date, row in worst_days.iterrows():
        print(f"  {date.strftime('%Y-%m-%d')}: {row['daily_loss_pct']*100:.2f}% (资金: ${row['capital']:.2f})")
    
    return analysis, daily_with_metrics

# 使用示例
if __name__ == "__main__":
    # 创建与backtest.py相同的配置（使用最新的手续费和滑点数据）
    base_config = {
        'data_path': 'qqq_longport.csv',  # 使用包含Turnover字段的longport数据
        # 'data_path': 'qqq_market_hours_with_indicators.csv',
        'ticker': 'QQQ',
        'initial_capital': 100000,
        'lookback_days': 1,
        'start_date': date(2024, 1, 1),   # 使用实际数据的开始日期
        'end_date': date(2025, 9, 30),     # 使用实际数据的结束日期
        # 'start_date': date(2020, 1, 1),   # 使用实际数据的开始日期
        # 'end_date': date(2025, 4, 30),     # 使用实际数据的结束日期
        'check_interval_minutes': 15,
        'enable_transaction_fees': True,  # 启用手续费计算
        'transaction_fee_per_share': 0.008166,  # 最新手续费配置
        'slippage_per_share': 0.01,  # 最新滑点配置：每股滑点，买入时多付，卖出时少收
        'enable_intraday_stop_loss': True,  # 🛡️ 启用4%日内止损功能
        'intraday_stop_loss_pct': 0.04,  # 🛡️ 日内止损阈值：4%
        'trading_start_time': (9, 40),
        'trading_end_time': (15, 40),
        'max_positions_per_day': 10,
        'print_daily_trades': False,
        'print_trade_details': False,
        'K1': 1,  # 上边界sigma乘数
        'K2': 1,  # 下边界sigma乘数
        'leverage': 1,  # 资金杠杆倍数，默认为1
        'use_vwap': True,  # VWAP开关，True为使用VWAP，False为不使用
    }
    
    # ===========================================
    # 可以在这里自定义分析参数
    # ===========================================
    
    # 模拟次数：建议快速测试用20-50次，精确分析用100-200次
    NUM_SIMULATIONS = 20  # 每个杠杆率的模拟次数
    
    # 杠杆率范围：测试1-10倍杠杆
    LEVERAGE_RANGE = [2,3, 4, 5]
    
    # 日内止损设置
    USE_DAILY_STOP_LOSS = True  # 是否启用日内止损
    DAILY_STOP_LOSS_THRESHOLD = 0.04 # 日内止损阈值（4.8%）
    
    # 分析模式选择
    ANALYSIS_MODE = "leverage_analysis"  # "leverage_analysis", "single" 或 "multi_timing"
    
    print("="*60)
    print("🚀 FTMO挑战通过率分析（优化版）")
    print("="*60)
    print(f"📊 数据文件: {base_config['data_path']}")
    print(f"📈 股票代码: {base_config['ticker']}")
    print(f"📅 数据范围: {base_config['start_date']} 至 {base_config['end_date']}")
    print(f"🔄 模拟次数: {NUM_SIMULATIONS}次/杠杆率")
    print(f"⚡ 杠杆率范围: {LEVERAGE_RANGE}")
    if USE_DAILY_STOP_LOSS:
        print(f"🛡️ 日内止损: 启用 ({DAILY_STOP_LOSS_THRESHOLD*100:.1f}%)")
    else:
        print(f"🛡️ 日内止损: 禁用")
    print(f"🎯 目标: 达到10%收益即通过（无时间限制）")
    analysis_mode_names = {
        'leverage_analysis': '不同杠杆倍数的通过率和爆仓率分析',
        'multi_timing': '多账户时间错配分析', 
        'single': '单一杠杆率分析'
    }
    print(f"📍 分析模式: {analysis_mode_names.get(ANALYSIS_MODE, '未知模式')}")
    print(f"💡 提示: 如需修改数据，请直接修改上面的base_config")
    print("="*60)
    
    # 预加载和处理数据（只需要一次）
    print("\n🔄 预加载数据...")
    try:
        get_processed_data(base_config)
        print("✅ 数据预加载完成，后续测试将使用缓存数据")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        exit(1)
    
    if ANALYSIS_MODE == "leverage_analysis":
        # 运行杠杆倍数分析
        print(f"\n开始不同杠杆倍数的通过率和爆仓率分析...")
        print(f"将测试以下杠杆倍数: {LEVERAGE_RANGE}")
        
        try:
            results_df = analyze_leverage_ftmo_performance(
                base_config,
                leverage_range=LEVERAGE_RANGE,
                num_simulations=NUM_SIMULATIONS,
                use_daily_stop_loss=USE_DAILY_STOP_LOSS,
                daily_stop_loss=DAILY_STOP_LOSS_THRESHOLD
            )
            
            if results_df is not None:
                print(f"\n\n🎆 分析完成！")
                print(f"• 最高通过率: {results_df['pass_rate'].max()*100:.1f}%")
                print(f"• 最低通过率: {results_df['pass_rate'].min()*100:.1f}%")
                print(f"• 平均通过率: {results_df['pass_rate'].mean()*100:.1f}%")
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  用户中断")
    
    elif ANALYSIS_MODE == "multi_timing":
        # 运行多账户时间错配分析
        print(f"\n开始多账户时间错配分析...")
        print(f"将同时测试三个账户，使用不同的交易时间:")
        print(f"  - 账户A: 9:40-15:40 (标准时间)")
        print(f"  - 账户B: 9:39-15:39 (提前1分钟)")
        print(f"  - 账户C: 9:41-15:41 (延后1分钟)")
        
        try:
            results_df, summary = monte_carlo_multi_timing_analysis(
                base_config,
                num_simulations=NUM_SIMULATIONS,
                leverage=LEVERAGE_RANGE[0],  # 使用第一个杠杆率
                use_daily_stop_loss=USE_DAILY_STOP_LOSS,
                daily_stop_loss=DAILY_STOP_LOSS_THRESHOLD
            )
            
            if results_df is not None:
                print(f"\n\n🎯 多账户时间错配分析总结:")
                print("="*60)
                print(f"通过微调交易时间来分散风险的效果:")
                print(f"  - 单个账户平均成功率: {(summary['account_1_pass_rate'] + summary['account_2_pass_rate'] + summary['account_3_pass_rate'])/3*100:.1f}%")
                print(f"  - 至少一个账户成功率: {summary['at_least_one_pass_rate']*100:.1f}%")
                print(f"  - 风险分散效果: {(summary['at_least_one_pass_rate'] - max(summary['account_1_pass_rate'], summary['account_2_pass_rate'], summary['account_3_pass_rate']))*100:.1f}% 提升")
                
                # 保存详细结果到CSV（可选）
                # results_df.to_csv('multi_timing_analysis_results.csv', index=False)
                # print(f"\n详细结果已保存到: multi_timing_analysis_results.csv")
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  用户中断")
    
    elif ANALYSIS_MODE == "single":
        # 单一杠杆率分析模式
        single_leverage = LEVERAGE_RANGE[0] if LEVERAGE_RANGE else 4
        print(f"\n开始单一杠杆率分析: {single_leverage}x")
        
        try:
            analysis, daily_metrics = analyze_single_leverage(
                base_config,
                leverage=single_leverage,
                use_daily_stop_loss=USE_DAILY_STOP_LOSS,
                daily_stop_loss=DAILY_STOP_LOSS_THRESHOLD
            )
            print(f"\n单一杠杆率分析完成")
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  用户中断")
    
    else:
        # 默认模式：运行杠杆分析
        print(f"\n未知的分析模式 '{ANALYSIS_MODE}'，使用默认的杠杆分析模式...")
        
        # 估算运行时间
        total_simulations = NUM_SIMULATIONS * len(LEVERAGE_RANGE)
        print(f"总计需要运行 {total_simulations} 次回测")
        print(f"预估运行时间: {total_simulations * 1:.0f}-{total_simulations * 2:.0f} 秒")
        print("提示: 可以随时按 Ctrl+C 终止\n")
        
        try:
            # 1. 分析不同杠杆率的通过率
            results_df = analyze_leverage_ftmo_performance(
                base_config, 
                leverage_range=LEVERAGE_RANGE,
                num_simulations=NUM_SIMULATIONS,
                use_daily_stop_loss=USE_DAILY_STOP_LOSS,
                daily_stop_loss=DAILY_STOP_LOSS_THRESHOLD
            )
            
            if results_df is not None:
                # 2. 打印结果表格
                print("\n\n📋 杠杆率与通过率关系汇总:")
                print("="*100)
                print(f"{'杠杆率':>6} | {'通过率':>7} | {'有效测试':>8} | {'成功次数':>8} | {'平均成功天数':>10} | {'平均有效天数':>10} | {'数据用完':>8} | {'日损失':>8} | {'总损失':>8}")
                print("="*100)
                
                for _, row in results_df.iterrows():
                    # 合并日内和收盘的失败次数
                    total_daily_failures = row['failure_daily_loss'] + row.get('failure_daily_loss_intraday', 0)
                    total_total_failures = row['failure_total_loss'] + row.get('failure_total_loss_intraday', 0)
                    print(f"{row['leverage']:>6}x | {row['pass_rate']*100:>6.1f}% | {row['valid_count']:>8} | {row['passed_count']:>8} | "
                          f"{row['avg_days_to_success']:>9.1f}天 | {row['avg_days_valid']:>9.1f}天 | {row['data_exhausted_count']:>8} | "
                          f"{total_daily_failures:>8} | {total_total_failures:>8}")
                
                # 3. 推荐配置
                print(f"\n💡 分析结果说明:")
                print(f"• 通过率基于有效测试计算（排除数据用完的情况）")
                print(f"• 有效测试 = 总测试 - 数据用完的测试")
                print(f"• 平均成功天数：成功案例达到10%收益的平均天数")
                print(f"• 平均有效天数：所有有效测试的平均持续天数")
                print(f"• 数据用完：因数据不足而无法完成测试的次数（不计入成功率）")
                print(f"• 日损失：违反5%日损失限制的次数")
                print(f"• 总损失：违反10%总损失限制的次数")
                print(f"• 重要：包含日内实时违规检测，更准确反映实际交易风险")
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  用户中断，显示已完成的结果:")
            # 显示已完成的结果
            if 'results_summary' in locals():
                print(f"\n📊 已完成的结果:")
                print(f"{'杠杆率':>6} | {'通过率':>7} | {'有效测试':>8} | {'成功次数':>8}")
                print("-"*40)
                # 这里不会执行，因为 results_summary 不存在
                pass
            else:
                print("没有完成任何分析")
    
    # 程序结束时提供缓存清理选项
    print(f"\n💾 数据缓存状态:")
    print(f"  原始数据缓存: {len(_data_cache)} 个文件")
    print(f"  处理数据缓存: {len(_processed_data_cache)} 个配置")
    
    # 如果需要清理缓存，可以取消下面的注释
    # clear_data_cache()
    # print("✅ 缓存已清理")
    