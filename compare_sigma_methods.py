import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date
import pytz

# 模拟simulate.py中的sigma计算方法
def calculate_sigma_simulate(df, target_date, target_time, lookback_days=10):
    """
    使用simulate.py中的方法计算sigma
    
    参数:
        df: 包含历史数据的DataFrame
        target_date: 目标日期
        target_time: 目标时间
        lookback_days: 回溯天数
    """
    # 创建数据副本
    df_copy = df.copy()
    
    # 获取唯一日期并排序
    unique_dates = sorted(df_copy["Date"].unique())
    
    # 假设最后一天是当前交易日，直接排除
    if len(unique_dates) > 1:
        target_date_obj = target_date
        history_dates = [d for d in unique_dates if d < target_date_obj]
        
        # 从剩余日期中选择最近的lookback_days天
        history_dates = history_dates[-lookback_days:] if len(history_dates) >= lookback_days else history_dates
    else:
        print(f"错误: 数据中只有一天或没有数据，无法计算噪声空间")
        return None
    
    # 为历史日期计算当日开盘价和相对变动率
    history_df = df_copy[df_copy["Date"].isin(history_dates)].copy()
    
    # 为每个历史日期计算当日开盘价
    day_opens = {}
    for date in history_dates:
        day_data = history_df[history_df["Date"] == date]
        if day_data.empty:
            print(f"错误: {date} 日期数据为空")
            return None
        day_opens[date] = day_data["Open"].iloc[0]
    
    # 为每个时间点计算相对于开盘价的绝对变动率
    history_df["move"] = 0.0
    for date in history_dates:
        day_open = day_opens[date]
        history_df.loc[history_df["Date"] == date, "move"] = abs(history_df.loc[history_df["Date"] == date, "Close"] / day_open - 1)
    
    # 获取历史数据中目标时间点的数据
    historical_moves = []
    for date in history_dates:
        hist_data = history_df[(history_df["Date"] == date) & (history_df["Time"] == target_time)]
        if not hist_data.empty:
            historical_moves.append(hist_data["move"].iloc[0])
    
    # 确保有足够的历史数据计算sigma
    if len(historical_moves) == 0:
        print(f"时间点 {target_time} 没有足够的历史数据")
        return None
    
    # 计算平均变动率作为sigma
    sigma = sum(historical_moves) / len(historical_moves)
    return sigma

# 模拟backtest.py中的sigma计算方法
def calculate_sigma_backtest(df, target_date, target_time, lookback_days=10):
    """
    使用backtest.py中的方法计算sigma
    
    参数:
        df: 包含历史数据的DataFrame
        target_date: 目标日期
        target_time: 目标时间
        lookback_days: 回溯天数
    """
    # 创建数据副本
    df_copy = df.copy()
    
    # 计算每分钟相对开盘的回报（使用day_open保持一致性）
    df_copy['ret'] = df_copy['Close'] / df_copy['day_open'] - 1 
    
    # 将时间点转为列
    pivot = df_copy.pivot(index='Date', columns='Time', values='ret').abs()
    
    # 计算每个时间点的绝对回报的滚动平均值
    sigma = pivot.rolling(window=lookback_days, min_periods=1).mean().shift(1)
    
    # 获取目标日期和时间的sigma值
    if target_date in sigma.index and target_time in sigma.columns:
        return sigma.loc[target_date, target_time]
    else:
        print(f"目标日期 {target_date} 或时间 {target_time} 不在数据中")
        return None

# 主函数
def main():
    # 读取分钟级k线数据
    # 注意：这里假设数据已经包含了Date和Time列
    # 如果没有，需要从DateTime列中提取
    df = pd.read_csv('tqqq_longport.csv', parse_dates=['DateTime'])
    
    # 确保数据包含必要的列
    if 'Date' not in df.columns:
        df['Date'] = df['DateTime'].dt.date
    if 'Time' not in df.columns:
        df['Time'] = df['DateTime'].dt.strftime('%H:%M')
    
    # 确保数据包含day_open列（每日开盘价）
    if 'day_open' not in df.columns:
        df['day_open'] = df.groupby('Date')['Open'].transform('first')
    
    # 设置目标日期和时间
    target_date = date(2025, 5, 15)
    target_time = '12:46'
    
    # 使用两种方法计算sigma
    sigma_simulate = calculate_sigma_simulate(df, target_date, target_time, lookback_days=10)
    sigma_backtest = calculate_sigma_backtest(df, target_date, target_time, lookback_days=10)
    
    # 打印结果
    print(f"\n目标日期时间: 2025-05-15 12:46:00")
    print(f"使用simulate.py方法计算的sigma值: {sigma_simulate:.6f}")
    print(f"使用backtest.py方法计算的sigma值: {sigma_backtest:.6f}")
    print(f"差异: {abs(sigma_simulate - sigma_backtest):.6f}")
    
    # 计算上下边界（可选）
    day_data = df[df['Date'] == target_date]
    if not day_data.empty:
        day_open = day_data['day_open'].iloc[0]
        
        # 获取前一日收盘价
        prev_date_data = df[df['Date'] < target_date].sort_values('DateTime').tail(1)
        if not prev_date_data.empty:
            prev_close = prev_date_data['Close'].iloc[0]
            
            # 计算参考价格
            upper_ref = max(day_open, prev_close)
            lower_ref = min(day_open, prev_close)
            
            # 计算边界
            print("\n边界计算:")
            print(f"当日开盘价: {day_open:.6f}")
            print(f"前一日收盘价: {prev_close:.6f}")
            print(f"上界参考价格: {upper_ref:.6f}")
            print(f"下界参考价格: {lower_ref:.6f}")
            
            # 使用simulate方法的sigma计算边界
            upper_bound_simulate = upper_ref * (1 + sigma_simulate)
            lower_bound_simulate = lower_ref * (1 - sigma_simulate)
            print(f"\n使用simulate.py方法:")
            print(f"上边界: {upper_bound_simulate:.6f}")
            print(f"下边界: {lower_bound_simulate:.6f}")
            
            # 使用backtest方法的sigma计算边界
            upper_bound_backtest = upper_ref * (1 + sigma_backtest)
            lower_bound_backtest = lower_ref * (1 - sigma_backtest)
            print(f"\n使用backtest.py方法:")
            print(f"上边界: {upper_bound_backtest:.6f}")
            print(f"下边界: {lower_bound_backtest:.6f}")

if __name__ == "__main__":
    main()
