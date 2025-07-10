"""
=============================================================================
股票数据处理工具 - 使用说明
=============================================================================

本脚本用于处理原始股票数据，将其转换为适合回测系统使用的格式。

🎯 主要功能：
-----------
1. 过滤市场交易时间数据（9:30 AM - 4:00 PM）
2. 计算每日开盘价和收盘价
3. 添加MACD技术指标
4. 生成适合回测系统使用的CSV文件

📊 数据转换：
-----------
输入格式（原始CSV）：
    DateTime,Open,High,Low,Close,Volume
    2000-01-03 09:31:00,192.375,192.375,191.5,192.0,376100

输出格式（处理后CSV）：
    DateTime,Open,High,Low,Close,Volume,Year,DayOpen,DayClose,Date,Time,EMA_fast,EMA_slow,MACD,MACD_signal,MACD_histogram

🚀 使用方法：
-----------
方法1：命令行使用
    python process_data.py qqq.csv
    # 输出：qqq_market_hours_with_indicators.csv

方法2：指定输出文件
    python process_data.py qqq.csv --output my_output.csv

方法3：自定义MACD参数
    python process_data.py qqq.csv --fast 12 --slow 26 --signal 9

📋 命令行参数：
-------------
- input_file: 输入CSV文件路径（必需）
- --output, -o: 输出CSV文件路径（可选，默认自动生成）
- --fast: MACD快速EMA周期（默认：12）
- --slow: MACD慢速EMA周期（默认：26）
- --signal: MACD信号线EMA周期（默认：9）

💡 使用示例：
-----------
# 基本用法 - 处理QQQ数据
python process_data.py qqq.csv

# 处理SPY数据并指定输出文件
python process_data.py spy.csv --output spy_processed.csv

# 使用自定义MACD参数
python process_data.py qqq.csv --fast 10 --slow 20 --signal 5

📁 文件说明：
-----------
- qqq.csv: 原始QQQ数据文件（包含盘前盘后数据）
- qqq_market_hours_with_indicators.csv: 处理后的文件（仅市场时间+指标）

⚠️ 注意事项：
-----------
1. 输入文件必须包含DateTime, Open, High, Low, Close, Volume列
2. DateTime格式应为 'YYYY-MM-DD HH:MM:SS'
3. 处理后的文件会过滤掉盘前盘后数据，只保留9:30-16:00的数据
4. 如果某些交易日缺少9:30或16:00的数据点，会在处理过程中显示警告

🔧 处理流程：
-----------
1. 读取原始CSV文件
2. 过滤市场交易时间（9:30 AM - 4:00 PM）
3. 计算每日开盘价（9:30 AM）和收盘价（4:00 PM）
4. 按日期分组计算MACD指标
5. 保存处理后的数据到新文件

=============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import time

def prepare_market_hours_data(df):
    """
    Process data to filter for market hours and calculate day open/close prices.
    
    Parameters:
        df (DataFrame): Input DataFrame with stock data
    
    Returns:
        DataFrame: Processed DataFrame with market hours data
    """
    # Extract date and time components if not already present
    if 'Date' not in df.columns:
        df['Date'] = df['DateTime'].dt.date
    if 'Time' not in df.columns:
        df['Time'] = df['DateTime'].dt.time

    # Filter data to include only regular market hours (9:30 AM - 4:00 PM)
    market_open = time(9, 30)
    market_close = time(16, 0)

    # Create a mask for regular market hours
    market_hours_mask = (df['Time'] >= market_open) & (df['Time'] <= market_close)
    df_market_hours = df[market_hours_mask].copy()

    print(f"Rows after filtering for market hours: {len(df_market_hours)}")

    # Group by date to find the first (9:30 AM) and last (4:00 PM) data points for each day
    # For each day, get the first row (9:30 AM opening price)
    opening_prices = df_market_hours.groupby('Date').first().reset_index()
    opening_prices = opening_prices[['Date', 'Open']].rename(columns={'Open': 'DayOpen'})

    # For each day, get the last row (4:00 PM closing price)
    closing_prices = df_market_hours.groupby('Date').last().reset_index()
    closing_prices = closing_prices[['Date', 'Close']].rename(columns={'Close': 'DayClose'})

    # Merge the opening and closing prices back to the main dataframe
    df_market_hours = pd.merge(df_market_hours, opening_prices, on='Date', how='left')
    df_market_hours = pd.merge(df_market_hours, closing_prices, on='Date', how='left')

    # Check if 'Year' column exists, if not, extract it from DateTime
    if 'Year' not in df_market_hours.columns:
        df_market_hours['Year'] = df_market_hours['DateTime'].dt.year

    # Create a new dataframe with the filtered data
    columns_to_keep = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Year', 'DayOpen', 'DayClose', 'Date', 'Time']
    filtered_df = df_market_hours[columns_to_keep].copy()

    # Display some statistics about the filtered data
    print(f"Total number of trading days: {filtered_df['Date'].nunique()}")
    print(f"Date range: {filtered_df['DateTime'].min().date()} to {filtered_df['DateTime'].max().date()}")
    print(f"Average number of data points per day: {len(filtered_df) / filtered_df['Date'].nunique():.2f}")

    # Check if there are any days with missing 9:30 AM or 4:00 PM data points
    market_open_time = time(9, 30)
    market_close_time = time(16, 0)

    # Group by date and check if each day has data at 9:30 AM and 4:00 PM
    day_times = df_market_hours.groupby('Date')['Time'].apply(list).reset_index()
    days_missing_open = []
    days_missing_close = []

    for _, row in day_times.iterrows():
        if market_open_time not in row['Time']:
            days_missing_open.append(row['Date'])
        if market_close_time not in row['Time']:
            days_missing_close.append(row['Date'])

    print(f"Number of days missing 9:30 AM data: {len(days_missing_open)}")
    print(f"Number of days missing 4:00 PM data: {len(days_missing_close)}")

    if days_missing_open:
        print("Sample of days missing 9:30 AM data:")
        print(days_missing_open[:5])
        
    if days_missing_close:
        print("Sample of days missing 4:00 PM data:")
        print(days_missing_close[:5])
        
    return filtered_df

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    计算MACD指标
    
    参数:
        df: 包含价格数据的DataFrame
        fast_period: 快速EMA周期
        slow_period: 慢速EMA周期
        signal_period: 信号线EMA周期
        
    返回:
        添加了MACD指标的DataFrame
    """
    # 复制DataFrame以避免修改原始数据
    df_copy = df.copy()
    
    # 计算快速和慢速EMA
    df_copy['EMA_fast'] = df_copy['Close'].ewm(span=fast_period, adjust=False).mean()
    df_copy['EMA_slow'] = df_copy['Close'].ewm(span=slow_period, adjust=False).mean()
    
    # 计算MACD线
    df_copy['MACD'] = df_copy['EMA_fast'] - df_copy['EMA_slow']
    
    # 计算信号线
    df_copy['MACD_signal'] = df_copy['MACD'].ewm(span=signal_period, adjust=False).mean()
    
    # 计算柱状图
    df_copy['MACD_histogram'] = df_copy['MACD'] - df_copy['MACD_signal']
    
    return df_copy

def process_data(input_file, output_file=None, fast_period=12, slow_period=26, signal_period=9):
    """
    处理CSV文件，过滤市场交易时间并添加技术指标
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径（如果为None，将基于输入文件名生成）
        fast_period: MACD快速EMA周期
        slow_period: MACD慢速EMA周期
        signal_period: MACD信号线EMA周期
        
    返回:
        输出文件路径
    """
    # 如果未指定输出文件，则基于输入文件名生成
    if output_file is None:
        base_name = os.path.basename(input_file)
        ticker = os.path.splitext(base_name)[0]
        output_dir = os.path.dirname(input_file)
        output_file = os.path.join(output_dir, f"{ticker}_market_hours_with_indicators.csv")
    
    print(f"读取数据文件: {input_file}")
    df = pd.read_csv(input_file)

    # 确保DateTime列是datetime格式
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # 显示原始数据样本
    print("原始数据样本:")
    print(df.head())
    print(f"原始数据总行数: {len(df)}")
    
    # 步骤1: 过滤市场交易时间数据
    print("\n步骤1: 过滤市场交易时间数据...")
    market_hours_df = prepare_market_hours_data(df)
    
    # 步骤2: 计算MACD指标
    print("\n步骤2: 计算MACD指标...")
    # 确保数据按时间排序
    market_hours_df.sort_values('DateTime', inplace=True)
    
    # 按日期分组计算MACD
    grouped = market_hours_df.groupby('Date')
    result_dfs = []
    
    for date, group in grouped:
        # 计算当天的MACD
        group_with_macd = calculate_macd(
            group, 
            fast_period=fast_period, 
            slow_period=slow_period, 
            signal_period=signal_period
        )
        result_dfs.append(group_with_macd)
    
    # 合并所有日期的结果
    result_df = pd.concat(result_dfs)
    
    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"\n处理完成! 带有市场交易时间和MACD指标的数据已保存至: {output_file}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理股票数据：过滤市场交易时间并添加技术指标')
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('--output', '-o', help='输出CSV文件路径（可选）')
    parser.add_argument('--fast', type=int, default=12, help='MACD快速EMA周期（默认：12）')
    parser.add_argument('--slow', type=int, default=26, help='MACD慢速EMA周期（默认：26）')
    parser.add_argument('--signal', type=int, default=9, help='MACD信号线EMA周期（默认：9）')
    
    args = parser.parse_args()
    
    process_data(
        args.input_file, 
        args.output, 
        fast_period=args.fast, 
        slow_period=args.slow, 
        signal_period=args.signal
    )
