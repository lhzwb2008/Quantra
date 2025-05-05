import pandas as pd
import numpy as np
import os

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    计算MACD指标，确保只对当天的数据进行计算
    
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
    
    # 确保按日期和时间排序
    if "DateTime" in df_copy.columns:
        df_copy = df_copy.sort_values("DateTime")
        # 提取日期以便按天分组
        if "Date" not in df_copy.columns:
            df_copy["Date"] = df_copy["DateTime"].dt.date
    elif "Date" in df_copy.columns and "Time" in df_copy.columns:
        df_copy = df_copy.sort_values(["Date", "Time"])
    
    # 按日期分组计算MACD
    result_dfs = []
    grouped = df_copy.groupby("Date")
    
    for date, group in grouped:
        # 对每天的数据单独计算MACD
        group = group.copy()
        
        # 计算快速和慢速EMA
        group["EMA_fast"] = group["Close"].ewm(span=fast_period, adjust=False).mean()
        group["EMA_slow"] = group["Close"].ewm(span=slow_period, adjust=False).mean()
        
        # 计算MACD线
        group["MACD"] = group["EMA_fast"] - group["EMA_slow"]
        
        # 计算信号线
        group["MACD_signal"] = group["MACD"].ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        group["MACD_histogram"] = group["MACD"] - group["MACD_signal"]
        
        result_dfs.append(group)
    
    # 合并所有日期的结果
    if result_dfs:
        return pd.concat(result_dfs).sort_values("DateTime" if "DateTime" in df_copy.columns else ["Date", "Time"])
    else:
        # 如果没有足够的数据分组（例如只有一天的数据），就直接对整个DataFrame计算
        print("警告：没有足够的数据分组，直接对整个数据集计算MACD")
        
        # 计算快速和慢速EMA
        df_copy["EMA_fast"] = df_copy["Close"].ewm(span=fast_period, adjust=False).mean()
        df_copy["EMA_slow"] = df_copy["Close"].ewm(span=slow_period, adjust=False).mean()
        
        # 计算MACD线
        df_copy["MACD"] = df_copy["EMA_fast"] - df_copy["EMA_slow"]
        
        # 计算信号线
        df_copy["MACD_signal"] = df_copy["MACD"].ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        df_copy["MACD_histogram"] = df_copy["MACD"] - df_copy["MACD_signal"]
        
        return df_copy

def process_file_with_indicators(input_file, output_file=None, fast_period=12, slow_period=26, signal_period=9):
    """
    处理CSV文件，添加技术指标
    
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
        name_without_ext = os.path.splitext(base_name)[0]
        output_dir = os.path.dirname(input_file)
        output_file = os.path.join(output_dir, f"{name_without_ext}_with_indicators.csv")
    
    print(f"读取数据文件: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['DateTime'])
    
    # 确保数据按时间排序
    df.sort_values('DateTime', inplace=True)
    
    # 按日期分组计算MACD
    # 这样可以确保每天的MACD计算是独立的，避免跨日期的影响
    grouped = df.groupby(df['DateTime'].dt.date)
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
    print(f"带有MACD指标的数据已保存至: {output_file}")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='计算股票技术指标')
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('--output', '-o', help='输出CSV文件路径（可选）')
    parser.add_argument('--fast', type=int, default=12, help='MACD快速EMA周期（默认：12）')
    parser.add_argument('--slow', type=int, default=26, help='MACD慢速EMA周期（默认：26）')
    parser.add_argument('--signal', type=int, default=9, help='MACD信号线EMA周期（默认：9）')
    
    args = parser.parse_args()
    
    process_file_with_indicators(
        args.input_file, 
        args.output, 
        fast_period=args.fast, 
        slow_period=args.slow, 
        signal_period=args.signal
    )
