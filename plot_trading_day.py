import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import os

def plot_trading_day(day_df, trades, save_path=None):
    """
    生成交易日的图表，显示价格、上下边界、VWAP和交易点
    
    参数:
        day_df: 包含单日数据的DataFrame
        trades: 当天交易列表
        save_path: 保存图表的路径（如果为None，则显示图表而不保存）
    """
    # 确保日期数据正确
    trade_date = day_df['Date'].iloc[0]
    
    # 计算VWAP
    prices = day_df['Close'].values
    volumes = day_df['Volume'].values
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
    
    day_df['VWAP'] = vwaps
    
    # 创建图表
    plt.figure(figsize=(14, 7))
    
    # 绘制价格线
    plt.plot(day_df['Time'], day_df['Close'], label='Price', color='black', linewidth=1.5)
    
    # 绘制上下边界
    plt.plot(day_df['Time'], day_df['UpperBound'], label='Upper Bound', color='green', linestyle='--', alpha=0.7)
    plt.plot(day_df['Time'], day_df['LowerBound'], label='Lower Bound', color='red', linestyle='--', alpha=0.7)
    
    # 绘制VWAP
    plt.plot(day_df['Time'], day_df['VWAP'], label='VWAP', color='blue', linestyle='-', alpha=0.7)
    
    # 标记交易点
    for trade in trades:
        # 获取入场和出场时间
        entry_time = trade['entry_time'].strftime('%H:%M')
        exit_time = trade['exit_time'].strftime('%H:%M')
        
        # 找到对应的数据点
        entry_idx = day_df[day_df['Time'] == entry_time].index
        exit_idx = day_df[day_df['Time'] == exit_time].index
        
        if len(entry_idx) > 0 and len(exit_idx) > 0:
            entry_idx = entry_idx[0]
            exit_idx = exit_idx[0]
            
            # 获取价格
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            
            # 标记入场点
            if trade['side'] == 'Long':
                plt.scatter(entry_time, entry_price, color='green', s=100, marker='^', label='Long Entry' if 'Long Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(entry_time, entry_price, color='red', s=100, marker='v', label='Short Entry' if 'Short Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            # 标记出场点
            if trade['side'] == 'Long':
                plt.scatter(exit_time, exit_price, color='red', s=100, marker='x', label='Long Exit' if 'Long Exit' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(exit_time, exit_price, color='green', s=100, marker='x', label='Short Exit' if 'Short Exit' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            # 连接入场和出场点
            plt.plot([entry_time, exit_time], [entry_price, exit_price], color='gray', linestyle='-', alpha=0.5)
            
            # 添加P&L标签
            mid_time_idx = (day_df.index.get_loc(entry_idx) + day_df.index.get_loc(exit_idx)) // 2
            if mid_time_idx < len(day_df):
                mid_time = day_df['Time'].iloc[mid_time_idx]
                mid_price = (entry_price + exit_price) / 2
                plt.annotate(f"P&L: ${trade['pnl']:.2f}", 
                             xy=(mid_time, mid_price),
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center',
                             fontsize=9,
                             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    # 设置图表标题和标签
    side_text = ""
    if trades:
        sides = [trade['side'] for trade in trades]
        if 'Long' in sides and 'Short' not in sides:
            side_text = "Long"
        elif 'Short' in sides and 'Long' not in sides:
            side_text = "Short"
        elif 'Long' in sides and 'Short' in sides:
            side_text = "Long/Short"
    
    plt.title(f"Trading Day: {trade_date} ({side_text})", fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    
    # 设置x轴刻度
    # 每30分钟一个刻度
    all_times = day_df['Time'].unique()
    time_ticks = [t for t in all_times if t.endswith(':00') or t.endswith(':30')]
    plt.xticks(time_ticks, rotation=45)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图表
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
