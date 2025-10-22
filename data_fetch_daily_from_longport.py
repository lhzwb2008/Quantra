#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从Longport获取日K线数据
用于对比日K的开盘价/收盘价 与 分钟K聚合的开盘价/收盘价的差异
"""

from datetime import date, timedelta
from zoneinfo import ZoneInfo
from longport.openapi import QuoteContext, Config, Period, AdjustType
import pandas as pd

# ———— 配置 & 初始化 ————
config = Config.from_env()
ctx    = QuoteContext(config)

# ———— 时区定义 ————
TZ_HK = ZoneInfo('Asia/Hong_Kong')
TZ_ET = ZoneInfo('US/Eastern')

# ———— 用户参数：美东起止日期（inclusive） ————
start_date = date(2025, 1, 1)
end_date   = date(2025, 10, 25)

print(f"正在获取 QQQ.US 的日K数据...")
print(f"日期范围: {start_date} 到 {end_date}")

# ———— 获取日K数据 ————
resp = ctx.history_candlesticks_by_date(
    "QQQ.US",
    Period.Day,  # 获取日K
    AdjustType.NoAdjust,
    start_date,
    end_date
)

print(f"获取到 {len(resp)} 条日K数据")

# ———— 转换时区 & 保存 ————
rows = []
for c in resp:
    # API 返回的 timestamp 是香港本地的 naive 时间
    dt_hk = c.timestamp.replace(tzinfo=TZ_HK)
    # 转到美东
    dt_et = dt_hk.astimezone(TZ_ET)
    rows.append({
        'Date': dt_et.strftime('%Y-%m-%d'),  # 日K只需要日期
        'Open':      c.open,
        'High':      c.high,
        'Low':       c.low,
        'Close':     c.close,
        'Volume':    c.volume,
        'Turnover':  c.turnover
    })

df = pd.DataFrame(rows)

# 检查并去除重复的日期，保留最后一条记录
initial_count = len(df)
df = df.drop_duplicates(subset=['Date'], keep='last')
final_count = len(df)

if initial_count > final_count:
    print(f"⚠️  发现并去除了 {initial_count - final_count} 条重复的日期记录")

# 按日期排序
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 保存到CSV
output_file = 'qqq_daily_longport.csv'
df.to_csv(output_file, index=False)
print(f"✔️ 已保存 {output_file}，共 {len(df)} 条记录")

# 显示前几条数据
print("\n前5条数据预览:")
print(df.head())

