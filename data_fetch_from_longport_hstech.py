#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import date, timedelta
from zoneinfo import ZoneInfo
from longport.openapi import QuoteContext, Config, Period, AdjustType
import pandas as pd

# ———— 配置 & 初始化 ————
config = Config.from_env()
ctx    = QuoteContext(config)

# ———— 时区定义 ————
TZ_HK = ZoneInfo('Asia/Hong_Kong')

# ———— 用户参数：香港时间起止日期（inclusive） ————
# 注意：history_candlesticks_by_date 接口接受 date 类型
start_date = date(2024, 1, 1)
end_date   = date(2025, 11, 12)

all_candles = []

# ———— 按天拉：每次用 history_candlesticks_by_date ————
current = start_date
while current <= end_date:
    resp = ctx.history_candlesticks_by_date(
        "03032.HK",  # 恒生科技ETF
        Period.Min_1,
        AdjustType.NoAdjust,
        current,
        current
    )
    print(f"{current} → 拉到 {len(resp)} 条")
    all_candles.extend(resp)
    current += timedelta(days=1)

# ———— 转换 & 保存（保持香港时间） ————
rows = []
for c in all_candles:
    # API 返回的 timestamp 是香港本地的 naive 时间
    dt_hk = c.timestamp.replace(tzinfo=TZ_HK)
    rows.append({
        'DateTime': dt_hk.strftime('%Y-%m-%d %H:%M:%S'),
        'Open':      c.open,
        'High':      c.high,
        'Low':       c.low,
        'Close':     c.close,
        'Volume':    c.volume,
        'Turnover':  c.turnover
    })

df = pd.DataFrame(rows)

# 检查并去除重复的时间戳，保留最后一条记录（通常是更新后的数据）
initial_count = len(df)
df = df.drop_duplicates(subset=['DateTime'], keep='last')
final_count = len(df)

if initial_count > final_count:
    print(f"⚠️  发现并去除了 {initial_count - final_count} 条重复的时间戳记录")

df.to_csv('hstech_etf_longport.csv', index=False)
print(f"✔️ 已保存 hstech_etf_longport.csv，共 {len(df)} 条记录（所有时间均为香港本地时间）。")

