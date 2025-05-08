#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from longport.openapi import Config, QuoteContext, Period, AdjustType
from datetime import date

# 初始化配置和上下文
config = Config.from_env()
quote_ctx = QuoteContext(config)

# 获取指定股票和日期的K线数据
symbol = "VXX.US"
target_date = date(2024, 4, 1)  # 指定日期

print(f"获取 {symbol} 在 {target_date} 的日K线数据...")

# 使用history_candlesticks_by_date方法获取指定日期的K线数据
candles = quote_ctx.history_candlesticks_by_date(
    symbol,
    Period.Day,  # 日K线
    AdjustType.ForwardAdjust,  # 前复权
    target_date,
    target_date
)

# 处理结果
if candles and len(candles) > 0:
    # 只取第一条数据
    candle = candles[0]
    
    # 打印结果
    print(f"\n{symbol} 在 {target_date} 的日K线数据:")
    print(f"交易日期: {target_date}")
    print(f"开盘价: {candle.open}")
    print(f"最高价: {candle.high}")
    print(f"最低价: {candle.low}")
    print(f"收盘价: {candle.close}")
    print(f"成交量: {candle.volume}")
    print(f"成交额: {candle.turnover}")
else:
    print(f"未能获取 {symbol} 在 {target_date} 的日K线数据") 