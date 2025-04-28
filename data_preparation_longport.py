# 获取标的历史 K 线
#
# 运行前请访问"开发者中心"确保账户有正确的行情权限。
# 如没有开通行情权限，可以通过"LongPort"手机客户端，并进入"我的 - 我的行情 - 行情商城"购买开通行情权限。
from datetime import datetime, date, timedelta
from longport.openapi import QuoteContext, Config, Period, AdjustType
import pandas as pd

config = Config.from_env()
ctx = QuoteContext(config)

# 设置查询的起始点
start_date = datetime(2025, 4, 21)
print(f"查询起始日期: {start_date.strftime('%Y-%m-%d')}")

# 获取历史K线数据
resp = ctx.history_candlesticks_by_offset("TQQQ.US", Period.Min_1, AdjustType.NoAdjust, True, 1000, start_date)
print(f"获取到 {len(resp)} 条K线数据")

# 将结果转换为DataFrame
data = []
for candle in resp:
    # 简单直接地将时间增加12小时
    adjusted_time = candle.timestamp - timedelta(hours=12)
    
    data.append({
        'DateTime': adjusted_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Open': candle.open,
        'High': candle.high,
        'Low': candle.low,
        'Close': candle.close,
        'Volume': candle.volume
    })

# 创建DataFrame并保存为CSV
df = pd.DataFrame(data)
df.to_csv('tqqq_longport.csv', index=True)
print(f"数据已保存到 tqqq_longport.csv，共 {len(df)} 条记录")
print("时间已简单调整（-12小时）")
