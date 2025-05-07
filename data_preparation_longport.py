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
start_date = datetime(2025, 3, 1)
print(f"查询起始日期: {start_date.strftime('%Y-%m-%d')}")

# 设置单次请求限制
batch_size = 1000    # 单次API请求的最大数量限制

# 用于存储所有K线数据
all_candles = []
current_date = start_date
total_fetched = 0
max_attempts = 100   # 设置一个最大尝试次数，防止无限循环

# 循环获取历史K线数据，直到无法获取更多数据
for attempt in range(max_attempts):
    # 获取当前批次的数据
    resp = ctx.history_candlesticks_by_offset("TQQQ.US", Period.Min_1, AdjustType.NoAdjust, True, 
                                             batch_size, current_date)
    
    print(f"第{attempt+1}次请求：获取到 {len(resp)} 条K线数据，起始时间: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 如果没有获取到数据，退出循环
    if len(resp) == 0:
        print("无法获取更多数据，结束获取")
        break
    
    # 将获取到的数据添加到总数据列表中
    all_candles.extend(resp)
    total_fetched += len(resp)
    
    # 如果获取的数据少于请求的数量，说明已经没有更多数据了
    if len(resp) < batch_size:
        print(f"获取的数据量({len(resp)})小于请求量({batch_size})，已到达数据末尾")
        break
    
    # 将最后一条K线的时间作为下一次请求的起始时间
    current_date = resp[-1].timestamp + timedelta(seconds=1)
    
    print(f"当前已获取总数据量: {total_fetched}，准备下一次请求...")

print(f"总共获取到 {len(all_candles)} 条K线数据")

# 将结果转换为DataFrame
data = []
for candle in all_candles:
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
