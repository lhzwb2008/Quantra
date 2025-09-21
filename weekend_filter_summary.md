# 周末数据过滤修改总结

## 修改目的
修改 `simulate.py` 中的数据处理逻辑，确保在计算噪声区域参数（sigma）时忽略所有周末的数据，只使用工作日（周一到周五）的数据进行计算。

## 修改的函数

### 1. `get_historical_data()` 函数
**位置**: 第127-133行
**修改内容**: 在获取历史数据时跳过周末日期
```python
# 跳过周末（周六=5, 周日=6）
if date_to_check.weekday() >= 5:
    if DEBUG_MODE:
        print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] 跳过周末: {date_to_check}")
    date_to_check -= timedelta(days=1)
    continue
```

### 2. `get_historical_data()` 函数 - 数据后处理
**位置**: 第243-249行
**修改内容**: 在数据处理的最后阶段再次过滤周末数据（双重保险）
```python
# 过滤周末数据（双重保险）
weekday_mask = df["Date"].apply(lambda x: x.weekday() < 5 if isinstance(x, date_type) else True)
df = df[weekday_mask]

if DEBUG_MODE and not df.empty:
    unique_dates = sorted(df["Date"].unique())
    print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 最终数据包含的日期: {unique_dates}")
```

### 3. `calculate_noise_area()` 函数
**位置**: 第292-308行
**修改内容**: 在计算噪声区域时过滤周末数据
```python
# 过滤周末数据：只保留周一到周五的数据
weekday_dates = []
for d in unique_dates:
    if isinstance(d, date_type):
        # weekday(): 0=Monday, 1=Tuesday, ..., 6=Sunday
        if d.weekday() < 5:  # 0-4 表示周一到周五
            weekday_dates.append(d)
    else:
        weekday_dates.append(d)  # 如果不是date类型，保留原样

unique_dates = weekday_dates
df_copy = df_copy[df_copy["Date"].isin(unique_dates)]

if DEBUG_MODE:
    print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] 过滤周末后的日期数量: {len(unique_dates)}")
    if len(unique_dates) > 0:
        print(f"[{now_et.strftime('%Y-%m-%d %H:%M:%S')}] 最近的交易日: {unique_dates[-5:]}")
```

## 过滤逻辑说明

### 周末判断标准
- 使用 Python 的 `date.weekday()` 方法
- 返回值: 0=周一, 1=周二, 2=周三, 3=周四, 4=周五, 5=周六, 6=周日
- 过滤条件: `weekday() >= 5` (周六和周日)
- 保留条件: `weekday() < 5` (周一到周五)

### 三层过滤保护
1. **API获取层**: 在 `get_historical_data()` 中跳过周末日期的API调用
2. **数据处理层**: 在数据处理完成后过滤掉任何残留的周末数据
3. **计算层**: 在 `calculate_noise_area()` 中再次确保只使用工作日数据

## 测试验证

### 测试用例
- 测试日期范围: 2025年8月1日-7日（包含一个完整的周末）
- 预期过滤: 2025-08-02（周六）和 2025-08-03（周日）
- 预期保留: 2025-08-01（周五）、2025-08-04（周一）到 2025-08-07（周四）

### 测试结果
✅ 所有测试通过
- 周末日期被正确识别和过滤
- DataFrame 处理逻辑正确
- 没有周末数据残留

## 影响分析

### 正面影响
1. **数据一致性**: 确保 simulate.py 和 backtest.py 使用相同的工作日数据
2. **计算准确性**: 避免周末的异常数据影响 sigma 计算
3. **策略稳定性**: 提高交易策略的可靠性和可重现性

### 性能影响
- **API调用减少**: 跳过周末日期的API调用，提高效率
- **数据处理**: 增加少量的日期判断逻辑，影响微乎其微
- **内存使用**: 过滤掉周末数据，略微减少内存使用

## 调试信息
当 `DEBUG_MODE = True` 时，会输出以下调试信息：
- 跳过的周末日期
- 过滤后的日期数量
- 最近的交易日列表
- 最终数据包含的日期

## 兼容性
- 向后兼容：不影响现有的交易逻辑
- 参数兼容：保持所有函数参数不变
- 数据格式兼容：输出的数据格式保持一致

## 建议
1. 在生产环境部署前，建议进行充分测试
2. 监控日志输出，确认周末过滤正常工作
3. 定期验证计算结果与 backtest.py 的一致性
