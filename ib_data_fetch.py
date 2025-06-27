#!/usr/bin/env python3
"""
IB 数据获取脚本
支持获取多种股票的历史数据，用于回测分析
"""

from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, date

# ==================== 配置参数 ====================
# 合约配置
SYMBOL = 'QQQ'                  # 合约代码: 'QQQ', 'SPY', 'AAPL', 'TSLA', 'NVDA', 'MSFT' 等
CONTRACT_TYPE = 'CFD'           # 合约类型: 'CFD', 'STK' (股票)
EXCHANGE = 'SMART'              # 交易所: 'SMART', 'NASDAQ', 'NYSE'
CURRENCY = 'USD'                # 货币: 'USD', 'HKD', 'CNH'

# 数据获取配置
START_DATE = date(2025, 6, 11)  # 开始日期
END_DATE = date(2025, 6, 20)    # 结束日期
BAR_SIZE = '1 min'              # K线周期: '1 min', '5 mins', '15 mins', '1 hour', '1 day'

# 文件保存配置
OUTPUT_FILENAME = f'{SYMBOL.lower()}_{CONTRACT_TYPE.lower()}_data.csv'  # 自动根据合约生成文件名

# 连接配置
IB_HOST = '127.0.0.1'       # IB Gateway地址
IB_PORT = 4002              # IB Gateway端口 (实盘:4001, 模拟:4002)
CLIENT_ID = 1               # 客户端ID

# ==================== 使用说明 ====================
# 1. 修改 SYMBOL 来设置要获取的合约:
#    美股CFD: 'QQQ', 'SPY', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN'
#    美股股票: 同上，但需要设置 CONTRACT_TYPE = 'STK'
#    港股CFD: '700' (腾讯), '9988' (阿里) - 需要设置 EXCHANGE='SEHK', CURRENCY='HKD'
#
# 2. 修改 CONTRACT_TYPE 来设置合约类型:
#    'CFD' = CFD差价合约
#    'STK' = 股票
#
# 3. 修改 START_DATE 和 END_DATE 来设置获取数据的日期范围:
#    START_DATE = date(2025, 6, 11)  # 开始日期
#    END_DATE = date(2025, 6, 30)    # 结束日期
#
# 4. 修改 BAR_SIZE 来设置K线周期:
#    - '1 min', '2 mins', '3 mins', '5 mins', '10 mins'
#    - '15 mins', '20 mins', '30 mins', '1 hour', '2 hours'
#    - '3 hours', '4 hours', '8 hours', '1 day', '1 week'
#
# 5. 输出文件名会自动生成，如: qqq_cfd_data.csv, aapl_stk_data.csv
# ================================================

def calculate_duration_from_dates(start_date, end_date):
    """根据开始和结束日期计算时间段"""
    delta = end_date - start_date
    days = delta.days
    
    if days <= 0:
        raise ValueError("结束日期必须晚于开始日期")
    
    # 根据天数选择合适的时间段格式
    if days <= 30:
        return f"{days} D"
    elif days <= 365:
        weeks = days // 7
        if weeks > 0:
            return f"{weeks} W"
        else:
            return f"{days} D"
    else:
        months = days // 30
        return f"{months} M"

def get_contract_data(symbol=SYMBOL, contract_type=CONTRACT_TYPE, exchange=EXCHANGE, currency=CURRENCY, start_date=START_DATE, end_date=END_DATE, bar_size=BAR_SIZE):
    """获取合约历史数据（支持CFD和股票）"""
    
    ib = IB()
    
    try:
        # 连接
        print(f"🔗 连接到IB Gateway: {IB_HOST}:{IB_PORT}")
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        print("✅ 连接成功")
        
        # 根据合约类型创建合约
        if contract_type.upper() == 'CFD':
            contract = CFD(symbol, exchange, currency)
            print(f"📄 创建CFD合约: {symbol}")
        elif contract_type.upper() == 'STK':
            contract = Stock(symbol, exchange, currency)
            print(f"📄 创建股票合约: {symbol}")
        else:
            print(f"❌ 不支持的合约类型: {contract_type}")
            return None
            
        # 完善合约信息
        qualified = ib.qualifyContracts(contract)
        
        if not qualified:
            print(f"❌ 无法创建{symbol} {contract_type}合约")
            return None
            
        contract = qualified[0]
        print(f"✅ {symbol} {contract_type}合约: {contract.conId}")
        
        # 显示合约详细信息
        if hasattr(contract, 'localSymbol'):
            print(f"   本地符号: {contract.localSymbol}")
        if hasattr(contract, 'tradingClass'):
            print(f"   交易类别: {contract.tradingClass}")
        
        # 计算时间段和结束时间
        duration = calculate_duration_from_dates(start_date, end_date)
        end_datetime = end_date.strftime("%Y%m%d") + " 16:00:00"  # 美股收盘时间
        
        print(f"\n📈 获取数据配置:")
        print(f"   合约代码: {symbol}")
        print(f"   合约类型: {contract_type}")
        print(f"   交易所: {exchange}")
        print(f"   货币: {currency}")
        print(f"   开始日期: {start_date}")
        print(f"   结束日期: {end_date}")
        print(f"   计算时间段: {duration}")
        print(f"   K线周期: {bar_size}")
        
        # 获取历史数据
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
            timeout=60
        )
        
        if bars and len(bars) > 0:
            print(f"✅ 成功获取 {len(bars)} 根K线数据")
            
            # 转换为DataFrame
            df = util.df(bars)
            
            # 处理时间索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # 过滤数据到指定日期范围
            start_datetime = pd.to_datetime(start_date).tz_localize('US/Eastern')
            end_datetime = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).tz_localize('US/Eastern')  # 包含结束日期
            
            df = df[(df.index >= start_datetime) & (df.index < end_datetime)]
            
            print(f"   过滤后数据: {len(df)} 根K线")
            if len(df) > 0:
                print(f"   数据时间范围: {df.index[0]} 到 {df.index[-1]}")
            
            return df
        else:
            print("❌ 未获取到CFD历史数据")
            print("💡 提示: CFD历史数据可能需要特殊的市场数据权限")
            return None
            
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        return None
        
    finally:
        if ib.isConnected():
            ib.disconnect()


def analyze_data(df):
    """分析数据"""
    if df is None or len(df) == 0:
        print("❌ 没有数据可分析")
        return
        
    print(f"\n📊 数据分析:")
    print(f"   数据点数量: {len(df)}")
    print(f"   时间跨度: {df.index[-1] - df.index[0]}")
    
    # 基本统计
    print(f"\n💰 价格统计:")
    print(f"   最新价格: ${df['close'].iloc[-1]:.2f}")
    print(f"   最高价格: ${df['high'].max():.2f}")
    print(f"   最低价格: ${df['low'].min():.2f}")
    print(f"   平均价格: ${df['close'].mean():.2f}")
    print(f"   价格标准差: ${df['close'].std():.2f}")
    
    # 成交量统计
    print(f"\n📈 成交量统计:")
    print(f"   总成交量: {df['volume'].sum():,.0f}")
    print(f"   平均成交量: {df['volume'].mean():,.0f}")
    print(f"   最大成交量: {df['volume'].max():,.0f}")
    
    # 计算简单技术指标
    print(f"\n📊 技术指标:")
    
    # 移动平均
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_60'] = df['close'].rolling(60).mean()
    
    current_price = df['close'].iloc[-1]
    ma20 = df['ma_20'].iloc[-1]
    ma60 = df['ma_60'].iloc[-1]
    
    if not pd.isna(ma20):
        print(f"   20分钟均线: ${ma20:.2f} ({'上方' if current_price > ma20 else '下方'})")
    if not pd.isna(ma60):
        print(f"   60分钟均线: ${ma60:.2f} ({'上方' if current_price > ma60 else '下方'})")
    
    # 计算收益率
    df['returns'] = df['close'].pct_change()
    
    # 波动率（年化）
    volatility = df['returns'].std() * np.sqrt(252 * 390)  # 252个交易日，每日390分钟
    print(f"   年化波动率: {volatility*100:.2f}%")
    
    # 最大回撤
    df['cumulative'] = (1 + df['returns']).cumprod()
    df['peak'] = df['cumulative'].expanding().max()
    df['drawdown'] = (df['cumulative'] - df['peak']) / df['peak']
    max_drawdown = df['drawdown'].min()
    print(f"   最大回撤: {max_drawdown*100:.2f}%")
    
    # 显示最近几分钟的数据
    if len(df) >= 5:
        print(f"\n⏰ 最近5分钟数据:")
        recent = df.tail(5)[['open', 'high', 'low', 'close', 'volume']]
        for idx, row in recent.iterrows():
            print(f"   {idx.strftime('%m-%d %H:%M')}: O=${row['open']:.2f} H=${row['high']:.2f} L=${row['low']:.2f} C=${row['close']:.2f} V={row['volume']:,.0f}")


def save_data(df, filename=OUTPUT_FILENAME):
    """保存数据到CSV，格式为：DateTime,Open,High,Low,Close,Volume"""
    if df is None or len(df) == 0:
        print("❌ 没有数据可保存")
        return
        
    try:
        # 准备输出数据，确保列名和格式正确
        output_df = df.copy()
        
        # 重置索引，将时间作为一列
        output_df.reset_index(inplace=True)
        
        # 重命名列以匹配目标格式
        column_mapping = {
            'date': 'DateTime',
            'open': 'Open', 
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # 只保留需要的列并重命名
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        output_df = output_df[required_columns].rename(columns=column_mapping)
        
        # 保存为CSV，不包含索引
        output_df.to_csv(filename, index=False)
        print(f"\n💾 数据已保存到: {filename}")
        print(f"   文件格式: DateTime,Open,High,Low,Close,Volume")
        print(f"   数据行数: {len(output_df)} 行")
        
        # 显示前几行数据作为示例
        print(f"\n📋 数据预览:")
        print(output_df.head(3).to_string(index=False))
        
        # 显示如何使用数据
        print(f"\n💡 使用方法:")
        print(f"   import pandas as pd")
        print(f"   df = pd.read_csv('{filename}', parse_dates=['DateTime'])")
        print(f"   print(df.head())")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("📊 IB 合约数据获取脚本")
    print("🎯 支持CFD和股票数据获取，用于回测分析")
    print("=" * 60)
    
    print(f"📈 当前配置:")
    print(f"   合约代码: {SYMBOL}")
    print(f"   合约类型: {CONTRACT_TYPE}")
    print(f"   交易所: {EXCHANGE}")
    print(f"   货币: {CURRENCY}")
    print(f"   日期范围: {START_DATE} 到 {END_DATE}")
    print(f"   K线周期: {BAR_SIZE}")
    print(f"   输出文件: {OUTPUT_FILENAME}")
    
    # 获取数据
    df = get_contract_data()
    
    if df is not None and len(df) > 0:
        # 分析数据
        analyze_data(df)
        
        # 保存数据
        save_data(df)
        
        print(f"\n🎉 {SYMBOL} {CONTRACT_TYPE} 数据获取和分析完成！")
        
    else:
        print(f"\n❌ {SYMBOL} {CONTRACT_TYPE} 数据获取失败")
        
        # 如果1分钟数据失败，尝试5分钟数据
        print(f"\n🔄 尝试获取5分钟数据...")
        df = get_contract_data(bar_size='5 mins')
        
        if df is not None and len(df) > 0:
            analyze_data(df)
            save_data(df, f"{SYMBOL.lower()}_{CONTRACT_TYPE.lower()}_5min.csv")
            print(f"\n🎉 {SYMBOL} {CONTRACT_TYPE} 5分钟数据获取成功！")


if __name__ == "__main__":
    main() 