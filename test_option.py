#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LongPort 期权报价功能测试脚本
用于测试期权数据权限和报价获取功能
"""

import os
import sys
from datetime import datetime, date
import pytz
from dotenv import load_dotenv
from longport.openapi import Config, QuoteContext

# 加载环境变量
load_dotenv()

def get_us_eastern_time():
    """获取美东时间"""
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

def test_option_quote_access():
    """测试期权报价权限"""
    print("=" * 60)
    print("LongPort 期权报价功能测试")
    print("=" * 60)
    
    try:
        # 创建API连接
        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 正在连接LongPort API...")
        config = Config.from_env()
        quote_ctx = QuoteContext(config)
        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] API连接成功")
        
        # 测试标的
        symbol_base = "QQQ"
        print(f"\n[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 测试标的: {symbol_base}")
        
        # 1. 获取期权到期日列表
        print(f"\n[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 步骤1: 获取期权到期日列表")
        expiry_dates = quote_ctx.option_chain_expiry_date_list(f"{symbol_base}.US")
        print(f"  - 找到 {len(expiry_dates)} 个到期日")
        print(f"  - 最近5个到期日: {expiry_dates[:5]}")
        
        # 2. 选择最近的到期日
        nearest_expiry = expiry_dates[0] if expiry_dates else None
        if not nearest_expiry:
            print("  - 错误: 没有找到期权到期日")
            return
            
        print(f"  - 选择到期日: {nearest_expiry}")
        
        # 3. 获取该到期日的期权链
        print(f"\n[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 步骤2: 获取期权链数据")
        option_chain_data = quote_ctx.option_chain_info_by_date(f"{symbol_base}.US", nearest_expiry)
        
        if not option_chain_data:
            print("  - 错误: 没有获取到期权链数据")
            return
            
        print(f"  - 获取到 {len(option_chain_data)} 个执行价格")
        
        # 4. 选择几个期权合约进行测试
        test_symbols = []
        for i, strike_info in enumerate(option_chain_data[:3]):  # 测试前3个执行价格
            if hasattr(strike_info, 'call_symbol') and strike_info.call_symbol:
                test_symbols.append(strike_info.call_symbol)
            if hasattr(strike_info, 'put_symbol') and strike_info.put_symbol:
                test_symbols.append(strike_info.put_symbol)
                
        print(f"  - 选择测试期权: {test_symbols[:6]}")  # 最多显示6个
        
        # 5. 测试期权报价获取
        print(f"\n[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 步骤3: 测试期权报价获取")
        
        success_count = 0
        error_count = 0
        
        for symbol in test_symbols[:6]:  # 测试前6个期权
            try:
                print(f"\n  测试期权: {symbol}")
                quotes = quote_ctx.option_quote([symbol])
                
                if quotes and len(quotes) > 0:
                    quote = quotes[0]
                    print(f"    ✓ 成功获取报价:")
                    print(f"      - 最新价: {quote.last_done}")
                    print(f"      - 买价: {getattr(quote, 'bid', 'N/A')}")
                    print(f"      - 卖价: {getattr(quote, 'ask', 'N/A')}")
                    print(f"      - 成交量: {quote.volume}")
                    success_count += 1
                else:
                    print(f"    ✗ 返回空数据")
                    error_count += 1
                    
            except Exception as e:
                error_msg = str(e)
                print(f"    ✗ 获取失败: {error_msg}")
                error_count += 1
                
                # 分析错误类型
                if "no quote access" in error_msg.lower():
                    print(f"      → 期权报价权限不足")
                elif "rate limit" in error_msg.lower():
                    print(f"      → API调用频率限制")
                elif "invalid symbol" in error_msg.lower():
                    print(f"      → 期权代码无效")
                else:
                    print(f"      → 其他错误")
        
        # 6. 测试结果总结
        print(f"\n[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 测试结果总结:")
        print(f"  - 成功获取报价: {success_count} 个")
        print(f"  - 获取失败: {error_count} 个")
        
        if success_count > 0:
            print(f"  - 状态: ✓ 期权报价功能正常")
        elif error_count > 0:
            print(f"  - 状态: ✗ 期权报价功能异常")
            print(f"  - 建议: 检查期权数据权限或联系券商")
        
        # 7. 额外测试：尝试获取股票报价作为对比
        print(f"\n[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 步骤4: 对比测试股票报价")
        try:
            stock_quotes = quote_ctx.quote([f"{symbol_base}.US"])
            if stock_quotes and len(stock_quotes) > 0:
                stock_quote = stock_quotes[0]
                print(f"  ✓ 股票报价获取成功:")
                print(f"    - {symbol_base}.US 最新价: {stock_quote.last_done}")
            else:
                print(f"  ✗ 股票报价获取失败")
        except Exception as e:
            print(f"  ✗ 股票报价获取异常: {str(e)}")
            
    except Exception as e:
        print(f"[{get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}] 测试过程中发生错误: {str(e)}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    print("LongPort 期权报价功能测试脚本")
    print("请确保已正确配置环境变量 (LONGPORT_APP_KEY, LONGPORT_APP_SECRET, LONGPORT_ACCESS_TOKEN)")
    print()
    
    # 检查环境变量
    required_vars = ['LONGPORT_APP_KEY', 'LONGPORT_APP_SECRET', 'LONGPORT_ACCESS_TOKEN']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"错误: 缺少环境变量: {', '.join(missing_vars)}")
        print("请在 .env 文件中配置这些变量")
        sys.exit(1)
    
    test_option_quote_access()