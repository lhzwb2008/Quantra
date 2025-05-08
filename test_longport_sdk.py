#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from longport.openapi import Config, QuoteContext

# 初始化配置和上下文
config = Config.from_env()
quote_ctx = QuoteContext(config)

# 获取VIXY.US的实时行情
symbol = "VIXY.US"
print(f"获取 {symbol} 实时行情...")

# 调用quote方法
quote_result = quote_ctx.quote([symbol])

# 打印结果
if quote_result and len(quote_result) > 0:
    quote = quote_result[0]
    print(f"代码: {symbol}")
    print(f"最新价: {quote.last_done}")
    
    # 打印所有属性，便于调试
    print("\n所有属性:")
    for attr in dir(quote):
        if not attr.startswith('_') and not callable(getattr(quote, attr)):
            print(f"{attr}: {getattr(quote, attr)}")
else:
    print("未获取到数据") 