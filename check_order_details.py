#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime, date
from dotenv import load_dotenv
from longport.openapi import Config, TradeContext

load_dotenv(override=True)

def create_trade_context():
    """创建交易上下文"""
    try:
        config = Config.from_env()
        trade_ctx = TradeContext(config)
        print("✅ 交易API连接成功")
        return trade_ctx
    except Exception as e:
        print(f"❌ 交易API连接失败: {str(e)}")
        return None

def check_order_details(trade_ctx, order_id):
    """检查指定订单的详细信息"""
    try:
        print(f"\n🔍 查询订单详情，订单ID: {order_id}")
        order_detail = trade_ctx.order_detail(order_id)
        
        print(f"订单详情:")
        print(f"  订单ID: {order_detail.order_id}")
        print(f"  股票代码: {order_detail.symbol}")
        print(f"  订单类型: {order_detail.order_type}")
        print(f"  买卖方向: {order_detail.side}")
        
        # 打印所有可用属性来调试
        print(f"  可用属性: {[attr for attr in dir(order_detail) if not attr.startswith('_')]}")
        
        # 尝试获取各种可能的属性名
        for attr_name in ['submitted_quantity', 'quantity', 'submitted_qty']:
            if hasattr(order_detail, attr_name):
                print(f"  提交数量: {getattr(order_detail, attr_name)}")
                break
        
        for attr_name in ['submitted_price', 'price', 'submitted_prc']:
            if hasattr(order_detail, attr_name):
                print(f"  提交价格: {getattr(order_detail, attr_name)}")
                break
                
        for attr_name in ['status', 'order_status']:
            if hasattr(order_detail, attr_name):
                print(f"  订单状态: {getattr(order_detail, attr_name)}")
                break
                
        for attr_name in ['executed_quantity', 'filled_quantity', 'executed_qty', 'filled_qty']:
            if hasattr(order_detail, attr_name):
                print(f"  已成交数量: {getattr(order_detail, attr_name)}")
                break
                
        for attr_name in ['executed_price', 'filled_price', 'avg_price', 'executed_prc']:
            if hasattr(order_detail, attr_name):
                print(f"  平均成交价: {getattr(order_detail, attr_name)}")
                break
        
        return order_detail
        
    except Exception as e:
        print(f"❌ 查询订单详情失败: {str(e)}")
        return None

def get_today_executions(trade_ctx, symbol="QQQ.US"):
    """获取今日成交记录"""
    try:
        print(f"\n📊 获取今日成交记录 ({symbol})")
        executions = trade_ctx.today_executions(symbol=symbol)
        
        if not executions:
            print("今日无成交记录")
            return []
            
        print(f"今日成交记录 ({len(executions)}笔):")
        for i, execution in enumerate(executions, 1):
            print(f"  成交记录 #{i}:")
            print(f"    订单ID: {execution.order_id}")
            print(f"    股票代码: {execution.symbol}")
            
            # 打印可用属性进行调试
            print(f"    可用属性: {[attr for attr in dir(execution) if not attr.startswith('_')]}")
            
            # 尝试获取各种可能的属性名
            for attr_name in ['side', 'order_side']:
                if hasattr(execution, attr_name):
                    print(f"    买卖方向: {getattr(execution, attr_name)}")
                    break
            
            for attr_name in ['quantity', 'qty', 'executed_quantity']:
                if hasattr(execution, attr_name):
                    print(f"    成交数量: {getattr(execution, attr_name)}")
                    break
                    
            for attr_name in ['price', 'executed_price', 'trade_price']:
                if hasattr(execution, attr_name):
                    print(f"    成交价格: {getattr(execution, attr_name)}")
                    break
                    
            for attr_name in ['trade_done_at', 'executed_at', 'trade_time']:
                if hasattr(execution, attr_name):
                    print(f"    成交时间: {getattr(execution, attr_name)}")
                    break
                    
            for attr_name in ['trade_id', 'execution_id']:
                if hasattr(execution, attr_name):
                    print(f"    成交ID: {getattr(execution, attr_name)}")
                    break
            print()
            
        return executions
        
    except Exception as e:
        print(f"❌ 获取今日成交记录失败: {str(e)}")
        return []

def main():
    """主函数"""
    print("🔍 检查longport API订单详情和成交价格")
    print("=" * 50)
    
    # 创建交易上下文
    trade_ctx = create_trade_context()
    if not trade_ctx:
        sys.exit(1)
    
    # 从simulate.log中找到的订单ID
    # 2025-09-25的开仓和平仓订单ID
    open_order_id = "1155861520790331392"  # 开仓订单ID
    close_order_id = "1155869095640436736"  # 平仓订单ID
    
    print(f"\n📝 检查2025-09-25的交易订单详情")
    
    # 检查开仓订单详情
    print(f"\n🟢 开仓订单详情:")
    open_order = check_order_details(trade_ctx, open_order_id)
    
    # 检查平仓订单详情
    print(f"\n🔴 平仓订单详情:")
    close_order = check_order_details(trade_ctx, close_order_id)
    
    # 获取今日所有成交记录
    executions = get_today_executions(trade_ctx, "QQQ.US")
    
    # 对比分析
    print(f"\n📊 价格对比分析:")
    print(f"=" * 50)
    
    if open_order and close_order:
        print(f"开仓订单:")
        print(f"  日志记录价格: 589.215")
        print(f"  API实际成交价: {open_order.executed_price}")
        print(f"  LongPort软件显示: 589.18")
        
        print(f"\n平仓订单:")
        print(f"  日志记录价格: 592.34") 
        print(f"  API实际成交价: {close_order.executed_price}")
        print(f"  LongPort软件显示: 592.26")
        
        # 计算差异
        if open_order.executed_price:
            open_diff_log = abs(float(open_order.executed_price) - 589.215)
            open_diff_software = abs(float(open_order.executed_price) - 589.18)
            print(f"\n开仓价格差异:")
            print(f"  API vs 日志: {open_diff_log:.3f}")
            print(f"  API vs 软件: {open_diff_software:.3f}")
        
        if close_order.executed_price:
            close_diff_log = abs(float(close_order.executed_price) - 592.34)
            close_diff_software = abs(float(close_order.executed_price) - 592.26)
            print(f"\n平仓价格差异:")
            print(f"  API vs 日志: {close_diff_log:.3f}")
            print(f"  API vs 软件: {close_diff_software:.3f}")

if __name__ == "__main__":
    main()
