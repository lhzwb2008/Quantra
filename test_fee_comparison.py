"""
测试比较真实费用和简化费用的差异
"""

from trading_fees import TradingFees
import pandas as pd

def test_fee_comparison():
    """比较不同交易规模下的费用差异"""
    
    # 初始化费用计算器
    calculator = TradingFees()
    
    # 测试不同的交易场景
    test_cases = [
        {"quantity": 10, "buy_price": 500, "sell_price": 510, "desc": "小额交易（10股）"},
        {"quantity": 100, "buy_price": 500, "sell_price": 510, "desc": "中等交易（100股）"},
        {"quantity": 500, "buy_price": 500, "sell_price": 510, "desc": "大额交易（500股）"},
        {"quantity": 1000, "buy_price": 500, "sell_price": 510, "desc": "超大额交易（1000股）"},
    ]
    
    # 简化费用率（每股）
    simple_fee_rates = [0.01, 0.013166]
    
    results = []
    
    print("=" * 80)
    print("交易费用对比分析")
    print("=" * 80)
    
    for case in test_cases:
        print(f"\n{case['desc']}:")
        print(f"  买入: {case['quantity']}股 @ ${case['buy_price']}")
        print(f"  卖出: {case['quantity']}股 @ ${case['sell_price']}")
        
        # 计算真实费用
        real_fees = calculator.calculate_round_trip_fees(
            case['quantity'], case['buy_price'], case['sell_price']
        )
        
        print(f"\n  真实费用明细:")
        print(f"    买入费用: ${real_fees['buy_fees']['total']:.2f}")
        print(f"    卖出费用: ${real_fees['sell_fees']['total']:.2f}")
        print(f"    总费用: ${real_fees['total_fees']:.2f}")
        
        # 计算盈亏
        gross_pnl = case['quantity'] * (case['sell_price'] - case['buy_price'])
        net_pnl_real = gross_pnl - real_fees['total_fees']
        
        print(f"\n  简化费用对比:")
        for rate in simple_fee_rates:
            simple_fee = max(case['quantity'] * rate * 2, 2.16)  # 买卖各收一次，最低2.16
            net_pnl_simple = gross_pnl - simple_fee
            fee_diff = real_fees['total_fees'] - simple_fee
            pnl_diff = net_pnl_real - net_pnl_simple
            
            print(f"    费率${rate:.6f}/股:")
            print(f"      简化费用: ${simple_fee:.2f}")
            print(f"      费用差异: ${fee_diff:+.2f} ({fee_diff/real_fees['total_fees']*100:+.1f}%)")
            print(f"      净盈亏差异: ${pnl_diff:+.2f}")
        
        print(f"\n  交易结果:")
        print(f"    毛利润: ${gross_pnl:.2f}")
        print(f"    真实净利润: ${net_pnl_real:.2f}")
        print(f"    费用占毛利润比例: {real_fees['total_fees']/gross_pnl*100:.1f}%")
        
        results.append({
            'quantity': case['quantity'],
            'buy_price': case['buy_price'],
            'sell_price': case['sell_price'],
            'gross_pnl': gross_pnl,
            'real_fees': real_fees['total_fees'],
            'net_pnl': net_pnl_real,
            'fee_ratio': real_fees['total_fees']/gross_pnl*100
        })
    
    # 创建汇总表
    print("\n" + "=" * 80)
    print("费用汇总表")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\n关键发现:")
    print(f"1. 平均费用占毛利润比例: {df['fee_ratio'].mean():.1f}%")
    print(f"2. 真实费用范围: ${df['real_fees'].min():.2f} - ${df['real_fees'].max():.2f}")
    print(f"3. 小额交易的费用影响更大（费用占比更高）")

if __name__ == "__main__":
    test_fee_comparison()
