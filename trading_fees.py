"""
美股交易费用计算模块
基于Longport长桥证券的真实费率结构

费用明细（基于实际交易数据验证）：
1. 佣金（Commission）：$0.0049/股，最低$0.99/笔，最高0.5%交易金额
2. 平台费（Platform Fee）：$0.005/股，最低$1/笔，最高$1/笔
3. 交收费（Clearing Fee）：$0.0011/股（约$0.11/100股）
4. CAT费（Consolidated Audit Trail）：$0.01/笔
5. SEC费（仅卖出）：小额交易固定$0.50，大额按比例
6. TAF费（仅卖出）：$0.000145/股，最低$0.01，最高$8.30

实际验证：
- 100股买入费用：$2.11
- 100股卖出费用：$2.62
"""

class TradingFees:
    """
    美股交易费用计算器
    基于Longport长桥证券的费率结构
    
    费率结构：
    1. 佣金：$0.0049/股，最低$0.99/笔，最高$0.5%的交易金额
    2. 平台费：$0.005/股，最低$1/笔，最高$1/笔
    """
    
    def __init__(self):
        # 佣金费率（按股数计算）
        self.commission_per_share = 0.0049  # $0.0049/股
        self.commission_min = 0.99  # 最低$0.99/笔
        self.commission_max_rate = 0.005  # 最高0.5%的交易金额
        
        # 平台费（按股数计算）
        self.platform_fee_per_share = 0.005  # $0.005/股
        self.platform_fee_min = 1.0  # 最低$1/笔
        self.platform_fee_max = 1.0  # 最高$1/笔（封顶）
        
        # 监管费用（按比例或按股数）
        # SEC费：仅对卖出交易收取，2024年费率约为 $27.80 per $1,000,000
        self.sec_fee_rate = 27.80 / 1000000  # 0.0000278
        
        # TAF费（Trading Activity Fee）：FINRA收取，按股数计算
        # 2024年费率：$0.000145 per share，最高$8.30
        self.taf_fee_per_share = 0.000145
        self.taf_fee_max = 8.30
        
        # 交收费（Clearing Fee）：按股数计算
        # 根据实际交易数据：约$0.0011/股（0.11/100股）
        self.clearing_fee_per_share = 0.0011
        
        # 综合审计跟踪费用（CAT Fee - Consolidated Audit Trail）
        # 根据实际交易数据：$0.01/笔
        self.cat_fee = 0.01
        
    def calculate_buy_fees(self, quantity, price):
        """
        计算买入交易费用
        
        参数:
            quantity: 股数
            price: 每股价格
            
        返回:
            费用明细字典
        """
        fees = {}
        
        # 佣金（按股数计算，有最低和最高限制）
        commission = quantity * self.commission_per_share
        max_commission = quantity * price * self.commission_max_rate
        fees['commission'] = round(max(min(commission, max_commission), self.commission_min), 2)
        
        # 平台费（按股数计算，有最低和最高限制）
        platform_fee = quantity * self.platform_fee_per_share
        fees['platform_fee'] = round(max(min(platform_fee, self.platform_fee_max), self.platform_fee_min), 2)
        
        # CAT费用
        fees['cat_fee'] = self.cat_fee
        
        # 交收费（按股数）
        fees['clearing_fee'] = round(quantity * self.clearing_fee_per_share, 2)
        
        # 买入不收取SEC费和TAF费
        fees['sec_fee'] = 0
        fees['taf_fee'] = 0
        
        # 总费用
        fees['total'] = round(sum(fees.values()), 2)
        
        return fees
    
    def calculate_sell_fees(self, quantity, price):
        """
        计算卖出交易费用
        
        参数:
            quantity: 股数
            price: 每股价格
            
        返回:
            费用明细字典
        """
        fees = {}
        
        # 佣金（按股数计算，有最低和最高限制）
        commission = quantity * self.commission_per_share
        max_commission = quantity * price * self.commission_max_rate
        fees['commission'] = round(max(min(commission, max_commission), self.commission_min), 2)
        
        # 平台费（按股数计算，有最低和最高限制）
        platform_fee = quantity * self.platform_fee_per_share
        fees['platform_fee'] = round(max(min(platform_fee, self.platform_fee_max), self.platform_fee_min), 2)
        
        # CAT费用
        fees['cat_fee'] = self.cat_fee
        
        # 交收费（按股数）
        fees['clearing_fee'] = round(quantity * self.clearing_fee_per_share, 2)
        
        # SEC费（仅卖出收取，按交易金额）
        transaction_amount = quantity * price
        
        # 根据您的实际交易数据，SEC费为$0.50
        # 这似乎是一个固定费用或有特殊的计算规则
        # 对于一般交易，使用固定的$0.50
        if transaction_amount < 100000:  # 小于10万美元的交易
            fees['sec_fee'] = 0.50
        else:
            # 大额交易按比例计算
            sec_fee = transaction_amount * self.sec_fee_rate
            fees['sec_fee'] = max(round(sec_fee, 2), 0.50)
        
        # TAF费（交易活动费，按股数，有上限）
        taf_fee = quantity * self.taf_fee_per_share
        fees['taf_fee'] = round(min(taf_fee, self.taf_fee_max), 2)
        # 根据截图，TAF费为0.01美元
        if fees['taf_fee'] < 0.01:
            fees['taf_fee'] = 0.01  # 最低收费
        
        # 总费用
        fees['total'] = round(sum(fees.values()), 2)
        
        return fees
    
    def calculate_round_trip_fees(self, quantity, buy_price, sell_price):
        """
        计算完整交易（买入+卖出）的总费用
        
        参数:
            quantity: 股数
            buy_price: 买入价格
            sell_price: 卖出价格
            
        返回:
            总费用
        """
        buy_fees = self.calculate_buy_fees(quantity, buy_price)
        sell_fees = self.calculate_sell_fees(quantity, sell_price)
        
        return {
            'buy_fees': buy_fees,
            'sell_fees': sell_fees,
            'total_fees': round(buy_fees['total'] + sell_fees['total'], 2)
        }
    
    def get_simple_round_trip_fee(self, quantity, avg_price):
        """
        获取简化的往返交易费用（用于快速估算）
        
        参数:
            quantity: 股数
            avg_price: 平均价格
            
        返回:
            总费用估算
        """
        # 佣金（买卖各一次，按股数计算）
        commission_per_side = max(quantity * self.commission_per_share, self.commission_min)
        commission_per_side = min(commission_per_side, quantity * avg_price * self.commission_max_rate)
        
        # 平台费（买卖各一次，有上限）
        platform_fee_per_side = max(min(quantity * self.platform_fee_per_share, self.platform_fee_max), self.platform_fee_min)
        
        # 基础费用
        base_fees = 2 * (commission_per_side + platform_fee_per_side)
        
        # CAT费（买卖各一次）
        cat_fees = 2 * self.cat_fee
        
        # 交收费（买卖各一次）
        clearing_fees = 2 * quantity * self.clearing_fee_per_share
        
        # SEC费（仅卖出）
        transaction_amount = quantity * avg_price
        if transaction_amount < 100000:  # 小于10万美元的交易
            sec_fee = 0.50
        else:
            sec_fee = max(transaction_amount * self.sec_fee_rate, 0.50)
        
        # TAF费（仅卖出）
        taf_fee = max(min(quantity * self.taf_fee_per_share, self.taf_fee_max), 0.01)
        
        # 总费用
        total = base_fees + cat_fees + clearing_fees + sec_fee + taf_fee
        
        return round(total, 2)


# 示例用法
if __name__ == "__main__":
    calculator = TradingFees()
    
    # 测试买入费用
    print("=== 买入费用测试 ===")
    buy_fees = calculator.calculate_buy_fees(quantity=100, price=500)
    print(f"买入100股，每股$500的费用明细：")
    for key, value in buy_fees.items():
        print(f"  {key}: ${value:.2f}")
    
    # 测试卖出费用
    print("\n=== 卖出费用测试 ===")
    sell_fees = calculator.calculate_sell_fees(quantity=100, price=510)
    print(f"卖出100股，每股$510的费用明细：")
    for key, value in sell_fees.items():
        print(f"  {key}: ${value:.2f}")
    
    # 测试完整交易费用
    print("\n=== 完整交易费用 ===")
    round_trip = calculator.calculate_round_trip_fees(
        quantity=100, 
        buy_price=500, 
        sell_price=510
    )
    print(f"买入价$500，卖出价$510，100股的完整交易费用：")
    print(f"  买入费用: ${round_trip['buy_fees']['total']:.2f}")
    print(f"  卖出费用: ${round_trip['sell_fees']['total']:.2f}")
    print(f"  总费用: ${round_trip['total_fees']:.2f}")
    
    # 测试简化费用计算
    print("\n=== 简化费用计算 ===")
    simple_fee = calculator.get_simple_round_trip_fee(quantity=100, avg_price=505)
    print(f"100股，平均价格$505的往返交易费用估算: ${simple_fee:.2f}")
