#!/usr/bin/env python3
"""
IBKR API 测试脚本
测试连接、账户信息和查询QQQ相关的CFD产品
"""

import sys
import time
from threading import Thread
from datetime import datetime

# 需要先安装: pip install ib_insync
try:
    from ib_insync import *
except ImportError:
    print("请先安装ib_insync: pip install ib_insync")
    sys.exit(1)


class IBKRTest:
    def __init__(self):
        self.ib = IB()
        
    def connect(self, host='127.0.0.1', port=4001, clientId=1):
        """
        连接到IB Gateway
        默认端口: 
        - TWS Paper Trading: 7497
        - TWS Live Trading: 7496
        - IB Gateway Paper Trading: 4002
        - IB Gateway Live Trading: 4001
        """
        try:
            print(f"正在连接到IB Gateway: {host}:{port}")
            self.ib.connect(host, port, clientId=clientId)
            print("✅ 连接成功!")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def test_connection(self):
        """测试连接状态"""
        print("\n=== 测试连接状态 ===")
        if self.ib.isConnected():
            print("✅ 已连接到IBKR")
            # 注释掉不存在的属性
            # print(f"服务器版本: {self.ib.serverVersion()}")
            # print(f"连接时间: {self.ib.connectTime}")
            
            # 使用client对象获取信息
            try:
                print(f"客户端ID: {self.ib.client.clientId}")
                print(f"服务器版本: {self.ib.client.serverVersion()}")
                print(f"连接时间: {self.ib.client.connTime}")
            except:
                print("基本连接信息获取成功")
        else:
            print("❌ 未连接到IBKR")
            
    def get_account_info(self):
        """获取账户信息"""
        print("\n=== 账户信息 ===")
        try:
            # 获取账户列表
            accounts = self.ib.managedAccounts()
            print(f"管理的账户: {accounts}")
            
            if accounts:
                # 获取账户摘要
                account = accounts[0]
                summary = self.ib.accountSummary(account)
                
                print(f"\n账户 {account} 摘要:")
                for item in summary[:10]:  # 只显示前10项
                    print(f"  {item.tag}: {item.value} {item.currency}")
                    
                # 获取账户余额
                values = self.ib.accountValues(account)
                print(f"\n账户余额信息 (前5项):")
                for value in values[:5]:
                    print(f"  {value.tag}: {value.value} {value.currency}")
                    
        except Exception as e:
            print(f"❌ 获取账户信息失败: {e}")
            
    def search_qqq_cfd(self):
        """搜索QQQ相关的CFD产品"""
        print("\n=== 搜索QQQ CFD产品 ===")
        
        try:
            # 搜索QQQ相关的CFD
            print("正在搜索QQQ CFD...")
            
            # 方法1: 直接创建CFD合约
            cfd_contracts = []
            
            # 尝试不同的交易所
            exchanges = ['SMART', 'IBCFD', 'CFD']
            
            for exchange in exchanges:
                try:
                    contract = CFD('QQQ', exchange=exchange)
                    details = self.ib.reqContractDetails(contract)
                    if details:
                        cfd_contracts.extend(details)
                        print(f"✅ 在 {exchange} 找到 {len(details)} 个QQQ CFD合约")
                except Exception as e:
                    print(f"  在 {exchange} 未找到QQQ CFD: {e}")
            
            # 方法2: 通过搜索功能查找
            print("\n使用搜索功能查找QQQ相关产品...")
            search_results = self.ib.reqMatchingSymbols('QQQ')
            
            cfd_results = []
            for result in search_results:
                if result.contract.secType == 'CFD':
                    cfd_results.append(result)
                    
            print(f"搜索到 {len(cfd_results)} 个QQQ相关的CFD产品")
            
            # 显示找到的CFD详情
            if cfd_contracts or cfd_results:
                print("\n=== QQQ CFD详细信息 ===")
                
                # 显示通过直接创建找到的CFD
                for i, detail in enumerate(cfd_contracts[:3]):  # 只显示前3个
                    contract = detail.contract
                    print(f"\nCFD #{i+1}:")
                    print(f"  符号: {contract.symbol}")
                    print(f"  交易所: {contract.exchange}")
                    print(f"  货币: {contract.currency}")
                    print(f"  合约ID: {contract.conId}")
                    print(f"  描述: {detail.longName}")
                    
                    # 获取市场数据
                    self.get_market_data(contract)
                    
                # 显示搜索到的CFD
                for i, result in enumerate(cfd_results[:3]):
                    contract = result.contract
                    print(f"\n搜索结果 CFD #{i+1}:")
                    print(f"  符号: {contract.symbol}")
                    print(f"  类型: {contract.secType}")
                    print(f"  交易所: {contract.primaryExchange}")
                    print(f"  描述: {result.contractDescription}")
                    
            else:
                print("\n未找到QQQ CFD产品，尝试查找QQQ股票...")
                self.search_qqq_stock()
                
        except Exception as e:
            print(f"❌ 搜索CFD失败: {e}")
            
    def search_qqq_stock(self):
        """搜索QQQ股票（作为备选）"""
        try:
            # 创建QQQ股票合约
            stock = Stock('QQQ', 'SMART', 'USD')
            details = self.ib.reqContractDetails(stock)
            
            if details:
                print(f"\n找到QQQ股票合约:")
                contract = details[0].contract
                print(f"  符号: {contract.symbol}")
                print(f"  交易所: {contract.exchange}")
                print(f"  货币: {contract.currency}")
                print(f"  合约ID: {contract.conId}")
                
                # 获取市场数据
                self.get_market_data(contract)
                
        except Exception as e:
            print(f"❌ 搜索股票失败: {e}")
            
    def get_market_data(self, contract):
        """获取合约的市场数据"""
        try:
            # 获取实时价格数据
            print(f"\n📊 获取延迟价格数据...")
            
            # 设置为延迟数据模式
            self.ib.reqMarketDataType(3)  # 3 = 延迟数据
            
            # 请求市场数据
            self.ib.reqMktData(contract, '', False, False)
            
            # 等待数据
            time.sleep(2)
            
            # 获取ticker
            ticker = self.ib.ticker(contract)
            
            if ticker.last:
                print(f"  最新价: {ticker.last}")
                print(f"  买价: {ticker.bid}")
                print(f"  卖价: {ticker.ask}")
                print(f"  成交量: {ticker.volume}")
                print(f"  时间: {ticker.time}")
            else:
                print("  暂无市场数据")
                
            # 取消市场数据订阅
            self.ib.cancelMktData(contract)
            
        except Exception as e:
            print(f"  获取市场数据失败: {e}")
            
    def test_order_capabilities(self):
        """测试下单功能（不实际下单）"""
        print("\n=== 测试下单功能 ===")
        try:
            # 创建一个测试合约（QQQ股票）
            contract = Stock('QQQ', 'SMART', 'USD')
            
            # 创建限价单
            order = LimitOrder('BUY', 1, 400.00)
            
            # 验证订单（不实际提交）
            trade = self.ib.whatIfOrder(contract, order)
            
            if trade:
                print("✅ 订单验证成功:")
                print(f"  预计佣金: {trade.commission}")
                print(f"  预计初始保证金: {trade.initMarginChange}")
                print(f"  预计维持保证金: {trade.maintMarginChange}")
            else:
                print("❌ 订单验证失败")
                
        except Exception as e:
            print(f"❌ 测试下单功能失败: {e}")
            
    def disconnect(self):
        """断开连接"""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("\n✅ 已断开连接")
            
    def get_qqq_cfd_price(self):
        """专门获取QQQ CFD的当前价格"""
        print("\n=== 获取QQQ CFD当前价格 ===")
        
        try:
            # 直接创建QQQ CFD合约
            print("🔍 创建QQQ CFD合约...")
            cfd_contract = CFD('QQQ', 'SMART', 'USD')
            
            # 完善合约信息
            qualified = self.ib.qualifyContracts(cfd_contract)
            
            if not qualified:
                print("❌ 无法创建QQQ CFD合约")
                return None
                
            contract = qualified[0]
            print(f"✅ QQQ CFD合约创建成功:")
            print(f"   合约ID: {contract.conId}")
            print(f"   本地符号: {contract.localSymbol}")
            print(f"   交易类别: {contract.tradingClass}")
            print(f"   交易所: {contract.exchange}")
            print(f"   货币: {contract.currency}")
            
            # 获取实时价格数据
            print(f"\n📊 获取实时价格数据...")
            
            # 请求市场数据
            self.ib.reqMktData(contract, '', False, False)
            
            # 等待数据更新
            print("⏳ 等待价格数据...")
            time.sleep(3)
            
            # 获取ticker数据
            ticker = self.ib.ticker(contract)
            
            print(f"\n💰 QQQ CFD 价格信息:")
            if ticker.last and ticker.last > 0:
                print(f"   最新价格: ${ticker.last:.2f}")
            else:
                print(f"   最新价格: N/A")
                
            if ticker.bid and ticker.bid > 0:
                print(f"   买一价格: ${ticker.bid:.2f}")
            else:
                print(f"   买一价格: N/A")
                
            if ticker.ask and ticker.ask > 0:
                print(f"   卖一价格: ${ticker.ask:.2f}")
            else:
                print(f"   卖一价格: N/A")
                
            if ticker.bidSize:
                print(f"   买一数量: {ticker.bidSize}")
            if ticker.askSize:
                print(f"   卖一数量: {ticker.askSize}")
                
            if ticker.volume and ticker.volume > 0:
                print(f"   成交量: {ticker.volume:,.0f}")
            else:
                print(f"   成交量: N/A")
                
            if ticker.time:
                print(f"   更新时间: {ticker.time}")
                
            # 计算买卖价差
            if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                spread = ticker.ask - ticker.bid
                spread_pct = (spread / ticker.ask) * 100
                print(f"   买卖价差: ${spread:.2f} ({spread_pct:.3f}%)")
            
            # 取消市场数据订阅
            self.ib.cancelMktData(contract)
            
            return ticker
            
        except Exception as e:
            print(f"❌ 获取QQQ CFD价格失败: {e}")
            return None
            
    def get_qqq_stock_price(self):
        """获取QQQ股票价格作为对比"""
        print("\n=== 获取QQQ股票当前价格（对比） ===")
        
        try:
            # 创建QQQ股票合约
            print("🔍 创建QQQ股票合约...")
            stock_contract = Stock('QQQ', 'SMART', 'USD')
            
            # 完善合约信息
            qualified = self.ib.qualifyContracts(stock_contract)
            
            if not qualified:
                print("❌ 无法创建QQQ股票合约")
                return None
                
            contract = qualified[0]
            print(f"✅ QQQ股票合约创建成功:")
            print(f"   合约ID: {contract.conId}")
            print(f"   交易所: {contract.exchange}")
            print(f"   货币: {contract.currency}")
            
            # 获取实时价格数据
            print(f"\n📊 获取股票延迟价格数据...")
            
            # 设置为延迟数据模式
            self.ib.reqMarketDataType(3)  # 3 = 延迟数据
            
            # 请求市场数据
            self.ib.reqMktData(contract, '', False, False)
            
            # 等待数据更新
            print("⏳ 等待价格数据...")
            time.sleep(3)
            
            # 获取ticker数据
            ticker = self.ib.ticker(contract)
            
            print(f"\n💰 QQQ股票 价格信息:")
            if ticker.last and ticker.last > 0:
                print(f"   最新价格: ${ticker.last:.2f}")
            else:
                print(f"   最新价格: N/A")
                
            if ticker.bid and ticker.bid > 0:
                print(f"   买一价格: ${ticker.bid:.2f}")
            else:
                print(f"   买一价格: N/A")
                
            if ticker.ask and ticker.ask > 0:
                print(f"   卖一价格: ${ticker.ask:.2f}")
            else:
                print(f"   卖一价格: N/A")
                
            if ticker.volume and ticker.volume > 0:
                print(f"   成交量: {ticker.volume:,.0f}")
            else:
                print(f"   成交量: N/A")
                
            if ticker.time:
                print(f"   更新时间: {ticker.time}")
            
            # 取消市场数据订阅
            self.ib.cancelMktData(contract)
            
            return ticker
            
        except Exception as e:
            print(f"❌ 获取QQQ股票价格失败: {e}")
            return None
            

def main():
    """主测试函数"""
    print("=== IBKR API 测试程序 ===")
    print("确保IB Gateway已启动并配置正确")
    print("默认连接到: 127.0.0.1:4002 (IB Gateway模拟)")
    
    # 创建测试实例
    tester = IBKRTest()
    
    try:
        # 1. 连接测试 - 使用Paper Trading端口4002
        if not tester.connect(port=4002):
            print("\n请检查:")
            print("1. IB Gateway是否已启动")
            print("2. API连接是否已在IB Gateway中启用")
            print("3. 端口号是否正确（实盘:4001, 模拟:4002）")
            return
            
        # 2. 测试连接状态
        tester.test_connection()
        
        # 3. 获取账户信息
        tester.get_account_info()
        
        # 4. 专门获取QQQ CFD当前价格
        tester.get_qqq_cfd_price()
        
        # 4.5. 获取QQQ股票价格作为对比
        tester.get_qqq_stock_price()
        
        # 5. 搜索QQQ CFD（更详细的搜索）
        tester.search_qqq_cfd()
        
        # 6. 测试下单功能
        tester.test_order_capabilities()
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
    finally:
        # 断开连接
        tester.disconnect()
        

if __name__ == "__main__":
    main()
