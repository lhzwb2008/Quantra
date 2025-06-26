#!/usr/bin/env python3
"""
查找和测试SPX (S&P 500) CFD产品
"""

from ib_insync import *
import time


def check_spx_cfd():
    """查找SPX CFD产品"""
    
    ib = IB()
    
    try:
        # 连接
        print("连接到IB Gateway...")
        ib.connect('127.0.0.1', 4001, clientId=1)
        print("✅ 连接成功\n")
        
        # 1. 搜索SPX相关产品
        print("=== 搜索SPX相关产品 ===")
        
        # 尝试多个可能的符号
        symbols_to_try = ['SPX', 'SPX500', 'US500', 'SP500', 'USA500']
        
        all_results = []
        for symbol in symbols_to_try:
            print(f"\n尝试搜索: {symbol}")
            try:
                results = ib.reqMatchingSymbols(symbol)
                if results:
                    print(f"  找到 {len(results)} 个结果")
                    all_results.extend(results)
            except Exception as e:
                print(f"  搜索失败: {e}")
        
        # 按类型分类
        by_type = {}
        for result in all_results:
            sec_type = result.contract.secType
            if sec_type not in by_type:
                by_type[sec_type] = []
            by_type[sec_type].append(result)
        
        print(f"\n找到的产品类型汇总:")
        for sec_type, items in by_type.items():
            print(f"  {sec_type}: {len(items)} 个")
        
        # 显示CFD产品详情
        if 'CFD' in by_type:
            print("\n=== SPX CFD产品详情 ===")
            for i, result in enumerate(by_type['CFD']):
                contract = result.contract
                print(f"\nCFD #{i+1}:")
                print(f"  符号: {contract.symbol}")
                print(f"  交易所: {contract.primaryExchange}")
                print(f"  货币: {contract.currency}")
                print(f"  合约ID: {contract.conId}")
                print(f"  描述: {result.contractDescription}")
                
                # 获取详细信息
                try:
                    details = ib.reqContractDetails(contract)
                    if details:
                        detail = details[0]
                        print(f"  最小变动: {detail.minTick}")
                        print(f"  交易时间: {detail.tradingHours[:60]}...")
                        
                        # 获取市场数据
                        print("\n  获取市场数据...")
                        ib.reqMktData(contract, '', True, False)
                        time.sleep(2)
                        
                        ticker = ib.ticker(contract)
                        if ticker.bid or ticker.ask:
                            print(f"  买价: {ticker.bid}")
                            print(f"  卖价: {ticker.ask}")
                            print(f"  价差: {ticker.ask - ticker.bid if ticker.ask and ticker.bid else 'N/A'}")
                        if ticker.last:
                            print(f"  最新价: {ticker.last}")
                        if ticker.close:
                            print(f"  昨收盘: {ticker.close}")
                except Exception as e:
                    print(f"  获取详情失败: {e}")
        
        # 2. 直接尝试创建SPX CFD
        print("\n=== 直接创建SPX CFD合约 ===")
        
        # 尝试不同的交易所和符号组合
        cfd_configs = [
            ('SPX', 'SMART'),
            ('SPX', 'CFD'),
            ('SPX500', 'SMART'),
            ('US500', 'SMART'),
            ('USA500', 'SMART')
        ]
        
        for symbol, exchange in cfd_configs:
            try:
                print(f"\n尝试: {symbol} @ {exchange}")
                cfd = CFD(symbol, exchange=exchange)
                details = ib.reqContractDetails(cfd)
                
                if details:
                    print(f"✅ 成功找到CFD!")
                    contract = details[0].contract
                    print(f"  合约ID: {contract.conId}")
                    print(f"  实际符号: {contract.symbol}")
                    print(f"  交易所: {contract.exchange}")
                    print(f"  货币: {contract.currency}")
                    print(f"  描述: {details[0].longName}")
                    
                    # 测试下单功能（仅验证）
                    print("\n  测试订单验证（不会真实下单）:")
                    test_order = LimitOrder('BUY', 1, 4000.00)
                    order_state = ib.whatIfOrder(contract, test_order)
                    
                    if order_state:
                        print(f"  ✅ 可以交易")
                        print(f"  预计佣金: ${order_state.commission}")
                        print(f"  初始保证金: ${order_state.initMarginChange}")
                        
                    break  # 找到一个就停止
                    
            except Exception as e:
                print(f"  失败: {str(e)[:50]}...")
        
        # 3. 查找相关的指数产品
        print("\n=== 其他S&P 500相关产品 ===")
        
        # SPY是S&P 500的ETF
        print("\n查找SPY (S&P 500 ETF):")
        try:
            spy_results = ib.reqMatchingSymbols('SPY')
            spy_cfds = [r for r in spy_results if r.contract.secType == 'CFD']
            if spy_cfds:
                print(f"✅ 找到SPY CFD: {len(spy_cfds)} 个")
                contract = spy_cfds[0].contract
                print(f"  符号: {contract.symbol}")
                print(f"  合约ID: {contract.conId}")
            else:
                print("❌ 未找到SPY CFD")
        except:
            pass
        
        # 4. 建议
        print("\n=== 交易建议 ===")
        if 'CFD' in by_type:
            print("✅ 找到SPX CFD产品，可以直接交易")
        else:
            print("如果没有找到SPX CFD，可以考虑:")
            print("1. SPY ETF - S&P 500指数ETF")
            print("2. ES期货 - E-mini S&P 500期货")
            print("3. SPX期权 - S&P 500指数期权")
            print("4. 杠杆ETF - SPXL (3倍做多) 或 SPXS (3倍做空)")
            
    except Exception as e:
        print(f"错误: {e}")
        
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\n✅ 已断开连接")


if __name__ == "__main__":
    print("=== SPX CFD 产品查找 ===")
    print("此脚本只做查询，不会进行任何交易")
    print("-" * 40)
    
    check_spx_cfd() 