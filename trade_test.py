import os
import sys
import time
from datetime import datetime
import pytz
from math import floor
from decimal import Decimal

# 导入长桥API
from longport.openapi import Config, TradeContext, QuoteContext, OrderSide, OrderType, TimeInForceType, OutsideRTH, OrderStatus

# 交易的股票代码
SYMBOL = "TQQQ.US"

def get_us_eastern_time():
    """获取当前的美东时间"""
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

def create_contexts():
    """创建长桥API的行情和交易上下文"""
    try:
        print("正在初始化长桥API连接...")
        # 从环境变量加载配置
        config = Config.from_env()
        print("成功从环境变量加载API配置")
        
        # 创建行情上下文
        quote_ctx = QuoteContext(config)
        print("成功创建行情上下文")
        
        # 创建交易上下文
        trade_ctx = TradeContext(config)
        print("成功创建交易上下文")
        
        return quote_ctx, trade_ctx
    except Exception as e:
        print(f"创建API上下文时出错: {e}")
        return None, None

def get_account_balance(trade_ctx):
    """获取账户余额"""
    try:
        if trade_ctx is None:
            print("交易上下文未初始化")
            return 0
            
        print("正在获取美元账户余额...")
        # 直接获取美元账户余额
        balance_list = trade_ctx.account_balance(currency="USD")
        
        # 处理返回结果
        if balance_list and len(balance_list) > 0:
            balance = balance_list[0]
            available_cash = float(balance.net_assets)
            print(f"美元账户余额: ${available_cash:.2f}")
            return available_cash
        else:
            print("未找到美元账户，请检查您的账户设置")
            return 0
    except Exception as e:
        print(f"获取账户余额时出错: {e}")
        return 0

def get_current_positions(trade_ctx):
    """获取当前持仓"""
    try:
        if trade_ctx is None:
            print("交易上下文未初始化")
            return {}
            
        # 获取股票持仓
        stock_positions_response = trade_ctx.stock_positions()
        
        # 提取持仓信息
        positions = {}
        
        # 正确处理StockPositionsResponse对象
        for channel in stock_positions_response.channels:
            for position in channel.positions:
                symbol = position.symbol
                quantity = int(position.quantity)
                cost_price = float(position.cost_price)
                positions[symbol] = {
                    "quantity": quantity,
                    "cost_price": cost_price
                }
        
        return positions
    except Exception as e:
        print(f"获取持仓时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}

def get_quote(quote_ctx, symbol):
    """获取股票实时报价"""
    try:
        if quote_ctx is None:
            print("行情上下文未初始化")
            return {}
            
        # 获取实时行情
        quotes = quote_ctx.quote([symbol])
        
        if not quotes:
            return {}
            
        # 转换为字典格式
        quote_data = {
            "symbol": quotes[0].symbol,
            "last_done": str(quotes[0].last_done),
            "open": str(quotes[0].open),
            "high": str(quotes[0].high),
            "low": str(quotes[0].low),
            "volume": str(quotes[0].volume),
            "turnover": str(quotes[0].turnover),
            "timestamp": quotes[0].timestamp.isoformat()
        }
        
        return quote_data
    except Exception as e:
        print(f"获取报价时出错: {e}")
        return {}

def is_overnight_session():
    """判断当前是否是夜盘交易时段"""
    et_now = get_us_eastern_time()
    et_hour = et_now.hour
    
    # 美股夜盘:
    # - 盘前 4:00 AM - 9:30 AM (ET)
    # - 盘后 4:00 PM - 8:00 PM (ET)
    return (4 <= et_hour < 9) or (16 <= et_hour < 20)

def submit_order(trade_ctx, symbol, side, quantity, order_type=OrderType.MO, price=None, outside_rth=None):
    """
    提交订单
    
    参数:
        trade_ctx: 交易上下文
        symbol: 股票代码
        side: "Buy" 或 "Sell"
        quantity: 订单数量
        order_type: 订单类型，默认为市价单 OrderType.MO
        price: 限价单价格 (限价单必需)
        outside_rth: 是否允许盘前盘后交易，None则根据当前时间自动判断
        
    返回:
        str: 成功则返回订单ID，失败返回None
    """
    try:
        if trade_ctx is None:
            print("交易上下文未初始化")
            return None
        
        # 打印参数类型以便调试    
        print(f"订单参数类型: order_type={type(order_type)}, outside_rth={type(outside_rth)}")
            
        # 转换side为SDK的OrderSide枚举
        sdk_side = OrderSide.Buy if side == "Buy" else OrderSide.Sell
        
        # 确保order_type是OrderType枚举
        if isinstance(order_type, str):
            # 如果传入的是字符串类型，尝试转换
            order_type_map = {
                "MO": OrderType.MO,
                "LO": OrderType.LO, 
                "ELO": OrderType.ELO,
                "AO": OrderType.AO,
                "ALO": OrderType.ALO
            }
            order_type = order_type_map.get(order_type, OrderType.MO)
            print(f"已将字符串订单类型 '{order_type}' 转换为枚举类型")
        
        # 设置time_in_force
        time_in_force = TimeInForceType.Day
        
        # 如果outside_rth为None，则根据当前时间自动判断
        if outside_rth is None:
            if is_overnight_session():
                # 如果是夜盘时段，则设置为允许夜盘交易
                outside_rth = OutsideRTH.Overnight
                print("当前为夜盘交易时段，设置为夜盘交易模式")
            else:
                # 否则设置为允许任何时间交易
                outside_rth = OutsideRTH.AnyTime
                print("当前为正常交易时段，设置为任何时间交易模式")
        
        # 将数量转换为Decimal类型
        dec_quantity = Decimal(str(quantity))
        
        print(f"提交订单: {symbol}, {side}, 数量: {quantity}, 订单类型: {order_type}, outside_rth: {outside_rth}")
        
        # 提交订单
        if order_type == OrderType.LO and price is not None:
            # 限价单需要价格 - 价格也转换为Decimal类型
            dec_price = Decimal(str(price))
            print(f"提交限价单，价格: {dec_price}")
            response = trade_ctx.submit_order(
                symbol=symbol,
                order_type=order_type,
                side=sdk_side,
                submitted_price=dec_price,
                submitted_quantity=dec_quantity,
                time_in_force=time_in_force,
                outside_rth=outside_rth
            )
        else:
            # 市价单不需要价格
            print(f"提交市价单")
            response = trade_ctx.submit_order(
                symbol=symbol,
                order_type=OrderType.MO,  # 强制使用市价单
                side=sdk_side,
                submitted_quantity=dec_quantity,
                time_in_force=time_in_force,
                outside_rth=outside_rth
            )
        
        # 从SubmitOrderResponse对象中提取order_id
        print(f"订单提交成功: {response}")
        return response.order_id
        
    except Exception as e:
        print(f"提交订单时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_order_status(trade_ctx, order_id):
    """获取订单状态"""
    try:
        if trade_ctx is None:
            print("交易上下文未初始化")
            return {}
            
        # 确保order_id是字符串类型
        if not isinstance(order_id, str):
            order_id = str(order_id)
        
        print(f"正在获取订单 {order_id} 的状态...")
            
        # 获取订单详情
        order_detail = trade_ctx.order_detail(order_id)
        
        # 转换OrderStatus为字符串
        status_str = str(order_detail.status)
        
        # 转换为字典格式
        order_info = {
            "order_id": order_detail.order_id,
            "status": status_str,
            "stock_name": order_detail.stock_name,
            "quantity": order_detail.quantity,
            "executed_quantity": order_detail.executed_quantity,
            "price": str(order_detail.price),
            "executed_price": str(order_detail.executed_price),
            "submitted_at": order_detail.submitted_at.isoformat(),
            "side": str(order_detail.side)  # 也转换为字符串
        }
        
        print(f"订单状态获取成功: {status_str}")
        return order_info
    except Exception as e:
        print(f"获取订单状态时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}

def cancel_order(trade_ctx, order_id):
    """取消订单"""
    try:
        if trade_ctx is None:
            print("交易上下文未初始化")
            return False
            
        # 确保order_id是字符串类型
        if not isinstance(order_id, str):
            order_id = str(order_id)
            
        # 取消订单
        trade_ctx.cancel_order(order_id)
        
        return True
    except Exception as e:
        print(f"取消订单时出错: {e}")
        return False

def test_trading():
    """测试交易功能"""
    # 初始化API上下文
    quote_ctx, trade_ctx = create_contexts()
    
    if quote_ctx is None or trade_ctx is None:
        print("无法创建API上下文，请检查API凭证是否正确设置")
        return
    
    try:
        # 获取当前美东时间
        now_et = get_us_eastern_time()
        print(f"当前美东时间: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 判断当前是否为夜盘时段
        is_night_session = is_overnight_session()
        print(f"当前{'是' if is_night_session else '不是'}夜盘交易时段")
        
        # 获取账户余额
        balance = get_account_balance(trade_ctx)
        if balance <= 0:
            print("账户余额不足或无法获取账户余额")
            return
        
        # 获取当前持仓
        positions = get_current_positions(trade_ctx)
        print("当前持仓:")
        if positions:
            for symbol, position in positions.items():
                print(f"  {symbol}: {position['quantity']}股, 成本价: ${position['cost_price']:.2f}")
        else:
            print("  暂无持仓")
        
        # 获取TQQQ的实时报价
        quote = get_quote(quote_ctx, SYMBOL)
        if not quote:
            print(f"无法获取{SYMBOL}的实时报价")
            return
        
        latest_price = float(quote["last_done"])
        print(f"{SYMBOL}当前价格: ${latest_price}")
        
        # 计算可买入的数量 (使用账户余额的10%进行测试)
        test_capital = balance * 0.1
        buy_quantity = floor(test_capital / latest_price)
        
        if buy_quantity <= 0:
            print("可买入的数量太少，请增加测试资金比例或检查账户余额")
            return
        
        print(f"测试资金: ${test_capital:.2f}, 计划买入: {buy_quantity}股 {SYMBOL}")
        
        # 确认是否继续
        confirm = input("是否继续测试交易? (y/n): ")
        if confirm.lower() != 'y':
            print("测试已取消")
            return
        
        # 设置适合当前时段的outside_rth参数
        outside_rth = OutsideRTH.Overnight if is_night_session else OutsideRTH.AnyTime
        print(f"使用交易时段参数: {outside_rth}")
        
        # 执行买入订单 (使用市价单)
        print(f"提交买入订单: {buy_quantity}股 {SYMBOL} (市价单)")
        buy_order_id = None
        try:
            buy_order_id = submit_order(
                trade_ctx, 
                SYMBOL, 
                "Buy", 
                buy_quantity, 
                order_type=OrderType.MO,  # 确保使用市价单
                outside_rth=outside_rth  # 设置合适的交易时段参数
            )
        except Exception as e:
            print(f"买入订单提交过程中出现异常: {e}")
            import traceback
            traceback.print_exc()
            
        if not buy_order_id:
            print("买入订单提交失败，是否继续测试其他功能?")
            confirm = input("继续测试? (y/n): ")
            if confirm.lower() != 'y':
                print("测试已取消")
                return
        else:
            print(f"买入订单已提交，订单ID: {buy_order_id}")
            
            # 等待订单执行并获取状态
            print("等待5秒后检查订单状态...")
            time.sleep(5)
            
            buy_order_status = {}
            try:
                buy_order_status = get_order_status(trade_ctx, buy_order_id)
            except Exception as e:
                print(f"获取订单状态过程中出现异常: {e}")
                
            if buy_order_status:
                print(f"买入订单状态: {buy_order_status}")
            else:
                print("无法获取买入订单状态，但继续执行")
        
        # 再次获取持仓确认
        positions = get_current_positions(trade_ctx)
        print("当前持仓:")
        if positions:
            for symbol, position in positions.items():
                print(f"  {symbol}: {position['quantity']}股, 成本价: ${position['cost_price']:.2f}")
        else:
            print("  暂无持仓")
        
        # 确认是否平仓
        confirm = input("是否测试平仓? (y/n): ")
        if confirm.lower() != 'y':
            print("平仓测试已取消")
            return
        
        # 获取TQQQ持仓数量
        tqqq_quantity = positions.get(SYMBOL, {}).get("quantity", 0)
        if tqqq_quantity <= 0:
            print(f"当前没有{SYMBOL}持仓，无法测试平仓")
            return
        
        # 执行卖出订单 (平仓) (使用市价单)
        print(f"提交卖出订单: {tqqq_quantity}股 {SYMBOL} (市价单)")
        sell_order_id = None
        try:
            sell_order_id = submit_order(
                trade_ctx, 
                SYMBOL, 
                "Sell", 
                tqqq_quantity, 
                order_type=OrderType.MO,  # 确保使用市价单
                outside_rth=outside_rth  # 设置合适的交易时段参数
            )
        except Exception as e:
            print(f"卖出订单提交过程中出现异常: {e}")
            import traceback
            traceback.print_exc()
            
        if not sell_order_id:
            print("卖出订单提交失败")
            return
        
        print(f"卖出订单已提交，订单ID: {sell_order_id}")
        
        # 等待订单执行并获取状态
        print("等待5秒后检查订单状态...")
        time.sleep(5)
        
        sell_order_status = {}
        try:
            sell_order_status = get_order_status(trade_ctx, sell_order_id)
        except Exception as e:
            print(f"获取卖出订单状态过程中出现异常: {e}")
            
        if sell_order_status:
            print(f"卖出订单状态: {sell_order_status}")
        else:
            print("无法获取卖出订单状态，但继续执行")
        
        # 最终持仓确认
        positions = get_current_positions(trade_ctx)
        print("最终持仓:")
        if positions:
            for symbol, position in positions.items():
                print(f"  {symbol}: {position['quantity']}股, 成本价: ${position['cost_price']:.2f}")
        else:
            print("  暂无持仓")
        
        print("交易测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 在Python SDK中，不需要显式关闭上下文
        pass

if __name__ == "__main__":
    print("\n" + "*"*70)
    print("* 长桥API交易测试程序")
    print("* 时间:", get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S'), "(美东时间)")
    print("*"*70 + "\n")
    
    # 检查环境变量
    required_vars = ["LONGPORT_APP_KEY", "LONGPORT_APP_SECRET", "LONGPORT_ACCESS_TOKEN"]
    missing_vars = [var for var in required_vars if var not in os.environ]
    
    if missing_vars:
        print("错误: 缺少以下环境变量:")
        for var in missing_vars:
            print(f"- {var}")
        print("\n请确保设置以下环境变量:")
        print("export LONGPORT_APP_KEY=你的应用键")
        print("export LONGPORT_APP_SECRET=你的应用密钥")
        print("export LONGPORT_ACCESS_TOKEN=你的访问令牌")
        sys.exit(1)
    
    # 运行测试
    test_trading() 