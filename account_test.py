from longport.openapi import Config, TradeContext, QuoteContext, Period, OrderSide, OrderType, TimeInForceType, AdjustType, OutsideRTH
from dotenv import load_dotenv

# 强制重新加载 .env 文件，覆盖现有环境变量
load_dotenv(override=True)
config = Config.from_env()
# 初始化交易上下文
trade = TradeContext(config)
account = trade.account_balance()
print(account)
