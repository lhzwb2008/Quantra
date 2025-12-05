#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生科技指数成分股反转对冲策略

策略逻辑：
1. 均值回归：买入近期相对弱势的股票
2. 对冲风险：同时反向对冲指数
3. 固定持仓期后平仓

核心发现：
- 恒生科技成分股存在强烈的均值回归特征
- 近期弱于指数的股票会在未来反弹
"""

# ==================== 策略参数配置 ====================
CONFIG = {
    # 回测时间范围
    'start_date': None,              # 回测开始日期（格式：'YYYY-MM-DD'，None表示从最早数据开始）
    'end_date': None,                # 回测结束日期（格式：'YYYY-MM-DD'，None表示到最新数据）
    
    # 策略参数（最优配置：三年都盈利，平均收益76.67%）
    'lookback_days': 10,             # 回看期（天）- 用于计算相对强度
    'holding_days': 7,               # 持仓期（天）- 7日给予更多反转时间
    'min_rel_weakness': 0.10,        # 最小相对弱度（10%）- 需低于指数10%以上
    
    # 对冲参数
    'hedge_ratio': 0.0,              # 对冲比例（0.0=不对冲，1.0=全对冲）
                                     # 建议：0.0-0.2 获得最佳收益，0.4-0.6 平衡风险
    
    # 资金管理
    'initial_capital': 1000000,      # 初始资金（元）- 个股和指数对冲各使用全部资金
    
    # 交易成本（长桥证券港股费率）
    'transaction_cost_rate': 0.001115,  # 交易费率（单边0.1115%）
    'fixed_cost_per_trade': 15,         # 平台使用费（15港元/笔）
}

# 说明：
# - 个股仓位 = initial_capital（全仓买入）
# - 指数对冲仓位 = initial_capital × hedge_ratio（按比例对冲）
# - 长桥证券交易成本：
#   • 佣金：免佣金
#   • 比例费用：0.1115%（单边）= 印花税0.1% + 交收费0.002% + 交易费0.00565% + 其他0.004%
#   • 固定费用：15港元/笔（平台使用费）
#   • 双边总成本：约0.223% + 30港元/次交易
# - 总杠杆 = 2倍（1倍个股 + 1倍指数对冲）

# 示例配置：
# 只回测2025年：'start_date': '2025-01-01', 'end_date': '2025-12-31'
# 回测最近一年：'start_date': '2024-01-01', 'end_date': None
# 全部数据：   'start_date': None, 'end_date': None
# ====================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import os


class DailyReversalHedgeStrategy:
    """基于日线的反转对冲策略"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 回测时间范围
        self.start_date = config.get('start_date', None)
        self.end_date = config.get('end_date', None)
        
        # 策略参数
        self.lookback_days = config['lookback_days']
        self.holding_days = config['holding_days']
        self.min_rel_weakness = config['min_rel_weakness']
        
        # 资金管理（个股和指数都使用全仓）
        self.initial_capital = config['initial_capital']
        self.stock_position = self.initial_capital      # 全仓买入个股
        self.hedge_ratio = config.get('hedge_ratio', 1.0)  # 对冲比例，默认全对冲
        self.index_position = self.initial_capital      # 全仓反向对冲指数
        
        # 交易成本（长桥证券费率）
        self.transaction_cost_rate = config['transaction_cost_rate']  # 费率（单边）
        self.fixed_cost_per_trade = config['fixed_cost_per_trade']    # 固定费用/笔
        
        # 交易记录
        self.trades = []
        self.current_position = None
        self.capital = self.initial_capital
        self.daily_returns = []
        
    def calculate_relative_strength(self, stock_prices: pd.Series, index_prices: pd.Series) -> float:
        """
        计算相对强度
        正值 = 强于指数，负值 = 弱于指数
        """
        if len(stock_prices) < 2 or len(index_prices) < 2:
            return 0.0
        
        stock_return = (stock_prices.iloc[-1] - stock_prices.iloc[0]) / stock_prices.iloc[0]
        index_return = (index_prices.iloc[-1] - index_prices.iloc[0]) / index_prices.iloc[0]
        
        return stock_return - index_return
    
    def find_weakest_stock(self, date: datetime, index_df: pd.DataFrame, components: Dict[str, pd.DataFrame]) -> Dict:
        """找出相对最弱的股票（均值回归标的）"""
        
        index_hist = index_df[index_df['Date'] <= date].tail(self.lookback_days)
        if len(index_hist) < self.lookback_days:
            return None
        
        index_prices = index_hist['Close']
        
        candidates = []
        for symbol, stock_df in components.items():
            stock_hist = stock_df[stock_df['Date'] <= date].tail(self.lookback_days)
            
            if len(stock_hist) < self.lookback_days:
                continue
            
            stock_prices = stock_hist['Close']
            rel_strength = self.calculate_relative_strength(stock_prices, index_prices)
            
            stock_return = (stock_prices.iloc[-1] - stock_prices.iloc[0]) / stock_prices.iloc[0]
            index_return = (index_prices.iloc[-1] - index_prices.iloc[0]) / index_prices.iloc[0]
            
            candidates.append({
                'symbol': symbol,
                'rel_strength': rel_strength,
                'stock_return': stock_return,
                'index_return': index_return,
                'stock_price': stock_hist.iloc[-1]['Close']
            })
        
        if not candidates:
            return None
        
        # 按相对强度排序，找最弱的
        candidates.sort(key=lambda x: x['rel_strength'])
        weakest = candidates[0]
        
        # 只有当相对弱度超过阈值时才开仓
        if weakest['rel_strength'] < -self.min_rel_weakness:
            return {
                'symbol': weakest['symbol'],
                'direction': 'long',
                'rel_strength': weakest['rel_strength'],
                'stock_return': weakest['stock_return'],
                'index_return': weakest['index_return'],
                'stock_price': weakest['stock_price']
            }
        
        return None
    
    def open_position(self, signal: Dict, date: datetime, index_price: float):
        """开仓"""
        stock_price = signal['stock_price']
        
        stock_shares = self.stock_position / stock_price
        index_shares = self.index_position / index_price
        
        self.current_position = {
            'symbol': signal['symbol'],
            'direction': 'long',
            'entry_date': date,
            'entry_stock_price': stock_price,
            'entry_index_price': index_price,
            'stock_shares': stock_shares,
            'index_shares': index_shares,
            'rel_strength': signal['rel_strength'],
            'stock_return': signal['stock_return'],
            'index_return': signal['index_return']
        }
        
        print(f"\n{date.strftime('%Y-%m-%d')} 开仓:")
        print(f"  股票: {signal['symbol']} 做多 @ {stock_price:.2f}")
        print(f"  指数: 03032.HK 做空 @ {index_price:.2f}")
        print(f"  相对弱度: {signal['rel_strength']*100:.2f}% "
              f"(个股{signal['stock_return']*100:.2f}% vs 指数{signal['index_return']*100:.2f}%)")
    
    def close_position(self, date: datetime, stock_price: float, index_price: float, reason: str = ''):
        """平仓"""
        if not self.current_position:
            return None
        
        pos = self.current_position
        
        # 计算个股盈亏（做多）
        stock_pnl = pos['stock_shares'] * (stock_price - pos['entry_stock_price'])
        stock_return = (stock_price - pos['entry_stock_price']) / pos['entry_stock_price']
        
        # 计算对冲盈亏（做空指数），按对冲比例调整
        index_pnl_full = pos['index_shares'] * (pos['entry_index_price'] - index_price)
        index_pnl = index_pnl_full * self.hedge_ratio
        index_return = (pos['entry_index_price'] - index_price) / pos['entry_index_price']
        
        total_pnl = stock_pnl + index_pnl
        
        # 计算成本（长桥证券费率：单边0.1115% + 15港元/笔，双边需要×2）
        # 个股成本：比例费用 + 固定费用
        stock_cost_rate = self.stock_position * self.transaction_cost_rate * 2  # 买入+卖出
        stock_cost_fixed = self.fixed_cost_per_trade * 2  # 买入+卖出各15港元
        stock_cost = stock_cost_rate + stock_cost_fixed
        
        # 指数成本（按对冲比例）
        if self.hedge_ratio > 0:
            index_cost_rate = self.index_position * self.transaction_cost_rate * 2 * self.hedge_ratio
            index_cost_fixed = self.fixed_cost_per_trade * 2 * self.hedge_ratio
            index_cost = index_cost_rate + index_cost_fixed
        else:
            index_cost = 0
        
        cost = stock_cost + index_cost
        net_pnl = total_pnl - cost
        
        holding_days = (date - pos['entry_date']).days
        
        trade = {
            'symbol': pos['symbol'],
            'direction': pos['direction'],
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'holding_days': holding_days,
            'entry_stock_price': pos['entry_stock_price'],
            'exit_stock_price': stock_price,
            'entry_index_price': pos['entry_index_price'],
            'exit_index_price': index_price,
            'stock_pnl': stock_pnl,
            'stock_return': stock_return,
            'index_pnl': index_pnl,
            'index_return': index_return,
            'gross_pnl': total_pnl,
            'transaction_cost': cost,
            'net_pnl': net_pnl,
            'rel_strength': pos['rel_strength'],
            'close_reason': reason
        }
        
        self.trades.append(trade)
        old_capital = self.capital
        self.capital += net_pnl
        
        # 记录日收益率（用于计算夏普比率）
        self.daily_returns.append(net_pnl / old_capital)
        
        # 打印交易详情
        status = "✅" if net_pnl > 0 else "❌"
        print(f"\n{date.strftime('%Y-%m-%d')} 平仓 {status}:")
        print(f"  股票盈亏: {stock_pnl:>10,.0f}元 ({stock_return*100:>6.2f}%)")
        if self.hedge_ratio > 0:
            print(f"  对冲盈亏: {index_pnl:>10,.0f}元 ({index_return*100:>6.2f}%) [比例{self.hedge_ratio*100:.0f}%]")
        else:
            print(f"  对冲盈亏: 未对冲")
        print(f"  合计盈亏: {total_pnl:>10,.0f}元")
        if self.hedge_ratio > 0:
            print(f"  交易成本: {cost:>10,.0f}元 (个股{stock_cost:.0f}元 + 指数{index_cost:.0f}元)")
            print(f"    费率: {self.transaction_cost_rate*100:.3f}%×2 + 固定费{self.fixed_cost_per_trade:.0f}港元×2")
        else:
            print(f"  交易成本: {cost:>10,.0f}元 (个股{stock_cost:.0f}元)")
            print(f"    费率: {self.transaction_cost_rate*100:.3f}%×2 + 固定费{self.fixed_cost_per_trade:.0f}港元×2")
        print(f"  净盈亏:   {net_pnl:>10,.0f}元 | 持仓{holding_days}日 | 账户: {self.capital:>12,.0f}元")
        
        self.current_position = None
        return trade
    
    def backtest(self, index_df: pd.DataFrame, components: Dict[str, pd.DataFrame]):
        """回测"""
        
        # 确保数据按日期排序
        index_df = index_df.sort_values('Date').reset_index(drop=True)
        for symbol in components:
            components[symbol] = components[symbol].sort_values('Date').reset_index(drop=True)
        
        # 过滤回测时间范围
        if self.start_date:
            index_df = index_df[index_df['Date'] >= self.start_date].reset_index(drop=True)
            for symbol in components:
                components[symbol] = components[symbol][components[symbol]['Date'] >= self.start_date].reset_index(drop=True)
        
        if self.end_date:
            index_df = index_df[index_df['Date'] <= self.end_date].reset_index(drop=True)
            for symbol in components:
                components[symbol] = components[symbol][components[symbol]['Date'] <= self.end_date].reset_index(drop=True)
        
        # 显示回测信息
        actual_start = index_df['Date'].min()
        actual_end = index_df['Date'].max()
        
        print("="*100)
        print("恒生科技指数成分股反转对冲策略回测")
        print("="*100)
        print(f"回测时间: {actual_start} ~ {actual_end}")
        print(f"回看期: {self.lookback_days}日 | 持仓期: {self.holding_days}日 | 最小相对弱度: {self.min_rel_weakness*100:.1f}%")
        if self.hedge_ratio > 0:
            print(f"初始资金: {self.initial_capital:,}元 | 个股仓位: {self.stock_position:,}元(全仓) | 指数对冲: {int(self.index_position*self.hedge_ratio):,}元({self.hedge_ratio*100:.0f}%)")
            print(f"交易费率: {self.transaction_cost_rate*100:.3f}%(单边) + {self.fixed_cost_per_trade}港元/笔 | 总杠杆: {1+self.hedge_ratio:.1f}倍")
        else:
            print(f"初始资金: {self.initial_capital:,}元 | 个股仓位: {self.stock_position:,}元(全仓) | 指数对冲: 不对冲")
            print(f"交易费率: {self.transaction_cost_rate*100:.3f}%(单边) + {self.fixed_cost_per_trade}港元/笔 | 总杠杆: 1倍")
        print("="*100)
        
        start_idx = self.lookback_days
        dates = index_df['Date'].tolist()
        
        for i in range(start_idx, len(dates)):
            current_date = dates[i]
            
            # 检查持仓是否到期
            if self.current_position:
                hold_days = (pd.to_datetime(current_date) - pd.to_datetime(self.current_position['entry_date'])).days
                
                if hold_days >= self.holding_days:
                    symbol = self.current_position['symbol']
                    stock_df = components[symbol]
                    stock_data = stock_df[stock_df['Date'] == current_date]
                    index_data = index_df[index_df['Date'] == current_date]
                    
                    if len(stock_data) > 0 and len(index_data) > 0:
                        self.close_position(
                            pd.to_datetime(current_date),
                            stock_data.iloc[0]['Close'],
                            index_data.iloc[0]['Close'],
                            f'{self.holding_days}日到期'
                        )
            
            # 如果没有持仓，寻找新机会
            if not self.current_position:
                signal = self.find_weakest_stock(current_date, index_df, components)
                
                if signal:
                    index_data = index_df[index_df['Date'] == current_date]
                    if len(index_data) > 0:
                        self.open_position(signal, pd.to_datetime(current_date), index_data.iloc[0]['Close'])
        
        # 强制平仓剩余持仓
        if self.current_position:
            last_date = dates[-1]
            symbol = self.current_position['symbol']
            stock_df = components[symbol]
            stock_data = stock_df[stock_df['Date'] == last_date]
            index_data = index_df[index_df['Date'] == last_date]
            
            if len(stock_data) > 0 and len(index_data) > 0:
                self.close_position(
                    pd.to_datetime(last_date),
                    stock_data.iloc[0]['Close'],
                    index_data.iloc[0]['Close'],
                    '回测结束'
                )
        
        print("\n" + "="*100)
    
    def calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(self.daily_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # 年化夏普比率（假设每年约50个交易周期）
        sharpe = (mean_return / std_return) * np.sqrt(50)
        return sharpe
    
    def calculate_max_drawdown(self, df):
        """计算最大回撤"""
        # 计算累计资金曲线
        df = df.copy()
        df['cumulative_capital'] = self.initial_capital + df['net_pnl'].cumsum()
        
        # 计算最大回撤
        df['peak'] = df['cumulative_capital'].cummax()
        df['drawdown'] = (df['cumulative_capital'] - df['peak']) / df['peak'] * 100
        
        max_drawdown = df['drawdown'].min()
        
        return max_drawdown, df
    
    def analyze_results(self):
        """分析回测结果"""
        if not self.trades:
            print("无交易记录")
            return None
        
        df = pd.DataFrame(self.trades)
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        trades_count = len(df)
        win_count = (df['net_pnl'] > 0).sum()
        win_rate = win_count / trades_count * 100
        
        avg_pnl = df['net_pnl'].mean()
        max_win = df['net_pnl'].max()
        max_loss = df['net_pnl'].min()
        
        stock_pnl_sum = df['stock_pnl'].sum()
        index_pnl_sum = df['index_pnl'].sum()
        gross_pnl = df['gross_pnl'].sum()
        total_cost = df['transaction_cost'].sum()
        
        sharpe_ratio = self.calculate_sharpe_ratio()
        
        # 计算年化收益
        start_date = pd.to_datetime(df['entry_date'].min())
        end_date = pd.to_datetime(df['exit_date'].max())
        years = (end_date - start_date).days / 365.25
        annual_return = (((self.capital / self.initial_capital) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        # 计算最大回撤
        max_drawdown, df_with_dd = self.calculate_max_drawdown(df)
        
        print("\n" + "="*100)
        print("回测结果")
        print("="*100)
        print(f"回测期间: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({years:.1f}年)")
        print(f"总收益率: {total_return:.2f}%")
        print(f"年化收益: {annual_return:.2f}%")
        print(f"夏普比率: {sharpe_ratio:.3f}")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"交易次数: {trades_count}笔")
        print(f"胜率: {win_rate:.1f}% ({win_count}胜/{trades_count-win_count}负)")
        print(f"平均盈亏: {avg_pnl:,.0f}元/笔")
        print(f"最大盈利: {max_win:,.0f}元")
        print(f"最大亏损: {max_loss:,.0f}元")
        
        print(f"\n组合分析:")
        print(f"  个股总盈亏: {stock_pnl_sum:>12,.0f}元 ({stock_pnl_sum/self.initial_capital*100:>7.2f}%)")
        print(f"  对冲总盈亏: {index_pnl_sum:>12,.0f}元 ({index_pnl_sum/self.initial_capital*100:>7.2f}%)")
        print(f"  毛盈亏合计: {gross_pnl:>12,.0f}元 ({gross_pnl/self.initial_capital*100:>7.2f}%)")
        print(f"  交易成本:   {total_cost:>12,.0f}元 ({total_cost/self.initial_capital*100:>7.2f}%)")
        print(f"  净盈亏:     {self.capital-self.initial_capital:>12,.0f}元 ({total_return:>7.2f}%)")
        
        # 按年份统计
        df_with_dd['year'] = pd.to_datetime(df_with_dd['entry_date']).dt.year
        
        print(f"\n年度统计:")
        for year in sorted(df_with_dd['year'].unique()):
            year_data = df_with_dd[df_with_dd['year'] == year]
            
            count = len(year_data)
            net = year_data['net_pnl'].sum()
            stock = year_data['stock_pnl'].sum()
            hedge = year_data['index_pnl'].sum()
            
            # 计算该年的最大回撤
            year_max_dd, _ = self.calculate_max_drawdown(year_data.reset_index(drop=True))
            
            status = "✅" if net > 0 else "❌"
            print(f"  {status} {year}: {count}笔, 个股{stock:>10,.0f}, 对冲{hedge:>10,.0f}, 合计{net:>10,.0f}元, 最大回撤{year_max_dd:>6.2f}%")
        
        print("="*100)
        
        return {
            'trades_df': df,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate
        }


if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'daily_data')
    
    # 加载数据
    print("加载数据...")
    print(f"数据目录: {data_dir}")
    
    index_df = pd.read_csv(os.path.join(data_dir, '03032_daily.csv'))
    
    components = {}
    for file in os.listdir(data_dir):
        if file.endswith('_daily.csv') and file != '03032_daily.csv':
            symbol = file.replace('_daily.csv', '') + '.HK'
            components[symbol] = pd.read_csv(os.path.join(data_dir, file))
    
    print(f"加载了指数和 {len(components)} 只成分股数据")
    print(f"指数数据: {len(index_df)}条, 从 {index_df['Date'].min()} 到 {index_df['Date'].max()}\n")
    
    # 运行策略
    strategy = DailyReversalHedgeStrategy(CONFIG)
    strategy.backtest(index_df, components)
    strategy.analyze_results()

