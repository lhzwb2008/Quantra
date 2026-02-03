import pandas as pd
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta, date
import random
import os
from plot_trading_day import plot_trading_day

def calculate_vwap(turnovers, volumes, prices):
    """
    Calculate VWAP using cumulative turnover / cumulative volume
    """
    total_volume = sum(volumes)
    if total_volume > 0:
        return sum(turnovers) / total_volume
    else:
        return prices[-1]

def calculate_vwap_with_hl_average(highs, lows, volumes):
    """
    使用High和Low的平均值计算VWAP的近似值
    平均价格 = (High + Low) / 2
    """
    total_volume = sum(volumes)
    if total_volume > 0:
        # 计算每个时间点的High和Low平均价格
        hl_average_prices = [(h + l) / 2 for h, l in zip(highs, lows)]
        # 计算近似成交额
        turnovers = [avg_price * v for avg_price, v in zip(hl_average_prices, volumes)]
        return sum(turnovers) / total_volume
    else:
        # 如果没有成交量，返回最后一个时间点的HL平均价
        return (highs[-1] + lows[-1]) / 2 if highs and lows else 0

def calculate_vwap_with_turnover(day_df, current_index):
    """
    使用真实的Turnover数据计算VWAP，与simulate.py保持一致
    """
    # 获取当天到当前时间点的所有数据
    current_day_data = day_df.iloc[:current_index + 1].copy()
    
    # 按时间排序确保正确累计
    current_day_data = current_day_data.sort_values('DateTime')
    
    # 计算累计成交量和成交额
    cumulative_volume = current_day_data['Volume'].cumsum()
    cumulative_turnover = current_day_data['Turnover'].cumsum()
    
    # 计算VWAP: 累计成交额 / 累计成交量
    if cumulative_volume.iloc[-1] > 0:
        vwap = cumulative_turnover.iloc[-1] / cumulative_volume.iloc[-1]
    else:
        # 处理成交量为0的情况，使用当前收盘价
        vwap = current_day_data['Close'].iloc[-1]
    
    return vwap

def simulate_day(day_df, prev_close, allowed_times, position_size, config, day_start_capital=None):
    """
    模拟单日交易，使用噪声空间策略 + VWAP
    
    参数:
        day_df: 包含日内数据的DataFrame
        prev_close: 前一日收盘价
        allowed_times: 允许交易的时间列表
        position_size: 仓位大小
        config: 配置字典，包含所有交易参数
    """
    # 从配置中提取参数
    transaction_fee_per_share = config.get('transaction_fee_per_share', 0.01)
    enable_transaction_fees = config.get('enable_transaction_fees', True)  # 新增手续费开关
    trading_end_time = config.get('trading_end_time', (15, 50))
    max_positions_per_day = config.get('max_positions_per_day', float('inf'))
    print_details = config.get('print_trade_details', False)
    debug_time = config.get('debug_time', None)
    use_vwap = config.get('use_vwap', True)  # 新增VWAP开关参数
    # 滑点配置 - 简化为直接的买卖价差
    slippage_per_share = config.get('slippage_per_share', 0.02)  # 每股滑点，买入时多付，卖出时少收
    
    # 🛡️ 日内止损配置 - 新增功能
    enable_intraday_stop_loss = config.get('enable_intraday_stop_loss', False)  # 是否启用日内止损
    intraday_stop_loss_pct = config.get('intraday_stop_loss_pct', 0.04)  # 日内止损阈值，默认4%
    initial_capital = config.get('initial_capital', 100000)  # 初始资金，用于计算日内损失
    
    # 🎯 动态追踪止盈配置 - 新增功能
    enable_trailing_take_profit = config.get('enable_trailing_take_profit', False)  # 是否启用动态追踪止盈
    trailing_tp_activation_pct = config.get('trailing_tp_activation_pct', 0.005)  # 激活追踪止盈的最低浮盈百分比，默认0.5%
    trailing_tp_callback_pct = config.get('trailing_tp_callback_pct', 0.5)  # 保护的利润比例，默认保护50%的浮盈
    
    def apply_slippage(price, is_buy, is_entry):
        """
        应用滑点到交易价格 - 简化版本
        参数:
            price: 原始价格
            is_buy: 是否为买入操作
            is_entry: 是否为开仓操作
        返回:
            调整后的价格
        """
        if slippage_per_share == 0:
            return price
        
        # 简化逻辑：买入时价格上升，卖出时价格下降
        if is_buy:
            return price + slippage_per_share  # 买入多付
        else:
            return price - slippage_per_share  # 卖出少收
    
    def check_intraday_stop_loss(pnl, current_time):
        """
        🛡️ 检查日内止损
        参数:
            pnl: 当前交易的盈亏
            current_time: 当前时间
        返回:
            是否触发止损
        """
        nonlocal current_day_pnl, intraday_stop_triggered
        
        current_day_pnl += pnl
        if enable_intraday_stop_loss and not intraday_stop_triggered:
            # 计算日内损失百分比（基于当日开始资金）
            intraday_loss_pct = current_day_pnl / day_start_capital
            if intraday_loss_pct < -intraday_stop_loss_pct:
                intraday_stop_triggered = True
                if print_details:
                    print(f"🛡️ 日内止损触发！时间: {current_time}, 当日损失: {intraday_loss_pct*100:.2f}%, 阈值: {intraday_stop_loss_pct*100:.1f}%")
                return True
        return False
    
    position = 0  # 0: 无仓位, 1: 多头, -1: 空头
    entry_price = np.nan
    trailing_stop = np.nan
    trade_entry_time = None
    trades = []
    positions_opened_today = 0  # 今日开仓计数器
    
    # 🎯 动态追踪止盈相关变量
    max_profit_price = np.nan  # 持仓期间的最优价格（多头：最高价，空头：最低价）
    trailing_tp_activated = False  # 追踪止盈是否已激活
    dynamic_take_profit_level = np.nan  # 动态止盈线
    
    # 🛡️ 日内止损监控变量
    if day_start_capital is None:
        day_start_capital = initial_capital  # 如果没有传入，使用初始资金
    current_day_pnl = 0  # 当日累计盈亏
    intraday_stop_triggered = False  # 是否已触发日内止损
    
    # 📊 日内资金回撤追踪
    intraday_capital_peak = day_start_capital  # 日内资金峰值
    intraday_max_drawdown = 0  # 日内最大回撤金额
    intraday_capital_low = day_start_capital  # 日内最低资金
    intraday_capital_high = day_start_capital  # 日内最高资金
    
    # 调试时间点标记，确保只打印一次
    debug_printed = False
    
    for idx, row in day_df.iterrows():
        current_time = row['Time']
        price = row['Close']
        high = row['High']
        low = row['Low']
        volume = row['Volume']
        upper = row['UpperBound']
        lower_bound = row['LowerBound']
        sigma = row.get('sigma', 0)
        
        # 📊 计算当前K线的账户资金（用于日内回撤计算）
        # 如果已触发止损，不再更新回撤统计
        if not intraday_stop_triggered:
            # 使用K线的High和Low来计算最好和最坏情况
            if position == 1:  # 多头持仓
                # 多头：高点是最好情况，低点是最坏情况
                best_unrealized = position_size * (high - entry_price)
                worst_unrealized = position_size * (low - entry_price)
            elif position == -1:  # 空头持仓
                # 空头：低点是最好情况，高点是最坏情况
                best_unrealized = position_size * (entry_price - low)
                worst_unrealized = position_size * (entry_price - high)
            else:  # 无持仓
                best_unrealized = 0
                worst_unrealized = 0
            
            # 计算当前K线的资金极值
            current_best_capital = day_start_capital + current_day_pnl + best_unrealized
            current_worst_capital = day_start_capital + current_day_pnl + worst_unrealized
            
            # 更新资金峰值（使用最好情况）
            if current_best_capital > intraday_capital_peak:
                intraday_capital_peak = current_best_capital
            
            # 计算当前回撤（使用最坏情况与峰值的差距）
            current_drawdown = intraday_capital_peak - current_worst_capital
            if current_drawdown > intraday_max_drawdown:
                intraday_max_drawdown = current_drawdown
            
            # 追踪日内资金的绝对最高和最低点
            if current_best_capital > intraday_capital_high:
                intraday_capital_high = current_best_capital
            if current_worst_capital < intraday_capital_low:
                intraday_capital_low = current_worst_capital
        
        # 🛡️ 实时监控：检查日内回撤（从峰值回落）
        if position != 0 and enable_intraday_stop_loss and not intraday_stop_triggered:
            # 计算当前回撤是否超过阈值（基于日内峰值的回撤）
            # current_drawdown 已经在上面计算过了
            drawdown_pct = current_drawdown / day_start_capital
            
            if drawdown_pct > intraday_stop_loss_pct:
                # 触发日内回撤止损
                # 计算止损价格（刚好触发阈值的价格）
                max_drawdown_amount = day_start_capital * intraday_stop_loss_pct
                # 允许的最低资金 = 峰值 - 最大回撤
                min_capital_allowed = intraday_capital_peak - max_drawdown_amount
                # 当前已实现盈亏后的基础资金
                base_capital = day_start_capital + current_day_pnl
                # 允许的最大浮亏
                max_unrealized_loss = base_capital - min_capital_allowed
                
                # 计算止损价格（在回撤刚好达到阈值时的价格）
                # max_unrealized_loss 是允许的最大浮亏（正数表示亏损金额）
                if position == 1:  # 多头
                    # 多头亏损 = position_size * (entry_price - exit_price)
                    # 所以 exit_price = entry_price - 亏损金额/position_size
                    stop_exit_price = entry_price - max_unrealized_loss / position_size - slippage_per_share
                else:  # 空头
                    # 空头亏损 = position_size * (exit_price - entry_price)
                    # 所以 exit_price = entry_price + 亏损金额/position_size
                    stop_exit_price = entry_price + max_unrealized_loss / position_size + slippage_per_share
                
                if print_details:
                    print(f"🛡️ 日内回撤止损触发！时间: {current_time}, 回撤: {drawdown_pct*100:.2f}%, 阈值: {intraday_stop_loss_pct*100:.1f}%")
                    print(f"   峰值资金: ${intraday_capital_peak:.2f}, 当前最差资金: ${current_worst_capital:.2f}")
                
                # 立即平仓
                exit_time = row['DateTime']
                exit_price = stop_exit_price
                
                # 计算交易费用
                if enable_transaction_fees:
                    transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)
                else:
                    transaction_fees = 0
                
                # 计算实际盈亏（包含交易费用）
                if position == 1:
                    pnl = position_size * (exit_price - entry_price) - transaction_fees
                else:
                    pnl = position_size * (entry_price - exit_price) - transaction_fees
                
                # 记录强制平仓交易
                trades.append({
                    'entry_time': trade_entry_time,
                    'exit_time': exit_time,
                    'side': 'Long' if position == 1 else 'Short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'Intraday Stop Loss',
                    'position_size': position_size,
                    'transaction_fees': transaction_fees,
                    'vwap_influenced': False,
                    'stop_level': intraday_stop_loss_pct,
                    'upper_bound': upper if position == 1 else np.nan,
                    'lower_bound': lower_bound if position == -1 else np.nan,
                    'vwap_value': np.nan
                })
                
                # 更新当日盈亏并标记止损已触发
                current_day_pnl += pnl
                intraday_stop_triggered = True
                position = 0
                trailing_stop = np.nan
                
                # 触发止损后，当日不再开仓
                continue
        
        # # 调试特定时间点
        # if debug_time is not None and current_time >= debug_time and not debug_printed:
        #     date_str = row['DateTime'].strftime('%Y-%m-%d')
        #     print(f"\n===== 调试信息 [{date_str} {current_time}] =====")
        #     print(f"价格: {price:.6f}")
        #     print(f"上边界: {upper:.6f}")
        #     print(f"下边界: {lower:.6f}")
        #     print(f"Sigma值: {sigma:.6f}")
        #     print(f"VWAP: {calculate_vwap(prices, volumes, prices):.6f}")
        #     print("=====================================\n")
        #     debug_printed = True  # 确保只打印一次
        
        # 计算当前VWAP（使用真实的Turnover数据）
        # 获取当前行在DataFrame中的位置索引
        current_index = day_df.index.get_loc(idx)
        
        # 检查是否有Turnover字段
        if 'Turnover' in day_df.columns:
            vwap = calculate_vwap_with_turnover(day_df, current_index)
        else:
            # 如果没有Turnover字段，回退到使用HL平均值的方法
            highs = day_df.iloc[:current_index + 1]['High'].tolist()
            lows = day_df.iloc[:current_index + 1]['Low'].tolist()
            volumes = day_df.iloc[:current_index + 1]['Volume'].tolist()
            vwap = calculate_vwap_with_hl_average(highs, lows, volumes)
        
        # 🛡️ 日内止损检查 - 如果已触发止损，当日不再开仓
        if enable_intraday_stop_loss and intraday_stop_triggered:
            # 已触发日内止损，跳过所有开仓逻辑
            pass
        # 在允许时间内的入场信号
        elif position == 0 and current_time in allowed_times and positions_opened_today < max_positions_per_day:
            # 检查潜在多头入场
            if use_vwap:
                # 使用VWAP条件
                long_entry_condition = price > upper and price > vwap
            else:
                # 不使用VWAP条件
                long_entry_condition = price > upper
                
            if long_entry_condition:
                # 打印边界计算详情（如果需要）
                if print_details:
                    date_str = row['DateTime'].strftime('%Y-%m-%d')
                    sigma = row.get('sigma', 0)
                    upper_ref = row.get('upper_ref', 0)
                    lower_ref = row.get('lower_ref', 0)
                    day_open = row.get('day_open', 0)
                    
                    print(f"\n交易点位详情 [{date_str} {current_time}] - 多头入场:")
                    print(f"  价格: {price:.2f} > 上边界: {upper:.2f} 且 > VWAP: {vwap:.2f}")
                    print(f"  边界计算详情:")
                    print(f"    - 日开盘价: {day_open:.2f}, 前日收盘价: {prev_close:.2f}")
                    print(f"    - 上边界参考价: max({day_open:.2f}, {prev_close:.2f}) = {upper_ref:.2f}")
                    print(f"    - 下边界参考价: min({day_open:.2f}, {prev_close:.2f}) = {lower_ref:.2f}")
                    print(f"    - Sigma值: {sigma:.6f}")
                    print(f"    - 上边界计算: {upper_ref:.2f} * (1 + {sigma:.6f}) = {upper:.2f}")
                    print(f"    - 下边界计算: {lower_ref:.2f} * (1 - {sigma:.6f}) = {lower_bound:.2f}")
                
                # 允许多头入场
                position = 1
                entry_price = apply_slippage(price, is_buy=True, is_entry=True)  # 多头开仓是买入
                trade_entry_time = row['DateTime']
                positions_opened_today += 1  # 增加开仓计数器
                # 初始止损设置
                if use_vwap:
                    trailing_stop = max(upper, vwap)
                else:
                    trailing_stop = upper
                    
            # 检查潜在空头入场
            if use_vwap:
                # 使用VWAP条件
                short_entry_condition = price < lower_bound and price < vwap
            else:
                # 不使用VWAP条件
                short_entry_condition = price < lower_bound
                
            if short_entry_condition:
                # 打印边界计算详情（如果需要）
                if print_details:
                    date_str = row['DateTime'].strftime('%Y-%m-%d')
                    sigma = row.get('sigma', 0)
                    upper_ref = row.get('upper_ref', 0)
                    lower_ref = row.get('lower_ref', 0)
                    day_open = row.get('day_open', 0)
                    
                    print(f"\n交易点位详情 [{date_str} {current_time}] - 空头入场:")
                    print(f"  价格: {price:.2f} < 下边界: {lower_bound:.2f} 且 < VWAP: {vwap:.2f}")
                    print(f"  边界计算详情:")
                    print(f"    - 日开盘价: {day_open:.2f}, 前日收盘价: {prev_close:.2f}")
                    print(f"    - 上边界参考价: max({day_open:.2f}, {prev_close:.2f}) = {upper_ref:.2f}")
                    print(f"    - 下边界参考价: min({day_open:.2f}, {prev_close:.2f}) = {lower_ref:.2f}")
                    print(f"    - Sigma值: {sigma:.6f}")
                    print(f"    - 上边界计算: {upper_ref:.2f} * (1 + {sigma:.6f}) = {upper:.2f}")
                    print(f"    - 下边界计算: {lower_ref:.2f} * (1 - {sigma:.6f}) = {lower_bound:.2f}")
                
                # 允许空头入场
                position = -1
                entry_price = apply_slippage(price, is_buy=False, is_entry=True)  # 空头开仓是卖出
                trade_entry_time = row['DateTime']
                positions_opened_today += 1  # 增加开仓计数器
                # 初始止损设置
                if use_vwap:
                    trailing_stop = min(lower_bound, vwap)
                else:
                    trailing_stop = lower_bound
        
        # 更新止损并检查出场信号
        if position != 0:
            if position == 1:  # 多头仓位
                # 计算当前时刻的止损水平
                if use_vwap:
                    current_stop = max(upper, vwap)
                    vwap_influenced = vwap > upper  # 如果VWAP > 上边界，则VWAP影响了止损
                else:
                    current_stop = upper
                    vwap_influenced = False  # 不使用VWAP时，VWAP不影响止损
                
                # 🎯 动态追踪止盈逻辑 - 多头
                trailing_tp_exit = False
                if enable_trailing_take_profit:
                    # 更新最优价格（使用K线的最高价）
                    if np.isnan(max_profit_price) or high > max_profit_price:
                        max_profit_price = high
                    
                    # 计算当前浮盈百分比（使用最高价）
                    current_profit_pct = (max_profit_price - entry_price) / entry_price
                    
                    # 检查是否激活追踪止盈
                    if not trailing_tp_activated and current_profit_pct >= trailing_tp_activation_pct:
                        trailing_tp_activated = True
                        if print_details:
                            print(f"🎯 追踪止盈激活！时间: {current_time}, 浮盈: {current_profit_pct*100:.2f}%, 最高价: {max_profit_price:.2f}")
                    
                    # 如果追踪止盈已激活，计算动态止盈线
                    if trailing_tp_activated:
                        # 动态止盈线 = 入场价 + (最大浮盈 * 保护比例)
                        # 保护比例为 trailing_tp_callback_pct（例如0.5表示保护50%的利润）
                        protected_profit = (max_profit_price - entry_price) * trailing_tp_callback_pct
                        dynamic_take_profit_level = entry_price + protected_profit
                        
                        # 检查是否触发动态止盈（使用当前收盘价）
                        if price <= dynamic_take_profit_level:
                            trailing_tp_exit = True
                            if print_details:
                                print(f"🎯 动态止盈触发！时间: {current_time}, 价格: {price:.2f} <= 止盈线: {dynamic_take_profit_level:.2f}")
                                print(f"   最高价: {max_profit_price:.2f}, 保护利润: {protected_profit:.2f}")
                
                # 如果价格跌破当前止损，则平仓
                exit_condition = price < current_stop or trailing_tp_exit
                
                # 检查是否出场
                if exit_condition and current_time in allowed_times:
                    # 确定出场原因
                    if trailing_tp_exit:
                        exit_reason = 'Trailing Take Profit'
                    else:
                        exit_reason = 'Stop Loss'
                    
                    # 打印出场详情（如果需要）
                    if print_details:
                        date_str = row['DateTime'].strftime('%Y-%m-%d')
                        print(f"\n交易点位详情 [{date_str} {current_time}] - 多头出场 ({exit_reason}):")
                        if trailing_tp_exit:
                            print(f"  价格: {price:.2f} <= 动态止盈线: {dynamic_take_profit_level:.2f}")
                            print(f"  最高价: {max_profit_price:.2f}, 保护比例: {trailing_tp_callback_pct*100:.0f}%")
                        else:
                            print(f"  价格: {price:.2f} < 当前止损: {current_stop:.2f}")
                            print(f"  止损计算: max(上边界={upper:.2f}, VWAP={vwap:.2f}) = {current_stop:.2f}")
                        print(f"  买入价: {entry_price:.2f}, 卖出价: {price:.2f}, 股数: {position_size}")
                    
                    # 平仓多头
                    exit_time = row['DateTime']
                    exit_price = apply_slippage(price, is_buy=False, is_entry=False)  # 多头平仓是卖出
                    # 计算交易费用（开仓和平仓）
                    if enable_transaction_fees:
                        transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
                    else:
                        transaction_fees = 0  # 关闭手续费
                    pnl = position_size * (exit_price - entry_price) - transaction_fees
                    
                    trades.append({
                        'entry_time': trade_entry_time,
                        'exit_time': exit_time,
                        'side': 'Long',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'position_size': position_size,
                        'transaction_fees': transaction_fees,
                        'vwap_influenced': vwap_influenced,  # 新增字段
                        'stop_level': current_stop,
                        'upper_bound': upper,
                        'vwap_value': vwap if use_vwap else np.nan,
                        'trailing_tp_activated': trailing_tp_activated,  # 🎯 动态止盈是否激活
                        'max_profit_price': max_profit_price if not np.isnan(max_profit_price) else np.nan,  # 🎯 最高价
                        'dynamic_tp_level': dynamic_take_profit_level if not np.isnan(dynamic_take_profit_level) else np.nan  # 🎯 动态止盈线
                    })
                    
                    # 🛡️ 检查日内止损
                    check_intraday_stop_loss(pnl, current_time)
                    
                    position = 0
                    trailing_stop = np.nan
                    # 🎯 重置动态追踪止盈变量
                    max_profit_price = np.nan
                    trailing_tp_activated = False
                    dynamic_take_profit_level = np.nan
                    
            elif position == -1:  # 空头仓位
                # 计算当前时刻的止损水平
                if use_vwap:
                    current_stop = min(lower_bound, vwap)
                    vwap_influenced = vwap < lower_bound  # 如果VWAP < 下边界，则VWAP影响了止损
                else:
                    current_stop = lower_bound
                    vwap_influenced = False  # 不使用VWAP时，VWAP不影响止损
                
                # 🎯 动态追踪止盈逻辑 - 空头
                trailing_tp_exit = False
                if enable_trailing_take_profit:
                    # 更新最优价格（使用K线的最低价，空头时低价是有利的）
                    if np.isnan(max_profit_price) or low < max_profit_price:
                        max_profit_price = low
                    
                    # 计算当前浮盈百分比（空头：入场价 - 最低价）
                    current_profit_pct = (entry_price - max_profit_price) / entry_price
                    
                    # 检查是否激活追踪止盈
                    if not trailing_tp_activated and current_profit_pct >= trailing_tp_activation_pct:
                        trailing_tp_activated = True
                        if print_details:
                            print(f"🎯 追踪止盈激活！时间: {current_time}, 浮盈: {current_profit_pct*100:.2f}%, 最低价: {max_profit_price:.2f}")
                    
                    # 如果追踪止盈已激活，计算动态止盈线
                    if trailing_tp_activated:
                        # 动态止盈线 = 入场价 - (最大浮盈 * 保护比例)
                        # 空头的止盈线在入场价下方，当价格上涨超过止盈线时触发
                        protected_profit = (entry_price - max_profit_price) * trailing_tp_callback_pct
                        dynamic_take_profit_level = entry_price - protected_profit
                        
                        # 检查是否触发动态止盈（使用当前收盘价）
                        if price >= dynamic_take_profit_level:
                            trailing_tp_exit = True
                            if print_details:
                                print(f"🎯 动态止盈触发！时间: {current_time}, 价格: {price:.2f} >= 止盈线: {dynamic_take_profit_level:.2f}")
                                print(f"   最低价: {max_profit_price:.2f}, 保护利润: {protected_profit:.2f}")
                
                # 如果价格涨破当前止损，则平仓
                exit_condition = price > current_stop or trailing_tp_exit
                
                # 检查是否出场
                if exit_condition and current_time in allowed_times:
                    # 确定出场原因
                    if trailing_tp_exit:
                        exit_reason = 'Trailing Take Profit'
                    else:
                        exit_reason = 'Stop Loss'
                    
                    # 打印出场详情（如果需要）
                    if print_details:
                        date_str = row['DateTime'].strftime('%Y-%m-%d')
                        print(f"\n交易点位详情 [{date_str} {current_time}] - 空头出场 ({exit_reason}):")
                        if trailing_tp_exit:
                            print(f"  价格: {price:.2f} >= 动态止盈线: {dynamic_take_profit_level:.2f}")
                            print(f"  最低价: {max_profit_price:.2f}, 保护比例: {trailing_tp_callback_pct*100:.0f}%")
                        else:
                            print(f"  价格: {price:.2f} > 当前止损: {current_stop:.2f}")
                            print(f"  止损计算: min(下边界={lower_bound:.2f}, VWAP={vwap:.2f}) = {current_stop:.2f}")
                        print(f"  卖出价: {entry_price:.2f}, 买入价: {price:.2f}, 股数: {position_size}")
                    
                    # 平仓空头
                    exit_time = row['DateTime']
                    exit_price = apply_slippage(price, is_buy=True, is_entry=False)  # 空头平仓是买入
                    # 计算交易费用（开仓和平仓）
                    if enable_transaction_fees:
                        transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
                    else:
                        transaction_fees = 0  # 关闭手续费
                    pnl = position_size * (entry_price - exit_price) - transaction_fees
                    
                    trades.append({
                        'entry_time': trade_entry_time,
                        'exit_time': exit_time,
                        'side': 'Short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'position_size': position_size,
                        'transaction_fees': transaction_fees,
                        'vwap_influenced': vwap_influenced,  # 新增字段
                        'stop_level': current_stop,
                        'lower_bound': lower_bound,
                        'vwap_value': vwap if use_vwap else np.nan,
                        'trailing_tp_activated': trailing_tp_activated,  # 🎯 动态止盈是否激活
                        'max_profit_price': max_profit_price if not np.isnan(max_profit_price) else np.nan,  # 🎯 最低价
                        'dynamic_tp_level': dynamic_take_profit_level if not np.isnan(dynamic_take_profit_level) else np.nan  # 🎯 动态止盈线
                    })
                    
                    # 🛡️ 检查日内止损
                    check_intraday_stop_loss(pnl, current_time)
                    
                    position = 0
                    trailing_stop = np.nan
                    # 🎯 重置动态追踪止盈变量
                    max_profit_price = np.nan
                    trailing_tp_activated = False
                    dynamic_take_profit_level = np.nan
    
    # 获取交易结束时间字符串，格式为HH:MM
    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    
    # 寻找结束时间的数据点（如果存在）
    close_time_rows = day_df[day_df['Time'] == end_time_str]
    
    # 如果有结束时间的数据点且仍有未平仓位，则平仓
    if not close_time_rows.empty and position != 0:
        close_row = close_time_rows.iloc[0]
        exit_time = close_row['DateTime']
        close_price = close_row['Close']
        
        if position == 1:  # 多头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {end_time_str}] - 多头收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {close_price:.2f}, 股数: {position_size}")
            
            # 计算交易费用（开仓和平仓）
            if enable_transaction_fees:
                transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
            else:
                transaction_fees = 0  # 关闭手续费
            pnl = position_size * (close_price - entry_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Long',
                'entry_price': entry_price,
                'exit_price': close_price,
                'pnl': pnl,
                'exit_reason': 'Intraday Close',
                'position_size': position_size,
                'transaction_fees': transaction_fees,
                'vwap_influenced': False,  # 收盘平仓不受VWAP影响
                'stop_level': np.nan,
                'upper_bound': np.nan,
                'vwap_value': np.nan
            })
            
            # 🛡️ 检查日内止损
            check_intraday_stop_loss(pnl, end_time_str)
            
            position = 0
            trailing_stop = np.nan
            # 🎯 重置动态追踪止盈变量
            max_profit_price = np.nan
            trailing_tp_activated = False
            dynamic_take_profit_level = np.nan
                
        else:  # 空头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {end_time_str}] - 空头收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {close_price:.2f}, 股数: {position_size}")
            
            # 计算交易费用（开仓和平仓）
            if enable_transaction_fees:
                transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
            else:
                transaction_fees = 0  # 关闭手续费
            pnl = position_size * (entry_price - close_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Short',
                'entry_price': entry_price,
                'exit_price': close_price,
                'pnl': pnl,
                'exit_reason': 'Intraday Close',
                'position_size': position_size,
                'transaction_fees': transaction_fees,
                'vwap_influenced': False,  # 收盘平仓不受VWAP影响
                'stop_level': np.nan,
                'lower_bound': np.nan,
                'vwap_value': np.nan
            })
            
            # 🛡️ 检查日内止损
            check_intraday_stop_loss(pnl, end_time_str)
            
            position = 0
            trailing_stop = np.nan
            # 🎯 重置动态追踪止盈变量
            max_profit_price = np.nan
            trailing_tp_activated = False
            dynamic_take_profit_level = np.nan
    
    # 如果仍有未平仓位且没有结束时间数据点，则在一天结束时平仓
    elif position != 0:
        exit_time = day_df.iloc[-1]['DateTime']
        last_price = day_df.iloc[-1]['Close']
        last_time = day_df.iloc[-1]['Time']
        
        if position == 1:  # 多头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {last_time}] - 多头市场收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {last_price:.2f}, 股数: {position_size}")
            
            # 应用滑点
            exit_price = apply_slippage(last_price, is_buy=False, is_entry=False)  # 多头平仓是卖出
            # 计算交易费用（开仓和平仓）
            if enable_transaction_fees:
                transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
            else:
                transaction_fees = 0  # 关闭手续费
            pnl = position_size * (exit_price - entry_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Long',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_reason': 'Market Close',
                'position_size': position_size,
                'transaction_fees': transaction_fees,
                'vwap_influenced': False,  # 市场收盘平仓不受VWAP影响
                'stop_level': np.nan,
                'upper_bound': np.nan,
                'vwap_value': np.nan
            })
            
            # 🛡️ 检查日内止损
            check_intraday_stop_loss(pnl, last_time)
            # 🎯 重置动态追踪止盈变量
            max_profit_price = np.nan
            trailing_tp_activated = False
            dynamic_take_profit_level = np.nan
                
        else:  # 空头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {last_time}] - 空头市场收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {last_price:.2f}, 股数: {position_size}")
            
            # 应用滑点
            exit_price = apply_slippage(last_price, is_buy=True, is_entry=False)  # 空头平仓是买入
            # 计算交易费用（开仓和平仓）
            if enable_transaction_fees:
                transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
            else:
                transaction_fees = 0  # 关闭手续费
            pnl = position_size * (entry_price - exit_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_reason': 'Market Close',
                'position_size': position_size,
                'transaction_fees': transaction_fees,
                'vwap_influenced': False,  # 市场收盘平仓不受VWAP影响
                'stop_level': np.nan,
                'lower_bound': np.nan,
                'vwap_value': np.nan
            })
            
            # 🛡️ 检查日内止损
            check_intraday_stop_loss(pnl, last_time)
            # 🎯 重置动态追踪止盈变量
            max_profit_price = np.nan
            trailing_tp_activated = False
            dynamic_take_profit_level = np.nan
    
    # 计算日内最大回撤百分比（基于当日起始资金）
    intraday_max_drawdown_pct = intraday_max_drawdown / day_start_capital if day_start_capital > 0 else 0
    
    # 返回交易、日内回撤百分比、日内最低资金、日内最高资金
    return trades, intraday_max_drawdown_pct, intraday_capital_low, intraday_capital_high

def run_backtest(config):
    """
    运行回测 - 噪声空间策略 + VWAP
    
    参数:
        config: 配置字典，包含所有回测参数
        
    返回:
        日度结果DataFrame
        月度结果DataFrame
        交易记录DataFrame
        性能指标字典
    """
    # 从配置中提取参数
    data_path = config.get('data_path')
    ticker = config.get('ticker')
    initial_capital = config.get('initial_capital', 100000)
    lookback_days = config.get('lookback_days', 90)
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    plot_days = config.get('plot_days')
    random_plots = config.get('random_plots', 0)
    plots_dir = config.get('plots_dir', 'trading_plots')
    check_interval_minutes = config.get('check_interval_minutes', 30)
    transaction_fee_per_share = config.get('transaction_fee_per_share', 0.01)
    enable_transaction_fees = config.get('enable_transaction_fees', True)  # 新增手续费开关
    trading_start_time = config.get('trading_start_time', (10, 00))
    trading_end_time = config.get('trading_end_time', (15, 40))
    max_positions_per_day = config.get('max_positions_per_day', float('inf'))
    print_daily_trades = config.get('print_daily_trades', True)
    print_trade_details = config.get('print_trade_details', False)
    debug_time = config.get('debug_time')
    leverage = config.get('leverage', 1)  # 资金杠杆倍数，默认为1
    
    # 如果未提供ticker，从文件名中提取
    if ticker is None:
        # 从文件名中提取ticker
        file_name = os.path.basename(data_path)
        # 移除_market_hours.csv（如果存在）
        ticker = file_name.replace('_market_hours.csv', '')
    
    # 加载和处理数据
    print(f"加载{ticker}数据，从{data_path}...")
    price_df = pd.read_csv(data_path, parse_dates=['DateTime'])
    price_df.sort_values('DateTime', inplace=True)
    
    # 提取日期和时间组件
    price_df['Date'] = price_df['DateTime'].dt.date
    price_df['Time'] = price_df['DateTime'].dt.strftime('%H:%M')
    
    # 按日期范围过滤数据（如果指定）
    if start_date is not None:
        price_df = price_df[price_df['Date'] >= start_date]
        print(f"筛选数据，开始日期为{start_date}")
    
    if end_date is not None:
        price_df = price_df[price_df['Date'] <= end_date]
        print(f"筛选数据，结束日期为{end_date}")
    
    # 检查DayOpen和DayClose列是否存在，如果不存在则创建
    if 'DayOpen' not in price_df.columns or 'DayClose' not in price_df.columns:
        # 对于每一天，获取第一行（9:30 AM开盘价）
        opening_prices = price_df.groupby('Date').first().reset_index()
        opening_prices = opening_prices[['Date', 'Open']].rename(columns={'Open': 'DayOpen'})

        # 对于每一天，获取最后一行（4:00 PM收盘价）
        closing_prices = price_df.groupby('Date').last().reset_index()
        closing_prices = closing_prices[['Date', 'Close']].rename(columns={'Close': 'DayClose'})

        # 将开盘价和收盘价合并回主DataFrame
        price_df = pd.merge(price_df, opening_prices, on='Date', how='left')
        price_df = pd.merge(price_df, closing_prices, on='Date', how='left')
    
    # 使用筛选后数据的DayOpen和DayClose
    # 这些代表9:30 AM开盘价和4:00 PM收盘价
    price_df['prev_close'] = price_df.groupby('Date')['DayClose'].transform('first').shift(1)
    
    # 使用9:30 AM价格作为当天的开盘价
    price_df['day_open'] = price_df.groupby('Date')['DayOpen'].transform('first')
    
    # 为每个交易日计算一次参考价格，并将其应用于该日的所有时间点
    # 这确保了整个交易日使用相同的参考价格
    unique_dates = price_df['Date'].unique()
    
    # 创建临时DataFrame来存储每个日期的参考价格
    date_refs = []
    for d in unique_dates:
        day_data = price_df[price_df['Date'] == d].iloc[0]  # 获取该日第一行数据
        day_open = day_data['day_open']
        prev_close = day_data['prev_close']
        
        # 计算该日的参考价格
        if not pd.isna(prev_close):
            upper_ref = max(day_open, prev_close)
            lower_ref = min(day_open, prev_close)
        else:
            upper_ref = day_open
            lower_ref = day_open
            
        date_refs.append({
            'Date': d,
            'upper_ref': upper_ref,
            'lower_ref': lower_ref
        })
    
    # 创建日期参考价格DataFrame
    date_refs_df = pd.DataFrame(date_refs)
    
    # 检查是否有足够的数据进行处理
    if date_refs_df.empty or len(date_refs) == 0:
        raise ValueError(f"在指定日期范围内没有找到有效的交易数据。请检查日期范围设置。")
    
    # 将参考价格合并回主DataFrame
    price_df = price_df.drop(columns=['upper_ref', 'lower_ref'], errors='ignore')
    
    # 检查Date列是否存在
    if 'Date' not in price_df.columns:
        raise ValueError("price_df中缺少Date列，无法进行合并操作")
    if 'Date' not in date_refs_df.columns:
        raise ValueError("date_refs_df中缺少Date列，无法进行合并操作")
    
    price_df = pd.merge(price_df, date_refs_df, on='Date', how='left')
    
    # 计算每分钟相对开盘的回报（使用day_open保持一致性）
    price_df['ret'] = price_df['Close'] / price_df['day_open'] - 1 

    # 计算噪声区域边界
    print(f"计算噪声区域边界...")
    # 将时间点转为列
    pivot = price_df.pivot(index='Date', columns='Time', values='ret').abs()
    
    # 重要修复：确保rolling基于实际交易日而不是日历日
    # 对于周一或节假日后的第一个交易日，应该使用前一个交易日的数据
    # 这里使用实际存在的交易日进行rolling计算
    sigma = pivot.rolling(window=lookback_days, min_periods=lookback_days).mean().shift(1)
    
    # 对于lookback_days=1的情况，特殊处理：直接使用前一个交易日的数据
    if lookback_days == 1:
        # shift(1)已经确保使用前一个交易日的数据
        # 因为pivot的索引只包含实际的交易日，所以shift(1)会自动跳过周末
        pass
    
    # 转回长格式
    sigma = sigma.stack().reset_index(name='sigma')
    
    
    # 保存一个原始数据的副本，用于计算买入持有策略
    price_df_original = price_df.copy()
    
    # 将sigma合并回主DataFrame
    price_df = pd.merge(price_df, sigma, on=['Date', 'Time'], how='left')
    
    # 检查每个交易日是否有足够的sigma数据
    # 创建一个标记，记录哪些日期的sigma数据严重不完整（缺失超过10%）
    incomplete_sigma_dates = set()
    for date in price_df['Date'].unique():
        day_data = price_df[price_df['Date'] == date]
        na_count = day_data['sigma'].isna().sum()
        total_count = len(day_data)
        missing_ratio = na_count / total_count if total_count > 0 else 1.0
        
        # 只有当缺失率超过10%时才过滤掉这一天
        if missing_ratio > 0.1:
            incomplete_sigma_dates.add(date)
            incomplete_sigma_dates.add(date)
            # print(f"{date} sigma缺失率过高: {na_count}/{total_count} ({missing_ratio*100:.1f}%) - 将被过滤")
        # elif na_count > 0:
            # 少量缺失值，填充为前值
            # print(f"{date} 有少量sigma缺失: {na_count}/{total_count} ({missing_ratio*100:.1f}%) - 将被保留")
    
    # 移除sigma数据严重不完整的日期
    # if incomplete_sigma_dates:
    #     print(f"sigma严重不完整的日期（将被过滤）: {sorted(incomplete_sigma_dates)}")
    price_df = price_df[~price_df['Date'].isin(incomplete_sigma_dates)]
    
    # 对于剩余的少量缺失值，使用前值填充（forward fill）
    price_df['sigma'] = price_df.groupby('Date')['sigma'].ffill()
    # 如果还有缺失（比如第一个值），使用后值填充
    price_df['sigma'] = price_df.groupby('Date')['sigma'].bfill()
    # 如果整个时间点都缺失，使用0填充（保守策略）
    price_df['sigma'] = price_df['sigma'].fillna(0)
    
    # 确保所有剩余的sigma值都有有效数据
    if price_df['sigma'].isna().any():
        print(f"警告: 仍有{price_df['sigma'].isna().sum()}个缺失的sigma值")
    
    
    # 使用正确的参考价格计算噪声区域的上下边界
    # 从配置中获取K1和K2参数
    K1 = config.get('K1', 1)  # 如果未设置，默认为1
    K2 = config.get('K2', 1)  # 如果未设置，默认为1
    
    print(f"使用上边界乘数K1={K1}，下边界乘数K2={K2}")
    
    # 将K1和K2应用于sigma进行边界计算
    price_df['UpperBound'] = price_df['upper_ref'] * (1 + K1 * price_df['sigma'])
    price_df['LowerBound'] = price_df['lower_ref'] * (1 - K2 * price_df['sigma'])
    
    # 根据检查间隔生成允许的交易时间
    allowed_times = []
    start_hour, start_minute = trading_start_time  # 使用可配置的开始时间
    end_hour, end_minute = trading_end_time        # 使用可配置的结束时间
    
    current_hour, current_minute = start_hour, start_minute
    while current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute):
        # 将当前时间添加到allowed_times
        allowed_times.append(f"{current_hour:02d}:{current_minute:02d}")
        
        # 增加check_interval_minutes
        current_minute += check_interval_minutes
        if current_minute >= 60:
            current_hour += current_minute // 60
            current_minute = current_minute % 60
    
    # 始终确保trading_end_time包含在内，用于平仓
    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    if end_time_str not in allowed_times:
        allowed_times.append(end_time_str)
        allowed_times.sort()
    
    print(f"使用{check_interval_minutes}分钟的检查间隔")
    
    # 初始化回测变量
    capital = initial_capital
    daily_results = []
    all_trades = []
    total_transaction_fees = 0  # 跟踪总交易费用
    
    # 添加交易日期统计变量
    trading_days = set()       # 有交易的日期集合
    non_trading_days = set()   # 无交易的日期集合
    
    # 追踪最大日内回撤
    max_intraday_mdd_pct = 0   # 最大日内波动百分比
    max_intraday_mdd_date = None  # 最大日内波动发生的日期
    
    # 📊 精确最大回撤追踪（考虑日内波动）
    capital_peak = initial_capital  # 资金峰值（包含日内高点）
    precise_max_drawdown = 0  # 精确最大回撤金额
    precise_max_drawdown_pct = 0  # 精确最大回撤百分比
    precise_mdd_date = None  # 精确最大回撤发生日期
    precise_mdd_peak_date = None  # 峰值日期
    
    # 如果指定了随机生成图表的数量，随机选择交易日
    days_with_trades = []
    if random_plots > 0:
        # 先运行回测，记录有交易的日期
        for trade_date in unique_dates:
            day_data = price_df[price_df['Date'] == trade_date].copy()
            # 设置数据点阈值：对于今天允许更少的数据点
            is_today = (day_data['Date'].iloc[0] == datetime.now().date()) if len(day_data) > 0 else False
            min_data_points = 1 if is_today else 10
            if len(day_data) < min_data_points:  # 跳过数据不足的日期
                continue
                
            prev_close = day_data['prev_close'].iloc[0] if not pd.isna(day_data['prev_close'].iloc[0]) else None
            if prev_close is None:
                continue
                
            # 模拟当天交易
            simulation_result = simulate_day(day_data, prev_close, allowed_times, 100, config, config.get('initial_capital', 100000))
            
            # 从结果中提取交易（忽略其他返回值）
            trades = simulation_result[0]
                
            if trades:  # 如果有交易
                days_with_trades.append(trade_date)
        
        # 如果有交易的日期少于请求的随机图表数量，调整随机图表数量
        random_plots = min(random_plots, len(days_with_trades))
        # 随机选择日期
        if random_plots > 0:
            random_plot_days = random.sample(days_with_trades, random_plots)
        else:
            random_plot_days = []
    else:
        random_plot_days = []
    
    # 合并指定的绘图日期和随机选择的日期
    if plot_days is None:
        plot_days = []
    all_plot_days = list(set(plot_days + random_plot_days))
    
    # 确保绘图目录存在
    if all_plot_days and plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    
    # 创建买入持有回测数据（使用原始数据，不受sigma筛选影响）
    buy_hold_data = []
    filtered_dates = price_df['Date'].unique()  # 策略交易使用的日期（经过sigma筛选）
    
    # 创建独立的买入持有数据，使用原始数据（未经过sigma筛选）
    for trade_date in unique_dates:
        # 获取当天的数据（从原始数据中）
        day_data = price_df_original[price_df_original['Date'] == trade_date].copy()

        # 跳过数据不足的日期
        is_today = (day_data['Date'].iloc[0] == datetime.now().date()) if len(day_data) > 0 else False
        min_data_points = 1 if is_today else 10
        if len(day_data) < min_data_points:  # 任意阈值
            continue
        
        # 获取当天的开盘价和收盘价（用于计算买入持有）
        open_price = day_data['day_open'].iloc[0]
        close_price = day_data['DayClose'].iloc[0]
        
        # 存储买入持有数据
        buy_hold_data.append({
            'Date': trade_date,
            'Open': open_price,
            'Close': close_price
        })
    
    # 处理策略交易部分
    
    for i, trade_date in enumerate(filtered_dates):
        # 获取当天的数据
        day_data = price_df[price_df['Date'] == trade_date].copy()
        day_data = day_data.sort_values('DateTime').reset_index(drop=True)
        
        
        # 跳过数据不足的日期
        is_today = (day_data['Date'].iloc[0] == datetime.now().date()) if len(day_data) > 0 else False
        min_data_points = 1 if is_today else 10
        if len(day_data) < min_data_points:  # 任意阈值
            if not is_today:
                daily_results.append({
                    'Date': trade_date,
                    'capital': capital,
                    'daily_return': 0
                })
                continue
        
        # 获取前一天的收盘价
        prev_close = day_data['prev_close'].iloc[0] if not pd.isna(day_data['prev_close'].iloc[0]) else None
        
        # 将trade_date转换为字符串格式以便统一显示
        date_str = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        
        # 获取当天的开盘价
        day_open_price = day_data['day_open'].iloc[0]
        
        # 计算仓位大小（应用杠杆）
        leveraged_capital = capital * leverage  # 应用杠杆倍数
        position_size = floor(leveraged_capital / day_open_price)
        
        # 如果资金不足，跳过当天
        if position_size <= 0:
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
                
        # 模拟当天的交易
        simulation_result = simulate_day(day_data, prev_close, allowed_times, position_size, config, capital)
        
        # 从结果中提取交易、日内回撤、日内最低/最高资金
        trades, intraday_mdd_pct, intraday_low, intraday_high = simulation_result
        
        # 更新交易日期统计
        if trades:  # 有交易的日期
            trading_days.add(trade_date)
        else:  # 无交易的日期
            non_trading_days.add(trade_date)
        
        # 追踪最大日内资金回撤
        if intraday_mdd_pct > max_intraday_mdd_pct:
            max_intraday_mdd_pct = intraday_mdd_pct
            max_intraday_mdd_date = trade_date
        
        # 📊 精确最大回撤计算（考虑日内波动）
        # 更新资金峰值（使用日内最高点）
        if intraday_high > capital_peak:
            capital_peak = intraday_high
            precise_mdd_peak_date = trade_date
        
        # 计算当前回撤（使用日内最低点与历史峰值的差距）
        current_precise_drawdown = capital_peak - intraday_low
        current_precise_drawdown_pct = current_precise_drawdown / capital_peak if capital_peak > 0 else 0
        
        if current_precise_drawdown_pct > precise_max_drawdown_pct:
            precise_max_drawdown = current_precise_drawdown
            precise_max_drawdown_pct = current_precise_drawdown_pct
            precise_mdd_date = trade_date
        
        # 打印每天的交易信息
        if trades and print_daily_trades:
            # 计算当天总盈亏
            day_total_pnl = sum(trade['pnl'] for trade in trades)
            
            # 创建交易方向与时间的简要信息
            trade_summary = []
            for trade in trades:
                direction = "多" if trade['side'] == 'Long' else "空"
                entry_time = trade['entry_time'].strftime('%H:%M')
                exit_time = trade['exit_time'].strftime('%H:%M')
                pnl = trade['pnl']
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                size = trade.get('position_size', position_size)
                trade_summary.append(f"{direction}({entry_time}->{exit_time}) 买:{entry_price:.2f} 卖:{exit_price:.2f} 股数:{size} 盈亏:${pnl:.2f}")
            
            # 打印单行交易日志
            trade_info = ", ".join(trade_summary)
            leverage_info = f" [杠杆{leverage}x]" if leverage != 1 else ""
            print(f"{date_str} | 交易数: {len(trades)} | 总盈亏: ${day_total_pnl:.2f} | 日内回撤: {intraday_mdd_pct*100:.2f}%{leverage_info} | {trade_info}")
        
        # 检查是否需要为这一天生成图表
        if trade_date in all_plot_days:
            # 为当天的交易生成图表
            plot_path = os.path.join(plots_dir, f"{ticker}_trade_visualization_{trade_date}")
            
            # 添加交易类型到文件名
            sides = [trade['side'] for trade in trades]
            if 'Long' in sides and 'Short' not in sides:
                plot_path += "_Long.png"
            elif 'Short' in sides and 'Long' not in sides:
                plot_path += "_Short.png"
            elif 'Long' in sides and 'Short' in sides:
                plot_path += "_Mixed.png"
            else:
                plot_path += ".png"  # 没有交易
                
            # 生成并保存图表
            plot_trading_day(day_data, trades, save_path=plot_path)
        
        # 计算每日盈亏和交易费用
        day_pnl = 0
        day_transaction_fees = 0
        for trade in trades:
            day_pnl += trade['pnl']
            # 从每笔交易中提取交易费用
            if 'transaction_fees' not in trade:
                # 如果交易数据中没有交易费用，则计算
                if enable_transaction_fees:
                    trade['transaction_fees'] = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
                else:
                    trade['transaction_fees'] = 0  # 关闭手续费
            day_transaction_fees += trade['transaction_fees']
        
        # 添加到总交易费用
        total_transaction_fees += day_transaction_fees
        
        # 更新资金并计算每日回报
        capital_start = capital
        capital += day_pnl
        daily_return = day_pnl / capital_start
        
        # 存储每日结果
        daily_results.append({
            'Date': trade_date,
            'capital': capital,
            'daily_return': daily_return
        })
        
        # 存储交易
        for trade in trades:
            trade['Date'] = trade_date
            all_trades.append(trade)
    
    # 创建每日结果DataFrame
    daily_df = pd.DataFrame(daily_results)
    
    # 检查daily_results是否为空或缺少Date列
    if daily_df.empty:
        raise ValueError("daily_results为空，无法创建日度结果DataFrame")
    if 'Date' not in daily_df.columns:
        raise ValueError("daily_results中缺少Date列，无法进行日期转换")
    
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df.set_index('Date', inplace=True)
    
    # 创建买入持有DataFrame
    buy_hold_df = pd.DataFrame(buy_hold_data)
    
    # 检查buy_hold_data是否为空
    if not buy_hold_data:
        print("警告: 没有足够的数据来计算买入持有策略的表现")
        buy_hold_df = pd.DataFrame()  # 创建一个空的DataFrame
    else:
        buy_hold_df['Date'] = pd.to_datetime(buy_hold_df['Date'])
        buy_hold_df.set_index('Date', inplace=True)
    
    # 计算买入持有策略的表现
    if not buy_hold_df.empty:
        # 计算每日收益率
        buy_hold_df['daily_return'] = buy_hold_df['Close'] / buy_hold_df['Close'].shift(1) - 1
        
        # 计算累积资本
        buy_hold_df['capital'] = initial_capital * (1 + buy_hold_df['daily_return']).cumprod().fillna(1)
    
    # 计算月度回报
    monthly = daily_df.resample('ME').first()[['capital']].rename(columns={'capital': 'month_start'})
    monthly['month_end'] = daily_df.resample('ME').last()['capital']
    monthly['monthly_return'] = monthly['month_end'] / monthly['month_start'] - 1
    
    # 打印月度回报
    print("\n月度回报:")
    # 设置pandas显示选项以显示所有行
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # 创建格式化的月度回报显示
    monthly_display = monthly[['month_start', 'month_end', 'monthly_return']].copy()
    monthly_display['monthly_return_pct'] = monthly_display['monthly_return'] * 100
    monthly_display = monthly_display.round({'month_start': 2, 'month_end': 2, 'monthly_return_pct': 2})
    
    print(monthly_display[['month_start', 'month_end', 'monthly_return_pct']].rename(columns={
        'month_start': '月初资金',
        'month_end': '月末资金', 
        'monthly_return_pct': '月度收益率(%)'
    }))
    
    # 打印月度回报统计信息
    monthly_returns = monthly['monthly_return'].dropna()
    if len(monthly_returns) > 0:
        print(f"\n月度回报统计:")
        print(f"  平均月度收益率: {monthly_returns.mean()*100:.2f}%")
        print(f"  月度收益率标准差: {monthly_returns.std()*100:.2f}%")
        print(f"  最佳月度收益率: {monthly_returns.max()*100:.2f}%")
        print(f"  最差月度收益率: {monthly_returns.min()*100:.2f}%")
        print(f"  正收益月份: {(monthly_returns > 0).sum()}个")
        print(f"  负收益月份: {(monthly_returns < 0).sum()}个")
        print(f"  胜率: {(monthly_returns > 0).mean()*100:.1f}%")
    
    # 恢复默认显示设置
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    # 计算总体表现
    total_return = capital / initial_capital - 1
    print(f"\n总回报: {total_return*100:.2f}%")
    
    # 创建交易DataFrame
    trades_df = pd.DataFrame(all_trades)
    
    # 计算策略性能指标
    print(f"\n计算策略性能指标...")
    metrics = calculate_performance_metrics(daily_df, trades_df, initial_capital, buy_hold_df=buy_hold_df)
    
    # 计算总滑点损耗
    slippage_per_share = config.get('slippage_per_share', 0.01)
    if len(trades_df) > 0:
        total_slippage_cost = (trades_df['position_size'] * slippage_per_share * 2).sum()
    else:
        total_slippage_cost = 0
    
    # 打印交易费用和滑点统计
    print(f"\n交易成本统计:")
    print(f"总手续费: ${total_transaction_fees:.2f}")
    print(f"总滑点损耗: ${total_slippage_cost:.2f}")
    total_trading_cost = total_transaction_fees + total_slippage_cost
    print(f"总交易成本: ${total_trading_cost:.2f}")
    if len(trades_df) > 0:
        print(f"平均每笔手续费: ${total_transaction_fees / len(trades_df):.2f}")
        print(f"平均每笔滑点: ${total_slippage_cost / len(trades_df):.2f}")
        print(f"平均每笔总成本: ${total_trading_cost / len(trades_df):.2f}")
    if len(daily_df) > 0:
        print(f"平均每日交易成本: ${total_trading_cost / len(daily_df):.2f}")
    print(f"交易成本占初始资金比例: {total_trading_cost / initial_capital * 100:.2f}%")
    if capital > initial_capital:
        print(f"交易成本占总收益比例: {total_trading_cost / (capital - initial_capital) * 100:.2f}%")
    else:
        print(f"交易成本占总收益比例: N/A (无盈利)")
    
    # 打印交易日期统计
    print(f"\n交易日期统计:")
    print(f"总交易日数: {len(trading_days) + len(non_trading_days)}")
    print(f"有交易的天数: {len(trading_days)} ({len(trading_days)/(len(trading_days) + len(non_trading_days))*100:.1f}%)")
    print(f"无交易的天数: {len(non_trading_days)} ({len(non_trading_days)/(len(trading_days) + len(non_trading_days))*100:.1f}%)")
    
    # 打印简化的性能指标
    print(f"\n策略性能指标:")
    leverage_text = f" (杠杆{leverage}x)" if leverage != 1 else ""
    strategy_name = f"{ticker} Curr.Band + VWAP{leverage_text}"
    print(f"策略: {strategy_name}")
    
    # 创建表格格式对比策略与买入持有的指标
    print("\n性能指标对比:")
    print(f"{'指标':<20} | {'策略':<15} | {f'{ticker} Buy & Hold':<15}")
    print("-" * 55)
    
    # 总回报率
    print(f"{'总回报率':<20} | {metrics['total_return']*100:>14.1f}% | {metrics['buy_hold_return']*100:>14.1f}%")
    
    # 年化收益率
    print(f"{'年化收益率':<20} | {metrics['irr']*100:>14.1f}% | {metrics['buy_hold_irr']*100:>14.1f}%")
    
    # 波动率
    print(f"{'波动率':<20} | {metrics['volatility']*100:>14.1f}% | {metrics['buy_hold_volatility']*100:>14.1f}%")
    
    # 夏普比率
    print(f"{'夏普比率':<20} | {metrics['sharpe_ratio']:>14.2f} | {metrics['buy_hold_sharpe']:>14.2f}")
    
    # 最大回撤
    print(f"{'最大回撤':<20} | {metrics['mdd']*100:>14.1f}% | {metrics['buy_hold_mdd']*100:>14.1f}%")
    
    # 打印最大回撤的详细信息
    if 'max_drawdown_start_date' in metrics and 'max_drawdown_date' in metrics:
        start_date = metrics['max_drawdown_start_date'].strftime('%Y-%m-%d')
        bottom_date = metrics['max_drawdown_date'].strftime('%Y-%m-%d')
        
        print(f"\n最大回撤详细信息:")
        print(f"  峰值日期: {start_date}")
        print(f"  最低点日期: {bottom_date}")
        
        if metrics['max_drawdown_end_date'] is not None:
            end_date = metrics['max_drawdown_end_date'].strftime('%Y-%m-%d')
            print(f"  恢复日期: {end_date}")
            
            # 计算回撤持续时间
            duration = (metrics['max_drawdown_end_date'] - metrics['max_drawdown_start_date']).days
            print(f"  回撤持续时间: {duration}天")
        else:
            print(f"  恢复日期: 尚未恢复")
            
            # 计算到目前为止的回撤持续时间
            duration = (metrics['max_drawdown_date'] - metrics['max_drawdown_start_date']).days
            print(f"  回撤持续时间: {duration}天 (仍在回撤中)")
    
    # 策略特有指标
    print(f"\n策略特有指标:")
    print(f"胜率: {metrics['hit_ratio']*100:.1f}%")
    print(f"总交易次数: {metrics['total_trades']}")
    
    # 计算做多和做空的笔数
    if len(trades_df) > 0:
        long_trades = len(trades_df[trades_df['side'] == 'Long'])
        short_trades = len(trades_df[trades_df['side'] == 'Short'])
        print(f"做多交易笔数: {long_trades}")
        print(f"做空交易笔数: {short_trades}")
    else:
        print(f"做多交易笔数: 0")
        print(f"做空交易笔数: 0")
    
    print(f"平均每日交易次数: {metrics['avg_daily_trades']:.2f}")
    
    # 打印最大单笔收益和亏损统计
    print(f"\n单笔交易统计:")
    print(f"最大单笔收益: ${metrics.get('max_single_gain', 0):.2f}")
    print(f"最大单笔亏损: ${metrics.get('max_single_loss', 0):.2f}")
    
    # 打印前10笔最大收益
    if metrics.get('top_10_gains'):
        print(f"\n前10笔最大收益:")
        print(f"{'排名':<4} | {'日期':<12} | {'方向':<6} | {'买入价':<8} | {'卖出价':<8} | {'盈亏':<10} | {'退出原因':<15}")
        print("-" * 85)
        for i, trade in enumerate(metrics['top_10_gains'], 1):
            date_str = pd.to_datetime(trade['Date']).strftime('%Y-%m-%d')
            side = '多' if trade['side'] == 'Long' else '空'
            print(f"{i:<4} | {date_str:<12} | {side:<6} | ${trade['entry_price']:<7.2f} | ${trade['exit_price']:<7.2f} | ${trade['pnl']:<9.2f} | {trade['exit_reason']:<15}")
    
    # 打印前10笔最大亏损
    if metrics.get('top_10_losses'):
        print(f"\n前10笔最大亏损:")
        print(f"{'排名':<4} | {'日期':<12} | {'方向':<6} | {'买入价':<8} | {'卖出价':<8} | {'盈亏':<10} | {'退出原因':<15}")
        print("-" * 85)
        for i, trade in enumerate(metrics['top_10_losses'], 1):
            date_str = pd.to_datetime(trade['Date']).strftime('%Y-%m-%d')
            side = '多' if trade['side'] == 'Long' else '空'
            print(f"{i:<4} | {date_str:<12} | {side:<6} | ${trade['entry_price']:<7.2f} | ${trade['exit_price']:<7.2f} | ${trade['pnl']:<9.2f} | {trade['exit_reason']:<15}")
    
    # 打印策略总结
    print(f"\n" + "="*50)
    print(f"策略回测总结 - {strategy_name}")
    print(f"="*50)
    
    # 打印杠杆信息
    if leverage != 1:
        final_capital = daily_df['capital'].iloc[-1]
        print(f"💰 资金杠杆倍数: {leverage}x")
        print(f"💵 初始资金: ${initial_capital:,.0f}")
        print(f"💸 杠杆后可用资金: ${initial_capital * leverage:,.0f}")
        print(f"💰 最终资金: ${final_capital:,.2f}")
        print(f"-"*50)
    
    # 核心表现指标
    print(f"📈 总回报率: {metrics['total_return']*100:.1f}%")
    print(f"📊 年化收益率: {metrics['irr']*100:.1f}%")
    print(f"⚡ 夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"📉 最大回撤: {metrics['mdd']*100:.1f}%")
    if 'max_drawdown_start_date' in metrics and 'max_drawdown_date' in metrics:
        start_date = metrics['max_drawdown_start_date'].strftime('%Y-%m-%d')
        bottom_date = metrics['max_drawdown_date'].strftime('%Y-%m-%d')
        print(f"   └─ 峰值: {start_date} → 最低点: {bottom_date}")
        
        if metrics['max_drawdown_end_date'] is not None:
            end_date = metrics['max_drawdown_end_date'].strftime('%Y-%m-%d')
            duration = (metrics['max_drawdown_end_date'] - metrics['max_drawdown_start_date']).days
            print(f"   └─ 恢复: {end_date} (持续{duration}天)")
        else:
            duration = (metrics['max_drawdown_date'] - metrics['max_drawdown_start_date']).days
            print(f"   └─ 尚未恢复 (已持续{duration}天)")
    print(f"🎯 胜率: {metrics['hit_ratio']*100:.1f}% | 总交易: {metrics['total_trades']}次")
    if max_intraday_mdd_date is not None:
        max_mdd_date_str = pd.to_datetime(max_intraday_mdd_date).strftime('%Y-%m-%d')
        print(f"📊 最大单日回撤: {max_intraday_mdd_pct*100:.2f}% ({max_mdd_date_str})")
    
    # 📊 精确最大回撤报告（考虑日内波动）
    if precise_mdd_date is not None:
        precise_mdd_date_str = pd.to_datetime(precise_mdd_date).strftime('%Y-%m-%d')
        precise_peak_date_str = pd.to_datetime(precise_mdd_peak_date).strftime('%Y-%m-%d') if precise_mdd_peak_date else "N/A"
        print(f"📉 精确最大回撤(含日内): {precise_max_drawdown_pct*100:.2f}%")
        print(f"   └─ 峰值: {precise_peak_date_str} (${capital_peak:,.2f}) → 最低点: {precise_mdd_date_str}")
        
        # 爆仓风险评估
        if precise_max_drawdown_pct >= 1.0:
            print(f"💥 爆仓状态: 已爆仓！回撤超过100%")
        elif precise_max_drawdown_pct >= 0.8:
            print(f"⚠️ 爆仓风险: 极高！回撤已达{precise_max_drawdown_pct*100:.1f}%")
        elif precise_max_drawdown_pct >= 0.5:
            print(f"⚠️ 爆仓风险: 较高，回撤已达{precise_max_drawdown_pct*100:.1f}%")
        else:
            safe_margin = (1 - precise_max_drawdown_pct) * 100
            print(f"✅ 爆仓风险: 安全，距爆仓还有{safe_margin:.1f}%空间")
    
    # 交易成本统计
    print(f"-"*50)
    print(f"💸 交易成本统计:")
    print(f"   ├─ 总手续费: ${total_transaction_fees:,.2f}")
    print(f"   ├─ 总滑点损耗: ${total_slippage_cost:,.2f}")
    print(f"   ├─ 总交易成本: ${total_trading_cost:,.2f}")
    print(f"   ├─ 占初始资金比例: {total_trading_cost / initial_capital * 100:.2f}%")
    if capital > initial_capital:
        print(f"   └─ 占总收益比例: {total_trading_cost / (capital - initial_capital) * 100:.2f}%")
    else:
        print(f"   └─ 占总收益比例: N/A (无盈利)")
    
    print(f"="*50)
    
    # 分析VWAP影响
    vwap_stats = analyze_vwap_impact(trades_df)
    
    # 🎯 分析动态追踪止盈影响
    trailing_tp_stats = analyze_trailing_take_profit_impact(trades_df, config)
    
    return daily_df, monthly, trades_df, metrics 

def calculate_performance_metrics(daily_df, trades_df, initial_capital, risk_free_rate=0.02, trading_days_per_year=252, buy_hold_df=None):
    """
    计算策略的性能指标
    
    参数:
        daily_df: 包含每日回测结果的DataFrame
        trades_df: 包含所有交易的DataFrame
        initial_capital: 初始资金
        risk_free_rate: 无风险利率，默认为2%
        trading_days_per_year: 一年的交易日数量，默认为252
        buy_hold_df: 买入持有策略的DataFrame
        
    返回:
        包含各种性能指标的字典
    """
    metrics = {}
    
    # 确保daily_df有数据
    if len(daily_df) == 0:
        print("警告: 没有足够的数据来计算性能指标")
        # 返回默认值
        return {
            'total_return': 0, 'irr': 0, 'volatility': 0, 'sharpe_ratio': 0,
            'hit_ratio': 0, 'mdd': 0, 'buy_hold_return': 0, 'buy_hold_irr': 0,
            'buy_hold_volatility': 0, 'buy_hold_sharpe': 0, 'buy_hold_mdd': 0
        }
    
    # 1. 总回报率 (Total Return)
    final_capital = daily_df['capital'].iloc[-1]
    metrics['total_return'] = final_capital / initial_capital - 1
    
    # 2. 年化收益率 (IRR - Internal Rate of Return)
    # 获取回测的开始和结束日期
    start_date = daily_df.index[0]
    end_date = daily_df.index[-1]
    # 计算实际年数（考虑实际日历日而不仅仅是交易日）
    years = (end_date - start_date).days / 365.25
    # 如果时间跨度太短，使用交易日计算
    if years < 0.1:  # 少于约36天
        trading_days = len(daily_df)
        years = trading_days / trading_days_per_year
    
    # 计算年化收益率 (CAGR - Compound Annual Growth Rate)
    if years > 0:
        metrics['irr'] = (1 + metrics['total_return']) ** (1 / years) - 1
    else:
        metrics['irr'] = 0
    
    # 3. 波动率 (Vol - Volatility)
    # 计算日收益率的标准差，然后年化
    daily_returns = daily_df['daily_return']
    # 移除异常值（如果有）
    daily_returns = daily_returns[daily_returns.between(daily_returns.quantile(0.001), 
                                                      daily_returns.quantile(0.999))]
    metrics['volatility'] = daily_returns.std() * np.sqrt(trading_days_per_year)
    
    # 4. 夏普比率 (Sharpe Ratio)
    if metrics['volatility'] > 0:
        metrics['sharpe_ratio'] = (metrics['irr'] - risk_free_rate) / metrics['volatility']
    else:
        metrics['sharpe_ratio'] = 0
    
    # 5. 胜率 (Hit Ratio)和交易统计
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        metrics['hit_ratio'] = len(winning_trades) / len(trades_df)
        
        # 计算平均盈利和平均亏损
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # 计算盈亏比
        metrics['profit_loss_ratio'] = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 总交易次数
        metrics['total_trades'] = len(trades_df)
        
        # 计算每日交易次数
        daily_trade_counts = trades_df.groupby('Date').size()
        metrics['avg_daily_trades'] = daily_trade_counts.mean() if len(daily_trade_counts) > 0 else 0
        metrics['max_daily_trades'] = daily_trade_counts.max() if len(daily_trade_counts) > 0 else 0
        
        # 计算每日盈亏
        daily_pnl = trades_df.groupby('Date')['pnl'].sum()
        metrics['max_daily_loss'] = daily_pnl.min() if len(daily_pnl) > 0 and daily_pnl.min() < 0 else 0
        metrics['max_daily_gain'] = daily_pnl.max() if len(daily_pnl) > 0 else 0
        
        # 计算最大单笔收益和最大单笔亏损
        # 按盈亏排序，获取前10笔最大收益
        top_gains = trades_df.nlargest(10, 'pnl')[['Date', 'side', 'entry_price', 'exit_price', 'pnl', 'exit_reason']]
        metrics['top_10_gains'] = top_gains.to_dict('records')
        
        # 获取前10笔最大亏损
        top_losses = trades_df.nsmallest(10, 'pnl')[['Date', 'side', 'entry_price', 'exit_price', 'pnl', 'exit_reason']]
        metrics['top_10_losses'] = top_losses.to_dict('records')
        
        # 最大单笔收益和亏损
        metrics['max_single_gain'] = trades_df['pnl'].max()
        metrics['max_single_loss'] = trades_df['pnl'].min()
    else:
        metrics['hit_ratio'] = 0
        metrics['profit_loss_ratio'] = 0
        metrics['total_trades'] = 0
        metrics['avg_daily_trades'] = 0
        metrics['max_daily_trades'] = 0
        metrics['max_daily_loss'] = 0
        metrics['max_daily_gain'] = 0
        metrics['top_10_gains'] = []
        metrics['top_10_losses'] = []
        metrics['max_single_gain'] = 0
        metrics['max_single_loss'] = 0
    
    # 6. 最大回撤 (MDD - Maximum Drawdown)
    # 计算每日资金的累计最大值
    daily_df['peak'] = daily_df['capital'].cummax()
    # 计算每日回撤
    daily_df['drawdown'] = (daily_df['capital'] - daily_df['peak']) / daily_df['peak']
    # 最大回撤
    metrics['mdd'] = daily_df['drawdown'].min() * -1
    
    # 找到最大回撤发生的日期
    max_drawdown_date = daily_df['drawdown'].idxmin()
    metrics['max_drawdown_date'] = max_drawdown_date
    
    # 找到最大回撤开始的日期（即达到峰值的日期）
    max_drawdown_peak = daily_df.loc[max_drawdown_date, 'peak']
    # 找到达到这个峰值的最后一个日期
    peak_dates = daily_df[daily_df['capital'] == max_drawdown_peak].index
    max_drawdown_start_date = peak_dates[peak_dates <= max_drawdown_date].max()
    metrics['max_drawdown_start_date'] = max_drawdown_start_date
    
    # 找到最大回撤结束的日期（资金重新达到峰值的日期）
    recovery_dates = daily_df[daily_df['capital'] >= max_drawdown_peak].index
    recovery_dates_after = recovery_dates[recovery_dates > max_drawdown_date]
    if len(recovery_dates_after) > 0:
        max_drawdown_end_date = recovery_dates_after.min()
        metrics['max_drawdown_end_date'] = max_drawdown_end_date
    else:
        metrics['max_drawdown_end_date'] = None  # 尚未恢复
    
    # 计算回撤持续时间
    # 找到每个回撤开始的点
    drawdown_begins = (daily_df['peak'] != daily_df['peak'].shift(1)) & (daily_df['peak'] != daily_df['capital'])
    # 找到每个回撤结束的点（资金达到新高）
    drawdown_ends = daily_df['capital'] == daily_df['peak']
    
    # 计算最长回撤持续时间（交易日）
    if drawdown_begins.any() and drawdown_ends.any():
        begin_dates = daily_df.index[drawdown_begins]
        end_dates = daily_df.index[drawdown_ends]
        
        max_duration = 0
        for begin_date in begin_dates:
            # 找到这个回撤之后的第一个结束点
            end_date = end_dates[end_dates > begin_date]
            if len(end_date) > 0:
                duration = (end_date.min() - begin_date).days
                max_duration = max(max_duration, duration)
        
        metrics['max_drawdown_duration'] = max_duration
    else:
        metrics['max_drawdown_duration'] = 0
    
    # 计算Calmar比率 (年化收益率/最大回撤)
    if metrics['mdd'] > 0:
        metrics['calmar_ratio'] = metrics['irr'] / metrics['mdd']
    else:
        metrics['calmar_ratio'] = float('inf')  # 如果没有回撤，设为无穷大
        
    # 计算曝光时间 (Exposure Time)
    if len(trades_df) > 0:
        # 计算每笔交易的持仓时间（以分钟为单位）
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
        
        # 计算总交易时间（分钟）
        total_trade_minutes = trades_df['duration'].sum()
        
        # 计算回测总时间（分钟）
        # 假设每个交易日有6.5小时（390分钟）
        trading_minutes_per_day = 390
        total_backtest_minutes = len(daily_df) * trading_minutes_per_day
        
        # 计算曝光时间百分比
        metrics['exposure_time'] = total_trade_minutes / total_backtest_minutes
    else:
        metrics['exposure_time'] = 0
    
    # 计算买入持有策略的表现
    if buy_hold_df is not None and not buy_hold_df.empty:
        # 计算买入持有策略的总回报率
        if 'capital' in buy_hold_df.columns:
            final_buy_hold_capital = buy_hold_df['capital'].iloc[-1]
            metrics['buy_hold_return'] = final_buy_hold_capital / initial_capital - 1
            
            # 计算买入持有策略的年化收益率
            if years > 0:
                metrics['buy_hold_irr'] = (1 + metrics['buy_hold_return']) ** (1 / years) - 1
            else:
                metrics['buy_hold_irr'] = 0
            
            # 计算买入持有策略的波动率
            if 'daily_return' in buy_hold_df.columns:
                buy_hold_returns = buy_hold_df['daily_return'].dropna()
                # 移除异常值
                buy_hold_returns = buy_hold_returns[buy_hold_returns.between(
                    buy_hold_returns.quantile(0.001), buy_hold_returns.quantile(0.999))]
                metrics['buy_hold_volatility'] = buy_hold_returns.std() * np.sqrt(trading_days_per_year)
                
                # 计算买入持有策略的夏普比率
                if metrics['buy_hold_volatility'] > 0:
                    metrics['buy_hold_sharpe'] = (metrics['buy_hold_irr'] - risk_free_rate) / metrics['buy_hold_volatility']
                else:
                    metrics['buy_hold_sharpe'] = 0
            else:
                metrics['buy_hold_volatility'] = 0
                metrics['buy_hold_sharpe'] = 0
            
            # 计算买入持有策略的最大回撤
            if 'capital' in buy_hold_df.columns:
                buy_hold_df['peak'] = buy_hold_df['capital'].cummax()
                buy_hold_df['drawdown'] = (buy_hold_df['capital'] - buy_hold_df['peak']) / buy_hold_df['peak']
                metrics['buy_hold_mdd'] = buy_hold_df['drawdown'].min() * -1
            else:
                metrics['buy_hold_mdd'] = 0
        else:
            # 如果buy_hold_df中没有capital列，则计算起始日期和结束日期的价格变化
            if 'Close' in buy_hold_df.columns:
                start_price = buy_hold_df['Close'].iloc[0]
                end_price = buy_hold_df['Close'].iloc[-1]
                metrics['buy_hold_return'] = end_price / start_price - 1
                
                # 计算买入持有策略的年化收益率
                if years > 0:
                    metrics['buy_hold_irr'] = (1 + metrics['buy_hold_return']) ** (1 / years) - 1
                else:
                    metrics['buy_hold_irr'] = 0
                
                # 其他指标设为0
                metrics['buy_hold_volatility'] = 0
                metrics['buy_hold_sharpe'] = 0
                metrics['buy_hold_mdd'] = 0
            else:
                # 没有足够的数据计算买入持有策略的表现
                metrics['buy_hold_return'] = 0
                metrics['buy_hold_irr'] = 0
                metrics['buy_hold_volatility'] = 0
                metrics['buy_hold_sharpe'] = 0
                metrics['buy_hold_mdd'] = 0
    else:
        # 没有买入持有的数据，设置默认值
        metrics['buy_hold_return'] = 0
        metrics['buy_hold_irr'] = 0
        metrics['buy_hold_volatility'] = 0
        metrics['buy_hold_sharpe'] = 0
        metrics['buy_hold_mdd'] = 0
    
    return metrics 

def analyze_vwap_impact(trades_df):
    """
    分析VWAP对交易平仓的影响
    """
    if len(trades_df) == 0:
        print("\n=== VWAP影响分析 ===")
        print("没有交易数据可供分析")
        return
    
    # 只分析止损平仓的交易
    stop_loss_trades = trades_df[trades_df['exit_reason'] == 'Stop Loss']
    
    if len(stop_loss_trades) == 0:
        print("\n=== VWAP影响分析 ===")
        print("没有止损平仓的交易")
        return
    
    # 统计VWAP影响的交易
    vwap_influenced_trades = stop_loss_trades[stop_loss_trades['vwap_influenced'] == True]
    
    total_stop_loss = len(stop_loss_trades)
    vwap_influenced_count = len(vwap_influenced_trades)
    vwap_influence_ratio = vwap_influenced_count / total_stop_loss * 100
    
    print("\n=== VWAP影响分析 ===")
    print(f"总止损平仓交易数: {total_stop_loss}")
    print(f"VWAP影响的平仓数: {vwap_influenced_count}")
    print(f"VWAP生效比例: {vwap_influence_ratio:.1f}%")
    
    # 分多头和空头分析
    long_stop_loss = stop_loss_trades[stop_loss_trades['side'] == 'Long']
    short_stop_loss = stop_loss_trades[stop_loss_trades['side'] == 'Short']
    
    if len(long_stop_loss) > 0:
        long_vwap_influenced = long_stop_loss[long_stop_loss['vwap_influenced'] == True]
        long_ratio = len(long_vwap_influenced) / len(long_stop_loss) * 100
        print(f"\n多头交易:")
        print(f"  止损平仓数: {len(long_stop_loss)}")
        print(f"  VWAP影响数: {len(long_vwap_influenced)}")
        print(f"  VWAP生效比例: {long_ratio:.1f}%")
    
    if len(short_stop_loss) > 0:
        short_vwap_influenced = short_stop_loss[short_stop_loss['vwap_influenced'] == True]
        short_ratio = len(short_vwap_influenced) / len(short_stop_loss) * 100
        print(f"\n空头交易:")
        print(f"  止损平仓数: {len(short_stop_loss)}")
        print(f"  VWAP影响数: {len(short_vwap_influenced)}")
        print(f"  VWAP生效比例: {short_ratio:.1f}%")
    
    return {
        'total_stop_loss': total_stop_loss,
        'vwap_influenced_count': vwap_influenced_count,
        'vwap_influence_ratio': vwap_influence_ratio
    }

def analyze_trailing_take_profit_impact(trades_df, config):
    """
    🎯 分析动态追踪止盈对交易的影响
    """
    if len(trades_df) == 0:
        print("\n=== 动态追踪止盈分析 ===")
        print("没有交易数据可供分析")
        return None
    
    enable_trailing_tp = config.get('enable_trailing_take_profit', False)
    if not enable_trailing_tp:
        print("\n=== 动态追踪止盈分析 ===")
        print("动态追踪止盈未启用")
        return None
    
    # 统计追踪止盈触发的交易
    trailing_tp_trades = trades_df[trades_df['exit_reason'] == 'Trailing Take Profit']
    stop_loss_trades = trades_df[trades_df['exit_reason'] == 'Stop Loss']
    intraday_close_trades = trades_df[trades_df['exit_reason'] == 'Intraday Close']
    market_close_trades = trades_df[trades_df['exit_reason'] == 'Market Close']
    
    total_trades = len(trades_df)
    trailing_tp_count = len(trailing_tp_trades)
    trailing_tp_ratio = trailing_tp_count / total_trades * 100 if total_trades > 0 else 0
    
    print("\n=== 🎯 动态追踪止盈分析 ===")
    print(f"配置参数:")
    print(f"  激活阈值: {config.get('trailing_tp_activation_pct', 0.005)*100:.2f}%")
    print(f"  保护比例: {config.get('trailing_tp_callback_pct', 0.5)*100:.0f}%")
    print(f"\n出场方式统计:")
    print(f"  总交易数: {total_trades}")
    print(f"  追踪止盈: {trailing_tp_count} ({trailing_tp_ratio:.1f}%)")
    print(f"  止损平仓: {len(stop_loss_trades)} ({len(stop_loss_trades)/total_trades*100:.1f}%)")
    print(f"  日内收盘: {len(intraday_close_trades)} ({len(intraday_close_trades)/total_trades*100:.1f}%)")
    print(f"  市场收盘: {len(market_close_trades)} ({len(market_close_trades)/total_trades*100:.1f}%)")
    
    # 计算各类出场的盈亏
    if trailing_tp_count > 0:
        trailing_tp_pnl = trailing_tp_trades['pnl'].sum()
        trailing_tp_avg_pnl = trailing_tp_trades['pnl'].mean()
        trailing_tp_win_rate = (trailing_tp_trades['pnl'] > 0).mean() * 100
        print(f"\n追踪止盈交易详情:")
        print(f"  总盈亏: ${trailing_tp_pnl:.2f}")
        print(f"  平均盈亏: ${trailing_tp_avg_pnl:.2f}")
        print(f"  胜率: {trailing_tp_win_rate:.1f}%")
    
    if len(stop_loss_trades) > 0:
        stop_loss_pnl = stop_loss_trades['pnl'].sum()
        stop_loss_avg_pnl = stop_loss_trades['pnl'].mean()
        stop_loss_win_rate = (stop_loss_trades['pnl'] > 0).mean() * 100
        print(f"\n止损平仓交易详情:")
        print(f"  总盈亏: ${stop_loss_pnl:.2f}")
        print(f"  平均盈亏: ${stop_loss_avg_pnl:.2f}")
        print(f"  胜率: {stop_loss_win_rate:.1f}%")
    
    # 分多头和空头分析
    long_trailing_tp = trailing_tp_trades[trailing_tp_trades['side'] == 'Long']
    short_trailing_tp = trailing_tp_trades[trailing_tp_trades['side'] == 'Short']
    
    if len(long_trailing_tp) > 0:
        print(f"\n多头追踪止盈:")
        print(f"  次数: {len(long_trailing_tp)}")
        print(f"  总盈亏: ${long_trailing_tp['pnl'].sum():.2f}")
        print(f"  平均盈亏: ${long_trailing_tp['pnl'].mean():.2f}")
    
    if len(short_trailing_tp) > 0:
        print(f"\n空头追踪止盈:")
        print(f"  次数: {len(short_trailing_tp)}")
        print(f"  总盈亏: ${short_trailing_tp['pnl'].sum():.2f}")
        print(f"  平均盈亏: ${short_trailing_tp['pnl'].mean():.2f}")
    
    return {
        'total_trades': total_trades,
        'trailing_tp_count': trailing_tp_count,
        'trailing_tp_ratio': trailing_tp_ratio,
        'trailing_tp_pnl': trailing_tp_trades['pnl'].sum() if trailing_tp_count > 0 else 0,
        'stop_loss_count': len(stop_loss_trades),
        'stop_loss_pnl': stop_loss_trades['pnl'].sum() if len(stop_loss_trades) > 0 else 0
    }

def plot_specific_days(config, dates_to_plot):
    """
    为指定的日期生成交易图表
    
    参数:
        config: 配置字典，包含所有回测参数
        dates_to_plot: 要绘制的日期列表 (datetime.date 对象列表)
    """
    # 创建配置的副本并更新plot_days
    plot_config = config.copy()
    plot_config['plot_days'] = dates_to_plot
    
    # 运行回测，指定要绘制的日期
    _, _, _, _ = run_backtest(plot_config)
    
    print(f"\n已为以下日期生成图表:")
    for d in dates_to_plot:
        print(f"- {d}")

# 示例用法
if __name__ == "__main__":  
    # 创建配置字典
    config = {
        # 'data_path': 'qqq_market_hours_with_indicators.csv',
        # 'data_path':'tqqq_market_hours_with_indicators.csv',
        'data_path': 'qqq_longport.csv',  # 使用包含Turnover字段的longport数据
        # 'data_path': 'tqqq_longport.csv',
        'ticker': 'QQQ',
        'initial_capital': 10000,
        'lookback_days':1,
        'start_date': date(2025, 1, 1),
        'end_date': date(2026, 2, 3),
        # 'start_date': date(2020, 1, 1),
        # 'end_date': date(2025, 2, 3),
        'check_interval_minutes': 15 ,
        'enable_transaction_fees': True,  # 是否启用手续费计算，False表示不计算手续费
        # 'transaction_fee_per_share': 0.008166,
        'transaction_fee_per_share': 0,
        'slippage_per_share': 0.005,  # 滑点设置，每股滑点金额，买入时多付，卖出时少收
                                     # 例如：0.02表示买入每股多付2美分，卖出每股少收2美分
        'trading_start_time': (9, 40),
        'trading_end_time': (15, 40),
        'max_positions_per_day': 10,
        # 'random_plots': 3,
        # 'plots_dir': 'trading_plots',
        'print_daily_trades': False,
        'print_trade_details': False,
        'K1': 1,  # 上边界sigma乘数
        'K2': 1,  # 下边界sigma乘数
        'leverage':3,  # 资金杠杆倍数，默认为1
        'use_vwap': False,  # VWAP开关，True为使用VWAP，False为不使用
        'enable_intraday_stop_loss': False,  # 是否启用日内止损
        'intraday_stop_loss_pct': 0.04,  # 日内止损阈值（4%）
        
        # 🎯 动态追踪止盈配置
        'enable_trailing_take_profit': True,  # 是否启用动态追踪止盈
        'trailing_tp_activation_pct': 0.01,  # 激活追踪止盈的最低浮盈百分比（0.5%）
        'trailing_tp_callback_pct': 0.7,  # 保护的利润比例（50%），即从最大浮盈回撤50%时触发止盈
    }
    
    # 运行回测
    daily_results, monthly_results, trades, metrics = run_backtest(config)
