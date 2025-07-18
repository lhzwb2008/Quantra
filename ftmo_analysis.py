import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from backtest import run_backtest
import warnings
import random
import sys
import os
warnings.filterwarnings('ignore')

def analyze_ftmo_compliance(daily_results, trades_df, initial_capital, max_daily_loss=0.05, max_total_loss=0.10):
    """
    分析交易结果是否符合FTMO规则
    
    参数:
        daily_results: 每日结果DataFrame
        trades_df: 交易记录DataFrame
        initial_capital: 初始资金
        max_daily_loss: 最大日损失限制 (5%)
        max_total_loss: 最大总损失限制 (10%)
    
    返回:
        分析结果字典
    """
    results = {}
    
    # 1. 计算每日损失
    daily_results = daily_results.copy()
    daily_results['daily_pnl'] = daily_results['capital'].diff()
    daily_results['daily_pnl'].iloc[0] = daily_results['capital'].iloc[0] - initial_capital
    daily_results['daily_loss_pct'] = daily_results['daily_pnl'] / initial_capital
    
    # 2. 计算从初始资金开始的回撤
    daily_results['drawdown_from_initial'] = (daily_results['capital'] - initial_capital) / initial_capital
    daily_results['min_drawdown_from_initial'] = daily_results['drawdown_from_initial'].cummin()
    
    # 3. 统计违规情况
    daily_violations = (daily_results['daily_loss_pct'] < -max_daily_loss).sum()
    total_violation = (daily_results['min_drawdown_from_initial'] < -max_total_loss).any()
    
    # 4. 找出最大日损失和最大总回撤
    max_daily_loss_observed = daily_results['daily_loss_pct'].min()
    max_total_drawdown = daily_results['min_drawdown_from_initial'].min()
    
    # 5. 计算风险指标
    results['daily_violations'] = daily_violations
    results['total_violation'] = total_violation
    results['max_daily_loss_pct'] = max_daily_loss_observed
    results['max_total_drawdown_pct'] = max_total_drawdown
    results['days_to_violation'] = None
    
    # 如果有违规，找出第一次违规的日期
    if total_violation:
        violation_date = daily_results[daily_results['min_drawdown_from_initial'] < -max_total_loss].index[0]
        results['days_to_violation'] = len(daily_results[:violation_date])
    
    # 6. 计算每日最大浮亏（日内回撤）
    if not trades_df.empty:
        # 按日期分组计算每日的交易统计
        daily_trades = trades_df.groupby('Date').agg({
            'pnl': ['sum', 'min', 'max', 'count']
        })
        daily_trades.columns = ['total_pnl', 'worst_trade', 'best_trade', 'num_trades']
        
        results['avg_trades_per_day'] = daily_trades['num_trades'].mean()
        results['worst_single_trade_pct'] = daily_trades['worst_trade'].min() / initial_capital
    
    return results, daily_results

def simulate_ftmo_challenge(config, start_date, profit_target=0.10, max_daily_loss=0.05, max_total_loss=0.10):
    """
    模拟单次FTMO挑战（无时间限制）
    
    参数:
        config: 配置字典
        start_date: 挑战开始日期
        profit_target: 盈利目标 (10%)
        max_daily_loss: 最大日损失 (5%)
        max_total_loss: 最大总损失 (10%)
    
    返回:
        (是否通过, 结束原因, 持续天数, 最终收益率)
    """
    # 设置一个较长的结束日期，让策略自然运行
    end_date = config['end_date']  # 使用配置中的结束日期
    
    # 更新配置
    challenge_config = config.copy()
    challenge_config['start_date'] = start_date
    challenge_config['end_date'] = end_date
    challenge_config['print_daily_trades'] = False
    challenge_config['print_trade_details'] = False
    
    # 运行回测，重定向输出到null
    try:
        # 临时重定向stdout来隐藏backtest输出
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        daily_results, _, trades_df, _ = run_backtest(challenge_config)
        
        # 恢复stdout
        sys.stdout.close()
        sys.stdout = original_stdout
        
    except Exception as e:
        # 确保stdout被恢复
        if sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        return False, 'error', 0, 0
    
    if len(daily_results) == 0:
        return False, 'no_data', 0, 0
    
    # 确保trades_df['Date']是Timestamp类型
    if not trades_df.empty and not isinstance(trades_df['Date'].iloc[0], pd.Timestamp):
        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
    
    initial_capital = config['initial_capital']
    
    # 逐日检查是否达到目标或违反规则
    for i in range(len(daily_results)):
        current_day = i + 1
        day_data = daily_results.iloc[:current_day]
        
        # 计算当前资金和收益率
        current_capital = day_data['capital'].iloc[-1]
        current_return = (current_capital - initial_capital) / initial_capital
        
        # 计算当日损失
        if i == 0:
            daily_loss = (current_capital - initial_capital) / initial_capital
        else:
            daily_loss = (current_capital - daily_results['capital'].iloc[i-1]) / initial_capital
        
        # 检查是否达到盈利目标
        if current_return >= profit_target:
            return True, 'profit_target', current_day, current_return
        
        # 检查是否违反日损失规则
        if daily_loss < -max_daily_loss:
            return False, 'daily_loss', current_day, current_return
        
        # 检查是否违反总损失规则
        if current_return < -max_total_loss:
            return False, 'total_loss', current_day, current_return
    
    # 数据用完但未达到目标（这种情况下返回最终收益率）
    final_return = (daily_results['capital'].iloc[-1] - initial_capital) / initial_capital
    return False, 'data_exhausted', len(daily_results), final_return

def save_intermediate_results(results_summary, filename='ftmo_intermediate_results.csv'):
    """保存中间结果"""
    # 不再保存文件，只返回DataFrame用于显示
    if results_summary:
        df = pd.DataFrame(results_summary)
        return df
    return None

def monte_carlo_ftmo_analysis(config, num_simulations=100, leverage_range=None):
    """
    使用蒙特卡洛方法分析FTMO挑战通过率
    
    参数:
        config: 基础配置字典
        num_simulations: 每个杠杆率的模拟次数
        leverage_range: 杠杆率范围
    """
    if leverage_range is None:
        leverage_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # 获取数据的时间范围
    start_date = config['start_date']
    end_date = config['end_date']
    total_days = (end_date - start_date).days
    
    # 确保有足够的数据进行抽样
    if total_days < 60:  # 至少需要60天数据
        print(f"警告: 数据时间范围太短，需要至少60天的数据")
        return None
    
    results_summary = []
    
    for leverage_idx, leverage in enumerate(leverage_range):
        print(f"\n[{leverage_idx+1}/{len(leverage_range)}] 分析杠杆率: {leverage}x")
        
        # 更新配置
        test_config = config.copy()
        test_config['leverage'] = leverage
        
        # 运行多次模拟
        simulation_results = []
        
        for sim in range(num_simulations):
            # 随机选择起始日期，确保至少有60天的数据可用
            max_start_offset = max(0, total_days - 60)
            start_offset = random.randint(0, max_start_offset)
            sim_start_date = start_date + timedelta(days=start_offset)
            
            # 模拟挑战
            passed, reason, days, final_return = simulate_ftmo_challenge(
                test_config, 
                sim_start_date
            )
            
            simulation_results.append({
                'passed': passed,
                'reason': reason,
                'days': days,
                'final_return': final_return,
                'start_date': sim_start_date
            })
            
            # 显示进度
            if (sim + 1) % 10 == 0:
                current_passed = sum(1 for r in simulation_results if r['passed'])
                current_rate = current_passed / (sim + 1) * 100
                print(f"  进度: {sim + 1}/{num_simulations} | 当前通过率: {current_rate:.1f}%")
        
        # 统计结果
        passed_count = sum(1 for r in simulation_results if r['passed'])
        # 只计算有效的测试（排除数据用完的情况）
        valid_results = [r for r in simulation_results if r['reason'] != 'data_exhausted']
        valid_count = len(valid_results)
        
        if valid_count > 0:
            valid_passed_count = sum(1 for r in valid_results if r['passed'])
            pass_rate = valid_passed_count / valid_count
        else:
            pass_rate = 0
            valid_passed_count = 0
        
        # 按失败原因分类
        failure_reasons = {}
        for r in simulation_results:
            if not r['passed']:
                reason = r['reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        # 计算平均完成天数（仅成功的）
        successful_runs = [r for r in simulation_results if r['passed']]
        avg_days_to_success = np.mean([r['days'] for r in successful_runs]) if successful_runs else 0
        
        # 计算所有模拟的平均天数（包括失败的）
        all_days = [r['days'] for r in simulation_results]
        avg_days_all = np.mean(all_days) if all_days else 0
        
        # 计算有效测试的平均天数
        valid_days = [r['days'] for r in valid_results]
        avg_days_valid = np.mean(valid_days) if valid_days else 0
        
        # 计算收益统计
        all_returns = [r['final_return'] for r in simulation_results]
        avg_return = np.mean(all_returns)
        
        summary = {
            'leverage': leverage,
            'pass_rate': pass_rate,
            'num_simulations': num_simulations,
            'valid_count': valid_count,
            'passed_count': valid_passed_count,
            'data_exhausted_count': failure_reasons.get('data_exhausted', 0),
            'avg_days_to_success': avg_days_to_success,
            'avg_days_all': avg_days_all,
            'avg_days_valid': avg_days_valid,
            'avg_return': avg_return,
            'failure_daily_loss': failure_reasons.get('daily_loss', 0),
            'failure_total_loss': failure_reasons.get('total_loss', 0),
            'failure_data_exhausted': failure_reasons.get('data_exhausted', 0),
            'failure_error': failure_reasons.get('error', 0),
            'failure_no_data': failure_reasons.get('no_data', 0)
        }
        
        results_summary.append(summary)
        
        # 打印当前杠杆率结果
        print(f"  ✓ 有效测试: {valid_count}/{num_simulations} (排除数据用完: {failure_reasons.get('data_exhausted', 0)}次)")
        print(f"  ✓ 通过率: {pass_rate*100:.1f}% ({valid_passed_count}/{valid_count})")
        if successful_runs:
            print(f"  ✓ 平均成功天数: {avg_days_to_success:.1f}天")
        print(f"  ✓ 平均有效测试天数: {avg_days_valid:.1f}天")
        print(f"  ✓ 失败原因: 日损失{failure_reasons.get('daily_loss', 0)}次 | 总损失{failure_reasons.get('total_loss', 0)}次")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results_summary)
    
    # 找出最优杠杆率（80%通过率以上的最高杠杆）
    qualified = results_df[results_df['pass_rate'] >= 0.8]
    if not qualified.empty:
        optimal = qualified.loc[qualified['leverage'].idxmax()]
        print(f"\n🎯 最优杠杆率（通过率≥80%）:")
        print(f"  杠杆率: {optimal['leverage']}x")
        print(f"  通过率: {optimal['pass_rate']*100:.1f}%")
        print(f"  平均成功天数: {optimal['avg_days_to_success']:.1f}天")
    else:
        print(f"\n⚠️  没有杠杆率能达到80%的通过率")
    
    return results_df

# 保留原有的函数以便兼容
def rolling_window_analysis(config, window_days=30, leverage_range=None):
    """
    保留原函数签名，但调用新的蒙特卡洛分析
    """
    return monte_carlo_ftmo_analysis(config, num_simulations=100, leverage_range=leverage_range)

def analyze_single_leverage(config, leverage):
    """
    详细分析单个杠杆率的表现
    
    参数:
        config: 基础配置字典
        leverage: 杠杆率
    """
    print(f"\n详细分析杠杆率: {leverage}x")
    
    # 更新配置
    test_config = config.copy()
    test_config['leverage'] = leverage
    test_config['print_daily_trades'] = False
    
    # 运行回测
    daily_results, monthly_results, trades_df, metrics = run_backtest(test_config)
    
    # 全局分析
    analysis, daily_with_metrics = analyze_ftmo_compliance(
        daily_results, 
        trades_df, 
        config['initial_capital']
    )
    
    print(f"\n整体表现:")
    print(f"  年化收益率: {metrics['irr']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {metrics['mdd']*100:.1f}%")
    print(f"  总交易次数: {metrics['total_trades']}")
    
    print(f"\nFTMO规则分析:")
    print(f"  最大日损失: {analysis['max_daily_loss_pct']*100:.2f}%")
    print(f"  最大总回撤: {analysis['max_total_drawdown_pct']*100:.2f}%")
    print(f"  违反5%日损失规则次数: {analysis['daily_violations']}")
    print(f"  是否违反10%总损失规则: {'是' if analysis['total_violation'] else '否'}")
    
    if analysis['days_to_violation']:
        print(f"  首次违规发生在第 {analysis['days_to_violation']} 天")
    
    # 找出最差的几天
    worst_days = daily_with_metrics.nsmallest(5, 'daily_loss_pct')[['daily_loss_pct', 'capital']]
    print(f"\n最差的5个交易日:")
    for date, row in worst_days.iterrows():
        print(f"  {date.strftime('%Y-%m-%d')}: {row['daily_loss_pct']*100:.2f}% (资金: ${row['capital']:.2f})")
    
    return analysis, daily_with_metrics

# 使用示例
if __name__ == "__main__":
    # 创建与backtest.py相同的配置
    base_config = {
        'data_path': 'qqq_market_hours_with_indicators.csv',
        # 'data_path': 'qqq_longport.csv',
        # 'data_path': 'spy_longport.csv',
        'ticker': 'QQQ',
        'initial_capital': 10000,
        'lookback_days': 1,
        'start_date': date(2020, 1, 1),
        'end_date': date(2025, 1, 1),
        'check_interval_minutes': 15,
        'transaction_fee_per_share': 0.008166,
        'trading_start_time': (9, 40),
        'trading_end_time': (15, 40),
        'max_positions_per_day': 10,
        'print_daily_trades': False,
        'print_trade_details': False,
        'K1': 1,  # 上边界sigma乘数
        'K2': 1,  # 下边界sigma乘数
        'leverage': 4  # 资金杠杆倍数，默认为1
    }
    
    # ===========================================
    # 可以在这里自定义分析参数
    # ===========================================
    
    # 模拟次数：建议快速测试用20-50次，精确分析用100-200次
    NUM_SIMULATIONS = 50  # 每个杠杆率的模拟次数
    
    # 杠杆率范围：测试1-10倍杠杆
    LEVERAGE_RANGE = [4, 5]
    
    print("="*60)
    print("🚀 FTMO挑战通过率分析")
    print("="*60)
    print(f"📊 数据文件: {base_config['data_path']}")
    print(f"📈 股票代码: {base_config['ticker']}")
    print(f"📅 数据范围: {base_config['start_date']} 至 {base_config['end_date']}")
    print(f"🔄 模拟次数: {NUM_SIMULATIONS}次/杠杆率")
    print(f"⚡ 杠杆率范围: {LEVERAGE_RANGE}")
    print(f"🎯 目标: 达到10%收益即通过（无时间限制）")
    print(f"💡 提示: 如需修改数据，请直接修改上面的base_config")
    print("="*60)
    
    # 估算运行时间
    total_simulations = NUM_SIMULATIONS * len(LEVERAGE_RANGE)
    print(f"总计需要运行 {total_simulations} 次回测")
    print(f"预估运行时间: {total_simulations * 1:.0f}-{total_simulations * 2:.0f} 秒")
    print("提示: 可以随时按 Ctrl+C 终止\n")
    
    try:
        # 1. 分析不同杠杆率的通过率
        results_df = monte_carlo_ftmo_analysis(
            base_config, 
            num_simulations=NUM_SIMULATIONS,
            leverage_range=LEVERAGE_RANGE
        )
        
        if results_df is not None:
            # 2. 打印结果表格
            print("\n\n📋 杠杆率与通过率关系汇总:")
            print("="*100)
            print(f"{'杠杆率':>6} | {'通过率':>7} | {'有效测试':>8} | {'成功次数':>8} | {'平均成功天数':>10} | {'平均有效天数':>10} | {'数据用完':>8} | {'日损失':>8} | {'总损失':>8}")
            print("="*100)
            
            for _, row in results_df.iterrows():
                print(f"{row['leverage']:>6}x | {row['pass_rate']*100:>6.1f}% | {row['valid_count']:>8} | {row['passed_count']:>8} | "
                      f"{row['avg_days_to_success']:>9.1f}天 | {row['avg_days_valid']:>9.1f}天 | {row['data_exhausted_count']:>8} | "
                      f"{row['failure_daily_loss']:>8} | {row['failure_total_loss']:>8}")
            
            # 3. 推荐配置
            print(f"\n💡 分析结果说明:")
            print(f"• 通过率基于有效测试计算（排除数据用完的情况）")
            print(f"• 有效测试 = 总测试 - 数据用完的测试")
            print(f"• 平均成功天数：成功案例达到10%收益的平均天数")
            print(f"• 平均有效天数：所有有效测试的平均持续天数")
            print(f"• 数据用完：因数据不足而无法完成测试的次数（不计入成功率）")
            print(f"• 日损失失败和总损失失败是需要重点关注的风险指标")
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  用户中断，显示已完成的结果:")
        # 显示已完成的结果
        if 'results_summary' in locals():
            print(f"\n📊 已完成的结果:")
            print(f"{'杠杆率':>6} | {'通过率':>7} | {'有效测试':>8} | {'成功次数':>8}")
            print("-"*40)
            for summary in results_summary:
                print(f"{summary['leverage']:>6}x | {summary['pass_rate']*100:>6.1f}% | {summary['valid_count']:>8} | {summary['passed_count']:>8}")
        else:
            print("没有完成任何分析")
    
    # 4. 分析特定杠杆率（可选）
    # 如果想深入分析特定杠杆率，取消下面的注释
    analyze_single_leverage(base_config, leverage=4) 