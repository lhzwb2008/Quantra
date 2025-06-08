import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from itertools import product
import calendar
from backtest import run_backtest
import sys
import io
import multiprocessing as mp
from functools import partial

def get_month_range(year, month):
    """获取指定月份的开始和结束日期"""
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])
    return first_day, last_day

def get_next_month(year, month):
    """获取下一个月的年份和月份"""
    if month == 12:
        return year + 1, 1
    else:
        return year, month + 1

def suppress_output(func):
    """装饰器：抑制函数的标准输出"""
    def wrapper(*args, **kwargs):
        # 保存原始stdout
        old_stdout = sys.stdout
        # 重定向stdout到一个字符串缓冲区
        sys.stdout = io.StringIO()
        try:
            # 执行函数
            result = func(*args, **kwargs)
            return result
        finally:
            # 恢复原始stdout
            sys.stdout = old_stdout
    return wrapper

def test_single_param_combination(args):
    """
    测试单个参数组合的函数，用于多进程
    
    参数:
        args: (params, param_names, base_config, start_date, end_date)
    
    返回:
        (params_dict, metric_value) 或 None
    """
    params, param_names, base_config, start_date, end_date = args
    
    # 创建当前参数组合的配置
    test_config = base_config.copy()
    test_config['start_date'] = start_date
    test_config['end_date'] = end_date
    
    # 更新参数
    params_dict = {}
    for j, param_name in enumerate(param_names):
        test_config[param_name] = params[j]
        params_dict[param_name] = params[j]
    
    # 关闭详细输出以加快优化速度
    test_config['print_daily_trades'] = False
    test_config['print_trade_details'] = False
    
    try:
        # 静默运行回测
        @suppress_output
        def run_silent_backtest(config):
            return run_backtest(config)
        
        daily_df, _, trades_df, metrics = run_silent_backtest(test_config)
        
        # 使用总回报率作为优化目标
        current_metric = metrics['total_return']
        
        return (params_dict, current_metric)
        
    except Exception as e:
        # 静默处理错误
        return None

def create_coarse_grid(param_grid, reduction_factor=2):
    """
    创建粗糙网格，减少参数数量
    
    参数:
        param_grid: 原始参数网格
        reduction_factor: 减少因子，每个参数保留 1/reduction_factor 的值
    
    返回:
        粗糙参数网格
    """
    coarse_grid = {}
    for param, values in param_grid.items():
        # 对每个参数，按reduction_factor间隔采样
        if len(values) <= reduction_factor:
            coarse_grid[param] = values
        else:
            step = max(1, len(values) // reduction_factor)
            coarse_grid[param] = values[::step]
    
    return coarse_grid

def create_fine_grid(param_grid, best_params, expansion_range=1):
    """
    基于最佳参数创建细化网格
    
    参数:
        param_grid: 原始参数网格
        best_params: 粗筛选出的最佳参数
        expansion_range: 扩展范围，在最佳参数周围扩展几个值
    
    返回:
        细化参数网格
    """
    fine_grid = {}
    
    for param, values in param_grid.items():
        best_value = best_params[param]
        
        # 找到最佳值在原网格中的索引
        try:
            best_idx = values.index(best_value)
        except ValueError:
            # 如果最佳值不在原网格中，使用最接近的值
            best_idx = min(range(len(values)), key=lambda i: abs(values[i] - best_value))
        
        # 在最佳值周围扩展
        start_idx = max(0, best_idx - expansion_range)
        end_idx = min(len(values), best_idx + expansion_range + 1)
        
        fine_grid[param] = values[start_idx:end_idx]
    
    return fine_grid

def optimize_parameters(base_config, param_grid, start_date, end_date, use_parallel=True, n_processes=None):
    """
    在指定日期范围内优化参数（使用预筛选和并行计算）
    
    参数:
        base_config: 基础配置字典
        param_grid: 参数网格字典
        start_date: 优化开始日期
        end_date: 优化结束日期
        use_parallel: 是否使用并行计算
        n_processes: 进程数，None表示使用CPU核心数
    
    返回:
        最佳参数组合和对应的性能指标
    """
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 8)  # 限制最大进程数
    
    # 第一阶段：粗糙网格筛选
    print(f"    第一阶段：粗糙网格筛选...")
    coarse_grid = create_coarse_grid(param_grid, reduction_factor=2)
    
    param_names = list(coarse_grid.keys())
    param_values = list(coarse_grid.values())
    coarse_combinations = list(product(*param_values))
    
    print(f"    粗筛选组合数: {len(coarse_combinations)}")
    
    # 准备并行计算的参数
    args_list = [(params, param_names, base_config, start_date, end_date) 
                 for params in coarse_combinations]
    
    # 并行测试粗糙网格
    if use_parallel and len(coarse_combinations) > 10:
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(test_single_param_combination, args_list)
    else:
        results = [test_single_param_combination(args) for args in args_list]
    
    # 筛选有效结果
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return None, None
    
    # 选择前几名进行细化
    top_n = min(5, len(valid_results))
    valid_results.sort(key=lambda x: x[1], reverse=True)
    top_results = valid_results[:top_n]
    
    print(f"    粗筛选完成，选出前{top_n}名进行细化")
    
    # 第二阶段：细化网格优化
    print(f"    第二阶段：细化网格优化...")
    best_overall_params = None
    best_overall_metric = -float('inf')
    
    for coarse_params, coarse_metric in top_results:
        # 为每个候选参数创建细化网格
        fine_grid = create_fine_grid(param_grid, coarse_params, expansion_range=1)
        
        param_names = list(fine_grid.keys())
        param_values = list(fine_grid.values())
        fine_combinations = list(product(*param_values))
        
        # 准备并行计算的参数
        args_list = [(params, param_names, base_config, start_date, end_date) 
                     for params in fine_combinations]
        
        # 并行测试细化网格
        if use_parallel and len(fine_combinations) > 5:
            with mp.Pool(processes=n_processes) as pool:
                fine_results = pool.map(test_single_param_combination, args_list)
        else:
            fine_results = [test_single_param_combination(args) for args in args_list]
        
        # 找到当前细化网格的最佳结果
        valid_fine_results = [r for r in fine_results if r is not None]
        if valid_fine_results:
            best_fine = max(valid_fine_results, key=lambda x: x[1])
            if best_fine[1] > best_overall_metric:
                best_overall_metric = best_fine[1]
                best_overall_params = best_fine[0]
    
    total_combinations = len(coarse_combinations) + sum(
        len(list(product(*create_fine_grid(param_grid, params, 1).values()))) 
        for params, _ in top_results
    )
    print(f"    细化完成，总测试组合数: {total_combinations}")
    
    return best_overall_params, best_overall_metric

def run_dynamic_backtest(config):
    """
    执行动态参数优化的滚动窗口回测
    
    参数:
        config: 包含所有配置的字典，包括参数网格
    """
    # 提取参数网格
    param_grid = {
        'lookback_days': config.get('lookback_days_grid', [2, 5, 10, 20]),
        'check_interval_minutes': config.get('check_interval_minutes_grid', [3, 5, 10, 15]),
        'max_positions_per_day': config.get('max_positions_per_day_grid', [1, 2, 3, 4]),
        'K1': config.get('K1_grid', [0.8, 0.9, 1, 1.1, 1.2]),
        'K2': config.get('K2_grid', [0.8, 0.9, 1, 1.1, 1.2])
    }
    
    # 获取优化窗口期和回测窗口期
    optimization_window_days = config.get('optimization_window_days', 30)
    backtest_window_days = config.get('backtest_window_days', optimization_window_days)  # 默认与优化窗口期相同
    use_parallel = config.get('use_parallel', True)  # 是否使用并行计算
    n_processes = config.get('n_processes', None)  # 进程数
    
    # 基础配置（不包含会被优化的参数）
    base_config = config.copy()
    for param in param_grid.keys():
        base_config.pop(param, None)
    base_config.pop('optimization_window_days', None)
    base_config.pop('backtest_window_days', None)
    base_config.pop('use_parallel', None)
    base_config.pop('n_processes', None)
    
    # 获取总的回测时间范围
    overall_start_date = config['start_date']
    overall_end_date = config['end_date']
    
    # 初始化结果存储
    all_period_results = []
    all_trades = []
    cumulative_capital = config['initial_capital']
    current_params = None
    
    print(f"\n动态参数优化回测（滚动窗口 + 并行优化）")
    print(f"时间范围: {overall_start_date} 至 {overall_end_date}")
    print(f"初始资金: ${config['initial_capital']:,.0f}")
    print(f"优化窗口期: {optimization_window_days}天")
    print(f"回测窗口期: {backtest_window_days}天")
    print(f"并行计算: {'开启' if use_parallel else '关闭'}")
    if use_parallel:
        actual_processes = n_processes if n_processes else min(mp.cpu_count(), 8)
        print(f"进程数: {actual_processes}")
    print("="*60)
    
    # 第一步：使用前optimization_window_days天进行初始优化
    opt_start = overall_start_date
    opt_end = min(opt_start + timedelta(days=optimization_window_days - 1), overall_end_date)
    
    print(f"\n初始优化:")
    print(f"  优化期间: {opt_start} 至 {opt_end} ({(opt_end - opt_start).days + 1}天)")
    best_params, best_metric = optimize_parameters(base_config, param_grid, opt_start, opt_end, use_parallel, n_processes)
    
    if best_params:
        current_params = best_params
        print(f"  最佳参数: lookback={current_params['lookback_days']}, "
              f"interval={current_params['check_interval_minutes']}min, "
              f"max_pos={current_params['max_positions_per_day']}, "
              f"K1={current_params['K1']}, K2={current_params['K2']}")
        print(f"  优化期收益率: {best_metric*100:+.2f}%")
    else:
        print("初始优化失败")
        return [], []
    
    # 开始连续的回测
    backtest_start = opt_end + timedelta(days=1)
    period_num = 0
    
    while backtest_start <= overall_end_date:
        period_num += 1
        
        # 计算当前回测期间
        backtest_end = min(backtest_start + timedelta(days=backtest_window_days - 1), overall_end_date)
        
        # 如果剩余天数太少，结束回测
        if (backtest_end - backtest_start).days < backtest_window_days // 2:
            print(f"\n剩余天数不足，结束回测")
            break
        
        print(f"\n第{period_num}期:")
        print(f"  使用参数: lookback={current_params['lookback_days']}, "
              f"interval={current_params['check_interval_minutes']}min, "
              f"max_pos={current_params['max_positions_per_day']}, "
              f"K1={current_params['K1']}, K2={current_params['K2']}")
        print(f"  回测期间: {backtest_start} 至 {backtest_end} ({(backtest_end - backtest_start).days + 1}天)")
        
        # 创建回测配置
        backtest_config = base_config.copy()
        backtest_config.update(current_params)
        backtest_config['start_date'] = backtest_start
        backtest_config['end_date'] = backtest_end
        backtest_config['initial_capital'] = cumulative_capital
        backtest_config['print_daily_trades'] = False
        
        try:
            # 静默运行回测
            @suppress_output
            def run_silent_backtest(config):
                return run_backtest(config)
            
            daily_df, _, trades_df, metrics = run_silent_backtest(backtest_config)
            
            # 检查是否有有效数据
            if len(daily_df) > 0:
                start_capital = cumulative_capital
                cumulative_capital = daily_df['capital'].iloc[-1]
                period_return = (cumulative_capital / start_capital - 1) * 100
                
                # 存储期间结果
                period_result = {
                    'period': period_num,
                    'backtest_start': backtest_start,
                    'backtest_end': backtest_end,
                    'start_capital': start_capital,
                    'end_capital': cumulative_capital,
                    'period_return': period_return,
                    'total_trades': len(trades_df),
                    'parameters': current_params.copy()
                }
                all_period_results.append(period_result)
                
                # 打印回测结果
                print(f"  期间收益率: {period_return:+.2f}%")
                print(f"  交易次数: {len(trades_df)}")
                print(f"  期末资金: ${cumulative_capital:,.0f}")
                
                # 存储交易记录
                if len(trades_df) > 0:
                    trades_df['period'] = period_num
                    all_trades.append(trades_df)
            else:
                print(f"  无有效数据")
                
        except Exception as e:
            print(f"  回测失败: {str(e)}")
        
        # 使用刚回测完的期间数据进行参数优化，用于下一期
        # 检查是否还有下一期
        next_backtest_start = backtest_end + timedelta(days=1)
        if next_backtest_start <= overall_end_date:
            print(f"\n  使用 {backtest_start} 至 {backtest_end} 的数据优化下期参数...")
            best_params, best_metric = optimize_parameters(base_config, param_grid, backtest_start, backtest_end, use_parallel, n_processes)
            
            if best_params:
                current_params = best_params
                print(f"  优化完成，下期参数: lookback={current_params['lookback_days']}, "
                      f"interval={current_params['check_interval_minutes']}min, "
                      f"max_pos={current_params['max_positions_per_day']}, "
                      f"K1={current_params['K1']}, K2={current_params['K2']}")
                print(f"  优化期收益率: {best_metric*100:+.2f}%")
            else:
                print(f"  优化失败，继续使用当前参数")
        
        # 移动到下一个期间（连续的）
        backtest_start = backtest_end + timedelta(days=1)
    
    # 汇总结果
    print("\n" + "="*60)
    print("回测完成 - 总体表现")
    print("="*60)
    
    if all_period_results:
        # 计算总体表现
        total_return = (cumulative_capital / config['initial_capital'] - 1) * 100
        print(f"初始资金: ${config['initial_capital']:,.0f}")
        print(f"最终资金: ${cumulative_capital:,.0f}")
        print(f"总回报率: {total_return:+.2f}%")
        
        # 计算年化收益率
        first_result = all_period_results[0]
        last_result = all_period_results[-1]
        days = (last_result['backtest_end'] - first_result['backtest_start']).days + 1
        years = days / 365.25
        if years > 0:
            annualized_return = ((cumulative_capital / config['initial_capital']) ** (1/years) - 1) * 100
            print(f"年化收益率: {annualized_return:+.2f}%")
        
        # 统计信息
        print(f"\n统计信息:")
        print(f"总期数: {len(all_period_results)}")
        print(f"平均期间收益率: {np.mean([r['period_return'] for r in all_period_results]):.2f}%")
        print(f"最佳期间收益率: {max(r['period_return'] for r in all_period_results):.2f}%")
        print(f"最差期间收益率: {min(r['period_return'] for r in all_period_results):.2f}%")
        
        # 打印参数使用频率
        print("\n最常用参数组合:")
        param_usage = {}
        for result in all_period_results:
            param_str = f"lookback={result['parameters']['lookback_days']}, " \
                       f"interval={result['parameters']['check_interval_minutes']}, " \
                       f"max_pos={result['parameters']['max_positions_per_day']}, " \
                       f"K1={result['parameters']['K1']}, K2={result['parameters']['K2']}"
            param_usage[param_str] = param_usage.get(param_str, 0) + 1
        
        sorted_params = sorted(param_usage.items(), key=lambda x: x[1], reverse=True)
        for param_str, count in sorted_params[:3]:
            print(f"  {param_str} (使用{count}次)")
    else:
        print("无有效回测结果")
    
    return all_period_results, all_trades

# 示例用法
if __name__ == "__main__":
    # 创建配置字典
    config = {
        # 数据和基础设置
        'data_path': 'tqqq_longport.csv',
        'ticker': 'TQQQ',
        'initial_capital': 100000,
        
        # 总的回测时间范围
        'start_date': date(2025, 1, 1),
        'end_date': date(2025, 5, 30),
        
        # 滚动窗口参数
        'optimization_window_days': 10,  # 优化窗口期
        'backtest_window_days': 10,      # 回测窗口期（可以与优化窗口期不同）
        
        # 并行计算参数
        'use_parallel': False,           # 是否使用并行计算
        'n_processes': 4,               # 进程数，None表示自动检测
        
        # 参数网格（用于优化）
        'lookback_days_grid': [1,2,3, 5, 10],
        'check_interval_minutes_grid': [5, 10, 15],
        'max_positions_per_day_grid': [2, 3, 4],
        'K1_grid': [0.8, 1, 1.2],
        'K2_grid': [0.8, 1, 1.2],
        
        # 固定参数
        'transaction_fee_per_share': 0.005,
        'trading_start_time': (9, 40),
        'trading_end_time': (15, 45),
        'print_daily_trades': False,
        'print_trade_details': False,
    }
    
    # 运行动态参数优化回测
    period_results, all_trades = run_dynamic_backtest(config) 