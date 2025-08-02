import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from itertools import product
import multiprocessing as mp
from functools import partial
import sys
import io
from backtest_old import run_backtest

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
        args: (params, param_names, base_config, idx, total)
    
    返回:
        (params_dict, metrics, idx) 或 None
    """
    params, param_names, base_config, idx, total = args
    
    # 创建当前参数组合的配置
    test_config = base_config.copy()
    
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
        
        # 返回参数和所有指标
        return (params_dict, metrics, idx)
        
    except Exception as e:
        # 静默处理错误
        return None

def print_result(result, optimization_metric, idx, total):
    """实时打印单个结果"""
    if result is None:
        print(f"[{idx+1}/{total}] 测试失败")
        return
    
    params_dict, metrics, _ = result
    metric_value = metrics[optimization_metric]
    
    print(f"[{idx+1}/{total}] "
          f"lookback={params_dict['lookback_days']:2d}, "
          f"interval={params_dict['check_interval_minutes']:2d}min, "
          f"max_pos={params_dict['max_positions_per_day']:2d}, "
          f"K1={params_dict['K1']:.1f}, "
          f"K2={params_dict['K2']:.1f} | "
          f"{optimization_metric}={metric_value*100:+6.2f}%, "
          f"夏普={metrics['sharpe_ratio']:5.2f}, "
          f"回撤={metrics['mdd']*100:5.2f}%, "
          f"胜率={metrics['hit_ratio']*100:4.1f}%")

def optimize_parameters(config):
    """
    在指定日期范围内优化参数
    
    参数:
        config: 完整配置字典，包含所有设置
    
    返回:
        最佳参数组合、对应的指标值和所有测试结果
    """
    # 从config中提取设置
    param_grid = config['param_grid']
    optimization_metric = config.get('optimization_metric', 'total_return')
    use_parallel = config.get('use_parallel', True)
    n_processes = config.get('n_processes', None)
    top_n = config.get('top_n', 10)
    
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 8)  # 限制最大进程数
    
    # 基础配置（不包含会被优化的参数和优化设置）
    base_config = {
        'data_path': config['data_path'],
        'ticker': config['ticker'],
        'initial_capital': config['initial_capital'],
        'start_date': config['start_date'],
        'end_date': config['end_date'],
        'transaction_fee_per_share': config['transaction_fee_per_share'],
        'trading_start_time': config['trading_start_time'],
        'trading_end_time': config['trading_end_time'],
        'print_daily_trades': False,
        'print_trade_details': False,
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    print(f"\n参数优化")
    print(f"时间范围: {config['start_date']} 至 {config['end_date']}")
    print(f"初始资金: ${config['initial_capital']:,.0f}")
    print(f"优化指标: {optimization_metric}")
    print(f"参数组合总数: {len(all_combinations)}")
    print(f"并行计算: {'开启' if use_parallel else '关闭'}")
    if use_parallel:
        actual_processes = n_processes if n_processes else min(mp.cpu_count(), 8)
        print(f"进程数: {actual_processes}")
    print("="*100)
    print("\n开始测试参数组合...")
    
    # 准备并行计算的参数
    args_list = [(params, param_names, base_config, i, len(all_combinations)) 
                 for i, params in enumerate(all_combinations)]
    
    # 存储所有结果
    all_results = []
    best_metric_value = -float('inf')
    best_result = None
    
    # 执行参数测试
    if use_parallel and len(all_combinations) > 10:
        # 使用进程池的imap方法，可以实时获取结果
        with mp.Pool(processes=n_processes) as pool:
            for result in pool.imap_unordered(test_single_param_combination, args_list):
                if result is not None:
                    params_dict, metrics, idx = result
                    print_result(result, optimization_metric, idx, len(all_combinations))
                    
                    # 检查是否是新的最佳结果
                    metric_value = metrics[optimization_metric]
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = (params_dict, metrics)
                        print(f"  *** 新的最佳参数！{optimization_metric}={metric_value*100:.2f}% ***")
                    
                    all_results.append((params_dict, metrics))
    else:
        # 串行执行
        for i, args in enumerate(args_list):
            result = test_single_param_combination(args)
            if result is not None:
                params_dict, metrics, idx = result
                print_result(result, optimization_metric, idx, len(all_combinations))
                
                # 检查是否是新的最佳结果
                metric_value = metrics[optimization_metric]
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_result = (params_dict, metrics)
                    print(f"  *** 新的最佳参数！{optimization_metric}={metric_value*100:.2f}% ***")
                
                all_results.append((params_dict, metrics))
    
    print("\n" + "="*100)
    
    if not all_results:
        print("没有找到有效的参数组合")
        return None, None, []
    
    print(f"有效结果数: {len(all_results)}/{len(all_combinations)}")
    
    # 按优化指标排序
    all_results.sort(key=lambda x: x[1][optimization_metric], reverse=True)
    
    # 获取最佳参数
    best_params, best_metrics = all_results[0]
    
    # 打印最佳参数
    print(f"\n最佳参数组合:")
    print(f"  lookback_days: {best_params['lookback_days']}")
    print(f"  check_interval_minutes: {best_params['check_interval_minutes']}")
    print(f"  max_positions_per_day: {best_params['max_positions_per_day']}")
    print(f"  K1: {best_params['K1']}")
    print(f"  K2: {best_params['K2']}")
    
    print(f"\n最佳参数的性能指标:")
    print(f"  总回报率: {best_metrics['total_return']*100:+.2f}%")
    print(f"  年化收益率: {best_metrics['irr']*100:+.2f}%")
    print(f"  夏普比率: {best_metrics['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {best_metrics['mdd']*100:.2f}%")
    print(f"  胜率: {best_metrics['hit_ratio']*100:.1f}%")
    print(f"  盈亏比: {best_metrics['profit_loss_ratio']:.2f}")
    
    # 打印前N个最佳参数组合
    if top_n > 1:
        print(f"\n前{min(top_n, len(all_results))}个最佳参数组合:")
        print("-"*120)
        print(f"{'排名':^4} | {'lookback':^8} | {'interval':^8} | {'max_pos':^7} | {'K1':^4} | {'K2':^4} | "
              f"{'总回报率':^10} | {'年化收益':^10} | {'夏普比率':^8} | {'最大回撤':^8} | {'胜率':^6}")
        print("-"*120)
        
        for i, (params, metrics) in enumerate(all_results[:top_n]):
            print(f"{i+1:^4} | {params['lookback_days']:^8} | {params['check_interval_minutes']:^8} | "
                  f"{params['max_positions_per_day']:^7} | {params['K1']:^4.1f} | {params['K2']:^4.1f} | "
                  f"{metrics['total_return']*100:^10.2f}% | {metrics['irr']*100:^10.2f}% | "
                  f"{metrics['sharpe_ratio']:^8.2f} | {metrics['mdd']*100:^8.2f}% | "
                  f"{metrics['hit_ratio']*100:^6.1f}%")
    
    # 分析参数敏感性
    print("\n参数敏感性分析:")
    for param_name in param_names:
        param_performance = {}
        for params, metrics in all_results:
            param_value = params[param_name]
            if param_value not in param_performance:
                param_performance[param_value] = []
            param_performance[param_value].append(metrics[optimization_metric])
        
        print(f"\n{param_name}:")
        sorted_values = sorted(param_performance.keys())
        for value in sorted_values:
            performances = param_performance[value]
            avg_performance = np.mean(performances) * 100
            std_performance = np.std(performances) * 100
            print(f"  {value}: 平均{optimization_metric} = {avg_performance:+.2f}% (±{std_performance:.2f}%)")
    
    return best_params, best_metrics, all_results

def run_backtest_with_best_params(config, best_params):
    """
    使用最佳参数运行详细回测
    
    参数:
        config: 基础配置
        best_params: 最佳参数字典
    """
    # 创建回测配置
    backtest_config = {
        'data_path': config['data_path'],
        'ticker': config['ticker'],
        'initial_capital': config['initial_capital'],
        'start_date': config['start_date'],
        'end_date': config['end_date'],
        'transaction_fee_per_share': config['transaction_fee_per_share'],
        'trading_start_time': config['trading_start_time'],
        'trading_end_time': config['trading_end_time'],
        'print_daily_trades': True,
        'print_trade_details': True,
    }
    backtest_config.update(best_params)
    
    print("\n" + "="*80)
    print("使用最佳参数运行详细回测")
    print("="*80)
    
    # 运行回测
    daily_df, summary_df, trades_df, metrics = run_backtest(backtest_config)
    
    return daily_df, summary_df, trades_df, metrics

# 示例用法
if __name__ == "__main__":
    # 集中配置 - 所有设置都在这里
    config = {
        # 数据和基础设置
        'data_path': 'tqqq_longport.csv',
        'ticker': 'TQQQ',
        'initial_capital': 100000,
        
        # 回测时间范围
        'start_date': date(2024, 1, 1),
        'end_date': date(2025, 6, 30),
        
        # 参数网格（用于优化）
        'param_grid': {
            'lookback_days': [1, 2, 5, 10, 30, 90],
            'check_interval_minutes': [5, 10, 15, 20],
            'max_positions_per_day': [3, 5, 10],
            'K1': [0.8, 1.0, 1.2],
            'K2': [0.8, 1.0, 1.2],
        },
        
        # 优化设置
        'optimization_metric': 'total_return',  # 可选: 'total_return', 'sharpe_ratio', 'annualized_return'
        'use_parallel': True,                   # 是否使用并行计算
        'n_processes': 4,                       # 进程数，None表示自动检测
        'top_n': 20,                           # 显示前N个最佳参数组合
        
        # 固定参数
        'transaction_fee_per_share': 0.005,
        'trading_start_time': (9, 40),
        'trading_end_time': (15, 45),
    }
    
    # 运行参数优化
    best_params, best_metrics, all_results = optimize_parameters(config)
    
    if best_params:
        # 询问是否使用最佳参数运行详细回测
        print("\n" + "="*80)
        user_input = input("是否使用最佳参数运行详细回测？(y/n): ")
        if user_input.lower() == 'y':
            daily_df, summary_df, trades_df, metrics = run_backtest_with_best_params(config, best_params) 