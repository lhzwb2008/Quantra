#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股特殊日期获取工具
用于获取美联储议息日、ETF分红日、美股节日等特殊日期
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import yfinance as yf
import requests
from typing import List, Dict, Optional, Set
import warnings
warnings.filterwarnings('ignore')

class USSpecialDates:
    """美股特殊日期获取类"""
    
    def __init__(self):
        """初始化"""
        self.fomc_dates = set()
        self.dividend_dates = {}
        self.market_holidays = set()
        
    def get_fomc_dates(self, start_year: int = 2020, end_year: int = 2025) -> Set[date]:
        """
        获取美联储FOMC会议日期
        
        参数:
            start_year: 开始年份
            end_year: 结束年份
            
        返回:
            FOMC会议日期集合
        """
        # 历史已知的FOMC会议日期（手动维护）
        fomc_dates_dict = {
            2020: [
                "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", 
                "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16"
            ],
            2021: [
                "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", 
                "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15"
            ],
            2022: [
                "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", 
                "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14"
            ],
            2023: [
                "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", 
                "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13"
            ],
            2024: [
                "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", 
                "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"
            ],
            2025: [
                "2025-01-29", "2025-03-19", "2025-04-30", "2025-06-11", 
                "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17"
            ]
        }
        
        fomc_dates = set()
        for year in range(start_year, end_year + 1):
            if year in fomc_dates_dict:
                for date_str in fomc_dates_dict[year]:
                    fomc_dates.add(datetime.strptime(date_str, "%Y-%m-%d").date())
        
        self.fomc_dates = fomc_dates
        print(f"获取到 {len(fomc_dates)} 个FOMC会议日期 ({start_year}-{end_year})")
        return fomc_dates
    
    def get_dividend_dates(self, symbol: str, start_date: date, end_date: date) -> Dict[date, float]:
        """
        获取ETF分红日期
        
        参数:
            symbol: 股票代码（如'QQQ'）
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            分红日期字典 {日期: 分红金额}
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # 获取分红历史
            dividends = ticker.dividends
            
            if dividends.empty:
                print(f"未找到 {symbol} 的分红记录")
                return {}
            
            # 过滤日期范围
            dividends = dividends[(dividends.index.date >= start_date) & 
                                (dividends.index.date <= end_date)]
            
            # 转换为字典
            dividend_dict = {}
            for div_date, div_amount in dividends.items():
                dividend_dict[div_date.date()] = float(div_amount)
            
            self.dividend_dates[symbol] = dividend_dict
            print(f"获取到 {symbol} 的 {len(dividend_dict)} 个分红日期")
            return dividend_dict
            
        except Exception as e:
            print(f"获取 {symbol} 分红数据时出错: {e}")
            return {}
    
    def get_us_market_holidays(self, start_year: int = 2020, end_year: int = 2025) -> Set[date]:
        """
        获取美股市场节日
        
        参数:
            start_year: 开始年份
            end_year: 结束年份
            
        返回:
            节日日期集合
        """
        holidays = set()
        
        # 美股主要节日（固定日期和相对日期）
        for year in range(start_year, end_year + 1):
            # 新年
            holidays.add(date(year, 1, 1))
            
            # 马丁·路德·金日（1月第三个星期一）
            mlk_day = self._get_nth_weekday(year, 1, 0, 3)  # 第三个星期一
            holidays.add(mlk_day)
            
            # 总统日（2月第三个星期一）
            presidents_day = self._get_nth_weekday(year, 2, 0, 3)
            holidays.add(presidents_day)
            
            # 阵亡将士纪念日（5月最后一个星期一）
            memorial_day = self._get_last_weekday(year, 5, 0)
            holidays.add(memorial_day)
            
            # 独立日
            independence_day = date(year, 7, 4)
            # 如果7月4日是周六，则周五休市；如果是周日，则周一休市
            if independence_day.weekday() == 5:  # 周六
                holidays.add(independence_day - timedelta(days=1))
            elif independence_day.weekday() == 6:  # 周日
                holidays.add(independence_day + timedelta(days=1))
            else:
                holidays.add(independence_day)
            
            # 劳动节（9月第一个星期一）
            labor_day = self._get_nth_weekday(year, 9, 0, 1)
            holidays.add(labor_day)
            
            # 感恩节（11月第四个星期四）
            thanksgiving = self._get_nth_weekday(year, 11, 3, 4)  # 第四个星期四
            holidays.add(thanksgiving)
            
            # 圣诞节
            christmas = date(year, 12, 25)
            # 如果圣诞节是周六，则周五休市；如果是周日，则周一休市
            if christmas.weekday() == 5:  # 周六
                holidays.add(christmas - timedelta(days=1))
            elif christmas.weekday() == 6:  # 周日
                holidays.add(christmas + timedelta(days=1))
            else:
                holidays.add(christmas)
        
        self.market_holidays = holidays
        print(f"获取到 {len(holidays)} 个美股节日 ({start_year}-{end_year})")
        return holidays
    
    def _get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> date:
        """获取某月第n个指定星期几的日期"""
        first_day = date(year, month, 1)
        first_weekday = first_day.weekday()
        
        # 计算第一个指定星期几的日期
        days_ahead = weekday - first_weekday
        if days_ahead < 0:
            days_ahead += 7
        
        first_target = first_day + timedelta(days=days_ahead)
        return first_target + timedelta(weeks=n-1)
    
    def _get_last_weekday(self, year: int, month: int, weekday: int) -> date:
        """获取某月最后一个指定星期几的日期"""
        # 获取下个月第一天
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        
        # 向前找到最后一个指定星期几
        last_day = next_month - timedelta(days=1)
        days_back = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=days_back)
    
    def get_all_special_dates(self, symbols: List[str] = ['QQQ', 'SPY'], 
                            start_date: date = date(2020, 1, 1), 
                            end_date: date = date(2025, 12, 31)) -> Dict[str, Set[date]]:
        """
        获取所有特殊日期
        
        参数:
            symbols: ETF代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            特殊日期字典
        """
        special_dates = {}
        
        # 获取FOMC会议日期
        fomc_dates = self.get_fomc_dates(start_date.year, end_date.year)
        special_dates['FOMC'] = fomc_dates
        
        # 获取市场节日
        holidays = self.get_us_market_holidays(start_date.year, end_date.year)
        special_dates['Market_Holidays'] = holidays
        
        # 获取各ETF分红日期
        for symbol in symbols:
            dividend_dates = self.get_dividend_dates(symbol, start_date, end_date)
            if dividend_dates:
                special_dates[f'{symbol}_Dividends'] = set(dividend_dates.keys())
        
        return special_dates
    
    def filter_trading_dates(self, trading_dates: List[date], 
                           exclude_types: List[str] = None) -> List[date]:
        """
        过滤交易日期，排除特殊日期
        
        参数:
            trading_dates: 原始交易日期列表
            exclude_types: 要排除的特殊日期类型列表
                          可选项: ['FOMC', 'Market_Holidays', 'Dividends', 'All']
                          
        返回:
            过滤后的交易日期列表
        """
        if exclude_types is None:
            return trading_dates
        
        # 获取所有特殊日期
        all_special = self.get_all_special_dates()
        
        # 构建要排除的日期集合
        exclude_dates = set()
        
        for exclude_type in exclude_types:
            if exclude_type == 'All':
                # 排除所有特殊日期
                for dates in all_special.values():
                    exclude_dates.update(dates)
            elif exclude_type == 'FOMC' and 'FOMC' in all_special:
                exclude_dates.update(all_special['FOMC'])
            elif exclude_type == 'Market_Holidays' and 'Market_Holidays' in all_special:
                exclude_dates.update(all_special['Market_Holidays'])
            elif exclude_type == 'Dividends':
                # 排除所有分红日期
                for key, dates in all_special.items():
                    if 'Dividends' in key:
                        exclude_dates.update(dates)
        
        # 过滤日期
        filtered_dates = [d for d in trading_dates if d not in exclude_dates]
        
        print(f"原始交易日: {len(trading_dates)} 天")
        print(f"排除特殊日期: {len(exclude_dates)} 天")
        print(f"过滤后交易日: {len(filtered_dates)} 天")
        
        return filtered_dates
    
    def print_special_dates_summary(self, start_date: date, end_date: date):
        """打印特殊日期汇总信息"""
        all_special = self.get_all_special_dates(start_date=start_date, end_date=end_date)
        
        print(f"\n=== 特殊日期汇总 ({start_date} 至 {end_date}) ===")
        
        total_special_days = 0
        for date_type, dates in all_special.items():
            # 过滤日期范围
            filtered_dates = [d for d in dates if start_date <= d <= end_date]
            total_special_days += len(filtered_dates)
            
            print(f"{date_type}: {len(filtered_dates)} 天")
            
            # 显示最近的几个日期作为示例
            if filtered_dates:
                sorted_dates = sorted(filtered_dates)
                sample_dates = sorted_dates[:5]  # 显示前5个
                date_strings = [d.strftime('%Y-%m-%d') for d in sample_dates]
                print(f"  示例日期: {', '.join(date_strings)}")
                if len(sorted_dates) > 5:
                    print(f"  ... 还有 {len(sorted_dates) - 5} 个日期")
        
        print(f"\n总计特殊日期: {total_special_days} 天")
        print("="*50)

# 示例用法
if __name__ == "__main__":
    # 创建特殊日期获取器
    special_dates = USSpecialDates()
    
    # 设置日期范围
    start_date = date(2023, 1, 1)
    end_date = date(2025, 12, 31)
    
    # 获取所有特殊日期
    all_special = special_dates.get_all_special_dates(
        symbols=['QQQ', 'SPY', 'TQQQ'], 
        start_date=start_date, 
        end_date=end_date
    )
    
    # 打印汇总信息
    special_dates.print_special_dates_summary(start_date, end_date)
    
    # 示例：过滤交易日期
    sample_trading_dates = pd.date_range(start_date, end_date, freq='B').date.tolist()  # 工作日
    
    print(f"\n=== 日期过滤示例 ===")
    
    # 只排除FOMC会议日期
    filtered_fomc = special_dates.filter_trading_dates(
        sample_trading_dates, 
        exclude_types=['FOMC']
    )
    
    # 排除所有特殊日期
    filtered_all = special_dates.filter_trading_dates(
        sample_trading_dates, 
        exclude_types=['All']
    )
