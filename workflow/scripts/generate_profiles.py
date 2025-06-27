#!/usr/bin/env python3
"""
第一步：加载、清洗并归一化处理从Excel中提取的日负荷曲线。

该脚本专门负责处理原始的Excel文件，并生成一个干净、可用的、
归一化后的负荷曲线CSV文件，供后续脚本使用。
"""

import pandas as pd
from pathlib import Path

# --- 配置 ---
LOAD_PROFILE_CSV_PATH = "resources/data/load/Fig3-2.Daily charging load from electric vehicles across different vehicle types and power level.csv"
OUTPUT_DIR = Path("workflow/output")
NORMALIZED_PROFILES_CSV_PATH = OUTPUT_DIR / "normalized_profiles.csv"

def generate_clean_profiles():
    """加载、清洗、聚合（5分钟->1小时）、归一化并保存负荷曲线。"""
    print("--- 2.1 开始生成归一化的负荷曲线 (from CSV, with cleaning) ---")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    try:
        df = pd.read_csv(LOAD_PROFILE_CSV_PATH, header=0)
        
        # --- 关键清洗步骤 ---
        # 获取所有可能的P列（功率列）
        p_cols = [col for col in df.columns if '_P' in col]
        
        # 创建掩码，只保留至少有一个功率列不为0的行
        mask = df[p_cols].ne(0).any(axis=1)
        
        # 应用掩码，提取前288个有效数据点，并立即重置索引
        load_data_5min = df[mask].head(288).copy().reset_index(drop=True)
        
        # 创建一个5分钟的时间索引 (现在基于干净、连续的索引)
        time_index = pd.to_datetime("2030-01-01") + pd.to_timedelta(load_data_5min.index * 5, unit='m')
        load_data_5min.index = time_index

        # 创建一个干净的DataFrame来存储最终的小时级profile
        profile_df_hourly = pd.DataFrame()

        vehicle_types = ['Private car', 'Taxi', 'Official car', 'Rental car', 'Bus', 'SPV']
        date_types = ['workday', 'weekend & holiday']
        
        for v_type in vehicle_types:
            for d_type in date_types:
                new_col_name = f'{v_type}_{d_type}'
                
                date_type_str = 'workday' if d_type == 'workday' else 'weekend & holiday'
                p_cols_to_sum = [f'{v_type}_P{i}_{date_type_str}' for i in [1, 2, 3]]
                p_cols_exist = [col for col in p_cols_to_sum if col in load_data_5min.columns]
                
                if not p_cols_exist:
                    continue
                
                total_load_5min = load_data_5min[p_cols_exist].sum(axis=1)
                
                hourly_load = total_load_5min.resample('h').sum()
                
                col_sum = hourly_load.sum()
                if col_sum > 0:
                    profile_df_hourly[new_col_name] = hourly_load / col_sum
                else:
                    profile_df_hourly[new_col_name] = 0.0

        profile_df_hourly.insert(0, 'hour', range(24))
        
        profile_df_hourly.to_csv(NORMALIZED_PROFILES_CSV_PATH, index=False)
        
        print(f"✓ 归一化的负荷曲线已成功保存到: {NORMALIZED_PROFILES_CSV_PATH}")
        
    except Exception as e:
        print(f"处理CSV文件时发生严重错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_clean_profiles() 