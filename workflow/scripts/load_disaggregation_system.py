#!/usr/bin/env python3
"""
通用多部门负荷分解系统

该系统支持从不同的IAM模型（如REMIND、MESSAGE、IMAGE等）中提取数据，
并将年度部门用电量分解为小时级负荷曲线。

主要功能：
1. 支持多种IAM模型的数据格式
2. 支持多个部门的负荷分解（Transport、Heating、Industry等）
3. 模块化的设计，易于扩展新的模型和部门
4. 统一的配置管理和数据处理流程
5. 集成Gompertz模型进行空间分解
"""

import pandas as pd
import polars as pl
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import yaml
import logging

# 添加Gompertz模型导入
import sys
sys.path.append(str(Path(__file__).parent))
from gompertz_transport_disaggregator import GompertzTransportSpatialDisaggregator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 基础配置类 ---
class LoadDisaggregationConfig:
    """负荷分解系统的配置管理类"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'general': {
                'output_dir': '../../workflow/output',
                'target_year': 2030,
                'weekdays_per_year': 260,
                'weekends_holidays_per_year': 105,
                'ej_to_twh': 277.778
            },
            'gompertz': {
                'enabled': True,
                'saturation_level': 500,
                'alpha': -5.58,
                'start_year': 2020,
                'end_year': 2060
            },
            'models': {
                'remind': {
                    'file_path': '/p/tmp/ivanra/REMIND/output/SSP2-PkBudg1000-NoExprt_2025-06-11_14.54.13/EDGE-T/Transport.mif',
                    'separator': ';',
                    'region': 'CHA'
                }
            },
            'departments': {
                'transport': {
                    'enabled': True,
                    'variable_prefix': 'FE|Transport',
                    'variable_contains': 'Electricity',
                    'profile_file': '../../workflow/output/normalized_profiles.csv',
                    'ev_types_file': '../../resources/data/load/EVtypes.xlsx',
                    'weekend_ratios': {
                        'Private car': 1.0290,
                        'Taxi': 0.9411,
                        'Official car': 1.0373,
                        'Rental car': 1.1713,
                        'Bus': 0.8772,
                        'SPV': 1.0948
                    }
                },
                'heating': {
                    'enabled': True,
                    'variable_prefix': 'FE|Buildings',
                    'variable_contains': 'Electricity',
                    'profile_file': '../../workflow/output/heating_profiles.csv',
                    'weekend_ratios': {
                        'Residential': 1.1,
                        'Commercial': 0.8,
                        'Industrial': 0.9
                    }
                }
            }
        }

# --- 抽象基类 ---
class IAMModel(ABC):
    """IAM模型的抽象基类"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    @abstractmethod
    def load_data(self, file_path: str) -> pl.DataFrame:
        """加载IAM模型数据"""
        pass
    
    @abstractmethod
    def filter_data(self, data: pl.DataFrame, department_config: Dict) -> pl.DataFrame:
        """根据部门配置过滤数据"""
        pass
    
    @abstractmethod
    def extract_annual_energy(self, data: pl.DataFrame, year: int, department_config: Dict) -> pd.DataFrame:
        """提取年度能源消耗数据"""
        pass

class DepartmentProcessor(ABC):
    """部门处理器的抽象基类"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    @abstractmethod
    def load_profiles(self, profile_file: str) -> pd.DataFrame:
        """加载负荷曲线"""
        pass
    
    @abstractmethod
    def disaggregate_load(self, annual_energy: pd.DataFrame, profiles: pd.DataFrame, 
                         target_year: int, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分解负荷"""
        pass

# --- REMIND模型实现 ---
class REMINDModel(IAMModel):
    """REMIND模型的具体实现"""
    
    def load_data(self, file_path: str) -> pl.DataFrame:
        """加载REMIND数据"""
        logger.info(f"加载REMIND数据: {file_path}")
        return pl.read_csv(file_path, separator=self.config.get('separator', ';'))
    
    def filter_data(self, data: pl.DataFrame, department_config: Dict) -> pl.DataFrame:
        """过滤REMIND数据"""
        region = self.config.get('region', 'CHA')
        variable_prefix = department_config.get('variable_prefix', '')
        variable_contains = department_config.get('variable_contains', '')
        
        filtered = data.filter(
            (pl.col("Region") == region) &
            (pl.col("Variable").str.starts_with(variable_prefix)) &
            (pl.col("Variable").str.contains(variable_contains))
        )
        
        logger.info(f"过滤后数据行数: {len(filtered)}")
        return filtered
    
    def extract_annual_energy(self, data: pl.DataFrame, year: int, department_config: Dict) -> pd.DataFrame:
        """提取年度能源消耗"""
        logger.info(f"提取 {year} 年年度能源消耗")
        
        # 安全地解析分类
        def get_part(s: str, index: int):
            try: 
                return s.split('|')[index]
            except IndexError: 
                return None

        df_categorized = data.with_columns(
            pl.col("Variable").map_elements(lambda s: get_part(s, 4), return_dtype=pl.Utf8).alias("Vehicle_Type")
        ).select(["Vehicle_Type", str(year)]).to_pandas().rename(columns={str(year): 'Energy_EJ'})
        
        # 根据部门类型进行不同的聚合逻辑
        if department_config.get('name') == 'transport':
            return self._aggregate_transport_energy(df_categorized, department_config)
        elif department_config.get('name') == 'heating':
            return self._aggregate_heating_energy(df_categorized, department_config)
        else:
            return df_categorized
    
    def _aggregate_transport_energy(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """聚合交通部门能源数据"""
        logger.info("聚合交通部门能源数据")
        
        # 公交车
        bus_energy = df[df['Vehicle_Type'] == 'Bus']['Energy_EJ'].sum()
        
        # SPV = Heavy + Light Trucks
        spv_energy = df[df['Vehicle_Type'].isin(['Heavy', 'Light'])]['Energy_EJ'].sum()
        
        # LDV
        ldv_total_energy = df[df['Vehicle_Type'] == 'LDV']['Energy_EJ'].sum()
        
        # 使用外部文件拆分LDV
        ev_types_file = config.get('ev_types_file')
        if ev_types_file and Path(ev_types_file).exists():
            ev_type_ratios = pd.read_excel(ev_types_file, index_col='Vehicle type')
            
            ldv_types_to_split = ['Private car', 'Taxi', 'Official car', 'Rental car']
            mapped_data_list = []
            for v_type in ldv_types_to_split:
                ratio = ev_type_ratios.loc[v_type, '%'] if v_type in ev_type_ratios.index else 0
                energy = ldv_total_energy * ratio
                mapped_data_list.append({'mapped_type': v_type, 'Energy_EJ': energy})
            
            # 添加其他类型
            mapped_data_list.append({'mapped_type': 'Bus', 'Energy_EJ': bus_energy})
            mapped_data_list.append({'mapped_type': 'SPV', 'Energy_EJ': spv_energy})
            
            result = pd.DataFrame(mapped_data_list)
            result = result[result['Energy_EJ'] > 0].reset_index(drop=True)
            result['Energy_TWh'] = result['Energy_EJ'] * self.config.get('ej_to_twh', 277.778)
            
            return result
        
        return df
    
    def _aggregate_heating_energy(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """聚合供热部门能源数据"""
        logger.info("聚合供热部门能源数据")
        # 这里可以根据具体的供热部门数据结构进行定制
        # 暂时返回原始数据
        df['Energy_TWh'] = df['Energy_EJ'] * self.config.get('ej_to_twh', 277.778)
        return df

# --- 交通部门处理器 ---
class TransportProcessor(DepartmentProcessor):
    """交通部门负荷处理器"""
    
    def load_profiles(self, profile_file: str) -> pd.DataFrame:
        """加载交通负荷曲线"""
        logger.info(f"加载交通负荷曲线: {profile_file}")
        return pd.read_csv(profile_file)
    
    def disaggregate_load(self, annual_energy: pd.DataFrame, profiles: pd.DataFrame, 
                         target_year: int, output_dir: Path, provincial_shares: Dict[str, float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分解交通负荷到各省份"""
        logger.info(f"开始分解 {target_year} 年交通负荷")
        
        weekdays_per_year = self.config.get('weekdays_per_year', 260)
        weekends_holidays_per_year = self.config.get('weekends_holidays_per_year', 105)
        weekend_ratios = self.config.get('weekend_ratios', {})
        
        hourly_index = pd.to_datetime(
            pd.date_range(start=f'{target_year}-01-01', end=f'{target_year}-12-31 23:00', freq='h')
        )
        
        # 创建全国总负荷
        detailed_hourly_load = pd.DataFrame(index=hourly_index)
        total_hourly_load = pd.DataFrame(index=hourly_index)
        
        # 创建各省份负荷
        provincial_hourly_loads = {}
        if provincial_shares:
            for province in provincial_shares.keys():
                provincial_hourly_loads[province] = pd.DataFrame(index=hourly_index)
        
        for _, row in annual_energy.iterrows():
            v_type = row['mapped_type']
            energy_twh = row['Energy_TWh']
            
            workday_col_name = f'{v_type}_workday'
            weekend_col_name = f'{v_type}_weekend & holiday'
            
            if workday_col_name not in profiles.columns or weekend_col_name not in profiles.columns:
                logger.warning(f"在CSV中找不到 '{v_type}' 的负荷曲线，将跳过。")
                continue
            
            # 计算加权平均
            ratio = weekend_ratios.get(v_type, 1.0)
            total_energy_mwh = energy_twh * 1e6
            denominator = weekdays_per_year + weekends_holidays_per_year * ratio
            
            if denominator > 0:
                daily_energy_mwh_workday = total_energy_mwh / denominator
                daily_energy_mwh_weekend = daily_energy_mwh_workday * ratio
            else:
                daily_energy_mwh_workday = 0
                daily_energy_mwh_weekend = 0
            
            # 获取负荷曲线
            workday_profile_np = profiles[workday_col_name].to_numpy()
            weekend_profile_np = profiles[weekend_col_name].to_numpy()
            
            # 计算小时负荷
            hours_of_year = detailed_hourly_load.index.hour
            is_weekday_mask = (detailed_hourly_load.index.dayofweek < 5)
            
            hourly_weights = np.where(
                is_weekday_mask, 
                workday_profile_np[hours_of_year], 
                weekend_profile_np[hours_of_year]
            )
            
            daily_energy_map = np.where(
                is_weekday_mask,
                daily_energy_mwh_workday,
                daily_energy_mwh_weekend
            )
            
            hourly_loads_for_type = daily_energy_map * hourly_weights
            detailed_hourly_load[v_type] = hourly_loads_for_type
            
            # 分解到各省份
            if provincial_shares:
                for province, share in provincial_shares.items():
                    provincial_energy = energy_twh * share
                    provincial_total_energy_mwh = provincial_energy * 1e6
                    
                    if denominator > 0:
                        provincial_daily_energy_mwh_workday = provincial_total_energy_mwh / denominator
                        provincial_daily_energy_mwh_weekend = provincial_daily_energy_mwh_workday * ratio
                    else:
                        provincial_daily_energy_mwh_workday = 0
                        provincial_daily_energy_mwh_weekend = 0
                    
                    provincial_daily_energy_map = np.where(
                        is_weekday_mask,
                        provincial_daily_energy_mwh_workday,
                        provincial_daily_energy_mwh_weekend
                    )
                    
                    provincial_hourly_loads_for_type = provincial_daily_energy_map * hourly_weights
                    provincial_hourly_loads[province][v_type] = provincial_hourly_loads_for_type
        
        # 计算全国总负荷
        total_hourly_load['total_load_mw'] = detailed_hourly_load.sum(axis=1)
        
        # 计算各省份总负荷
        if provincial_shares:
            for province in provincial_shares.keys():
                provincial_hourly_loads[province]['total_load_mw'] = provincial_hourly_loads[province].sum(axis=1)
        
        return detailed_hourly_load, total_hourly_load, provincial_hourly_loads

# --- 供热部门处理器 ---
class HeatingProcessor(DepartmentProcessor):
    """供热部门负荷处理器"""
    
    def load_profiles(self, profile_file: str) -> pd.DataFrame:
        """加载供热负荷曲线"""
        logger.info(f"加载供热负荷曲线: {profile_file}")
        # 这里需要根据实际的供热负荷曲线文件格式进行调整
        return pd.read_csv(profile_file)
    
    def disaggregate_load(self, annual_energy: pd.DataFrame, profiles: pd.DataFrame, 
                         target_year: int, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分解供热负荷"""
        logger.info(f"开始分解 {target_year} 年供热负荷")
        
        # 这里需要根据供热部门的具体需求实现负荷分解逻辑
        # 暂时返回空DataFrame作为占位符
        hourly_index = pd.to_datetime(
            pd.date_range(start=f'{target_year}-01-01', end=f'{target_year}-12-31 23:00', freq='h')
        )
        
        detailed_hourly_load = pd.DataFrame(index=hourly_index)
        total_hourly_load = pd.DataFrame(index=hourly_index)
        total_hourly_load['total_load_mw'] = 0
        
        return detailed_hourly_load, total_hourly_load

# --- 主系统类 ---
class LoadDisaggregationSystem:
    """负荷分解主系统"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = LoadDisaggregationConfig(config_file)
        self.models = self._initialize_models()
        self.processors = self._initialize_processors()
        self.gompertz_disaggregator = self._initialize_gompertz_disaggregator()
    
    def _initialize_models(self) -> Dict[str, IAMModel]:
        """初始化IAM模型"""
        models = {}
        for model_name, model_config in self.config.config['models'].items():
            if model_name == 'remind':
                models[model_name] = REMINDModel(model_config)
            # 可以在这里添加其他模型
        return models
    
    def _initialize_processors(self) -> Dict[str, DepartmentProcessor]:
        """初始化部门处理器"""
        processors = {}
        for dept_name, dept_config in self.config.config['departments'].items():
            if dept_config.get('enabled', False):
                if dept_name == 'transport':
                    processors[dept_name] = TransportProcessor(dept_config)
                elif dept_name == 'heating':
                    processors[dept_name] = HeatingProcessor(dept_config)
                # 可以在这里添加其他部门
        return processors
    
    def _initialize_gompertz_disaggregator(self) -> Optional[GompertzTransportSpatialDisaggregator]:
        """初始化Gompertz空间分解器"""
        if not self.config.config.get('gompertz', {}).get('enabled', False):
            logger.info("Gompertz模型未启用")
            return None
        
        try:
            gompertz_config = self.config.config['gompertz']
            disaggregator = GompertzTransportSpatialDisaggregator(gompertz_config)
            
            # 加载历史数据并拟合模型
            logger.info("初始化Gompertz模型...")
            historical_data = disaggregator.load_historical_data()
            if not historical_data.empty:
                success = disaggregator.gompertz_model.fit_model(historical_data)
                if success:
                    logger.info("Gompertz模型拟合成功")
                    return disaggregator
                else:
                    logger.warning("Gompertz模型拟合失败")
            else:
                logger.warning("无法加载历史数据，Gompertz模型初始化失败")
            
            return None
            
        except Exception as e:
            logger.error(f"初始化Gompertz模型失败: {e}")
            return None
    
    def get_provincial_shares(self, target_year: int) -> Dict[str, float]:
        """获取各省份的车辆分配比例"""
        if self.gompertz_disaggregator is None:
            logger.warning("Gompertz模型未初始化，使用均匀分配")
            # 返回均匀分配（31个省份）
            provinces = [
                'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Innermongolia', 'Liaoning', 
                'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 
                'Fujian', 'Jiangxi', 'Shandong', 'Henan', 'Hubei', 'Hunan', 
                'Guangdong', 'Guangxi', 'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 
                'Yunnan', 'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
            ]
            uniform_share = 1.0 / len(provinces)
            return {province: uniform_share for province in provinces}
        
        try:
            # 确保有未来预测数据
            if self.gompertz_disaggregator.future_projections is None:
                self.gompertz_disaggregator.create_future_projections(
                    self.config.config['gompertz']['start_year'],
                    self.config.config['gompertz']['end_year']
                )
            
            # 预测车辆拥有量时间线
            if self.gompertz_disaggregator.vehicle_predictions is None:
                self.gompertz_disaggregator.predict_vehicle_ownership_timeline()
            
            # 获取省份分配比例
            shares = self.gompertz_disaggregator.calculate_provincial_shares(target_year)
            logger.info(f"成功获取 {target_year} 年各省份车辆分配比例")
            return shares
            
        except Exception as e:
            logger.error(f"获取省份分配比例失败: {e}")
            # 返回均匀分配作为后备
            provinces = [
                'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Innermongolia', 'Liaoning', 
                'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 
                'Fujian', 'Jiangxi', 'Shandong', 'Henan', 'Hubei', 'Hunan', 
                'Guangdong', 'Guangxi', 'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 
                'Yunnan', 'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
            ]
            uniform_share = 1.0 / len(provinces)
            return {province: uniform_share for province in provinces}
    
    def run_disaggregation(self, model_name: str = 'remind', departments: Optional[List[str]] = None):
        """运行负荷分解"""
        logger.info("开始运行负荷分解系统")
        
        if model_name not in self.models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model = self.models[model_name]
        output_dir = Path(self.config.config['general']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        target_year = self.config.config['general']['target_year']
        
        # 确定要处理的部门
        if departments is None:
            departments = list(self.processors.keys())
        
        for dept_name in departments:
            if dept_name not in self.processors:
                logger.warning(f"部门 {dept_name} 未启用或不存在，跳过")
                continue
            
            logger.info(f"处理部门: {dept_name}")
            
            # 1. 加载模型数据
            model_file = self.config.config['models'][model_name]['file_path']
            data = model.load_data(model_file)
            
            # 2. 过滤数据
            dept_config = self.config.config['departments'][dept_name].copy()
            dept_config['name'] = dept_name
            filtered_data = model.filter_data(data, dept_config)
            
            # 3. 提取年度能源数据
            annual_energy = model.extract_annual_energy(filtered_data, target_year, dept_config)
            
            # 4. 加载负荷曲线
            processor = self.processors[dept_name]
            profile_file = dept_config.get('profile_file')
            if profile_file and Path(profile_file).exists():
                profiles = processor.load_profiles(profile_file)
            else:
                logger.warning(f"负荷曲线文件不存在: {profile_file}")
                continue
            
            # 5. 分解负荷
            detailed_load, total_load, provincial_loads = processor.disaggregate_load(
                annual_energy, profiles, target_year, output_dir, self.get_provincial_shares(target_year)
            )
            
            # 6. 保存结果
            detailed_file = output_dir / f"{dept_name}_hourly_load_detailed.csv"
            summary_file = output_dir / f"{dept_name}_hourly_load_summary.csv"
            
            detailed_load.to_csv(detailed_file)
            total_load.to_csv(summary_file)
            
            # 7. 保存各省份负荷曲线
            if provincial_loads:
                provincial_dir = output_dir / "provincial_loads"
                provincial_dir.mkdir(exist_ok=True)
                
                for province, load_data in provincial_loads.items():
                    # 保存详细负荷
                    provincial_detailed_file = provincial_dir / f"{dept_name}_{province}_hourly_load_detailed.csv"
                    load_data.to_csv(provincial_detailed_file)
                    
                    # 保存汇总负荷
                    provincial_summary_file = provincial_dir / f"{dept_name}_{province}_hourly_load_summary.csv"
                    load_data[['total_load_mw']].to_csv(provincial_summary_file)
                
                logger.info(f"已保存 {len(provincial_loads)} 个省份的负荷曲线到: {provincial_dir}")
            
            logger.info(f"部门 {dept_name} 处理完成")
            logger.info(f"详细结果: {detailed_file}")
            logger.info(f"汇总结果: {summary_file}")
            if provincial_loads:
                logger.info(f"省份结果: {output_dir / 'provincial_loads'}")
        
        logger.info("负荷分解系统运行完成")

def main():
    """主函数"""
    # 可以指定配置文件路径，如果不指定则使用默认配置
    system = LoadDisaggregationSystem()
    
    # 运行所有启用的部门
    system.run_disaggregation()
    
    # 或者只运行特定部门
    # system.run_disaggregation(departments=['transport'])
    # system.run_disaggregation(departments=['heating'])

if __name__ == "__main__":
    main() 