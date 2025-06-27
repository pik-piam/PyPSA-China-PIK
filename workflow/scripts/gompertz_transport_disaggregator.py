#!/usr/bin/env python3
"""
基于Gompertz模型的交通部门空间分解器

该模块将Gompertz车辆拥有量预测模型集成到空间分解系统中，
生成从历史到2060年的车辆数量预测曲线，并计算各省份的分配比例。

主要功能：
1. 基于Gompertz函数预测车辆拥有量
2. 生成历史到2060年的预测曲线
3. 计算各省份的车辆分配比例
4. 与负荷分解系统集成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class GompertzVehicleModel:
    """Gompertz车辆拥有量预测模型"""
    
    def __init__(self, saturation_level: float = 500, alpha: float = -5.58):
        """
        初始化Gompertz模型
        
        Args:
            saturation_level: 车辆拥有量饱和水平 (每1000人车辆数)
            alpha: α参数，决定低收入水平的车辆需求
        """
        self.saturation_level = saturation_level
        self.alpha = alpha
        self.beta = None  # 需要通过拟合确定
        self.fitted = False
        
    def gompertz_function(self, pgdp: np.ndarray, beta: float) -> np.ndarray:
        """
        Gompertz函数
        
        Args:
            pgdp: 人均GDP数组
            beta: β参数
            
        Returns:
            人均车辆拥有量数组
        """
        return self.saturation_level * np.exp(self.alpha * np.exp(beta * pgdp))
    
    def fit_beta(self, pgdp_data: np.ndarray, vehicle_data: np.ndarray) -> float:
        """
        拟合β参数
        
        Args:
            pgdp_data: 人均GDP数据
            vehicle_data: 人均车辆拥有量数据
            
        Returns:
            拟合得到的β参数
        """
        def objective_function(pgdp, beta):
            return self.gompertz_function(pgdp, beta)
        
        try:
            # 使用curve_fit拟合β参数
            popt, _ = curve_fit(objective_function, pgdp_data, vehicle_data, 
                              p0=[-0.0001], bounds=([-1], [0]))
            beta = popt[0]
            logger.info(f"β参数拟合成功: {beta:.6f}")
            return beta
        except Exception as e:
            logger.warning(f"β参数拟合失败，使用默认值: {e}")
            return -0.0001  # 默认值
    
    def fit_model(self, historical_data: pd.DataFrame) -> bool:
        """
        拟合模型参数
        
        Args:
            historical_data: 包含GDP、人口、车辆数据的多年份历史数据
            
        Returns:
            拟合是否成功
        """
        try:
            # 使用所有历史数据点进行拟合
            pgdp_data = historical_data['pgdp'].values  # 人均GDP (万元/人)
            vehicle_data = historical_data['vehicle_per_capita'].values  # 每1000人车辆数
            
            logger.info(f"使用 {len(pgdp_data)} 个数据点进行拟合")
            logger.info(f"人均GDP范围: {pgdp_data.min():.2f}-{pgdp_data.max():.2f} 万元/人")
            logger.info(f"人均车辆拥有量范围: {vehicle_data.min():.1f}-{vehicle_data.max():.1f} 每1000人车辆数")
            
            if len(pgdp_data) < 10:
                logger.warning("数据点不足，无法进行可靠拟合")
                return False
            
            # 拟合β参数
            self.beta = self.fit_beta(pgdp_data, vehicle_data)
            self.fitted = True
            
            logger.info(f"模型拟合完成 - α: {self.alpha}, β: {self.beta:.6f}")
            
            # 计算拟合质量指标
            predicted = self.gompertz_function(pgdp_data, self.beta)
            residuals = vehicle_data - predicted
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            
            logger.info(f"拟合质量指标:")
            logger.info(f"  MSE: {mse:.2f}")
            logger.info(f"  RMSE: {rmse:.2f}")
            logger.info(f"  MAE: {mae:.2f}")
            logger.info(f"  平均相对误差: {np.mean(np.abs(residuals/vehicle_data))*100:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"模型拟合失败: {e}")
            return False
    
    def predict_vehicle_ownership(self, pgdp: float) -> float:
        """
        预测车辆拥有量
        
        Args:
            pgdp: 人均GDP (万元/人)
            
        Returns:
            人均车辆拥有量 (每1000人车辆数)
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit_model()")
        
        return self.gompertz_function(np.array([pgdp]), self.beta)[0]
    
    def predict_total_vehicles(self, pgdp: float, population: float) -> float:
        """
        预测总车辆数
        
        Args:
            pgdp: 人均GDP (万元/人)
            population: 人口数 (万人)
            
        Returns:
            总车辆数 (万辆)
        """
        vehicle_per_capita = self.predict_vehicle_ownership(pgdp)
        total_vehicles = vehicle_per_capita * population / 1000  # 转换为万辆
        return total_vehicles

class GompertzTransportSpatialDisaggregator:
    """基于Gompertz模型的交通部门空间分解器"""
    
    def __init__(self, config: Dict):
        """
        初始化分解器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.gompertz_model = GompertzVehicleModel(
            saturation_level=config.get('saturation_level', 500),
            alpha=config.get('alpha', -5.58)
        )
        self.historical_data = None
        self.future_projections = None
        self.vehicle_predictions = None
        
    def load_historical_data(self) -> pd.DataFrame:
        """加载历史数据"""
        logger.info("加载历史数据")
        
        try:
            # 使用绝对路径加载数据
            gdp_file = "/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/historty_GDP.csv"
            pop_file = "/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/History_POP.csv"
            car_file = "/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/History_private_car.csv"
            
            # 加载数据
            gdp_data = pd.read_csv(gdp_file, index_col=0, encoding="gbk")
            pop_data = pd.read_csv(pop_file, index_col=0, encoding="gbk")
            car_data = pd.read_csv(car_file, index_col=0, encoding="gbk")
            
            # 省份名称映射（历史数据到SSP2数据的映射）
            province_mapping = {
                'Beijing': 'Beijing',
                'Tianjin': 'Tianjin', 
                'Hebei': 'Hebei',
                'Shanxi': 'Shanxi',
                'Innermonglia': 'Innermongolia',  # SSP2数据中是Innermongolia（多一个i）
                'Liaoning': 'Liaoning',
                'Jilin': 'Jilin',
                'Heilongjiang': 'Heilongjiang',
                'Shanghai': 'Shanghai',
                'Jiangsu': 'Jiangsu',
                'Zhejiang': 'Zhejiang',
                'Anhui': 'Anhui',
                'Fujian': 'Fujian',
                'Jiangxi': 'Jiangxi',
                'Shandong': 'Shandong',
                'Henan': 'Henan',
                'Hubei': 'Hubei',
                'Hunan': 'Hunan',
                'Guangdong': 'Guangdong',
                'Guangxi': 'Guangxi',
                'Hainan': 'Hainan',
                'Chongqing': 'Chongqing',
                'Sichuan': 'Sichuan',
                'Guizhou': 'Guizhou',
                'Yunnan': 'Yunnan',
                'Tibet': 'Tibet',
                'Shaanxi': 'Shaanxi',
                'Gansu': 'Gansu',
                'Qinghai': 'Qinghai',
                'Ningxia': 'Ningxia',
                'Xinjiang': 'Xinjiang'
            }
            
            # 转换省份名称
            gdp_data.index = gdp_data.index.map(lambda x: province_mapping.get(x, x))
            pop_data.index = pop_data.index.map(lambda x: province_mapping.get(x, x))
            car_data.index = car_data.index.map(lambda x: province_mapping.get(x, x))
            
            # 获取所有年份
            years = [col for col in gdp_data.columns if col.isdigit()]
            years.sort()
            
            logger.info(f"历史数据年份范围: {min(years)}-{max(years)}")
            logger.info(f"包含 {len(years)} 个年份的数据")
            
            # 创建多年份历史数据用于拟合
            historical_fit_data = []
            
            for year in years:
                for province in gdp_data.index:
                    if province in pop_data.index and province in car_data.index:
                        try:
                            gdp = float(gdp_data.loc[province, year])  # 亿元
                            population = float(pop_data.loc[province, year])  # 万人
                            
                            # 检查车辆数据是否存在
                            if year in car_data.columns:
                                private_cars = float(car_data.loc[province, year])  # 万辆
                            else:
                                continue  # 跳过没有车辆数据的年份
                            
                            # 计算人均GDP (万元/人) 和人均车辆拥有量 (每1000人车辆数)
                            pgdp = gdp / population  # 人均GDP (万元/人)
                            vehicle_per_capita = private_cars / population * 1000  # 每1000人车辆数
                            
                            historical_fit_data.append({
                                'province': province,
                                'year': int(year),
                                'gdp': gdp,  # 亿元
                                'population': population,  # 万人
                                'private_cars': private_cars,  # 万辆
                                'pgdp': pgdp,  # 人均GDP (万元/人)
                                'vehicle_per_capita': vehicle_per_capita  # 每1000人车辆数
                            })
                        except (ValueError, KeyError) as e:
                            logger.warning(f"跳过 {province} {year} 年数据: {e}")
                            continue
            
            self.historical_data = pd.DataFrame(historical_fit_data)
            
            if self.historical_data.empty:
                logger.error("无法加载历史数据")
                return pd.DataFrame()
            
            logger.info(f"成功加载 {len(self.historical_data)} 条历史数据记录")
            logger.info(f"数据年份范围: {self.historical_data['year'].min()}-{self.historical_data['year'].max()}")
            logger.info(f"包含 {self.historical_data['province'].nunique()} 个省份")
            
            # 显示数据摘要
            print("\n=== 历史数据摘要 ===")
            print(f"数据点数量: {len(self.historical_data)}")
            print(f"年份范围: {self.historical_data['year'].min()}-{self.historical_data['year'].max()}")
            print(f"省份数量: {self.historical_data['province'].nunique()}")
            print(f"GDP范围: {self.historical_data['gdp'].min():.1f}-{self.historical_data['gdp'].max():.1f} 亿元")
            print(f"人口范围: {self.historical_data['population'].min():.1f}-{self.historical_data['population'].max():.1f} 万人")
            print(f"私家车范围: {self.historical_data['private_cars'].min():.1f}-{self.historical_data['private_cars'].max():.1f} 万辆")
            print(f"人均GDP范围: {self.historical_data['pgdp'].min():.2f}-{self.historical_data['pgdp'].max():.2f} 万元/人")
            print(f"人均车辆拥有量范围: {self.historical_data['vehicle_per_capita'].min():.1f}-{self.historical_data['vehicle_per_capita'].max():.1f} 每1000人车辆数")
            
            return self.historical_data
            
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            return pd.DataFrame()
    
    def create_future_projections(self, start_year: int = 2020, end_year: int = 2060) -> pd.DataFrame:
        """创建未来预测数据"""
        logger.info("创建未来预测数据")
        
        try:
            # 使用绝对路径加载SSP2数据
            pop_file = "/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/SSPs_POP_Prov_v2.xlsx"
            gdp_file = "/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/SSPs_GDP_Prov_v2.xlsx"
            
            # 检查文件是否存在
            if not Path(pop_file).exists():
                logger.warning(f"SSP2人口数据文件不存在: {pop_file}")
                logger.info("使用简化预测数据")
                return self._create_simplified_projections(start_year, end_year)
            
            if not Path(gdp_file).exists():
                logger.warning(f"SSP2 GDP数据文件不存在: {gdp_file}")
                logger.info("使用简化预测数据")
                return self._create_simplified_projections(start_year, end_year)
            
            # 加载SSP2数据
            pop_data = pd.read_excel(pop_file, sheet_name='SSP2', index_col=0)
            gdp_data = pd.read_excel(gdp_file, sheet_name='SSP2', index_col=0)
            
            logger.info(f"成功加载SSP2数据")
            logger.info(f"人口数据形状: {pop_data.shape}")
            logger.info(f"GDP数据形状: {gdp_data.shape}")
            
            # 确保年份列名为字符串
            pop_data.columns = pop_data.columns.astype(str)
            gdp_data.columns = gdp_data.columns.astype(str)
            
            # 获取目标年份范围
            target_years = [str(year) for year in range(start_year, end_year + 1)]
            available_years = [col for col in pop_data.columns if col.isdigit() and int(col) >= start_year and int(col) <= end_year]
            
            logger.info(f"目标年份范围: {start_year}-{end_year}")
            logger.info(f"可用年份: {available_years}")
            
            if not available_years:
                logger.warning("SSP2数据中没有目标年份范围的数据")
                return self._create_simplified_projections(start_year, end_year)
            
            # 创建未来预测数据
            future_data = []
            
            for year in available_years:
                for province in pop_data.index:
                    if province in gdp_data.index:
                        try:
                            population = float(pop_data.loc[province, year])  # 人
                            gdp = float(gdp_data.loc[province, year])  # 亿元
                            
                            # 转换单位
                            population_wan = population / 10000  # 转换为万人
                            gdp_yi = gdp  # 保持亿元单位
                            
                            future_data.append({
                                'province': province,
                                'year': int(year),
                                'population': population_wan,  # 万人
                                'gdp': gdp_yi,  # 亿元
                                'pgdp': gdp_yi / population_wan  # 人均GDP (万元/人)
                            })
                        except (ValueError, KeyError) as e:
                            logger.warning(f"跳过 {province} {year} 年数据: {e}")
                            continue
            
            self.future_projections = pd.DataFrame(future_data)
            
            if self.future_projections.empty:
                logger.warning("SSP2数据处理失败，使用简化预测数据")
                return self._create_simplified_projections(start_year, end_year)
            
            logger.info(f"成功创建 {len(self.future_projections)} 条未来预测数据")
            logger.info(f"预测年份范围: {self.future_projections['year'].min()}-{self.future_projections['year'].max()}")
            logger.info(f"包含 {self.future_projections['province'].nunique()} 个省份")
            
            # 显示数据摘要
            print("\n=== 未来预测数据摘要 ===")
            print(f"数据点数量: {len(self.future_projections)}")
            print(f"年份范围: {self.future_projections['year'].min()}-{self.future_projections['year'].max()}")
            print(f"省份数量: {self.future_projections['province'].nunique()}")
            print(f"GDP范围: {self.future_projections['gdp'].min():.1f}-{self.future_projections['gdp'].max():.1f} 亿元")
            print(f"人口范围: {self.future_projections['population'].min():.1f}-{self.future_projections['population'].max():.1f} 万人")
            print(f"人均GDP范围: {self.future_projections['pgdp'].min():.2f}-{self.future_projections['pgdp'].max():.2f} 万元/人")
            
            return self.future_projections
            
        except Exception as e:
            logger.error(f"创建未来预测数据失败: {e}")
            logger.info("使用简化预测数据")
            return self._create_simplified_projections(start_year, end_year)
    
    def _create_simplified_projections(self, start_year: int = 2020, end_year: int = 2060) -> pd.DataFrame:
        """创建简化的增长率预测（备用方法）"""
        logger.info(f"使用简化增长率预测 {start_year}-{end_year} 年")
        
        future_data = []
        years = range(start_year, end_year + 1)
        
        for _, row in self.historical_data.iterrows():
            province = row['province']
            current_gdp = row['gdp']
            current_population = row['population']
            
            # 简化的增长率预测
            # GDP年增长率：根据省份发展水平调整
            if province in ['Beijing', 'Shanghai', 'Guangdong', 'Jiangsu', 'Zhejiang']:
                gdp_growth_rate = 0.05  # 发达省份
            elif province in ['Tibet', 'Qinghai', 'Ningxia', 'Gansu']:
                gdp_growth_rate = 0.08  # 欠发达省份
            else:
                gdp_growth_rate = 0.06  # 中等发展省份
            
            # 人口年增长率：考虑人口老龄化
            if province in ['Beijing', 'Shanghai', 'Guangdong']:
                pop_growth_rate = 0.01  # 人口流入省份
            else:
                pop_growth_rate = 0.002  # 其他省份
            
            # 计算每年的值
            for year in years:
                years_ahead = year - start_year
                future_gdp = current_gdp * (1 + gdp_growth_rate) ** years_ahead
                future_population = current_population * (1 + pop_growth_rate) ** years_ahead
                
                future_data.append({
                    'province': province,
                    'year': year,
                    'gdp': future_gdp,
                    'population': future_population
                })
        
        df = pd.DataFrame(future_data)
        logger.info(f"创建了 {len(df)} 条简化预测数据")
        
        return df
    
    def predict_vehicle_ownership_timeline(self) -> pd.DataFrame:
        """预测车辆拥有量时间线"""
        logger.info("预测车辆拥有量时间线")
        
        if self.historical_data is None or self.future_projections is None:
            logger.error("请先加载历史数据和未来预测数据")
            return pd.DataFrame()
        
        # 拟合Gompertz模型
        success = self.gompertz_model.fit_model(self.historical_data)
        if not success:
            logger.error("Gompertz模型拟合失败")
            return pd.DataFrame()
        
        # 预测车辆拥有量
        predictions = []
        
        for _, row in self.future_projections.iterrows():
            province = row['province']
            year = row['year']
            future_gdp = row['gdp']
            future_population = row['population']
            
            # 计算人均GDP
            future_pgdp = future_gdp / future_population
            
            # 预测车辆拥有量
            vehicle_per_capita = self.gompertz_model.predict_vehicle_ownership(future_pgdp)
            total_vehicles = self.gompertz_model.predict_total_vehicles(future_pgdp, future_population)
            
            predictions.append({
                'province': province,
                'year': year,
                'pgdp': future_pgdp,
                'population': future_population,
                'vehicle_per_capita': vehicle_per_capita,
                'total_vehicles': total_vehicles
            })
        
        df = pd.DataFrame(predictions)
        self.vehicle_predictions = df
        
        logger.info(f"生成了 {len(df)} 条车辆拥有量预测数据")
        return df
    
    def plot_vehicle_ownership_timeline(self, output_file: str = None):
        """绘制车辆拥有量时间线图（包含历史数据对比）"""
        if self.vehicle_predictions is None:
            logger.error("请先预测车辆拥有量")
            return
        
        # 设置图形样式
        plt.style.use('default')
        
        # 获取所有省份
        provinces = self.vehicle_predictions['province'].unique()
        n_provinces = len(provinces)
        
        # 计算子图布局
        cols = 4  # 每行4个子图
        rows = (n_provinces + cols - 1) // cols  # 向上取整
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 为每个省份创建子图
        for i, province in enumerate(provinces):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # 获取该省份的预测数据
            province_data = self.vehicle_predictions[self.vehicle_predictions['province'] == province]
            
            # 获取该省份的历史数据
            if self.historical_data is not None:
                hist_data = self.historical_data[self.historical_data['province'] == province]
                if not hist_data.empty:
                    # 绘制历史数据点
                    ax.scatter(hist_data['year'], hist_data['vehicle_per_capita'], 
                             color='red', s=50, zorder=5, label='Historical', alpha=0.7)
            
            # 绘制预测数据
            ax.plot(province_data['year'], province_data['vehicle_per_capita'], 
                   color='blue', linewidth=2, marker='o', markersize=4, 
                   label='Prediction', alpha=0.8)
            
            # 设置子图属性
            ax.set_title(f'{province}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('Vehicles per 1000 people', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # 设置x轴刻度
            all_years = sorted(list(set(hist_data['year'].tolist() + province_data['year'].tolist())))
            if len(all_years) > 10:
                # 如果年份太多，只显示部分年份
                step = len(all_years) // 10
                tick_years = all_years[::step]
            else:
                tick_years = all_years
            
            ax.set_xticks(tick_years)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 隐藏多余的子图
        for i in range(n_provinces, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {output_file}")
        
        plt.show()
    
    def plot_vehicle_ownership_comparison(self, output_file: str = None):
        """绘制车辆拥有量对比图（历史vs预测）"""
        if self.vehicle_predictions is None or self.historical_data is None:
            logger.error("请先加载历史数据和预测数据")
            return
        
        # 设置图形样式
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 准备历史数据
        historical_comparison = []
        for _, row in self.historical_data.iterrows():
            province = row['province']
            hist_vehicle_per_capita = row['private_cars'] / row['population'] * 1000
            historical_comparison.append({
                'province': province,
                'year': 2020,
                'vehicle_per_capita': hist_vehicle_per_capita,
                'type': 'Historical'
            })
        
        # 准备预测数据
        prediction_comparison = []
        for _, row in self.vehicle_predictions.iterrows():
            prediction_comparison.append({
                'province': row['province'],
                'year': row['year'],
                'vehicle_per_capita': row['vehicle_per_capita'],
                'type': 'Prediction'
            })
        
        # 合并数据
        comparison_data = pd.DataFrame(historical_comparison + prediction_comparison)
        
        # 绘制总车辆数对比
        provinces = comparison_data['province'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(provinces)))
        
        for i, province in enumerate(provinces):
            province_data = comparison_data[comparison_data['province'] == province]
            hist_data = province_data[province_data['type'] == 'Historical']
            pred_data = province_data[province_data['type'] == 'Prediction']
            
            # 绘制历史点
            if not hist_data.empty:
                ax1.scatter(hist_data['year'], hist_data['vehicle_per_capita'], 
                           color=colors[i], s=100, zorder=5, alpha=0.8)
            
            # 绘制预测线
            if not pred_data.empty:
                ax1.plot(pred_data['year'], pred_data['vehicle_per_capita'], 
                        color=colors[i], linewidth=2, alpha=0.8, label=province)
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Vehicles per 1000 people', fontsize=12)
        ax1.set_title('Vehicle Ownership Prediction vs Historical Data (2020)', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 绘制拟合精度对比
        hist_2020 = comparison_data[comparison_data['type'] == 'Historical']
        pred_2020 = comparison_data[comparison_data['type'] == 'Prediction']
        pred_2020 = pred_2020[pred_2020['year'] == 2020]
        
        # 合并历史数据和2020年预测数据
        comparison_2020 = pd.merge(hist_2020, pred_2020, on='province', suffixes=('_hist', '_pred'))
        
        # 计算拟合误差
        comparison_2020['error'] = comparison_2020['vehicle_per_capita_pred'] - comparison_2020['vehicle_per_capita_hist']
        comparison_2020['error_percent'] = (comparison_2020['error'] / comparison_2020['vehicle_per_capita_hist']) * 100
        
        # 绘制拟合误差
        bars = ax2.bar(range(len(comparison_2020)), comparison_2020['error_percent'], 
                      color=['red' if x < 0 else 'green' for x in comparison_2020['error_percent']])
        
        ax2.set_xlabel('Provinces', fontsize=12)
        ax2.set_ylabel('Prediction Error (%)', fontsize=12)
        ax2.set_title('Gompertz Model Fitting Accuracy (2020)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(comparison_2020)))
        ax2.set_xticklabels(comparison_2020['province'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 添加误差数值标签
        for i, (bar, error) in enumerate(zip(bars, comparison_2020['error_percent'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if error > 0 else -1), 
                    f'{error:.1f}%', ha='center', va='bottom' if error > 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"对比图表已保存: {output_file}")
        
        plt.show()
        
        # 打印拟合统计信息
        print("\n=== Gompertz Model Fitting Statistics ===")
        print(f"Mean Absolute Error: {abs(comparison_2020['error_percent']).mean():.2f}%")
        print(f"Root Mean Square Error: {np.sqrt((comparison_2020['error_percent']**2).mean()):.2f}%")
        print(f"R-squared: {1 - (comparison_2020['error_percent']**2).sum() / ((comparison_2020['vehicle_per_capita_hist'] - comparison_2020['vehicle_per_capita_hist'].mean())**2).sum():.4f}")
        
        return comparison_2020
    
    def calculate_provincial_shares(self, target_year: int = 2030) -> Dict[str, float]:
        """计算各省份的车辆分配比例"""
        if self.vehicle_predictions is None:
            logger.error("请先预测车辆拥有量")
            return {}
        
        # 获取目标年份的数据
        year_data = self.vehicle_predictions[self.vehicle_predictions['year'] == target_year]
        
        if len(year_data) == 0:
            logger.error(f"没有找到 {target_year} 年的预测数据")
            return {}
        
        # 计算总车辆数
        total_vehicles = year_data['total_vehicles'].sum()
        
        # 计算各省份比例
        shares = {}
        for _, row in year_data.iterrows():
            province = row['province']
            share = row['total_vehicles'] / total_vehicles
            shares[province] = share
        
        logger.info(f"计算了 {len(shares)} 个省份的分配比例")
        return shares
    
    def disaggregate_national_energy(self, national_energy: pd.DataFrame, 
                                   target_year: int = 2030) -> pd.DataFrame:
        """分解国家能源消耗到各省份"""
        logger.info(f"分解国家能源消耗到各省份 (目标年份: {target_year})")
        
        # 计算各省份比例
        provincial_shares = self.calculate_provincial_shares(target_year)
        
        if not provincial_shares:
            logger.error("无法计算省份分配比例")
            return pd.DataFrame()
        
        results = []
        
        for _, row in national_energy.iterrows():
            vehicle_type = row['vehicle_type']
            national_energy_twh = row['Energy_TWh']
            
            # 为每个省份分配能源消耗
            for province, share in provincial_shares.items():
                energy_twh = national_energy_twh * share
                
                results.append({
                    'province': province,
                    'vehicle_type': vehicle_type,
                    'Energy_TWh': energy_twh,
                    'share': share,
                    'year': target_year
                })
        
        df = pd.DataFrame(results)
        logger.info(f"生成了 {len(df)} 条省份能源消耗数据")
        return df
    
    def save_predictions(self, output_dir: str = "workflow/output"):
        """保存预测结果"""
        if self.vehicle_predictions is None:
            logger.error("没有预测数据可保存")
            return
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存车辆预测数据
        vehicle_file = f"{output_dir}/gompertz_vehicle_predictions.csv"
        self.vehicle_predictions.to_csv(vehicle_file, index=False)
        logger.info(f"车辆预测数据已保存: {vehicle_file}")
        
        # 保存省份分配比例
        shares_2030 = self.calculate_provincial_shares(2030)
        shares_2050 = self.calculate_provincial_shares(2050)
        
        shares_df = pd.DataFrame({
            'province': list(shares_2030.keys()),
            'share_2030': list(shares_2030.values()),
            'share_2050': [shares_2050.get(p, 0) for p in shares_2030.keys()]
        })
        
        shares_file = f"{output_dir}/provincial_vehicle_shares.csv"
        shares_df.to_csv(shares_file, index=False)
        logger.info(f"省份分配比例已保存: {shares_file}")
        
        # 保存图表
        plot_file = f"{output_dir}/vehicle_ownership_timeline.png"
        self.plot_vehicle_ownership_timeline(plot_file)

def main():
    """主函数 - 演示Gompertz交通分解器的使用"""
    logger.info("开始Gompertz交通分解器演示")
    
    # 配置
    config = {
        'saturation_level': 500,
        'alpha': -5.58
    }
    
    # 创建分解器
    disaggregator = GompertzTransportSpatialDisaggregator(config)
    
    # 1. 加载历史数据
    historical_data = disaggregator.load_historical_data()
    
    if historical_data.empty:
        logger.error("无法加载历史数据")
        return
    
    # 2. 创建未来预测数据
    future_projections = disaggregator.create_future_projections(2020, 2060)
    
    # 3. 预测车辆拥有量时间线
    vehicle_predictions = disaggregator.predict_vehicle_ownership_timeline()
    
    # 4. 绘制时间线图
    disaggregator.plot_vehicle_ownership_timeline()
    
    # 5. 保存结果
    disaggregator.save_predictions()
    
    # 6. 演示能源分解
    # 模拟国家能源消耗数据
    national_energy = pd.DataFrame({
        'vehicle_type': ['Private car', 'Taxi', 'Official car', 'Rental car', 'Bus', 'SPV'],
        'Energy_TWh': [100, 20, 15, 10, 30, 25]
    })
    
    # 分解到各省份
    provincial_energy = disaggregator.disaggregate_national_energy(national_energy, 2030)
    
    print("\n=== 省份能源消耗分解结果 (2030年) ===")
    print(provincial_energy.head(20))
    
    logger.info("Gompertz交通分解器演示完成")

if __name__ == "__main__":
    main() 