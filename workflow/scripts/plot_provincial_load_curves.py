#!/usr/bin/env python3
"""
绘制各省份负荷曲线总图

该脚本读取各省份的小时负荷曲线数据，绘制一个总图，
每个子图显示一个省份的不同类型汽车累加的负荷曲线。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProvincialLoadPlotter:
    """省份负荷曲线绘图器"""
    
    def __init__(self, output_dir: str = "../../workflow/output"):
        self.output_dir = Path(output_dir)
        self.provincial_dir = self.output_dir / "provincial_loads"
        self.provinces = [
            'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Innermongolia', 'Liaoning', 
            'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 
            'Fujian', 'Jiangxi', 'Shandong', 'Henan', 'Hubei', 'Hunan', 
            'Guangdong', 'Guangxi', 'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 
            'Yunnan', 'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
        ]
        self.vehicle_types = ['Private car',
                            #    'Taxi', 'Official car', 'Rental car', 
                               'Bus', 'SPV']
        
    def load_provincial_data(self) -> Dict[str, pd.DataFrame]:
        """加载各省份的负荷数据"""
        logger.info("加载各省份负荷数据")
        
        provincial_data = {}
        
        for province in self.provinces:
            detailed_file = self.provincial_dir / f"transport_{province}_hourly_load_detailed.csv"
            
            if detailed_file.exists():
                try:
                    data = pd.read_csv(detailed_file, index_col=0, parse_dates=True)
                    provincial_data[province] = data
                    logger.info(f"成功加载 {province} 数据: {data.shape}")
                except Exception as e:
                    logger.warning(f"加载 {province} 数据失败: {e}")
            else:
                logger.warning(f"文件不存在: {detailed_file}")
        
        return provincial_data
    
    def plot_provincial_load_curves(self, provincial_data: Dict[str, pd.DataFrame], 
                                  output_file: str = None, 
                                  sample_days: int = 1) -> None:
        """Plot provincial load curves with weekday and weekend separation (all in English)"""
        logger.info("Plotting provincial load curves (weekday & weekend, English labels)")
        
        if not provincial_data:
            logger.error("No provincial data to plot")
            return
        
        # Calculate subplot layout
        n_provinces = len(provincial_data)
        cols = 5  # 5 subplots per row
        rows = (n_provinces + cols - 1) // cols  # Round up
        
        # Create 2*rows rows, cols columns subplots
        fig, axes = plt.subplots(2 * rows, cols, figsize=(25, 10 * rows))
        
        # Set color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.vehicle_types)))
        color_map = dict(zip(self.vehicle_types, colors))
        
        # Process each province's data
        for idx, (province, data) in enumerate(provincial_data.items()):
            if idx >= n_provinces:
                break
                
            # Ensure index is datetime format
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Select sample data (first sample_days days)
            sample_hours = sample_days * 24
            sample_data = data.head(sample_hours)
            
            # 选取第一个有数据的工作日
            weekday_days = data[data.index.weekday < 5].index.normalize().unique()
            if len(weekday_days) > 0:
                first_weekday = weekday_days[0]
                weekday_data = data[data.index.normalize() == first_weekday]
            else:
                weekday_data = pd.DataFrame()

            # 选取第一个有数据的周末
            weekend_days = data[data.index.weekday >= 5].index.normalize().unique()
            if len(weekend_days) > 0:
                first_weekend = weekend_days[0]
                weekend_data = data[data.index.normalize() == first_weekend]
            else:
                weekend_data = pd.DataFrame()
            
            # Get subplot position
            row = idx // cols
            col = idx % cols
            
            # Weekday subplot
            ax_weekday = axes[row, col]
            if not weekday_data.empty:
                for v_type in self.vehicle_types:
                    if v_type in weekday_data.columns:
                        ax_weekday.plot(weekday_data.index, weekday_data[v_type], 
                                       label=v_type, color=color_map[v_type], 
                                       linewidth=1.5, alpha=0.8)
                
                total_weekday = weekday_data[self.vehicle_types].sum(axis=1)
                ax_weekday.plot(weekday_data.index, total_weekday, 
                               label='Total', color='black', linewidth=2.5, alpha=0.9)
                
                # Add weekday statistics
                max_load = total_weekday.max()
                avg_load = total_weekday.mean()
                ax_weekday.text(0.02, 0.98, f'Max: {max_load:.1f} MW\nAvg: {avg_load:.1f} MW', 
                               transform=ax_weekday.transAxes, fontsize=8, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                logger.warning(f"No weekday data for {province}")
                ax_weekday.text(0.5, 0.5, "No weekday data", ha='center', va='center', 
                               fontsize=12, color='gray')
            
            # Set weekday subplot properties
            ax_weekday.set_title(f'{province} (Weekday)', fontsize=12, fontweight='bold')
            ax_weekday.set_xlabel('Time', fontsize=10)
            ax_weekday.set_ylabel('Load (MW)', fontsize=10)
            ax_weekday.grid(True, alpha=0.3)
            ax_weekday.legend(fontsize=8, loc='upper center', ncol=len(self.vehicle_types) + 1, bbox_to_anchor=(0.5, 1.15))
            ax_weekday.tick_params(axis='both', which='major', labelsize=8)
            plt.setp(ax_weekday.get_xticklabels(), rotation=45, ha='right')
            
            # Weekend subplot
            ax_weekend = axes[row + rows, col]
            if not weekend_data.empty:
                for v_type in self.vehicle_types:
                    if v_type in weekend_data.columns:
                        ax_weekend.plot(weekend_data.index, weekend_data[v_type], 
                                       label=v_type, color=color_map[v_type], 
                                       linewidth=1.5, alpha=0.8)
                
                total_weekend = weekend_data[self.vehicle_types].sum(axis=1)
                ax_weekend.plot(weekend_data.index, total_weekend, 
                               label='Total', color='black', linewidth=2.5, alpha=0.9)
                
                # Add weekend statistics
                max_load = total_weekend.max()
                avg_load = total_weekend.mean()
                ax_weekend.text(0.02, 0.98, f'Max: {max_load:.1f} MW\nAvg: {avg_load:.1f} MW', 
                               transform=ax_weekend.transAxes, fontsize=8, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                logger.warning(f"No weekend data for {province}")
                ax_weekend.text(0.5, 0.5, "No weekend data", ha='center', va='center', 
                               fontsize=12, color='gray')
            
            # Set weekend subplot properties
            ax_weekend.set_title(f'{province} (Weekend)', fontsize=12, fontweight='bold')
            ax_weekend.set_xlabel('Time', fontsize=10)
            ax_weekend.set_ylabel('Load (MW)', fontsize=10)
            ax_weekend.grid(True, alpha=0.3)
            ax_weekend.legend(fontsize=8, loc='upper center', ncol=len(self.vehicle_types) + 1, bbox_to_anchor=(0.5, 1.15))
            ax_weekend.tick_params(axis='both', which='major', labelsize=8)
            plt.setp(ax_weekend.get_xticklabels(), rotation=45, ha='right')
        
        # Hide unused subplots
        for i in range(n_provinces, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
            axes[row + rows, col].set_visible(False)
        
        # Set main title
        fig.suptitle('Provincial Load Curves Comparison (Weekday vs Weekend)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved: {output_file}")
        
        plt.show()
    
    def plot_provincial_comparison(self, provincial_data: Dict[str, pd.DataFrame], 
                                 output_file: str = None) -> None:
        """Plot provincial load comparison (all in English)"""
        logger.info("Plotting provincial load comparison")
        
        if not provincial_data:
            logger.error("No provincial data to plot")
            return
        
        # Set figure style
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Calculate statistics for each province
        province_stats = []
        for province, data in provincial_data.items():
            total_load = data[self.vehicle_types].sum(axis=1)
            province_stats.append({
                'province': province,
                'max_load': total_load.max(),
                'avg_load': total_load.mean(),
                'total_energy': total_load.sum() / 1000  # GWh
            })
        
        stats_df = pd.DataFrame(province_stats)
        stats_df = stats_df.sort_values('max_load', ascending=False)
        
        # Plot maximum load comparison
        bars1 = ax1.bar(range(len(stats_df)), stats_df['max_load'], 
                       color='skyblue', alpha=0.8, label='Maximum Load')
        
        ax1.set_xlabel('Provinces', fontsize=12)
        ax1.set_ylabel('Maximum Load (MW)', fontsize=12)
        ax1.set_title('Provincial Maximum Load Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(stats_df)))
        ax1.set_xticklabels(stats_df['province'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, stats_df['max_load'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot average load comparison
        bars2 = ax2.bar(range(len(stats_df)), stats_df['avg_load'], 
                       color='lightcoral', alpha=0.8, label='Average Load')
        
        ax2.set_xlabel('Provinces', fontsize=12)
        ax2.set_ylabel('Average Load (MW)', fontsize=12)
        ax2.set_title('Provincial Average Load Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(stats_df)))
        ax2.set_xticklabels(stats_df['province'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, stats_df['avg_load'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Provincial comparison figure saved: {output_file}")
        
        plt.show()
        
        # Print statistical summary
        print("\n=== Provincial Load Statistics Summary ===")
        print(f"Total provinces: {len(stats_df)}")
        print(f"Maximum load range: {stats_df['max_load'].min():.1f} - {stats_df['max_load'].max():.1f} MW")
        print(f"Average load range: {stats_df['avg_load'].min():.1f} - {stats_df['avg_load'].max():.1f} MW")
        print(f"Total energy range: {stats_df['total_energy'].min():.1f} - {stats_df['total_energy'].max():.1f} GWh")
        
        print("\n=== Top 10 Provinces by Maximum Load ===")
        print(stats_df.head(10)[['province', 'max_load', 'avg_load', 'total_energy']].to_string(index=False))
        
        return stats_df
    
    def plot_weekday_weekend_comparison(self, provincial_data: Dict[str, pd.DataFrame], 
                                       output_file: str = None) -> None:
        """Plot weekday vs weekend load comparison analysis (all in English)"""
        logger.info("Plotting weekday vs weekend load comparison analysis")
        
        if not provincial_data:
            logger.error("No provincial data to plot")
            return
        
        # Set figure style
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Calculate weekday and weekend statistics for each province
        comparison_data = []
        for province, data in provincial_data.items():
            # Ensure index is datetime format
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Separate weekday and weekend data
            weekday_data = data[data.index.weekday < 5]  # Monday to Friday
            weekend_data = data[data.index.weekday >= 5]  # Saturday and Sunday
            
            if not weekday_data.empty and not weekend_data.empty:
                total_weekday = weekday_data[self.vehicle_types].sum(axis=1)
                total_weekend = weekend_data[self.vehicle_types].sum(axis=1)
                
                comparison_data.append({
                    'province': province,
                    'weekday_max': total_weekday.max(),
                    'weekday_avg': total_weekday.mean(),
                    'weekend_max': total_weekend.max(),
                    'weekend_avg': total_weekend.mean(),
                    'weekday_energy': total_weekday.sum() / 1000,  # GWh
                    'weekend_energy': total_weekend.sum() / 1000,  # GWh
                    'max_ratio': total_weekday.max() / total_weekend.max() if total_weekend.max() > 0 else 0,
                    'avg_ratio': total_weekday.mean() / total_weekend.mean() if total_weekend.mean() > 0 else 0
                })
        
        comp_df = pd.DataFrame(comparison_data)
        comp_df = comp_df.sort_values('weekday_max', ascending=False)
        
        # 1. Weekday vs Weekend Maximum Load Comparison
        x_pos = np.arange(len(comp_df))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, comp_df['weekday_max'], width, 
                       label='Weekday', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, comp_df['weekend_max'], width, 
                       label='Weekend', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Provinces', fontsize=12)
        ax1.set_ylabel('Maximum Load (MW)', fontsize=12)
        ax1.set_title('Weekday vs Weekend Maximum Load Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(comp_df['province'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Weekday vs Weekend Average Load Comparison
        bars3 = ax2.bar(x_pos - width/2, comp_df['weekday_avg'], width, 
                       label='Weekday', color='skyblue', alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, comp_df['weekend_avg'], width, 
                       label='Weekend', color='lightcoral', alpha=0.8)
        
        ax2.set_xlabel('Provinces', fontsize=12)
        ax2.set_ylabel('Average Load (MW)', fontsize=12)
        ax2.set_title('Weekday vs Weekend Average Load Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(comp_df['province'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Maximum Load Ratio Distribution
        ax3.hist(comp_df['max_ratio'], bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.axvline(comp_df['max_ratio'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {comp_df["max_ratio"].mean():.2f}')
        ax3.set_xlabel('Weekday/Weekend Maximum Load Ratio', fontsize=12)
        ax3.set_ylabel('Number of Provinces', fontsize=12)
        ax3.set_title('Weekday vs Weekend Maximum Load Ratio Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Average Load Ratio Distribution
        ax4.hist(comp_df['avg_ratio'], bins=15, color='lightblue', alpha=0.7, edgecolor='black')
        ax4.axvline(comp_df['avg_ratio'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {comp_df["avg_ratio"].mean():.2f}')
        ax4.set_xlabel('Weekday/Weekend Average Load Ratio', fontsize=12)
        ax4.set_ylabel('Number of Provinces', fontsize=12)
        ax4.set_title('Weekday vs Weekend Average Load Ratio Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Weekday vs weekend comparison figure saved: {output_file}")
        
        plt.show()
        
        # Print statistical summary
        print("\n=== Weekday vs Weekend Load Comparison Summary ===")
        print(f"Total provinces: {len(comp_df)}")
        print(f"Weekday maximum load range: {comp_df['weekday_max'].min():.1f} - {comp_df['weekday_max'].max():.1f} MW")
        print(f"Weekend maximum load range: {comp_df['weekend_max'].min():.1f} - {comp_df['weekend_max'].max():.1f} MW")
        print(f"Weekday average load range: {comp_df['weekday_avg'].min():.1f} - {comp_df['weekday_avg'].max():.1f} MW")
        print(f"Weekend average load range: {comp_df['weekend_avg'].min():.1f} - {comp_df['weekend_avg'].max():.1f} MW")
        print(f"Maximum load ratio range: {comp_df['max_ratio'].min():.2f} - {comp_df['max_ratio'].max():.2f}")
        print(f"Average load ratio range: {comp_df['avg_ratio'].min():.2f} - {comp_df['avg_ratio'].max():.2f}")
        
        print("\n=== Top 10 Provinces by Weekday Load ===")
        print(comp_df.head(10)[['province', 'weekday_max', 'weekday_avg', 'max_ratio']].to_string(index=False))
        
        return comp_df
    
    def plot_vehicle_type_composition(self, provincial_data: Dict[str, pd.DataFrame], 
                                    output_file: str = None) -> None:
        """Plot vehicle type composition analysis (all in English)"""
        logger.info("Plotting vehicle type composition analysis")
        
        if not provincial_data:
            logger.error("No provincial data to plot")
            return
        
        # Calculate total energy for each vehicle type in each province
        composition_data = []
        for province, data in provincial_data.items():
            for v_type in self.vehicle_types:
                if v_type in data.columns:
                    total_energy = data[v_type].sum() / 1000  # GWh
                    composition_data.append({
                        'province': province,
                        'vehicle_type': v_type,
                        'total_energy': total_energy
                    })
        
        comp_df = pd.DataFrame(composition_data)
        
        # Set figure style
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot vehicle type composition by province
        pivot_df = comp_df.pivot(index='province', columns='vehicle_type', values='total_energy')
        pivot_df = pivot_df.fillna(0)
        
        # Select top 15 provinces for visualization
        top_provinces = pivot_df.sum(axis=1).sort_values(ascending=False).head(15).index
        top_pivot_df = pivot_df.loc[top_provinces]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.vehicle_types)))
        bottom = np.zeros(len(top_provinces))
        
        for i, v_type in enumerate(self.vehicle_types):
            if v_type in top_pivot_df.columns:
                values = top_pivot_df[v_type].values
                ax1.bar(range(len(top_provinces)), values, bottom=bottom, 
                       label=v_type, color=colors[i], alpha=0.8)
                bottom += values
        
        ax1.set_xlabel('Provinces', fontsize=12)
        ax1.set_ylabel('Total Energy (GWh)', fontsize=12)
        ax1.set_title('Vehicle Type Composition by Province (Top 15)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(top_provinces)))
        ax1.set_xticklabels(top_provinces, rotation=45, ha='right')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot national vehicle type composition
        national_composition = comp_df.groupby('vehicle_type')['total_energy'].sum()
        
        ax2.pie(national_composition.values, labels=national_composition.index, 
               autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('National Vehicle Type Composition', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Vehicle type composition figure saved: {output_file}")
        
        plt.show()
        
        # Print national composition statistics
        print("\n=== National Vehicle Type Composition ===")
        total_national = national_composition.sum()
        for v_type, energy in national_composition.items():
            percentage = (energy / total_national) * 100
            print(f"{v_type}: {energy:.1f} GWh ({percentage:.1f}%)")

def main():
    """Main function"""
    logger.info("=== Starting Provincial Load Curves Plotting ===")
    
    try:
        # Initialize plotter
        plotter = ProvincialLoadPlotter()
        
        # Load provincial data
        provincial_data = plotter.load_provincial_data()
        
        if not provincial_data:
            logger.error("No provincial load data found. Please run the integrated disaggregation system first.")
            return
        
        logger.info(f"Successfully loaded data for {len(provincial_data)} provinces")
        
        # 1. Plot provincial load curves (weekday and weekend)
        logger.info("Step 1: Plotting provincial load curves (weekday and weekend)")
        plotter.plot_provincial_load_curves(
            provincial_data, 
            output_file=plotter.output_dir / "provincial_load_curves_overview.png"
        )
        
        # 2. Plot weekday vs weekend load comparison analysis
        logger.info("Step 2: Plotting weekday vs weekend load comparison analysis")
        weekday_weekend_df = plotter.plot_weekday_weekend_comparison(
            provincial_data,
            output_file=plotter.output_dir / "weekday_weekend_comparison.png"
        )
        
        # 3. Plot provincial load comparison
        logger.info("Step 3: Plotting provincial load comparison")
        stats_df = plotter.plot_provincial_comparison(
            provincial_data,
            output_file=plotter.output_dir / "provincial_load_comparison.png"
        )
        
        # 4. Plot vehicle type composition analysis
        logger.info("Step 4: Plotting vehicle type composition analysis")
        plotter.plot_vehicle_type_composition(
            provincial_data,
            output_file=plotter.output_dir / "vehicle_type_composition.png"
        )
        
        logger.info("=== Provincial Load Curves Plotting Completed ===")
        logger.info(f"Output files:")
        logger.info(f"- Provincial load curves overview (weekday & weekend): {plotter.output_dir / 'provincial_load_curves_overview.png'}")
        logger.info(f"- Weekday vs weekend load comparison: {plotter.output_dir / 'weekday_weekend_comparison.png'}")
        logger.info(f"- Provincial load comparison: {plotter.output_dir / 'provincial_load_comparison.png'}")
        logger.info(f"- Vehicle type composition analysis: {plotter.output_dir / 'vehicle_type_composition.png'}")
        
    except Exception as e:
        logger.error(f"Error occurred during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 