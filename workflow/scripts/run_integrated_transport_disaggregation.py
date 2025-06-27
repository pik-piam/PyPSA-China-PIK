#!/usr/bin/env python3
"""
集成Gompertz模型的交通部门负荷分解脚本

该脚本将Gompertz车辆拥有量预测模型与负荷分解系统集成，
生成基于真实省份分配比例的各省份电动汽车小时负荷曲线。

主要功能：
1. 使用Gompertz模型计算各省份车辆分配比例
2. 从REMIND模型读取国家年度能源消耗数据
3. 将国家数据分解到各省份
4. 生成各省份的小时负荷曲线
"""

import sys
import logging
from pathlib import Path

# 添加脚本目录到路径
sys.path.append(str(Path(__file__).parent))

from load_disaggregation_system import LoadDisaggregationSystem
from gompertz_transport_disaggregator import GompertzTransportSpatialDisaggregator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("=== 开始集成Gompertz模型的交通负荷分解 ===")
    
    try:
        # 1. 初始化负荷分解系统
        logger.info("步骤1: 初始化负荷分解系统")
        config_file = "config/load_disaggregation_config.yaml"
        system = LoadDisaggregationSystem(config_file)
        
        # 2. 验证Gompertz模型状态
        logger.info("步骤2: 验证Gompertz模型状态")
        if system.gompertz_disaggregator is None:
            logger.error("Gompertz模型初始化失败，无法继续")
            return
        
        # 3. 显示Gompertz模型信息
        logger.info("步骤3: 显示Gompertz模型信息")
        target_year = system.config.config['general']['target_year']
        shares = system.get_provincial_shares(target_year)
        
        print(f"\n=== {target_year}年各省份车辆分配比例 ===")
        for province, share in sorted(shares.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {province}: {share:.4f}")
        
        # 4. 运行负荷分解
        logger.info("步骤4: 运行负荷分解")
        system.run_disaggregation(departments=['transport'])
        
        # 5. 显示结果摘要
        logger.info("步骤5: 显示结果摘要")
        output_dir = Path(system.config.config['general']['output_dir'])
        
        print(f"\n=== 输出文件 ===")
        print(f"全国详细负荷: {output_dir / 'transport_hourly_load_detailed.csv'}")
        print(f"全国汇总负荷: {output_dir / 'transport_hourly_load_summary.csv'}")
        print(f"省份负荷目录: {output_dir / 'provincial_loads'}")
        
        # 检查省份文件
        provincial_dir = output_dir / "provincial_loads"
        if provincial_dir.exists():
            provincial_files = list(provincial_dir.glob("transport_*_hourly_load_summary.csv"))
            print(f"已生成 {len(provincial_files)} 个省份的负荷曲线")
            
            # 显示前几个省份的文件
            for file in provincial_files[:5]:
                print(f"  {file.name}")
            if len(provincial_files) > 5:
                print(f"  ... 还有 {len(provincial_files) - 5} 个省份")
        
        logger.info("=== 集成Gompertz模型的交通负荷分解完成 ===")
        
    except Exception as e:
        logger.error(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 