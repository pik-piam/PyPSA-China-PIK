#!/usr/bin/env python3
"""
供热部门负荷分解脚本

使用通用负荷分解系统专门处理供热部门的负荷分解。
这是一个示例脚本，展示如何为新的部门创建专用处理器。
"""

import sys
from pathlib import Path

# 添加脚本目录到路径，以便导入通用系统
sys.path.append(str(Path(__file__).parent))

from load_disaggregation_system import LoadDisaggregationSystem
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """运行供热部门负荷分解"""
    logger.info("=== 开始供热部门负荷分解 ===")
    
    # 使用配置文件初始化系统
    config_file = "config/load_disaggregation_config.yaml"
    system = LoadDisaggregationSystem(config_file)
    
    # 只运行供热部门
    system.run_disaggregation(departments=['heating'])
    
    logger.info("=== 供热部门负荷分解完成 ===")
    logger.info("输出文件:")
    logger.info("- 详细负荷: workflow/output/heating_hourly_load_detailed.csv")
    logger.info("- 汇总负荷: workflow/output/heating_hourly_load_summary.csv")

if __name__ == "__main__":
    main() 