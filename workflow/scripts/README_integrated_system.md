# 集成Gompertz模型的交通负荷分解系统

## 概述

本系统将Gompertz车辆拥有量预测模型与负荷分解系统集成，实现了从国家年度能源消耗数据到各省份小时负荷曲线的完整分解流程。

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   REMIND模型    │    │  Gompertz模型   │    │   负荷曲线      │
│   国家年度数据   │    │  省份分配比例    │    │   时间分解      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   集成系统       │
                    │  各省份小时负荷  │
                    └─────────────────┘
```

## 主要组件

### 1. Gompertz车辆拥有量预测模型
- **功能**: 基于历史GDP、人口、车辆数据拟合Gompertz函数
- **输入**: 历史数据 (2005-2024年)
- **输出**: 各省份车辆分配比例 (2020-2060年)
- **文件**: `gompertz_transport_disaggregator.py`

### 2. 负荷分解系统
- **功能**: 从IAM模型读取数据，进行时间分解
- **输入**: REMIND模型数据、负荷曲线
- **输出**: 小时级负荷曲线
- **文件**: `load_disaggregation_system.py`

### 3. 集成系统
- **功能**: 将空间分解和时间分解结合
- **输入**: 国家年度数据 + 省份分配比例
- **输出**: 各省份小时负荷曲线
- **文件**: `run_integrated_transport_disaggregation.py`

## 数据流程

### 步骤1: 历史数据拟合
```
历史GDP数据 (亿元) + 历史人口数据 (万人) + 历史车辆数据 (万辆)
    ↓
Gompertz函数拟合 (α=-5.58, β=拟合值, 饱和水平=500)
    ↓
各省份车辆拥有量预测模型
```

### 步骤2: 未来预测
```
SSP2 GDP预测 + SSP2人口预测
    ↓
各省份未来车辆拥有量预测 (2020-2060)
    ↓
各省份车辆分配比例
```

### 步骤3: 负荷分解
```
REMIND国家年度能源消耗 + 省份分配比例 + 负荷曲线
    ↓
各省份年度能源消耗
    ↓
各省份小时负荷曲线
```

## 文件结构

```
workflow/scripts/
├── config/
│   └── load_disaggregation_config.yaml    # 配置文件
├── gompertz_transport_disaggregator.py    # Gompertz模型
├── load_disaggregation_system.py          # 负荷分解系统
├── run_integrated_transport_disaggregation.py  # 集成运行脚本
└── README_integrated_system.md            # 本文档
```

## 使用方法

### 1. 准备数据文件
确保以下数据文件存在：
- `/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/historty_GDP.csv`
- `/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/History_POP.csv`
- `/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/History_private_car.csv`
- `/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/SSPs_POP_Prov_v2.xlsx`
- `/p/tmp/yanleizh/PyPSA-China-PIK/resources/data/load/SSPs_GDP_Prov_v2.xlsx`

### 2. 运行集成系统
```bash
cd /p/tmp/yanleizh/PyPSA-China-PIK/workflow/scripts
python run_integrated_transport_disaggregation.py
```

### 3. 查看结果
输出文件位于 `workflow/output/` 目录：
- `transport_hourly_load_detailed.csv` - 全国详细负荷
- `transport_hourly_load_summary.csv` - 全国汇总负荷
- `provincial_loads/` - 各省份负荷曲线目录
  - `transport_Beijing_hourly_load_detailed.csv`
  - `transport_Beijing_hourly_load_summary.csv`
  - `transport_Shanghai_hourly_load_detailed.csv`
  - `transport_Shanghai_hourly_load_summary.csv`
  - ... (31个省份)

## 配置说明

### Gompertz模型参数
```yaml
gompertz:
  enabled: true
  saturation_level: 500    # 车辆拥有量饱和水平 (每1000人车辆数)
  alpha: -5.58             # α参数
  start_year: 2020         # 预测起始年份
  end_year: 2060           # 预测结束年份
```

### 负荷分解参数
```yaml
general:
  target_year: 2030        # 目标分解年份
  weekdays_per_year: 260   # 工作日天数
  weekends_holidays_per_year: 105  # 周末和节假日天数
  ej_to_twh: 277.778       # 能量单位转换系数
```

## 技术特点

### 1. 数据驱动
- 使用真实历史数据拟合Gompertz模型
- 基于SSP2情景的未来预测数据
- 从REMIND模型读取实际能源消耗数据

### 2. 模块化设计
- 空间分解和时间分解分离
- 易于扩展新的IAM模型和部门
- 配置驱动的系统架构

### 3. 质量保证
- 完整的日志记录
- 错误处理和后备方案
- 数据验证和统计信息

## 扩展性

### 添加新的IAM模型
1. 继承 `IAMModel` 抽象类
2. 实现 `load_data()`, `filter_data()`, `extract_annual_energy()` 方法
3. 在 `_initialize_models()` 中注册新模型

### 添加新的部门
1. 继承 `DepartmentProcessor` 抽象类
2. 实现 `load_profiles()`, `disaggregate_load()` 方法
3. 在 `_initialize_processors()` 中注册新部门

### 修改空间分解方法
1. 替换 `GompertzTransportSpatialDisaggregator`
2. 实现新的空间分解器
3. 在 `_initialize_gompertz_disaggregator()` 中使用新分解器

## 故障排除

### 常见问题

1. **Gompertz模型拟合失败**
   - 检查历史数据文件是否存在
   - 验证数据格式和编码
   - 查看日志中的具体错误信息

2. **SSP2数据加载失败**
   - 确认Excel文件路径正确
   - 检查SSP2 sheet是否存在
   - 验证省份名称映射

3. **REMIND数据读取失败**
   - 检查文件路径是否正确
   - 验证文件格式和分隔符
   - 确认变量名称匹配

### 调试模式
设置日志级别为DEBUG以获取详细信息：
```python
logging.basicConfig(level=logging.DEBUG)
```

## 版本历史

- **v1.0**: 初始版本，集成Gompertz模型和负荷分解系统
- **v1.1**: 添加省份负荷曲线输出
- **v1.2**: 改进错误处理和日志记录 