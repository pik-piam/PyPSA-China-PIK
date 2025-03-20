# snakemake rules for datainput check

from os.path import join

# 全局变量定义，提供所有必要的通配符
scenario_ph = {"planning_horizons": config["scenario"]["planning_horizons"]}
# 包含除了 planning_horizons 之外的所有通配符
scenario_wildcards = {k: v for k, v in config["scenario"].items() if k != "planning_horizons"}

# 不要使用 all_parameters 规则，这会导致通配符问题
# 直接定义 plot_parameters 规则

if config["foresight"] in ["None", "overnight", "non-pathway", "myopic"]:
    rule plot_parameters:
        input:
            costs = expand("resources/data/costs/costs_{planning_horizons}.csv", **scenario_ph)
        output:
            # 使用 expand 明确指定所有通配符的值
            cost_map = expand(RESULTS_DIR + "/plots/costs/parameters_comparison.pdf", **scenario_wildcards)
        log:
            # 日志文件也需要展开通配符
            expand(LOG_DIR + "/plot_costs/parameters_comparison.log", **scenario_wildcards)
        resources:
            mem_mb = 2000
        script:
            "../scripts/plot_parameters.py"
else:
    raise NotImplementedError(
        f"Plotting fororesight {config['foresight']} not implemented"
    )
