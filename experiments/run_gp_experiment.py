import time
import json
import os

from gp_engine.gp_operators import create_gp_toolbox, create_pset, parse_llm_expressions
from utils.readAndwrite import write_json
from gp_engine.gp_core import run_gp
from gp_engine.gp_core import compute_test_fitness
from utils.config_loader import generate_file_paths

N_GENERATIONS = 30
POPULATION_SIZE = 500
FUNCTION_ID = 4
NUM_EXPERIMENTS = 1


# **🔹 解析实验时间日志文件路径**
INIT_METHOD = "gp"
BASE_PATH = "../gp_gp_records"
LLM_PATH = "gp"
TIME_LOG_PATH = f"{BASE_PATH}/timelogs/func{FUNCTION_ID}/experiment_time_log_func{FUNCTION_ID}.json"
HEIGHT_LIMIT = 6

# **🔹 确保路径存在**
os.makedirs(os.path.dirname(TIME_LOG_PATH), exist_ok=True)

def run_gp_experiment():
    """ 运行 GP 进化实验 """
    experiment_times = {}

    for experiment_id in range(1, NUM_EXPERIMENTS + 1):
        print(f"\n🚀 Running Experiment {experiment_id}/{NUM_EXPERIMENTS}...")

        # **🔹 生成文件路径**
        file_paths = generate_file_paths(FUNCTION_ID, experiment_id, base_path=BASE_PATH, llm_path=LLM_PATH)

        pset = create_pset()

        if INIT_METHOD == "llm":
            parsed_trees = parse_llm_expressions(file_paths["init_expressions"], pset)
        else:
            parsed_trees = None

        toolbox = create_gp_toolbox(HEIGHT_LIMIT,init_method=INIT_METHOD,parsed_trees=parsed_trees, pset=pset)
        # **🔹 运行 GP 进化**
        experiment_start_time = time.time()
        best_individual = run_gp(N_GENERATIONS, POPULATION_SIZE, toolbox, pset, file_paths)

        # **🔹 计算测试适应度**
        compute_test_fitness(file_paths, toolbox, pset)
        experiment_end_time = time.time()

        # **🔹 记录实验耗时**
        experiment_times[f"experiment_{experiment_id}"] = experiment_end_time - experiment_start_time
        print(f"✅ Experiment {experiment_id} completed in {experiment_times[f'experiment_{experiment_id}']:.2f} seconds.")

    # **🔹 保存实验时间日志**
    write_json(TIME_LOG_PATH, experiment_times)
    print("\n🎉 All experiments completed. Execution times saved.")

if __name__ == "__main__":
    run_gp_experiment()
