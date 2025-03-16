import time
import json
import os

from gp_engine.gp_operators import create_gp_toolbox
from utils.readAndwrite import write_json
from gp_engine.gp_core import run_gp
from gp_engine.gp_core import compute_test_fitness
from utils.config_loader import generate_file_paths

N_GENERATIONS = 10
POPULATION_SIZE = 50
FUNCTION_ID = 4
NUM_EXPERIMENTS = 1

# **🔹 解析实验时间日志文件路径**
BASE_PATH = "../gp_records"
TIME_LOG_PATH = f"{BASE_PATH}/timelogs/func{FUNCTION_ID}/experiment_time_log_func{FUNCTION_ID}.json"

# **🔹 确保路径存在**
os.makedirs(os.path.dirname(TIME_LOG_PATH), exist_ok=True)

def run_experiment():
    """ 运行 GP 进化实验 """
    HEIGHT_LIMIT = 6
    toolbox, pset = create_gp_toolbox(HEIGHT_LIMIT)
    experiment_times = {}

    for experiment_id in range(1, NUM_EXPERIMENTS + 1):
        print(f"\n🚀 Running Experiment {experiment_id}/{NUM_EXPERIMENTS}...")

        # **🔹 生成文件路径**
        file_paths = generate_file_paths(FUNCTION_ID, experiment_id)

        # **🔹 运行 GP 进化**
        experiment_start_time = time.time()
        best_individual = run_gp(N_GENERATIONS, POPULATION_SIZE, FUNCTION_ID, experiment_id, toolbox, pset)

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
    run_experiment()
