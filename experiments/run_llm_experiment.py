import time

from llm_engine.llm_core import run_llm_gp, compute_test_fitness
from llm_engine.llm_operators import create_pset, create_llm_toolbox, parse_llm_expressions, parse_gp_expressions, \
    load_all_expressions
from utils.config_loader import generate_file_paths
from utils.openai_interface import OpenAIInterface
from utils.readAndwrite import write_json, read_json, read_jsonl

N_GENERATIONS = 30
POPULATION_SIZE = 500
FUNCTION_ID = 4
NUM_EXPERIMENTS = 1
HEIGHT_LIMIT = 6
# **🔹 解析实验时间日志文件路径**
INIT_METHOD = "gp"        # gp / llm
BASE_PATH = "../gp_llm_records"  # gp_records / llm_llm_records
LLM_PATH = "gp"           # gp / qwen /deepseek / chatgpt ....
TIME_LOG_PATH = f"{BASE_PATH}/timelogs/func{FUNCTION_ID}/experiment_time_log_func{FUNCTION_ID}.json"
llm_interface = OpenAIInterface()

def run_llm_experiment():
    """ 运行LLM GP实验 """
    experiment_times = {}

    for experiment_id in range(1, NUM_EXPERIMENTS + 1):
        print(f"\n🚀 Running Experiment {experiment_id}/{NUM_EXPERIMENTS}...")

        # **🔹 生成文件路径**
        file_paths = generate_file_paths(FUNCTION_ID, experiment_id, base_path=BASE_PATH, llm_path=LLM_PATH)

        pset = create_pset()


        # parsed_trees = load_all_expressions(file_paths["init_expressions"], pset)
        parsed_trees  = load_all_expressions(file_paths["inheritance_expressions"], pset)

        toolbox = create_llm_toolbox(init_method=INIT_METHOD, parsed_trees=parsed_trees, pset=pset)
        # **🔹 运行 GP 进化**
        experiment_start_time = time.time()

        best_individual = run_llm_gp(N_GENERATIONS, POPULATION_SIZE, toolbox, pset, file_paths, parsed_trees, llm_interface)

        # **🔹 计算测试适应度**
        compute_test_fitness(file_paths, toolbox, pset)
        experiment_end_time = time.time()

        # **🔹 记录实验耗时**
        experiment_times[f"experiment_{experiment_id}"] = experiment_end_time - experiment_start_time
        print(
            f"✅ Experiment {experiment_id} completed in {experiment_times[f'experiment_{experiment_id}']:.2f} seconds.")

    # **🔹 保存实验时间日志**
    write_json(TIME_LOG_PATH, experiment_times)
    print("\n🎉 All experiments completed. Execution times saved.")


if __name__ == "__main__":
    run_llm_experiment()
