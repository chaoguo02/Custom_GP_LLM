import os

import yaml

from utils.readAndwrite import ensure_directory_exists, write_json

# **🔹 基础路径**
BASE_PATH = "../gp_records"

# **🔹 文件路径格式**
PATH_TEMPLATES = {
    "train_fitness_cache": f"{BASE_PATH}/caches/func{{function_id}}/train_fitness_func{{function_id}}_exp{{experiment_id}}.json",
    "test_fitness_cache": f"{BASE_PATH}/caches/func{{function_id}}/test_fitness_func{{function_id}}_exp{{experiment_id}}.json",
    "results": f"{BASE_PATH}/results/func{{function_id}}/holdout_func{{function_id}}_exp{{experiment_id}}.jsonl",
    "first_generation_cache": f"{BASE_PATH}/records/func{{function_id}}/first_generation_func{{function_id}}_exp{{experiment_id}}.json",
    "experiment_time_log": f"{BASE_PATH}/timelogs/func{{function_id}}/experiment_time_log_func{{function_id}}.json",
    "train_data": "../datasets/fitness_cases{function_id}.csv",
    "test_data": "../datasets/hold_out{function_id}.csv",
}

def load_config(config_path):
    """ 加载 YAML 配置文件 """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 配置文件未找到: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"❌ YAML 解析错误: {e}")
    except Exception as e:
        raise RuntimeError(f"❌ 加载配置文件时发生未知错误: {e}")


def generate_file_paths(function_id, experiment_id):
    """ 生成 GP 相关文件路径，并确保路径存在 """

    # **🔹 格式化所有路径**
    paths = {key: value.format(function_id=function_id, experiment_id=experiment_id) for key, value in
             PATH_TEMPLATES.items()}

    # **🔹 确保路径存在**
    for path in paths.values():
        ensure_directory_exists(path)
        if not os.path.exists(path):
            write_json(path, {} if path.endswith(".json") else [])

    # **🔹 调试输出**
    print("\n🔹 Generated File Paths:")
    for key, value in paths.items():
        print(f"{key}: {value}")

    return paths