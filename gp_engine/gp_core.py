import operator
import random
import time
import json
import os
import numpy as np
import deap.gp as gp
import deap.base as base
import deap.creator as creator
import deap.tools as tools
import pandas as pd

from gp_engine.gp_operators import create_gp_toolbox
from utils.config_loader import generate_file_paths
from utils.data_loader import load_data
from utils.evaluation import evalSymbReg
from utils.readAndwrite import read_json, write_json, write_jsonl


def run_gp(n_gen, pop_size, function_id, experiment_id, toolbox, pset):
    start_time = time.time()
    HEIGHT_LIMIT = 6  # 限制最大树高
    ELITISM_RATE = 0.01
    elite_size = max(1, int(pop_size * ELITISM_RATE))
    # 生成文件路径
    file_paths = generate_file_paths(function_id, experiment_id)
    # 创建GP算法工具箱
    # 生成初始种群
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # 记录最优个体
    # 加载数据
    X_train, y_train, X_test, y_test = load_data(file_paths)

    # **Step 0: 加载训练适应度缓存**
    cache_train_fitness = read_json(file_paths["train_fitness_cache"])

    first_generation_saved = False  # 确保第一代只存储一次
    results_data = []


    for gen in range(n_gen):
        generation_data = []

        for ind in pop:
            expression = str(ind)
            print(f"Expression: {expression}")

            if expression in cache_train_fitness:
                train_fitness = cache_train_fitness[expression]
            else:
                train_fitness, _ = evalSymbReg(ind, pset, X_train, y_train, X_test, y_test)  # 计算适应度
                cache_train_fitness[expression] = train_fitness

            ind.fitness.values = (train_fitness,)

        # **Step 1.5: 记录第一代种群（仅存 expression）**
        if gen == 0 and not first_generation_saved:
            first_generation_data = [str(ind) for ind in pop]
            write_json(file_paths["first_generation_cache"], first_generation_data)
            first_generation_saved = True
            print("第一代种群表达式已存储！")

        # 记录当前代的适应度信息
        for ind in pop:
            generation_data.append({
                "generation": gen,
                "expression": str(ind),
                "train_fitness": ind.fitness.values[0]
            })

        # **按 `train_fitness` 排序**
        generation_data.sort(key=lambda x: x["train_fitness"])

        # **写入 JSONL 文件**
        results_data.extend(generation_data)

        print(f"Generation {gen} logged.")

        # **Step 3: 进行选择、交叉和变异**
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # **交叉**
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                child1, child2 = toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        # **变异**
        for mutant in offspring:
            if random.random() < 0.2:
                mutant, = toolbox.mutate(mutant)
                del mutant.fitness.values

        # **Step 3.5: 限制树高**(超限个体用父代替换)
        valid_offspring = []
        for i, ind in enumerate(offspring):
            if ind.height <= HEIGHT_LIMIT:
                valid_offspring.append(ind)
            else:
                valid_offspring.append(pop[i])  # 以父代替换超高个体
        offspring = valid_offspring

        elites = tools.selBest(pop, elite_size)
        remaining_size = max(0, pop_size - elite_size)
        offspring = tools.selBest(offspring, min(len(offspring), remaining_size))

        pop[:] = elites + offspring  # **更新种群**
        hof.update(pop)  # **确保最优个体被记录**

    # **Step 4: 保存训练适应度缓存**
    write_json(file_paths["train_fitness_cache"], cache_train_fitness)
    write_jsonl(file_paths["results"], results_data)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("\nBest Individual:", hof[0] if len(hof) > 0 else "None")

    return hof[0] if len(hof) > 0 else None



# **计算测试适应度**
def compute_test_fitness(file_paths, toolbox, pset):
    start_time = time.time()
    X_train, y_train, X_test, y_test = load_data(file_paths)

    # **Step 0: 加载测试适应度缓存**
    cache_test_fitness = read_json(file_paths["test_fitness_cache"])

    updated_data = []
    jsonl_data = []
    with open(file_paths["results"], "r") as f:
        for line in f:
            entry = json.loads(line)
            expression = entry["expression"]

            if expression in cache_test_fitness:
                test_fitness = cache_test_fitness[expression]
            else:
                try:
                    tree = gp.PrimitiveTree.from_string(expression, pset)
                    _, test_fitness = toolbox.evaluate(tree, pset, X_train, y_train, X_test, y_test)
                except Exception as e:
                    print(f"❌ Error processing expression {expression}: {e}")
                    test_fitness = float("inf")  # 处理异常情况

                cache_test_fitness[expression] = test_fitness

            entry["test_fitness"] = test_fitness
            updated_data.append(entry)
            jsonl_data.append(entry)

    # **Step 4: 重新写回 JSONL 文件**
    write_jsonl(file_paths["results"], jsonl_data)

    # **Step 5: 保存测试适应度缓存**
    write_json(file_paths["test_fitness_cache"], cache_test_fitness)

    print(f"Test fitness computed in {time.time() - start_time:.2f} seconds")

