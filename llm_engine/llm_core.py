import json
import random
import time

from deap import tools, gp

from utils.data_loader import load_data
from utils.evaluation import evalSymbReg
from utils.readAndwrite import read_json, write_json, write_jsonl


def run_llm_gp(n_gen, pop_size, toolbox, pset, file_paths, parsed_trees, llm_interface):
    start_time = time.time()
    HEIGHT_LIMIT = 6  # 限制最大树高
    ELITISM_RATE = 0.01
    elite_size = max(1, int(pop_size * ELITISM_RATE))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # 记录最优个体
    # 加载数据
    X_train, y_train, X_test, y_test = load_data(file_paths)

    # **Step 0: 加载训练适应度缓存**
    cache_train_fitness = read_json(file_paths["train_fitness_cache"])

    results_data = []

    for gen in range(n_gen):
        generation_data = []
        cnt = 0
        for ind in pop:
            expression = str(ind)
            print(f"Expression: {expression}")

            if expression in cache_train_fitness:
                train_fitness = cache_train_fitness[expression]
            else:
                train_fitness, _ = evalSymbReg(ind, pset, X_train, y_train, X_test, y_test)  # 计算适应度
                cache_train_fitness[expression] = train_fitness

            ind.fitness.values = (train_fitness,)

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
        write_jsonl(file_paths["results"], results_data)
        write_json(file_paths["train_fitness_cache"], cache_train_fitness)
        print(f"Generation {gen} logged.")

        # **Step 3: 进行选择、交叉和变异**
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        '''
        # **交叉**
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                child1, child2 = toolbox.mate(child1, child2, parsed_trees=parsed_trees, llm_interface=llm_interface)
                del child1.fitness.values, child2.fitness.values

        # **变异**
        for mutant in offspring:
            if random.random() < 0.2:
                mutant, = toolbox.mutate(mutant, llm_interface=llm_interface)
                del mutant.fitness.values

        # **Step 3.5: 限制树高**(超限个体用父代替换)
        valid_offspring = []
        for i, ind in enumerate(offspring):
            if ind.height <= HEIGHT_LIMIT:
                valid_offspring.append(ind)
            else:
                cnt = cnt + 1
                valid_offspring.append(pop[i])  # 以父代替换超高个体
        offspring = valid_offspring
        '''
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.8:
                offspring[i], offspring[i + 1] = toolbox.mate(offspring[i], offspring[i + 1], parsed_trees=parsed_trees, llm_interface=llm_interface, pset=pset)
                del offspring[i].fitness.values, offspring[i + 1].fitness.values

        for i in range(len(offspring)):  # 直接索引 `offspring`
            if random.random() < 0.2:
                offspring[i], = toolbox.mutate(offspring[i], llm_interface=llm_interface)  # 变异
                del offspring[i].fitness.values  # 清除适应度，以便重新计算

        offspring[:] = [ind if ind.height <= HEIGHT_LIMIT else pop[i] for i, ind in enumerate(offspring)]

        # elites = tools.selBest(pop, elite_size)
        # remaining_size = max(0, pop_size - elite_size)
        # offspring = tools.selTournament(offspring, remaining_size, 3)

        # pop[:] = elites + offspring  # **更新种群**
        pop[:] = offspring
        # hof.update(pop)  # **确保最优个体被记录**
        print(f"超过树高的次数：{cnt}")
    # **Step 4: 保存训练适应度缓存**
    # write_json(file_paths["train_fitness_cache"], cache_train_fitness)
    # write_jsonl(file_paths["results"], results_data)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("\nBest Individual:", hof[0] if len(hof) > 0 else "None")

    return hof[0] if len(hof) > 0 else None

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