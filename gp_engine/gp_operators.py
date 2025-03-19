import operator
import random
import numpy as np
import deap.gp as gp
import deap.base as base
import deap.creator as creator
import deap.tools as tools

from utils.convert_tree2expression import expression_to_tree
from utils.evaluation import evalSymbReg
from utils.readAndwrite import read_jsonl


# **保护性操作**
def protect_sqrt(x):
    return np.sqrt(x) if x >= 0 else 0

def protect_div(x, y):
    return x / y if y != 0 else 1

def square(x):
    return x ** 2

# 将llm生成的表达式转化为parsed_trees
def parse_llm_expressions(jsonl_file, pset):
    parsed_trees = []
    expressions = read_jsonl(jsonl_file)  # 读取 JSONL 文件

    if not expressions:
        print(f"⚠️ Warning: No expressions found in {jsonl_file}")
        return parsed_trees

    for entry in expressions:
        expr = entry.get("expression", "")
        try:
            parsed_expr = expression_to_tree(expr)
            tree = gp.PrimitiveTree.from_string(parsed_expr, pset)
            parsed_trees.append(tree)
        except Exception as e:
            print(f"❌ Error parsing expression {expr}: {e}")

    print(f"✅ Loaded {len(parsed_trees)} expressions from {jsonl_file}.")
    return parsed_trees

def initIndividual(parsed_trees):
    return creator.Individual(random.choice(parsed_trees))

def cxOnePointListOfTrees(ind1, ind2):
    HEIGHT_LIMIT = 6
    dec = gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT)
    print(f"ind 1: {ind1}, ind 2: {ind2}")
    # print(f"type(ind1): {type(ind1)}, type(ind2): {type(ind2)}")
    ind1, ind2 = dec(gp.cxOnePoint)(ind1, ind2)
    # print(f"type(ind1): {type(ind1)}, type(ind2): {type(ind2)}")
    return ind1, ind2

def mutUniformListOfTrees(individual, pset):
    HEIGHT_LIMIT = 6
    dec = gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT)
    expr = lambda pset, type_: gp.genFull(pset=pset, min_=1, max_=2)
    individual, = dec(gp.mutUniform)(individual, expr=expr, pset=pset)
    return (individual,)

def create_pset():
    # 定义GP语法树
    pset = gp.PrimitiveSet("MAIN", 2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(protect_div, 2)
    pset.addPrimitive(protect_sqrt, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(square, 1)
    pset.renameArguments(ARG0='x1', ARG1='x2')
    pset.addEphemeralConstant("rand", lambda: round(random.uniform(0, 1), 2))
    pset.addTerminal(-1)
    pset.addTerminal(1)
    return pset

def create_gp_toolbox(HEIGHT_LIMIT, init_method="gp", parsed_trees=None, pset=None):
    if pset is None:
        raise ValueError("❌ `pset` 不能为空！请先调用 `create_pset()` 生成 `pset`")

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # 创建Toolbox
    toolbox = base.Toolbox()
    if init_method == "gp":
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    elif init_method == "llm":
        if parsed_trees is None or len(parsed_trees) == 0:
            raise ValueError("❌ LLM 模式需要提供 parsed_trees 作为初始化表达式！")
        toolbox.register("individual", initIndividual, parsed_trees)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gp.compile, pset=pset)
    toolbox.register("evaluate", evalSymbReg)
    toolbox.register("select", tools.selTournament, tournsize=1)
    toolbox.register("mate", cxOnePointListOfTrees)
    toolbox.register("mutate", mutUniformListOfTrees, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))

    return toolbox


