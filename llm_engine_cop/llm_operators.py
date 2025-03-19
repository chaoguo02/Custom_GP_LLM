import json
import operator
import random
import numpy as np
import deap.gp as gp
import deap.base as base
import deap.creator as creator
import deap.tools as tools


from gp_engine.gp_operators import protect_div, protect_sqrt, square
from llm_engine.llm_evolutionary_operators import llm_crossover_expressions, llm_mutated_expressions
from utils.convert_tree2expression import expression_to_tree, tree_to_expression
from utils.evaluation import evalSymbReg
from utils.readAndwrite import read_jsonl, read_json


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

def parse_gp_expressions(jsonl_file):
    parsed_trees = []
    expressions = read_jsonl(jsonl_file)  # 读取 JSONL 文件

    if not expressions:
        print(f"⚠️ Warning: No expressions found in {jsonl_file}")
        return parsed_trees

    for entry in expressions:
        expr = entry.get("expression", "")
        try:
            parsed_trees.append(expr)
        except Exception as e:
            print(f"❌ Error parsing expression {expr}: {e}")

    print(f"✅ Loaded {len(parsed_trees)} expressions from {jsonl_file}.")
    return parsed_trees

def load_all_expressions(jsonl_file, pset=None):
    """ 读取 JSONL 文件中的所有表达式并转换为 GP 树 """
    global parsed_trees
    parsed_trees = []  # 先清空
    expressions = []

    # 读取所有 JSONL 记录
    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            expressions.append(data["expression"])

    # 转换为 GP 树
    parsed_expressions = [expression_to_tree(expr) for expr in expressions]
    parsed_trees = [gp.PrimitiveTree.from_string(expr, pset) for expr in parsed_expressions]

    print(f"Loaded {len(parsed_trees)} expressions into parsed_trees.")
    return parsed_trees

def initIndividual(parsed_trees):
    return creator.Individual(random.choice(parsed_trees))

## 🔥🔥🔥两个重要的函数
def cxOnePointListOfTrees(ind1, ind2, parsed_trees, llm_interface=None, pset=None):
    # **确保 `ind1` 和 `ind2` 是 PrimitiveTree**
    assert pset is not None, "❌ `pset` 不能为 None！"
    HEIGHT_LIMIT = 6
    ind1_tree = gp.PrimitiveTree(ind1) if isinstance(ind1, creator.Individual) else ind1
    ind2_tree = gp.PrimitiveTree(ind2) if isinstance(ind2, creator.Individual) else ind2

    print(f"Before Crossover: ind1 Tree: {ind1_tree}, ind2 Tree: {ind2_tree}")

    # **转换为数学表达式**
    expr1 = tree_to_expression(ind1_tree)
    expr2 = tree_to_expression(ind2_tree)
    print(f"Converted Expressions: expr1: {expr1}, expr2: {expr2}")

    tree_expressions = []
    tree_expressions.append(str(ind1_tree))
    tree_expressions.append(str(ind2_tree))
    # **调用 LLM 交叉**
    new_expressions = llm_crossover_expressions(llm_interface, tree_expressions)

    # **转换回 GP 结构**
    try:
        new_tree1 = gp.PrimitiveTree(new_expressions[0])
        new_tree2 = gp.PrimitiveTree(new_expressions[1])
    except Exception as e:
        print(f"交叉因为异常返回父代")
        return ind1, ind2
    print(f"New Trees: new_tree1.height: {new_tree1.height}, new_tree2.height: {new_tree2.height}")

    # **手动检查树高**
    if new_tree1.height > HEIGHT_LIMIT:
        print(f"⚠️ new_tree1 超出高度限制 ({new_tree1.height} > {HEIGHT_LIMIT})，使用父代 ind1 替换")
        new_tree1 = ind1_tree

    if new_tree2.height > HEIGHT_LIMIT:
        print(f"⚠️ new_tree2 超出高度限制 ({new_tree2.height} > {HEIGHT_LIMIT})，使用父代 ind2 替换")
        new_tree2 = ind2_tree
    new_individual1 = creator.Individual(new_tree1)
    new_individual2 = creator.Individual(new_tree2)

    print(f"After Crossover: new_individual1: {new_individual1}, new_individual2: {new_individual2}")

    return new_individual1, new_individual2

def mutUniformListOfTrees(ind, pset, parsed_trees=None, llm_interface=None):
    HEIGHT_LIMIT = 6
    ind_tree = gp.PrimitiveTree(ind) if isinstance(ind, creator.Individual) else ind
    print(f"Before Mutation: ind Tree: {ind_tree}")
    expr1 = tree_to_expression(ind_tree)
    new_expression = llm_mutated_expressions(llm_interface, expr1)
    try:
        new_tree1 = gp.PrimitiveTree.from_string(expression_to_tree(new_expression), pset)

    except Exception as e:
        print(f"变异因为异常，返回父代")
        return (ind,)

    print(f"New Mutated Tree Height: {new_tree1.height}")

    # **手动检查树高**
    if new_tree1.height > HEIGHT_LIMIT:
        print(f"⚠️ new_tree1 超出高度限制 ({new_tree1.height} > {HEIGHT_LIMIT})，使用父代 ind 替换")
        return ind,

    # **转换回 Individual**
    new_individual = creator.Individual(new_tree1)

    print(f"After Mutation: new_individual: {new_individual}")
    return new_individual,


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

def create_llm_toolbox(init_method="gp", parsed_trees=None, pset=None):
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
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cxOnePointListOfTrees)
    toolbox.register("mutate", mutUniformListOfTrees, pset=pset)

    return toolbox