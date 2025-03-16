import operator
import random
import numpy as np
import deap.gp as gp
import deap.base as base
import deap.creator as creator
import deap.tools as tools

from utils.evaluation import evalSymbReg


# **保护性操作**
def protect_sqrt(x):
    return np.sqrt(x) if x >= 0 else 0

def protect_div(x, y):
    return x / y if y != 0 else 1

def square(x):
    return x ** 2

def cxOnePointListOfTrees(ind1, ind2):
    HEIGHT_LIMIT = 6
    dec = gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT)
    ind1, ind2 = dec(gp.cxOnePoint)(ind1, ind2)
    return ind1, ind2

def mutUniformListOfTrees(individual, pset):
    HEIGHT_LIMIT = 6
    dec = gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT)
    expr = lambda pset, type_: gp.genFull(pset=pset, min_=1, max_=2)
    individual, = dec(gp.mutUniform)(individual, expr=expr, pset=pset)
    return (individual,)

def create_gp_toolbox(HEIGHT_LIMIT):
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

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gp.compile, pset=pset)
    toolbox.register("evaluate", evalSymbReg)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cxOnePointListOfTrees)
    toolbox.register("mutate", mutUniformListOfTrees, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))

    return toolbox,pset


