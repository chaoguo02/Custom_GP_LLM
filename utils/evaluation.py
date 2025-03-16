import numpy as np
from deap import gp

def evalSymbReg(individual, pset, X_train, y_train, X_test, y_test):
    func = gp.compile(individual, pset)
    predictions_train = np.array([func(x1, x2) for x1, x2 in X_train])
    predictions_test = np.array([func(x1, x2) for x1, x2 in X_test])
    train_fitness = np.mean((predictions_train - y_train) ** 2)
    test_fitness = np.mean((predictions_test - y_test) ** 2)
    return train_fitness, test_fitness