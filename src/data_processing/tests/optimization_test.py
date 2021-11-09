# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import numpy as np
from scipy.optimize import minimize

def objective(x):
    c = np.array([0.3, 0.4])
    x = np.array(x)
    return np.dot(c, x) - 0.9

def constraint_1(x):
    sum = 1
    for i in range(2):
        sum = sum - x[i]
    return sum

def constraint_2(x):
    return np.dot(np.array([0.7, 0.6]), x) - 0.1

def optimize():
    x0 = np.array([0.5, 0.5])
    b = (0.0, 1.0)
    bnds = (b,b)
    con1 = {'type':'eq', 'fun':constraint_1}
    con2 = {'type':'eq', 'fun':constraint_2}
    cons = [con1, con2]
    sol = minimize(objective,x0, method='SLSQP', bounds=bnds, constraints=cons)
    print(objective(x0))
    print(sol)

optimize()
