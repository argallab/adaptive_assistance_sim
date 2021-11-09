#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import os
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import collections
import rospkg
from scipy.optimize import minimize


sys.path.append(os.path.join(rospkg.RosPack().get_path('simulators'), 'scripts'))
from utils import TRUE_ACTION_TO_COMMAND
from utils import LOW_LEVEL_COMMANDS

class IntendedCommandGivenActionAnalysis(object):
    def __init__(self, args):


        # load p_um_given_a and p_um_given_ui
        self.id = args.id
        self.personalized_distributions_dir = os.path.join(rospkg.RosPack().get_path('inference_and_correction'), 'personalized_distributions')

        assert os.path.exists(os.path.join(self.personalized_distributions_dir, str(self.id)+'_p_um_given_a.pkl'))
        assert os.path.exists(os.path.join(self.personalized_distributions_dir, str(self.id)+'_p_um_given_ui.pkl'))

        print('LOADING PERSONALIZED P_UM_GIVEN_A')
        with open(os.path.join(self.personalized_distributions_dir, str(self.id)+'_p_um_given_a.pkl'), 'rb') as fp:
            self.P_UM_GIVEN_A = pickle.load(fp)#assumes that the conditional probability distribution is stored as a collections.OrderedDict conditioned on the mode

        print('LOADING PERSONALIZED P_UM_GIVEN_UI')
        with open(os.path.join(self.personalized_distributions_dir, str(self.id)+'_p_um_given_ui.pkl'), 'rb') as fp:
            self.P_UM_GIVEN_UI = pickle.load(fp) #assumes that the conditional probability distribution is stored as a collections.OrderedDict





    def compute_p_ui_given_a(self):

        keys = LOW_LEVEL_COMMANDS
        A = np.array([self.P_UM_GIVEN_UI[LOW_LEVEL_COMMANDS[0]].values(), self.P_UM_GIVEN_UI[LOW_LEVEL_COMMANDS[1]].values(), self.P_UM_GIVEN_UI[LOW_LEVEL_COMMANDS[2]].values(), self.P_UM_GIVEN_UI[LOW_LEVEL_COMMANDS[3]].values()])

        # embed(banner1='check dicts')


        p_ui = collections.OrderedDict()
        for mode in TRUE_ACTION_TO_COMMAND.keys():

            p_ui[mode] = collections.OrderedDict()

            for action in TRUE_ACTION_TO_COMMAND[mode].keys():

                p_ui[mode][action] = collections.OrderedDict()
                B = np.array(self.P_UM_GIVEN_A[mode][action].values())


                def objective(x):
                    return np.dot(A[0, :], x) - B[0]

                def constraint_1(x):
                    sum = 1
                    for i in range(x.size):
                        sum = sum - x[i]
                    return sum

                def constraint_2(x):
                    return np.dot(A[1, :], x) - B[1]

                def constraint_3(x):
                    return np.dot(A[2, :], x) - B[2]

                def constraint_4(x):
                    return np.dot(A[3, :], x) - B[3]


                b = (0.0, 1.0)
                bnds = (b,b,b,b)
                # x0 = np.array([0.25, 0.25, 0.25, 0.25])
                x0 = np.random.random((4,))
                x0 = x0/np.sum(x0)
                x0 = x0/np.sum(x0)
                con1 = {'type':'eq', 'fun':constraint_1}
                con2 = {'type':'eq', 'fun':constraint_2}
                con3 = {'type':'eq', 'fun':constraint_3}
                con4 = {'type':'eq', 'fun':constraint_4}
                cons = [con1, con2,con3, con4]
                sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 2000})
                p_ui_given_a_dist = sol.x/np.sum(sol.x)

                for ind, key in enumerate(LOW_LEVEL_COMMANDS):
                    p_ui[mode][action][key] = p_ui_given_a_dist[ind]


        pickle.dump(p_ui, open(os.path.join(self.personalized_distributions_dir, self.id+'_p_ui_given_a_optim.pkl'), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', help='subject id', type=str)
    args = parser.parse_args()
    puia = IntendedCommandGivenActionAnalysis(args)
    puia.compute_p_ui_given_a()
    # python p_ui_given_a_distribution_preprocessing.py -id mahdieh
