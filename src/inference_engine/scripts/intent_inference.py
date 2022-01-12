# this is just an pure class abstraction of the goal_inference.py ROSnode. To be used with pure simulation
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from random import sample
import numpy as np
import collections
import os
import pickle
from scipy import special
from scipy.stats import entropy
import itertools
import sys
import math

from adaptive_assistance_sim_utils import TRUE_ACTION_TO_COMMAND, LOW_LEVEL_COMMANDS
from adaptive_assistance_sim_utils import (
    AssistanceType,
    TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP,
    INTERFACE_LEVEL_ACTIONS,
    TASK_LEVEL_ACTIONS,
)


class IntentInference(object):
    def __init__(self, intent_inference_params):
        assert intent_inference_params is not None
        assert "mdps_for_goals" in intent_inference_params
        assert intent_inference_params["mdps_for_goals"] is not None  # list of mdps
        assert len(intent_inference_params["mdps_for_goals"]) >= 1  # need to have at least one potential goal

        self.P_PHI_GIVEN_A = None
        self.P_PHM_GIVEN_PHI = None
        self.PHI_SPARSE_LEVEL = 0.0
        self.PHM_SPARSE_LEVEL = 0.0
        self.DEFAULT_PHI_GIVEN_A_NOISE = intent_inference_params.get("phi_given_a_noise", 0.1)
        self.DEFAULT_PHM_GIVEN_PHI_NOISE = intent_inference_params.get("phm_given_phi_noise", 0.1)

        self.DELAYED_DECAY_THRESHOLD = 10

        self.P_PHI_GIVEN_A = collections.OrderedDict()
        self.init_P_PHI_GIVEN_A()

        self.P_PHM_GIVEN_PHI = collections.OrderedDict()
        self.init_P_PHM_GIVEN_PHI()

        self.mdps_for_goals = intent_inference_params["mdps_for_goals"]
        self.num_goals = len(self.mdps_for_goals)

        self.p_g_given_phm = np.array([1.0 / self.num_goals] * self.num_goals, dtype=np.float32)

    def perform_inference(self, inference_info_dict):
        self.compute_ii_recursive_bayes_interface_level(inference_info_dict)

    def compute_ii_recursive_bayes_interface_level(self, inference_info_dict):
        assert "phm" in inference_info_dict
        assert "state" in inference_info_dict
        phm = inference_info_dict["phm"]
        state = inference_info_dict["state"]
        current_mode = state[-1]
        if phm != "None":
            for g in range(self.num_goals):
                mdp_g = self.mdps_for_goals[g]
                likelihood = 0.0
                for a in self.P_PHI_GIVEN_A[current_mode].keys():
                    for phi in self.P_PHM_GIVEN_PHI.keys():
                        likelihood += (
                            self.P_PHM_GIVEN_PHI[phi][phm]
                            * self.P_PHI_GIVEN_A[current_mode][a][phi]
                            * mdp_g.get_prob_a_given_s(state, a)
                        )
                self.p_g_given_phm[g] = self.p_g_given_phm[g] * (1e-7 + likelihood)

            if np.sum(self.p_g_given_phm) == 0.0:
                import IPython

                IPython.embed(banner1="check belief zero")
            self.p_g_given_phm = self.p_g_given_phm / np.sum(self.p_g_given_phm)
        else:
            print("No change in belief phm is None")
            pass

    def get_current_p_g_given_phm(self):
        return self.p_g_given_phm

    def init_P_PHI_GIVEN_A(self):
        # only to be done at the beginning of a session for a subject. No updating between trials
        self.P_PHI_GIVEN_A = collections.OrderedDict()
        for mode in [1, 2]:  # hard coded modes for R2
            self.P_PHI_GIVEN_A[mode] = collections.OrderedDict()
            for k in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.keys():  # task level action
                self.P_PHI_GIVEN_A[mode][k] = collections.OrderedDict()
                for u in INTERFACE_LEVEL_ACTIONS:
                    if u == TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP[k]:
                        # try to weight the true command more for realistic purposes. Can be offset by using a high PHI_GIVEN_A_NOISE
                        self.P_PHI_GIVEN_A[mode][k][u] = 1.0
                    else:
                        self.P_PHI_GIVEN_A[mode][k][u] = 0.0

                delta_dist = np.array(list(self.P_PHI_GIVEN_A[mode][k].values()))
                uniform_dist = (1.0 / len(INTERFACE_LEVEL_ACTIONS)) * np.ones(len(INTERFACE_LEVEL_ACTIONS))
                blended_dist = (
                    1 - self.DEFAULT_PHI_GIVEN_A_NOISE
                ) * delta_dist + self.DEFAULT_PHI_GIVEN_A_NOISE * uniform_dist  # np.array
                for index, u in enumerate(INTERFACE_LEVEL_ACTIONS):
                    self.P_PHI_GIVEN_A[mode][k][u] = blended_dist[index]

    def init_P_PHM_GIVEN_PHI(self):
        """
        Generates a random p(um|ui). key = ui, subkey = um
        """
        self.P_PHM_GIVEN_PHI = collections.OrderedDict()
        for i in INTERFACE_LEVEL_ACTIONS:  # ui
            self.P_PHM_GIVEN_PHI[i] = collections.OrderedDict()
            for j in INTERFACE_LEVEL_ACTIONS:  # um
                if i == j:
                    # try to weight the true command more for realistic purposes. Can be offset by using a high UM_GIVEN_UI_NOISE
                    self.P_PHM_GIVEN_PHI[i][j] = 1.0
                else:
                    # P_PHM_GIVEN_PHI[i][j] = np.random.random()*UM_GIVEN_UI_NOISE#IF UM_GIVEN_UI_NOISE is 0, then the p(um|ui) is a deterministic mapping
                    self.P_PHM_GIVEN_PHI[i][j] = 0.0

            delta_dist = np.array(list(self.P_PHM_GIVEN_PHI[i].values()))
            uniform_dist = (1.0 / len(INTERFACE_LEVEL_ACTIONS)) * np.ones(len(INTERFACE_LEVEL_ACTIONS))
            blended_dist = (
                1 - self.DEFAULT_PHM_GIVEN_PHI_NOISE
            ) * delta_dist + self.DEFAULT_PHM_GIVEN_PHI_NOISE * uniform_dist  # np.array
            for index, j in enumerate(INTERFACE_LEVEL_ACTIONS):
                self.P_PHM_GIVEN_PHI[i][j] = blended_dist[index]
