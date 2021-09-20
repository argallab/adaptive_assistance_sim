#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import collections
import os
import pickle
from scipy import special
from scipy.stats import entropy
import itertools

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))

from adaptive_assistance_sim_utils import TRUE_ACTION_TO_COMMAND, LOW_LEVEL_COMMANDS
from adaptive_assistance_sim_utils import (
    AssistanceType,
    TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP,
    INTERFACE_LEVEL_ACTIONS,
    TASK_LEVEL_ACTIONS,
)


class DiscreteMIDisambAlgo(object):
    def __init__(self, env_params):

        self.env_params = env_params
        assert self.env_params is not None

        assert "mdp_list" in self.env_params
        assert "spatial_window_half_length" in self.env_params

        self.mdp_list = self.env_params["mdp_list"]
        self.SPATIAL_WINDOW_HALF_LENGTH = self.env_params["spatial_window_half_length"]
        self.P_PHI_GIVEN_A = None
        self.P_PHM_GIVEN_PHI = None
        self.DEFAULT_PHI_GIVEN_A_NOISE = 0.1
        self.DEFAULT_PHM_GIVEN_PHI_NOISE = 0.1

        self.distribution_directory_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "se2_personalized_distributions"
        )
        # init all distributions from file
        if os.path.exists(os.path.join(self.distribution_directory_path, str(self.subject_id) + "_p_phi_given_a.pkl")):
            print("LOADING PERSONALIZED P_PHI_GIVEN_A")
            with open(
                os.path.join(self.distribution_directory_path, str(self.subject_id) + "_p_phi_given_a.pkl"), "rb"
            ) as fp:
                self.P_PHI_GIVEN_A = pickle.load(fp)
        else:
            self.P_PHI_GIVEN_A = collections.OrderedDict()
            self.init_P_PHI_GIVEN_A()

        if os.path.exists(
            os.path.join(self.distribution_directory_path, str(self.subject_id) + "_p_phm_given_phi.pkl")
        ):
            print("LOADING PERSONALIZED P_PHM_GIVEN_PHI")
            with open(
                os.path.join(self.distribution_directory_path, str(self.subject_id) + "_p_phm_given_phi.pkl"), "rb"
            ) as fp:
                self.P_PHM_GIVEN_PHI = pickle.load(fp)
        else:
            self.P_PHM_GIVEN_PHI = collections.OrderedDict()
            self.init_P_PHM_GIVEN_PHI()

    def _compute_spatial_window_around_current_state(self, current_state):
        current_grid_loc = np.array(current_state[0:2])
        states_in_local_spatial_window = []

        # Add todo to ensure that self.mdp list is not None
        all_state_coords = self.mdp_list[0].get_all_state_coords()
        window_coordinates = itertools.product(
            range(-self.SPATIAL_WINDOW_HALF_LENGTH + 1, self.SPATIAL_WINDOW_HALF_LENGTH),
            range(-self.SPATIAL_WINDOW_HALF_LENGTH + 1, self.SPATIAL_WINDOW_HALF_LENGTH),
        )
        for wc in window_coordinates:
            vs = current_grid_loc + np.array(wc)
            for mode in range(self.num_modes):  # for 2d
                vs_mode = (vs[0], vs[1], mode + 1)
                if vs_mode in all_state_coords:
                    states_in_local_spatial_window.append(vs_mode)

        return states_in_local_spatial_window

    # TODO consolidate the following two functions so that both goal inference and
    # goal disamb both have the same set of information regarding interface noise
    def init_P_PHI_GIVEN_A(self):
        # only to be done at the beginning of a session for a subject. No updating between trials
        self.P_PHI_GIVEN_A = collections.OrderedDict()
        for k in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.keys():  # task level action
            self.P_PHI_GIVEN_A[k] = collections.OrderedDict()
            for u in INTERFACE_LEVEL_ACTIONS:
                if u == TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP[k]:
                    # try to weight the true command more for realistic purposes. Can be offset by using a high PHI_GIVEN_A_NOISE
                    self.P_PHI_GIVEN_A[k][u] = 1.0
                else:
                    self.P_PHI_GIVEN_A[k][u] = 0.0

            delta_dist = np.array(list(self.P_PHI_GIVEN_A[k].values()))
            uniform_dist = (1.0 / len(INTERFACE_LEVEL_ACTIONS)) * np.ones(len(INTERFACE_LEVEL_ACTIONS))
            blended_dist = (
                1 - self.DEFAULT_PHI_GIVEN_A_NOISE
            ) * delta_dist + self.DEFAULT_PHI_GIVEN_A_NOISE * uniform_dist  # np.array
            for index, u in enumerate(INTERFACE_LEVEL_ACTIONS):
                self.P_PHI_GIVEN_A[k][u] = blended_dist[index]

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
