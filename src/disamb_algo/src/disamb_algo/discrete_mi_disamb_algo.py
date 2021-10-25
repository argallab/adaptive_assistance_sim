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
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))

from adaptive_assistance_sim_utils import TRUE_ACTION_TO_COMMAND, LOW_LEVEL_COMMANDS
from adaptive_assistance_sim_utils import (
    AssistanceType,
    TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP,
    TRUE_INTERFACE_ACTION_TO_TASK_ACTION_MAP,
    INTERFACE_LEVEL_ACTIONS,
    TASK_LEVEL_ACTIONS,
    INTERFACE_LEVEL_ACTIONS_TO_NUMBER_ID,
    CARTESIAN_MODE_SET_OPTIONS,
)


class DiscreteMIDisambAlgo(object):
    def __init__(self, env_params, subject_id):

        self.env_params = env_params
        assert self.env_params is not None

        assert "mdp_list" in self.env_params
        assert "spatial_window_half_length" in self.env_params

        self.mdp_env_params = self.env_params["all_mdp_env_params"]
        self.grid_width = self.mdp_env_params["grid_width"]
        self.grid_height = self.mdp_env_params["grid_height"]

        self.grid_scale = np.linalg.norm([self.grid_width, self.grid_height])
        self.mdp_list = self.env_params["mdp_list"]
        assert self.mdp_list is not None
        assert len(self.mdp_list) > 0

        self.subject_id = subject_id
        self.num_goals = len(self.mdp_list)
        self.SPATIAL_WINDOW_HALF_LENGTH = self.env_params["spatial_window_half_length"]
        self.P_PHI_GIVEN_A = None
        self.P_PHM_GIVEN_PHI = None
        self.PHI_SPARSE_LEVEL = 0.0
        self.PHM_SPARSE_LEVEL = 0.0
        self.DEFAULT_PHI_GIVEN_A_NOISE = 0.1
        self.DEFAULT_PHM_GIVEN_PHI_NOISE = 0.1

        self.num_sample_trajectories = self.env_params.get("num_sample_trajectories", 250)
        self.mode_set_type = self.env_params["mode_set_type"]
        self.robot_type = self.env_params["robot_type"]
        self.mode_set = CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type]
        self.num_modes = len(self.mode_set)
        self.num_discrete_orientations = self.mdp_env_params["num_discrete_orientations"]

        self.num_modes = self.env_params.get("num_modes", 3)
        self.kl_coeff = self.env_params.get("kl_coeff", 0.8)
        self.dist_coeff = self.env_params.get("dist_coeff", 0.2)
        print(self.kl_coeff, self.dist_coeff)

        self.distribution_directory_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "se2_personalized_distributions"
        )
        # unify the initialization of these distribution between different classes
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

        print("Finished initializing DISAMB CLASS")

    def get_local_disamb_state(self, prior, current_state):
        # compute window around current_state
        states_in_local_spatial_window = self._compute_spatial_window_around_current_state(current_state)
        # print(states_in_local_spatial_window)
        # print(len(states_in_local_spatial_window))
        # # perform mi computation for all states in spatial window
        self._compute_mi(prior, states_in_local_spatial_window)
        # # pick argmax among this list
        max_disamb_state = self._max_disambiguating_state()
        return max_disamb_state

    def _max_disambiguating_state(self):
        rewards = self.avg_total_reward_for_valid_states.values()
        print("REWARDS FOR ALL NEIGHBORING STATES ", rewards)
        amax = np.argmax(rewards)
        max_disamb_state = list(self.avg_total_reward_for_valid_states.keys())[amax]
        return max_disamb_state

    def _compute_mi(self, prior, states_for_disamb_computation=None):
        self.avg_mi_for_valid_states = collections.OrderedDict()
        self.avg_dist_for_valid_states_from_goals = collections.OrderedDict()
        self.avg_total_reward_for_valid_states = collections.OrderedDict()

        assert len(prior) == self.num_goals
        prior = prior / np.sum(prior)  # normalizing to make sure random choice works #todo maybe add some minor noise
        for i, vs in enumerate(states_for_disamb_computation):
            # print("Computing MI for ", vs)
            traj_list = collections.defaultdict(list)
            for t in range(self.num_sample_trajectories):
                sampled_goal_index = np.random.choice(self.num_goals, p=prior)
                mdp_for_sampled_goal = self.mdp_list[sampled_goal_index]
                # sub optimal a_sampled
                a_sampled = mdp_for_sampled_goal.get_optimal_action(vs, return_optimal=False)
                # sampled corrupted interface level action corresponding to task-level action, could be None
                phi = self.sample_phi_given_a(a_sampled)
                # corrupted interface level action, could be None
                phm = self.sample_phm_given_phi(phi)
                if phm != "None":
                    applied_a = TRUE_INTERFACE_ACTION_TO_TASK_ACTION_MAP[phm]
                else:
                    applied_a = "None"

                next_state = mdp_for_sampled_goal.get_next_state_from_state_action(vs, applied_a)
                traj_tuple = (vs, a_sampled, phi, phm, applied_a, next_state)
                traj_list[sampled_goal_index].append(traj_tuple)

            p_phm_g_s0 = collections.defaultdict(list)  # p(phm | g, s0)
            for g in traj_list.keys():
                for traj_g in traj_list[g]:
                    (vs, a_sampled, phi, phm, applied_a, next_state) = traj_g
                    p_phm_g_s0[g].append(INTERFACE_LEVEL_ACTIONS_TO_NUMBER_ID[phm])

            # p(phm|s). is a list instead of defaultdict(list) because all actions are just combinaed
            p_phm_s0 = []
            for g in p_phm_g_s0.keys():
                p_phm_s0.extend(p_phm_g_s0[g])

            ph_actions_ids = INTERFACE_LEVEL_ACTIONS_TO_NUMBER_ID.values()

            # histogram
            p_phm_s0_hist = collections.Counter(p_phm_s0)
            # to make sure that all interface level actions are present in the histogram
            for ph_action_id in ph_actions_ids:
                if ph_action_id not in p_phm_s0_hist.keys():
                    p_phm_s0_hist[ph_action_id] = 0

            p_phm_s = np.array(p_phm_s0_hist.values(), dtype=np.float32)
            p_phm_s = p_phm_s / np.sum(p_phm_s)
            kl_list = []
            for g in p_phm_g_s0.keys():
                p_phm_g_s_hist = collections.Counter(p_phm_g_s0[g])
                for ph_action_id in ph_actions_ids:
                    if ph_action_id not in p_phm_g_s_hist.keys():
                        p_phm_g_s_hist[ph_action_id] = 0

                assert len(p_phm_g_s_hist) == len(p_phm_s)
                p_phm_g_s = np.array(p_phm_g_s_hist.values(), dtype=np.float32)
                p_phm_g_s = p_phm_g_s / np.sum(p_phm_g_s)
                kl = np.sum(special.rel_entr(p_phm_g_s, p_phm_s))
                kl_list.append(kl)

            # normalized to grid dimensions
            dist_of_vs_from_goals = []
            for goal in self.mdp_env_params["all_goals"]:

                dist_of_vs_from_goal = np.linalg.norm(np.array(goal[:2]) - np.array(vs[:2]))
                dist_of_vs_from_goal = dist_of_vs_from_goal / self.grid_scale
                dist_of_vs_from_goals.append(dist_of_vs_from_goal)

            self.avg_dist_for_valid_states_from_goals[vs] = np.mean(dist_of_vs_from_goals)
            self.avg_mi_for_valid_states[vs] = np.mean(kl_list)  # averaged over goals.
            self.avg_total_reward_for_valid_states[vs] = self.kl_coeff * (
                self.avg_mi_for_valid_states[vs]
            ) - self.dist_coeff * (self.avg_dist_for_valid_states_from_goals[vs])

    def _compute_spatial_window_around_current_state(self, current_state):
        current_grid_loc = np.array(current_state[0:2])  # (x,y)
        states_in_local_spatial_window = []
        current_orientation = current_state[2]

        # Add todo to ensure that self.mdp list is not None
        # all states except goal states
        all_state_coords = self.mdp_list[0].get_all_state_coords_with_grid_locs_diff_from_goals_and_obs()
        window_coordinates = list(
            itertools.product(
                range(-self.SPATIAL_WINDOW_HALF_LENGTH + 1, self.SPATIAL_WINDOW_HALF_LENGTH),
                range(-self.SPATIAL_WINDOW_HALF_LENGTH + 1, self.SPATIAL_WINDOW_HALF_LENGTH),
            )
        )
        print("LOCAL 2D WINDOW ", window_coordinates)
        for wc in window_coordinates:
            vs = current_grid_loc + np.array(wc)  # 2d grid loc
            for mode in range(self.num_modes):  #
                vs_mode = (
                    vs[0],
                    vs[1],
                    current_orientation,
                    mode + 1,
                )  # same orientation as the current state for all states under consideration
                if vs_mode in all_state_coords:
                    states_in_local_spatial_window.append(vs_mode)

        print("LOCAL WINDOW ", states_in_local_spatial_window)

        assert len(states_in_local_spatial_window) > 0, current_state
        return states_in_local_spatial_window

    def sample_phi_given_a(self, a):  # sample from p(phii|a)
        d = np.random.rand()

        if d < self.PHI_SPARSE_LEVEL:
            phi = "None"
        else:
            p_vector = self.P_PHI_GIVEN_A[a].values()  # list of probabilities for phii
            # sample from the multinomial distribution with distribution p_vector
            phi_index_vector = np.random.multinomial(1, p_vector)
            phi_index = np.nonzero(phi_index_vector)[0][
                0
            ]  # grab the index of the index_vector which had a nonzero entry
            phi = self.P_PHI_GIVEN_A[a].keys()[phi_index]  # retrieve phii using the phi_index
            # will be not None

        return phi

    def sample_phm_given_phi(self, phi):  # sample from p(phm|phi)
        d = np.random.rand()
        if phi != "None":
            if d < self.PHM_SPARSE_LEVEL:
                phm = "None"
            else:
                p_vector = self.P_PHM_GIVEN_PHI[phi].values()  # list of probabilities for phm given phi
                phm_index_vector = np.random.multinomial(1, p_vector)  # sample from the multinomial distribution
                # grab the index of the index_vector which had a nonzero entry
                phm_index = np.nonzero(phm_index_vector)[0][0]
                phm = self.P_PHM_GIVEN_PHI[phi].keys()[phm_index]  # retrieve phm
        else:
            print("Sampled phi is None, therefore phm is None")
            phm = "None"

        return phm

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
