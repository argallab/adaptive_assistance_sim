#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
import numpy as np


class IntentInference(object):
    def __init__(self, intent_inference_params):
        assert intent_inference_params is not None
        assert "mdps_for_goals" in intent_inference_params
        assert "p_phi_given_a" in intent_inference_params
        assert "p_phm_given_phi" in intent_inference_params
        assert "inference_type" in intent_inference_params
        assert intent_inference_params["mdps_for_goals"] is not None  # list of mdps
        assert len(intent_inference_params["mdps_for_goals"]) >= 1  # need to have at least one potential goal

        self.mdps_for_goals = intent_inference_params["mdps_for_goals"]
        self.num_goals = len(self.mdps_for_goals)

        # initialize with uniform distribution over goals.
        self.p_g_given_a_s = np.array([1.0 / self.num_goals] * self.num_goals, dtype=np.float32)
        self.p_g_given_phm = np.array([1.0 / self.num_goals] * self.num_goals, dtype=np.float32)
        if (
            "use_state_information" in intent_inference_params
            and intent_inference_params["use_state_information"] is not None
        ):
            self.use_state_information = intent_inference_params["use_state_information"]
        else:
            self.use_state_information = False

        self.P_PHI_GIVEN_A = intent_inference_params["p_phi_given_a"]
        self.P_PHM_GIVEN_PHI = intent_inference_params["p_phm_given_phi"]
        self.inference_type = intent_inference_params["inference_type"]

    def perform_inference(self, inference_info_dict):

        # potential use functools.partial to create a handle
        if self.inference_type == "bayes":
            self.compute_ii_recursive_bayes(inference_info_dict)
        elif self.inference_type == "bayes_interface":
            self.compute_ii_recursive_bayes_interface_level(inference_info_dict)
        elif self.inference_type == "dft":
            self.compute_ii_dft(action, state)
        elif self.inference_type == "heuristic":
            self.compute_ii_heuristic_confidence(action, state)
        else:
            import IPython

            IPython.embed(banner1="check invalid inference type")

    def compute_ii_dft(self, action, state):
        pass

    def _compute_state_based_weighting_factor(self, curr_loc, g_loc):
        # distance between current location and goal state for mdp.
        d = np.linalg.norm(np.array(curr_loc) - np.array(g_loc))
        # print(curr_loc, g_loc, d)
        return 1.0 / (d + 1e-2)  # closer the curr loc to the goal higher the weight

    def compute_ii_recursive_bayes_interface_level(self, inference_info_dict):
        assert "phm" in inference_info_dict
        assert "state" in inference_info_dict
        phm = inference_info_dict["phm"]
        state = inference_info_dict["state"]

        if phm != "None":
            for g in range(self.num_goals):
                mdp_g = self.mdps_for_goals[g]
                likelihood = 0.0
                for a in self.P_PHI_GIVEN_A.keys():  # task_level action
                    for phi in self.P_PHM_GIVEN_PHI.keys():
                        likelihood += (
                            self.P_PHM_GIVEN_PHI[phi][phm]
                            * self.P_PHI_GIVEN_A[a][phi]
                            * mdp_g.get_prob_a_given_s(state, a)
                        )

                self.p_g_given_phm[g] = self.p_g_given_phm[g] * likelihood

            self.p_g_given_phm = self.p_g_given_phm / np.sum(self.p_g_given_phm)
        else:
            # potentially add decay?
            print("PHM NONe, therefore no goal belief update")

    def compute_ii_recursive_bayes(self, inference_info_dict):
        """
        perform bayesian inference of goals. 
        """
        assert "action" in inference_info_dict
        assert "state" in inference_info_dict
        for g in range(self.num_goals):
            # assumes the use of eps greedy policy
            p_a_given_sg = self.mdps_for_goals[g].get_prob_a_given_s(state, action)
            if not self.use_state_information:
                # add noise to avoid collapse of the posterior
                self.p_g_given_a_s[g] = (p_a_given_sg + 1e-6) * self.p_g_given_a_s[g]
            else:

                p_s_given_g = self._compute_state_based_weighting_factor(
                    self.mdps_for_goals[g].get_location(state), self.mdps_for_goals[g].get_goal_state()
                )
                self.p_g_given_a_s[g] = p_a_given_sg * p_s_given_g * self.p_g_given_a_s[g]

        self.p_g_given_a_s = self.p_g_given_a_s / np.sum(self.p_g_given_a_s)  # normalize

    def compute_ii_heuristic_confidence(self, action, state):
        pass

    def get_current_p_g_given_a_s(self):
        return self.p_g_given_a_s

    def get_current_p_g_given_phm(self):
        return self.p_g_given_phm
