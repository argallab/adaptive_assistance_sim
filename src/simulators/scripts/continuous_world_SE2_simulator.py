#!/usr/bin/env python

import collections
import rospy
import time
from sensor_msgs.msg import Joy
from envs.continuous_world_SE2_env import ContinuousWorldSE2Env
from disamb_algo.discrete_mi_disamb_algo import DiscreteMIDisambAlgo
from sim_pfields.msg import CuboidObs
from sim_pfields.srv import CuboidObsList, CuboidObsListRequest, CuboidObsListResponse
from sim_pfields.srv import AttractorPos, AttractorPosRequest, AttractorPosResponse
from sim_pfields.srv import ComputeVelocity, ComputeVelocityRequest, ComputeVelocityResponse

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension, String, Int8
from teleop_nodes.msg import InterfaceSignal
from simulators.msg import State
from simulators.srv import InitBelief, InitBeliefRequest, InitBeliefResponse
from simulators.srv import ResetBelief, ResetBeliefRequest, ResetBeliefResponse
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse


from inference_engine.msg import BeliefInfo
from teleop_nodes.srv import SetMode, SetModeRequest, SetModeResponse
from mdp.mdp_discrete_SE2_gridworld_with_modes import MDPDiscreteSE2GridWorldWithModes
from pyglet.window import key
import numpy as np
import pickle
from pyglet.window import key
import random
import sys
import os
import copy
import itertools
from mdp.mdp_utils import *
from adaptive_assistance_sim_utils import *

GRID_WIDTH = 10
GRID_HEIGHT = 10
NUM_ORIENTATIONS = 8
NUM_GOALS = 3
OCCUPANCY_LEVEL = 0.0

SPARSITY_FACTOR = 0.0
RAND_DIRECTION_FACTOR = 0.1


class Simulator(object):
    def __init__(self, subject_id, algo_condition_block, block_id, training):
        super(Simulator, self).__init__()
        rospy.init_node("Simulator")
        rospy.on_shutdown(self.shutdown_hook)

        self.called_shutdown = False
        self.shutdown_pub = rospy.Publisher("/shutdown", String, queue_size=1)
        self.trial_marker_pub = rospy.Publisher("/trial_marker", String, queue_size=1)
        self.trial_index_pub = rospy.Publisher("/trial_index", Int8, queue_size=1)
        self.robot_state_pub = rospy.Publisher("/robot_state", State, queue_size=1)

        self.robot_state = State()
        self.dim = 3
        user_vel = InterfaceSignal()

        self.input_action = {}
        self.input_action["full_control_signal"] = user_vel

        rospy.Subscriber("/user_vel", InterfaceSignal, self.joy_callback)
        rospy.Subscriber("/belief_info", BeliefInfo, self.belief_info_callback)
        self.trial_index = 0

        self.env_params = None
        self.trial_info_dir_path = os.path.join(os.path.dirname(__file__), "trial_folders", "trial_dir")
        self.metadata_dir = os.path.join(os.path.dirname(__file__),"trial_folders", "metadata_dir")

        self.subject_id = subject_id
        self.algo_condition_block = algo_condition_block  # pass these things from launch file
        self.block_id = block_id
        self.training = training
        self.total_blocks = 6

        self.testing_block_filename = (
            self.subject_id
            + "_"
            + self.algo_condition_block
            + "_condition_"
            + self.block_id
            + "_num_blocks_"
            + str(self.total_blocks)
            + ".pkl"
        )
        print "TRAINING BLOCK FILENAME and IS TRAINING MODE", self.testing_block_filename, self.training

        rospy.loginfo("Waiting for sim_pfields node")
        rospy.wait_for_service("/sim_pfields_multiple/init_obstacles")
        rospy.wait_for_service("/sim_pfields_multiple/update_ds")
        rospy.wait_for_service("/sim_pfields_multiple/compute_velocity")
        rospy.loginfo("sim pfields node services found! ")

        self.init_obstacles_srv = rospy.ServiceProxy("/sim_pfields_multiple/init_obstacles", CuboidObsList)
        self.init_obstacles_request = CuboidObsListRequest()

        self.update_attractor_ds_srv = rospy.ServiceProxy("/sim_pfields_multiple/update_ds", AttractorPos)
        self.update_attractor_ds_request = AttractorPosRequest()

        self.compute_velocity_srv = rospy.ServiceProxy("/sim_pfields_multiple/compute_velocity", ComputeVelocity)
        self.compute_velocity_request = ComputeVelocityRequest()

        self.terminate = False
        self.restart = False
        if self.trial_info_dir_path is not None and os.path.exists(self.trial_info_dir_path) and not self.training:
            # if trials are pregenerated then load from there.
            print ("PRECACHED TRIALS")
            self.metadata_index_path = os.path.join(self.metadata_dir, self.testing_block_filename)
            print(self.metadata_index_path)
            assert os.path.exists(self.metadata_index_path)
            with open(self.metadata_index_path, "rb") as fp:
                self.metadata_index = pickle.load(fp)

            trial_info_filename_index = self.metadata_index[self.trial_index]
            trial_info_filepath = os.path.join(self.trial_info_dir_path, str(trial_info_filename_index) + ".pkl")
            print(trial_info_filename_index, self.trial_info_dir_path)
            assert os.path.exists(trial_info_filepath)
            with open(trial_info_filepath, "rb") as fp:
                self.env_params = pickle.load(fp)

            mdp_env_params = self.env_params['all_mdp_env_params']
            world_bounds  = self.env_params['world_bounds']
            self.algo_condition = self.env_params['algo_condition']
            
            print ("ALGO CONDITION", self.algo_condition)
        else:
            pass
        
        self._init_goal_pfields(
            self.env_params["goal_poses"],
            mdp_env_params["dynamic_obs_specs"],
            mdp_env_params["cell_size"],
            world_bounds,
        )
        self._init_other_pfields(
            self.env_params["goal_poses"],
            mdp_env_params["dynamic_obs_specs"],
            mdp_env_params["cell_size"],
            world_bounds,
            pfield_id="disamb",
        )
        self._init_other_pfields(
            self.env_params["goal_poses"],
            mdp_env_params["dynamic_obs_specs"],
            mdp_env_params["cell_size"],
            world_bounds,
            pfield_id="generic",
        )
        
        # alpha from confidence function parameters
        self.confidence_threshold = 1.05 / len(self.env_params['goal_poses'])
        self.confidence_max =  1.14 / len(self.env_params['goal_poses'])
        self.alpha_max = 0.8
        if self.confidence_max != self.confidence_threshold:
            self.confidence_slope = float(self.alpha_max) / (self.confidence_max - self.confidence_threshold)
        else:
            self.confidence_slope = -1.0

        self.ENTROPY_THRESHOLD = 0.7

        # instantiate the environement
        self.env_params["start"] = False
        self.env = ContinuousWorldSE2Env(self.env_params)
        self.env.initialize()
        self.env.initialize_viewer()
        self.env.reset()
        self.env.viewer.window.on_key_press = self.key_press

        self.disamb_algo = DiscreteMIDisambAlgo(self.env_params, subject_id)

        # setup all services
        rospy.loginfo("Waiting for goal inference node")
        rospy.wait_for_service("/goal_inference/init_belief")
        rospy.wait_for_service("/goal_inference/reset_belief")
        rospy.wait_for_service("/goal_inference/freeze_update")
        rospy.loginfo("goal inference node service found! ")

        self.init_belief_srv = rospy.ServiceProxy("/goal_inference/init_belief", InitBelief)
        self.init_belief_request = InitBeliefRequest()
        self.init_belief_request.num_goals = self.env_params["num_goals"]
        status = self.init_belief_srv(self.init_belief_request)

        self.reset_belief_srv = rospy.ServiceProxy("/goal_inference/reset_belief", ResetBelief)
        self.reset_belief_request = ResetBeliefRequest()

        self.freeze_update_srv = rospy.ServiceProxy("/goal_inference/freeze_update", SetBool)
        self.freeze_update_request = SetBoolRequest()

        # unfreeze belief update
        self.freeze_update_request.data = False
        self.freeze_update_srv(self.freeze_update_request)

        r = rospy.Rate(100)
        self.trial_start_time = time.time()
        if not self.training:
            self.max_time = 1000
        else:
            self.max_time = 1000

        is_done = False
        first_trial = True
        self.start = False

        self.p_g_given_phm = (1.0 / self.env_params["num_goals"]) * np.ones(self.env_params["num_goals"])
        self.autonomy_activate_ctr = 0
        
        self.is_autonomy_turn = False
        self.has_human_initiated = False
        self.DISAMB_ACTIVATE_THRESHOLD = 100
        self.BLENDING_THRESHOLD = 15  # distance within which thje blending kicks in

        while not rospy.is_shutdown():
            if not self.start:
                self.start = self.env.start_countdown()
            else:
                if first_trial:
                    time.sleep(2)

                    self.env.reset()
                    self.env.render()
                    self.env.set_information_text("Waiting....")
                    self.trial_start_time = time.time()
                    self.trial_marker_pub.publish("start")
                    self.trial_index_pub.publish(trial_info_filename_index)
                    first_trial = False
                else:
                    if is_done:
                        if not self.training:
                            print ("Move to NEXT TRIAL")
                            self.trial_marker_pub.publish("end")
                            self.env.render_clear("Loading next trial ...")
                            time.sleep(5.0)  # sleep before the next trial happens
                            self.trial_index += 1
                            if self.trial_index == len(self.metadata_index):
                                self.shutdown_hook("Reached end of trial list. End of session")
                                break  # experiment is done
                            trial_info_filename_index = self.metadata_index[self.trial_index]
                            trial_info_filepath = os.path.join(self.trial_info_dir_path, str(trial_info_filename_index) + ".pkl")
                            assert os.path.exists(trial_info_filepath) is not None
                            with open(trial_info_filepath, "rb") as fp:
                                self.env_params = pickle.load(fp)
                            
                            mdp_env_params = self.env_params['all_mdp_env_params']
                            world_bounds  = self.env_params['world_bounds']
                            self.algo_condition = self.env_params['algo_condition']
                            print ("ALGO CONDITION", self.algo_condition)

                            self._prepare_trial_setup(mdp_env_params, world_bounds)
                            self.trial_start_time = time.time()
                            self.trial_index_pub.publish("start")
                            self.trial_index_pub.publish(trial_info_filename_index)
                            is_done = False
                            self.is_restart = False
                        else:
                            self.shutdown_hook("Reached end of training")
                            break

                # if during human turn
                if not self.is_autonomy_turn:
                    # get current belief. and entropy of belief. compute argmax g
                    (
                        inferred_goal_id_str,
                        inferred_goal_id,
                        inferred_goal_prob,
                        normalized_h_of_p_g_given_phm,
                        argmax_goal_id,
                        argmax_goal_id_str,
                    ) = self._get_most_confident_goal()
                    # get human control action in full robot control space
                    human_vel = self.env.get_mode_conditioned_velocity(
                        self.input_action["human"].interface_signal
                    )  # robot_dim
                    is_mode_switch = self.input_action["human"].mode_switch

                    # autonomy inferred a valid goal
                    if inferred_goal_id_str is not None and inferred_goal_prob is not None:
                        # get pfield vel for argmax g and current robot pose
                        robot_continuous_position = self.env.get_robot_position()
                        robot_continuous_orientation = self.env.get_robot_orientation()
                        self.compute_velocity_request.current_pos = robot_continuous_position
                        self.compute_velocity_request.current_orientation = robot_continuous_orientation
                        self.compute_velocity_request.pfield_id = (
                            inferred_goal_id_str  # get pfeild vel corresponding to inferred goal
                        )
                        vel_response = self.compute_velocity_srv(self.compute_velocity_request)
                        autonomy_vel = list(vel_response.velocity_final)
                        inferred_goal_pose = self.env_params["goal_poses"][inferred_goal_id]
                        inferred_goal_position = inferred_goal_pose[:-1]
                        dist_weight = self._dist_based_weight(
                            self.env_params["goal_poses"][inferred_goal_id][:-1], robot_continuous_position
                        )
                        self.alpha = self._compute_alpha(inferred_goal_prob) * dist_weight
                    else:
                        # no inferred goal due to high entropy of goal belief, hence autonomy vel is zero
                        autonomy_vel = list([0.0] * np.array(human_vel).shape[0])  # 0.0 autonomy vel
                        self.alpha = 0.0  # no autonomy, purely human vel

                    # blend autonomy velocity by combining it with user vel.
                    blend_vel = self._blend_velocities(np.array(human_vel), np.array(autonomy_vel))
                    # apply blend velocity to robot.
                    self.input_action["full_control_signal"] = blend_vel
                    # print(is_mode_switch)
                    if np.linalg.norm(np.array(human_vel)) < 1e-5 and is_mode_switch == False:
                        if self.has_human_initiated:
                            # zero human vel and human has already issued some non-zero velocities during their turn,
                            # in which case keep track of inactivity time
                            self.autonomy_activate_ctr += 1
                    else:  # if non-zero human velocity
                        if not self.has_human_initiated:
                            print ("HUMAN INITIATED DURING THEIR TURN")
                            self.env.set_information_text("Your TURN")
                            self.has_human_initiated = True

                            # unfreeze belief update
                            self.freeze_update_request.data = False
                            self.freeze_update_srv(self.freeze_update_request)

                        # reset the activate ctr because human is providing non zero commands.
                        self.autonomy_activate_ctr = 0

                    if self.restart:
                        if not self.training:
                            print("RESTART INITIATED")
                            self.trial_marker_pub.publish("restart")
                            self.restart = False
                            time.sleep(5.0)
                            # TODO should I be incrementing trial index here or should I just restart the same trial?
                            if self.trial_index == len(self.metadata_index):
                                self.shutdown_hook("Reached end of trial list. End of session")
                                break
                            trial_info_filename_index = self.metadata_index[self.trial_index]
                            trial_info_filepath = os.path.join(
                                self.trial_info_dir_path, str(trial_info_filename_index) + ".pkl"
                            )
                            assert os.path.exists(trial_info_filepath) is not None
                            with open(trial_info_filepath, "rb") as fp:
                                self.env_params = pickle.load(fp)
                            
                            mdp_env_params = self.env_params['all_mdp_env_params']
                            world_bounds = self.env_params['world_bounds']
                            self.algo_condition = self.env_params['algo_condition']
                            print ("ALGO CONDITION", self.algo_condition)
                            self._prepare_trial_setup(mdp_env_params, world_bounds)
                            
                            self.trial_marker_pub.publish("start")
                            self.trial_index_pub.publish(trial_info_filename_index)
                            self.trial_start_time = time.time()
                            is_done = False

                    (
                        robot_continuous_position,
                        robot_continuous_orientation,
                        robot_discrete_state,
                        is_done,
                    ) = self.env.step(self.input_action)

                    if self.terminate:
                        self.shutdown_hook("Session terminated")
                        break

                    self.env.render()

                    # checks to whether trigger the autonomy turn
                    if self.algo_condition == "disamb":
                        # maybe add other conditions such as high entropy for belief as a way to trigger
                        if self.autonomy_activate_ctr > self.DISAMB_ACTIVATE_THRESHOLD:
                            if normalized_h_of_p_g_given_phm > self.ENTROPY_THRESHOLD:
                                print ("ACTIVATING DISAMB")
                                self.env.set_information_text("DISAMB")

                                # Freeze belief update during autonomy's turn up until human activates again
                                self.freeze_update_request.data = True
                                self.freeze_update_srv(self.freeze_update_request)

                                # get current discrete state
                                robot_discrete_state = self.env.get_robot_current_discrete_state()  # tuple (x,y,t,m)
                                current_mode = robot_discrete_state[-1]
                                belief_at_disamb_time = self.p_g_given_phm
                                max_disamb_state = self.disamb_algo.get_local_disamb_state(
                                    self.p_g_given_phm, robot_discrete_state
                                )
                                print ("MAX DISAMB STATE", max_disamb_state)
                                # convert discrete disamb state to continous attractor position for disamb pfield
                                (
                                    max_disamb_continuous_position,
                                    max_disamb_continuous_orientation,
                                    _,
                                ) = self._convert_discrete_state_to_continuous_pose(
                                    max_disamb_state, mdp_env_params["cell_size"], world_bounds
                                )
                                # update disamb pfield attractor
                                self.update_attractor_ds_request.pfield_id = "disamb"
                                self.update_attractor_ds_request.attractor_position = list(
                                    max_disamb_continuous_position
                                )  # dummy attractor pos. Will be update after disamb computation
                                self.update_attractor_ds_request.attractor_orientation = float(
                                    max_disamb_continuous_orientation
                                )
                                self.update_attractor_ds_srv(self.update_attractor_ds_request)
                                # activate disamb flag so that autonomy can drive the robot to the disamb location
                                self.is_autonomy_turn = True

                                # switch current mode to disamb mode. #if disamb mdoe different from current update the message
                                disamb_state_mode_index = max_disamb_state[-1]
                                if disamb_state_mode_index != current_mode:
                                    self.env.set_information_text("Disamb - MODE SWITCHED")
                                self.env.set_mode_in_robot(disamb_state_mode_index)
                            else:
                                # human has stopped. autonomy' turn. Upon waiting, the confidence is still high. Therefore, move the robot to current confident goal.
                                print ("ACTIVATING AUTONOMY")
                                self.env.set_information_text("AUTONOMY")

                                # Freeze belief update during autonomy's turn up until human activates again
                                self.freeze_update_request.data = True
                                self.freeze_update_srv(self.freeze_update_request)
                                belief_at_disamb_time = self.p_g_given_phm
                                # if so, get current inferred goal pose
                                inferred_goal_pose = self.env_params["goal_poses"][inferred_goal_id]
                                inferred_goal_position = inferred_goal_pose[:-1]
                                # connect the line joining current position and inferref goal position.
                                target_point = self._get_target_along_line(
                                    robot_continuous_position, inferred_goal_position
                                )
                                max_disamb_continuous_position = target_point
                                self.update_attractor_ds_request.pfield_id = "disamb"
                                self.update_attractor_ds_request.attractor_position = list(target_point)
                                self.update_attractor_ds_request.attractor_orientation = float(inferred_goal_pose[-1])
                                self.update_attractor_ds_srv(self.update_attractor_ds_request)
                                self.is_autonomy_turn = True

                    elif self.algo_condition == "control":
                        # check if ctr is past the threshold
                        if self.autonomy_activate_ctr > self.DISAMB_ACTIVATE_THRESHOLD:
                            print ("ACTIVATING AUTONOMY")
                            self.env.set_information_text("AUTONOMY")

                            # Freeze belief update during autonomy's turn up until human activates again
                            self.freeze_update_request.data = True
                            self.freeze_update_srv(self.freeze_update_request)

                            belief_at_disamb_time = self.p_g_given_phm
                            # if so, get current argmax goal pose
                            argmax_goal_pose = self.env_params["goal_poses"][argmax_goal_id]
                            argmax_goal_position = argmax_goal_pose[:-1]
                            # connect the line joining current position and inferref goal position.
                            target_point = self._get_target_along_line(robot_continuous_position, argmax_goal_position)
                            self.update_attractor_ds_request.pfield_id = "generic"
                            self.update_attractor_ds_request.attractor_position = list(target_point)
                            self.update_attractor_ds_request.attractor_orientation = float(robot_continuous_orientation)
                            self.update_attractor_ds_srv(self.update_attractor_ds_request)
                            self.is_autonomy_turn = True

                            # compute the point at distance R along the line.
                        # update generic pfield position.
                        # set autonomy turn flag.
                        # update mode if necessary

                    # print(robot_discrete_state[:-1], self.env_params['all_mdp_env_params']['all_goals']) #(x,y,t,m)
                    if tuple(robot_discrete_state[:-1]) in self.env_params["all_mdp_env_params"]["all_goals"]:
                        index_goal = self.env_params["all_mdp_env_params"]["all_goals"].index(
                            tuple(robot_discrete_state[:-1])
                        )
                        goal_continuous_position = self.env_params["goal_poses"][index_goal][:-1]
                        if (
                            np.linalg.norm(np.array(robot_continuous_position) - np.array(goal_continuous_position)) < 1.0
                        ):  # determine threshold
                            is_done = True
                else:
                    # what to do during autonomy turn

                    robot_continuous_position = self.env.get_robot_position()
                    robot_continuous_orientation = self.env.get_robot_orientation()
                    self.compute_velocity_request.current_pos = robot_continuous_position
                    self.compute_velocity_request.current_orientation = robot_continuous_orientation
                    if self.algo_condition == "disamb":
                        self.compute_velocity_request.pfield_id = "disamb"  # get disamb pfield vel
                    elif self.algo_condition == "control":
                        self.compute_velocity_request.pfield_id = "generic"

                    vel_response = self.compute_velocity_srv(self.compute_velocity_request)
                    autonomy_vel = list(vel_response.velocity_final)
                    # print(np.linalg.norm(np.array(robot_continuous_position) - np.array(max_disamb_continuous_position)))
                    # print(np.linalg.norm(autonomy_vel))
                    if self.algo_condition == "disamb":
                        # print (
                        #     "dist to disamb ",
                        #     np.linalg.norm(
                        #         np.array(robot_continuous_position) - np.array(max_disamb_continuous_position)
                        #     ),
                        # )
                        if (
                            np.linalg.norm(
                                np.array(robot_continuous_position) - np.array(max_disamb_continuous_position)
                            )
                            < 0.1
                        ):
                            print ("DONE WITH DISAMB PHASE")
                            # reset belief to what it was when the disamb mode was activated.
                            print ("RESET BELIEF")
                            self.reset_belief_request.num_goals = self.env_params["num_goals"]
                            self.reset_belief_request.p_g_given_phm = list(belief_at_disamb_time)
                            self.reset_belief_srv(self.reset_belief_request)

                            self.is_autonomy_turn = False
                            # reset human initiator flag for next turn.
                            self.has_human_initiated = False
                            self.env.set_information_text("Waiting...")
                            self.autonomy_activate_ctr = 0

                        else:
                            # use only autonomy vel to move towards the local disamb state.
                            self.input_action["full_control_signal"] = np.array(autonomy_vel)

                            (
                                robot_continuous_position,
                                robot_continuous_orientation,
                                robot_discrete_state,
                                is_done,
                            ) = self.env.step(self.input_action)
                    elif self.algo_condition == "control":
                        # print (
                        #     " dist to target",
                        #     np.linalg.norm(np.array(robot_continuous_position) - np.array(target_point)),
                        # )
                        if np.linalg.norm(np.array(robot_continuous_position) - np.array(target_point)) < 2.0:
                            print ("DONE WITH AUTONOMY PHASE")
                            # reset belief to what it was when the disamb mode was activated.
                            print ("RESET BELIEF")
                            self.reset_belief_request.num_goals = self.env_params["num_goals"]
                            self.reset_belief_request.p_g_given_phm = list(belief_at_disamb_time)
                            self.reset_belief_srv(self.reset_belief_request)

                            self.is_autonomy_turn = False
                            # reset human initiator flag for next turn.
                            self.has_human_initiated = False
                            self.env.set_information_text("Waiting...")
                            self.autonomy_activate_ctr = 0

                        else:
                            # use only autonomy vel to move towards the local disamb state.
                            self.input_action["full_control_signal"] = np.array(autonomy_vel)

                            (
                                robot_continuous_position,
                                robot_continuous_orientation,
                                robot_discrete_state,
                                is_done,
                            ) = self.env.step(self.input_action)

                    self.env.render()

                
            r.sleep()

    def _prepare_trial_setup(self, mdp_env_params, world_bounds):
        #init pfields
        self._init_goal_pfields(
            self.env_params["goal_poses"],
            mdp_env_params["dynamic_obs_specs"],
            mdp_env_params["cell_size"],
            world_bounds,
        )
        self._init_other_pfields(
            self.env_params["goal_poses"],
            mdp_env_params["dynamic_obs_specs"],
            mdp_env_params["cell_size"],
            world_bounds,
            pfield_id="disamb",
        )
        self._init_other_pfields(
            self.env_params["goal_poses"],
            mdp_env_params["dynamic_obs_specs"],
            mdp_env_params["cell_size"],
            world_bounds,
            pfield_id="generic",
        )

        self.confidence_threshold = 1.05 / len(self.env_params['goal_poses'])
        self.confidence_max =  1.14 / len(self.env_params['goal_poses'])
        self.alpha_max = 0.8
        if self.confidence_max != self.confidence_threshold:
            self.confidence_slope = float(self.alpha_max) / (self.confidence_max - self.confidence_threshold)
        else:
            self.confidence_slope = -1.0

        self.ENTROPY_THRESHOLD = 0.7

        # reinit the environement
        self.env_params["start"] = False
        self.env.update_params(self.env_params)
        self.env.reset()
        self.env.render()
        self.env.set_information_text("Waiting....")

        #update the disamb node
        self.disamb_algo.update_params(self.env_params)

        #reset inference node
        self.init_belief_request.num_goals = self.env_params["num_goals"]
        status = self.init_belief_srv(self.init_belief_request)
        # unfreeze belief update
        self.freeze_update_request.data = False
        self.freeze_update_srv(self.freeze_update_request)

        self.p_g_given_phm = (1.0 / self.env_params["num_goals"]) * np.ones(self.env_params["num_goals"])
        self.autonomy_activate_ctr = 0
        
        self.is_autonomy_turn = False
        self.has_human_initiated = False
        

    def _convert_discrete_state_to_continuous_pose(self, discrete_state, cell_size, world_bounds):
        # could be moved to utils
        x_coord = discrete_state[0]
        y_coord = discrete_state[1]
        theta_coord = discrete_state[2]
        mode = discrete_state[3] - 1  # minus one because the dictionary is 0-indexed

        robot_position = [
            x_coord * cell_size["x"] + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"],
            y_coord * cell_size["y"] + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"],
        ]
        robot_orientation = (theta_coord * 2 * PI) / NUM_ORIENTATIONS
        start_mode = MODE_INDEX_TO_DIM[mode]

        return robot_position, robot_orientation, start_mode

    def joy_callback(self, msg):
        self.input_action["human"] = msg
        # if msg.mode_switch == True:
        #     print(msg)

    def belief_info_callback(self, msg):
        self.p_g_given_phm = np.array(msg.p_g_given_phm)
        # print(msg)

    def shutdown_hook(self, msg_string="DONE"):
        if not self.called_shutdown:
            self.called_shutdown = True
            self.shutdown_pub.publish("shutdown")
            # clear screen
            self.env.render_clear("End of trial...")
            self.env.close_window()
            print ("Shutting down")

    def key_press(self, k, mode):
        if k == key.SPACE:
            self.terminate = True
        if k == key.R:
            self.restart = True
    def _blend_velocities(self, human_vel, autonomy_vel):
        if np.linalg.norm(human_vel) > 1e-5:
            blend_vel = self.alpha * autonomy_vel + (1.0 - self.alpha) * human_vel
        else:
            blend_vel = np.zeros_like(human_vel)
        return blend_vel

    # INITIALIZE PFIELDS
    def _init_pfield_obs_desc(self, obs_param_dict_list, cell_size, world_bounds):
        cell_size_x = cell_size["x"]
        cell_size_y = cell_size["y"]

        common_obs_descs_list = []
        for obs_param_dict in obs_param_dict_list:

            obs_desc = CuboidObs()

            # TODO add world bounds offset
            bottom_left_cell_x = obs_param_dict["bottom_left_cell_x"]
            bottom_left_cell_y = obs_param_dict["bottom_left_cell_y"]
            true_width_of_obs_in_cells = obs_param_dict["true_width_of_obs_in_cells"]
            true_height_of_obs_in_cells = obs_param_dict["true_height_of_obs_in_cells"]

            center_position_x = (bottom_left_cell_x + true_width_of_obs_in_cells * 0.5) * cell_size_x + world_bounds[
                "xrange"
            ]["lb"]
            center_position_y = (bottom_left_cell_y + true_height_of_obs_in_cells * 0.5) * cell_size_y + world_bounds[
                "yrange"
            ]["lb"]
            axes_0 = true_width_of_obs_in_cells * cell_size_x  # assuming axis 0 is the width
            axes_1 = true_height_of_obs_in_cells * cell_size_y  # assuming axis 1 is height

            # populate the obs desc msg
            obs_desc.center_position = [center_position_x, center_position_y]
            obs_desc.orientation = 0.0
            obs_desc.axes_length = [axes_0, axes_1]
            obs_desc.is_boundary = False

            common_obs_descs_list.append(obs_desc)

        # self.init_obstacles_srv(self.init_obstacles_request)

        return common_obs_descs_list

    def _init_other_pfields(
        self, continuous_goal_poses, obs_param_dict_list, cell_size, world_bounds, pfield_id="generic"
    ):
        common_obs_descs_list = self._init_pfield_obs_desc(obs_param_dict_list, cell_size, world_bounds)
        cell_size_x = cell_size["x"]
        cell_size_y = cell_size["y"]
        num_goals = len(continuous_goal_poses)
        self.init_obstacles_request.num_obstacles = len(obs_param_dict_list) + num_goals
        self.init_obstacles_request.obs_descs = []
        self.init_obstacles_request.obs_descs = copy.deepcopy(common_obs_descs_list)
        self.init_obstacles_request.pfield_id = pfield_id
        # add all goals as "obstacles" to the disamb pfield
        for goal_pose in continuous_goal_poses:
            goal_obs_desc = CuboidObs()
            bottom_left_cell_x = goal_pose[0]
            bottom_left_cell_y = goal_pose[1]
            true_width_of_obs_in_cells = 1
            true_height_of_obs_in_cells = 1

            center_position_x = (bottom_left_cell_x + true_width_of_obs_in_cells * 0.5) * cell_size_x + world_bounds[
                "xrange"
            ]["lb"]
            center_position_y = (bottom_left_cell_y + true_height_of_obs_in_cells * 0.5) * cell_size_y + world_bounds[
                "yrange"
            ]["lb"]
            axes_0 = true_width_of_obs_in_cells * cell_size_x  # assuming axis 0 is the width
            axes_1 = true_height_of_obs_in_cells * cell_size_y  # assuming axis 1 is height

            # populate the obs desc msg for other goal pose
            goal_obs_desc.center_position = [center_position_x, center_position_y]
            goal_obs_desc.orientation = 0.0
            goal_obs_desc.axes_length = [axes_0, axes_1]
            goal_obs_desc.is_boundary = False

            self.init_obstacles_request.obs_descs.append(goal_obs_desc)

        assert len(self.init_obstacles_request.obs_descs) == self.init_obstacles_request.num_obstacles

        self.init_obstacles_srv(self.init_obstacles_request)

        self.update_attractor_ds_request.pfield_id = pfield_id
        # dummy attractor pos. Will be update after point towards goal  computation
        self.update_attractor_ds_request.attractor_position = [0.0, 0.0]
        self.update_attractor_ds_request.attractor_orientation = 0.0
        self.update_attractor_ds_srv(self.update_attractor_ds_request)

    def _init_goal_pfields(self, continuous_goal_poses, obs_param_dict_list, cell_size, world_bounds):
        print ("CONT GOAL POSES ", continuous_goal_poses)
        cell_size_x = cell_size["x"]
        cell_size_y = cell_size["y"]
        num_goals = len(continuous_goal_poses)
        common_obs_descs_list = self._init_pfield_obs_desc(obs_param_dict_list, cell_size, world_bounds)
        # init pfield for each goal.
        for pfield_id, goal_pose in enumerate(continuous_goal_poses):
            self.init_obstacles_request.num_obstacles = (
                len(obs_param_dict_list) + num_goals - 1
            )  # common obstacles + other_goals
            self.init_obstacles_request.obs_descs = []
            self.init_obstacles_request.obs_descs = copy.deepcopy(common_obs_descs_list)
            self.init_obstacles_request.pfield_id = "goal_" + str(pfield_id)
            # for loop through all OTHER goals. Create obs_desc for each those other goals. append to obs_descs.
            for other_goal_pose in continuous_goal_poses:
                if np.all(np.equal(other_goal_pose[:-1], goal_pose[:-1])):
                    # if other goal position is same current goal position skip and continue
                    continue
                # create obs_desc for other goal pose
                other_goal_obs_desc = CuboidObs()

                bottom_left_cell_x = other_goal_pose[0]
                bottom_left_cell_y = other_goal_pose[1]
                true_width_of_obs_in_cells = 1
                true_height_of_obs_in_cells = 1

                center_position_x = (
                    bottom_left_cell_x + true_width_of_obs_in_cells * 0.5
                ) * cell_size_x + world_bounds["xrange"]["lb"]
                center_position_y = (
                    bottom_left_cell_y + true_height_of_obs_in_cells * 0.5
                ) * cell_size_y + world_bounds["yrange"]["lb"]
                axes_0 = true_width_of_obs_in_cells * cell_size_x  # assuming axis 0 is the width
                axes_1 = true_height_of_obs_in_cells * cell_size_y  # assuming axis 1 is height

                # populate the obs desc msg for other goal pose
                other_goal_obs_desc.center_position = [center_position_x, center_position_y]
                other_goal_obs_desc.orientation = 0.0
                other_goal_obs_desc.axes_length = [axes_0, axes_1]
                other_goal_obs_desc.is_boundary = False

                self.init_obstacles_request.obs_descs.append(other_goal_obs_desc)

            assert len(self.init_obstacles_request.obs_descs) == self.init_obstacles_request.num_obstacles
            self.init_obstacles_srv(self.init_obstacles_request)

            # init obs_setrvice with pfield_id
            # init attractor with goal_pose[:-1] and pfield_id
            self.update_attractor_ds_request.pfield_id = "goal_" + str(pfield_id)
            self.update_attractor_ds_request.attractor_position = goal_pose[:-1]
            self.update_attractor_ds_request.attractor_orientation = goal_pose[-1]  # goal orientation
            self.update_attractor_ds_srv(self.update_attractor_ds_request)

    def _get_target_along_line(self, start_point, end_point, R=10.0):

        D = np.linalg.norm(np.array(end_point) - np.array(start_point))
        # if D < 10.0: #if autonomy is within 20 pixel go for the goal
        #     R = D
        # else:
        R = min(R, D / 2)
        target_x = start_point[0] + (R / D) * (end_point[0] - start_point[0])
        target_y = start_point[1] + (R / D) * (end_point[1] - start_point[1])

        target = np.array([target_x, target_y])
        print ("DISTANCE ", D, R, start_point, end_point, target)
        return target

    def _get_most_confident_goal(self):
        p_g_given_phm_vector = self.p_g_given_phm + np.finfo(self.p_g_given_phm.dtype).tiny
        uniform_distribution = np.array([1.0 / p_g_given_phm_vector.size] * p_g_given_phm_vector.size)
        max_entropy = -np.dot(uniform_distribution, np.log2(uniform_distribution))
        normalized_h_of_p_g_given_phm = -np.dot(p_g_given_phm_vector, np.log2(p_g_given_phm_vector)) / max_entropy
        argmax_goal_id = np.argmax(p_g_given_phm_vector)
        argmax_goal_id_str = "goal_" + str(argmax_goal_id)
        if normalized_h_of_p_g_given_phm <= self.ENTROPY_THRESHOLD:
            inferred_goal_id = np.argmax(p_g_given_phm_vector)
            goal_id_str = "goal_" + str(inferred_goal_id)
            inferred_goal_prob = p_g_given_phm_vector[inferred_goal_id]
            return (
                goal_id_str,
                inferred_goal_id,
                inferred_goal_prob,
                normalized_h_of_p_g_given_phm,
                argmax_goal_id,
                argmax_goal_id_str,
            )
        else:
            # if entropy not greater than threshold return None as there is no confident goal
            return None, None, None, normalized_h_of_p_g_given_phm, argmax_goal_id, argmax_goal_id_str

    def _compute_alpha(self, inferred_goal_prob):
        if self.confidence_slope != -1.0:
            if inferred_goal_prob <= self.confidence_threshold:
                return 0.0
            elif inferred_goal_prob > self.confidence_threshold and inferred_goal_prob <= self.confidence_max:
                return self.confidence_slope * (inferred_goal_prob - self.confidence_threshold)
            elif inferred_goal_prob > self.confidence_max and inferred_goal_prob <= 1.0:
                return self.alpha_max
        else:
            if inferred_goal_prob <= self.confidence_threshold:
                return 0.0
            else:
                return self.alpha_max

    def _dist_based_weight(self, inferred_goal_position, robot_continuous_position, D=35.0, scale_factor=5):

        d = np.linalg.norm(inferred_goal_position - np.array(robot_continuous_position))
        weight_D = 0.6
        if d <= D:
            slope = -((1.0 - weight_D) / D)
            dist_weight = slope * d + 1.0
        elif d > D:
            dist_weight = weight_D * np.exp(-(d - D))
        return dist_weight

    


if __name__ == "__main__":
    subject_id = sys.argv[1]
    algo_condition_block = sys.argv[2]
    block_id = sys.argv[3]
    training = int(sys.argv[4])

    print type(subject_id), type(block_id), type(training)
    Simulator(subject_id, algo_condition_block, block_id, training)
    rospy.spin()
