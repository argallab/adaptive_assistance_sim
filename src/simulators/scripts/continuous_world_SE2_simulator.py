#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import collections
import rospy
import time
from sensor_msgs.msg import Joy
from envs.continuous_world_SE2_env import ContinuousWorldSE2Env
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension, String, Int8
from teleop_nodes.msg import CartVelCmd
from simulators.msg import State
from simulators.srv import InitBelief, InitBeliefRequest, InitBeliefResponse
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
from corrective_mode_switch_utils import (
    MODE_INDEX_TO_DIM,
    SCALE,
    DIM_TO_MODE_INDEX,
    VIEWPORT_W,
    VIEWPORT_H,
    PI,
    ROBOT_RADIUS_S,
)


GRID_WIDTH = 10
GRID_HEIGHT = 10
NUM_ORIENTATIONS = 8
NUM_GOALS = 3
OCCUPANCY_LEVEL = 0.1

SPARSITY_FACTOR = 0.0
RAND_DIRECTION_FACTOR = 0.1


class Simulator(object):
    def __init__(self, subject_id, assistance_block, block_id, training):
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
        user_vel = CartVelCmd()

        _dim = [MultiArrayDimension()]
        _dim[0].label = "cartesian_velocity"
        _dim[0].size = self.dim
        _dim[0].stride = self.dim
        user_vel.velocity.layout.dim = _dim
        user_vel.velocity.data = np.zeros(self.dim)
        user_vel.header.stamp = rospy.Time.now()
        user_vel.header.frame_id = "human_control"

        self.input_action = {}
        self.input_action["human"] = user_vel
        rospy.Subscriber("/user_vel", CartVelCmd, self.joy_callback)
        self.trial_index = 0

        self.env_params = None
        self.trial_info_dir_path = os.path.join(os.path.dirname(__file__), "se2_trial_dir")
        self.metadata_dir = os.path.join(os.path.dirname(__file__), "se2_metadata_dir")

        self.subject_id = subject_id
        self.assistance_block = assistance_block  # pass these things from launch file
        self.block_id = block_id
        self.training = training
        self.total_blocks = 6

        self.testing_block_filename = (
            self.subject_id
            + "_"
            + self.assistance_block
            + "_assistance_"
            + self.block_id
            + "_num_blocks_"
            + str(self.total_blocks)
            + ".pkl"
        )
        print "TRAINING BLOCK FILENAME and IS TRAINING MODE", self.testing_block_filename, self.training

        self.terminate = False
        self.restart = False
        if self.trial_info_dir_path is not None and os.path.exists(self.trial_info_dir_path) and not self.training:
            pass
        else:
            if not self.training:
                mdp_env_params = self._create_mdp_env_param_dict()
                mdp_env_params["cell_size"] = ROBOT_RADIUS_S * 2.1

                # create mdp list here. Select start positoin from valid stats.
                # pass mdp_list so that env doesn't waste time creating it.
                self.env_params = dict()
                self.env_params["all_mdp_env_params"] = mdp_env_params
                # generate continuous world bounds from width and height and cell size, and offset info
                world_bounds = collections.OrderedDict()
                world_bounds["xrange"] = collections.OrderedDict()
                world_bounds["yrange"] = collections.OrderedDict()
                # bottom left corner in continuous space
                world_bounds["xrange"]["lb"] = 0.1 * VIEWPORT_W / SCALE
                world_bounds["yrange"]["lb"] = 0.1 * VIEWPORT_H / SCALE

                world_bounds["xrange"]["ub"] = world_bounds["xrange"]["lb"] + (
                    mdp_env_params["grid_width"] * mdp_env_params["cell_size"]
                )
                world_bounds["yrange"]["ub"] = world_bounds["yrange"]["lb"] + (
                    mdp_env_params["grid_height"] * mdp_env_params["cell_size"]
                )

                self.env_params["world_bounds"] = world_bounds

                mdp_list = self.create_mdp_list(self.env_params["all_mdp_env_params"])
                self.env_params["mdp_list"] = mdp_list
                self.env_params["num_goals"] = NUM_GOALS

                print ("GOALS", mdp_env_params["all_goals"])
                discrete_robot_state = mdp_list[0].get_random_valid_state()
                robot_position, robot_orientation, start_mode = self._convert_discrete_state_to_continuous_pose(
                    discrete_robot_state, mdp_env_params["cell_size"], world_bounds
                )
                self.env_params["robot_position"] = robot_position
                self.env_params["robot_orientation"] = robot_orientation
                self.env_params["start_mode"] = start_mode

                # generate continuous obstacle bounds based
                self.env_params["obstacles"] = []
                for o in mdp_env_params["original_mdp_obstacles"]:
                    obs = collections.OrderedDict()
                    obs["bottom_left"] = (
                        o[0] * mdp_env_params["cell_size"] + world_bounds["xrange"]["lb"],
                        o[1] * mdp_env_params["cell_size"] + world_bounds["yrange"]["lb"],
                    )
                    obs["top_right"] = (
                        (o[0] + 1) * mdp_env_params["cell_size"] + world_bounds["xrange"]["lb"],
                        (o[1] + 1) * mdp_env_params["cell_size"] + world_bounds["yrange"]["lb"],
                    )
                    self.env_params["obstacles"].append(obs)

                self.env_params["goal_poses"] = self._generate_continuous_goal_poses(
                    mdp_env_params["all_goals"], mdp_env_params["cell_size"], self.env_params["world_bounds"]
                )
                self.env_params['assistance_type'] = 1
            else:
                pass

        rospy.set_param("assistance_type", self.env_params["assistance_type"])
        rospy.loginfo("Waiting for teleop_node ")
        rospy.wait_for_service("/teleop_node/set_mode")
        rospy.loginfo("teleop_node node service found! ")

        # set starting mode for the trial
        self.set_mode_srv = rospy.ServiceProxy("/teleop_node/set_mode", SetMode)
        self.set_mode_request = SetModeRequest()
        self.set_mode_request.mode_index = DIM_TO_MODE_INDEX[self.env_params["start_mode"]]
        status = self.set_mode_srv(self.set_mode_request)

        rospy.loginfo("Waiting for inference node")
        rospy.wait_for_service("/goal_inference_and_correction/init_belief")
        rospy.loginfo("inference node service found! ")

        self.init_belief_srv = rospy.ServiceProxy("/goal_inference_and_correction/init_belief", InitBelief)
        self.init_belief_request = InitBeliefRequest()
        self.init_belief_request.num_goals = self.env_params["num_goals"]
        status = self.init_belief_srv(self.init_belief_request)

        # instantiate the environement
        self.env_params["start"] = False
        self.env = ContinuousWorldSE2Env(self.env_params)
        self.env.initialize()
        self.env.initialize_viewer()
        self.env.viewer.window.on_key_press = self.key_press

        r = rospy.Rate(100)
        self.trial_start_time = time.time()
        if not self.training:
            self.max_time = 1000
        else:
            self.max_time = 1000
        is_done = False
        first_trial = True
        self.start = False

        while not rospy.is_shutdown():
            if not self.start:
                self.start = self.env.start_countdown()
            else:
                if first_trial:
                    time.sleep(2)
                    self.trial_marker_pub.publish("start")
                    # self.trial_index_pub.publish(trial_info_filename_index)
                    self.env.reset()
                    self.env.render()
                    self.trial_start_time = time.time()
                    first_trial = False
                else:
                    if (time.time() - self.trial_start_time) > self.max_time or is_done:
                        if not self.training:
                            print ("Move to NEXT TRIAL")
                            self.trial_marker_pub.publish("end")
                            self.env.render_clear("Loading next trial ...")
                            time.sleep(5.0)  # sleep before the next trial happens
                            self.trial_index += 1
                            if self.trial_index == 2:
                                self.shutdown_hook("Reached end of trial list. End of session")
                                break  # experiment is done

                        else:
                            self.shutdown_hook("Reached end of training")
                            break

                if self.restart:
                    pass

                (
                    robot_continuous_position,
                    robot_continuous_orientation,
                    robot_linear_velocity,
                    robot_angular_velocity,
                    robot_current_mode,
                    is_done,
                ) = self.env.step(self.input_action)

                self.robot_state.header.stamp = rospy.Time.now()
                self.robot_state.robot_continuous_position = robot_continuous_position
                self.robot_state.robot_continuous_orientation = robot_continuous_orientation
                self.robot_state.robot_linear_velocity = robot_linear_velocity
                self.robot_state.robot_angular_velocity = robot_angular_velocity

                self.robot_state.robot_discrete_state.mode = robot_current_mode

                self.robot_state_pub.publish(self.robot_state)
                if self.terminate:
                    self.shutdown_hook("Session terminated")
                    break

                self.env.render()

            r.sleep()

    def _generate_continuous_goal_poses(self, discrete_goal_list, cell_size, world_bounds):
        goal_poses = []
        for dg in discrete_goal_list:
            goal_pose = [0, 0, 0]
            goal_pose[0] = (dg[0] * cell_size) + cell_size / 2.0 + world_bounds["xrange"]["lb"]
            goal_pose[1] = (dg[1] * cell_size) + cell_size / 2.0 + world_bounds["yrange"]["lb"]
            goal_pose[2] = (float(dg[2]) / NUM_ORIENTATIONS) * 2 * PI
            goal_poses.append(goal_pose)

        return goal_poses

    def _random_robot_pose(self):
        robot_position = [0.5 * VIEWPORT_W / SCALE, 0.25 * VIEWPORT_H / SCALE]
        robot_orientation = 0.0
        # add proximity checks to any goals
        return (robot_position, robot_orientation)

    def _convert_discrete_state_to_continuous_pose(self, discrete_state, cell_size, world_bounds):
        x_coord = discrete_state[0]
        y_coord = discrete_state[1]
        theta_coord = discrete_state[2]
        mode = discrete_state[3] - 1  # minus one because the dictionary is 0-indexed

        robot_position = [
            x_coord * cell_size + cell_size / 2.0 + world_bounds["xrange"]["lb"],
            y_coord * cell_size + cell_size / 2.0 + world_bounds["yrange"]["lb"],
        ]
        robot_orientation = (theta_coord * 2 * PI) / NUM_ORIENTATIONS
        start_mode = MODE_INDEX_TO_DIM[mode]

        return robot_position, robot_orientation, start_mode

    def joy_callback(self, msg):
        self.input_action["human"] = msg

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

    def _create_mdp_env_param_dict(self):
        mdp_env_params = collections.OrderedDict()
        mdp_env_params["rl_algo_type"] = RlAlgoType.ValueIteration
        mdp_env_params["gamma"] = 0.96
        mdp_env_params["grid_width"] = GRID_WIDTH
        mdp_env_params["grid_height"] = GRID_HEIGHT
        mdp_env_params["num_discrete_orientations"] = NUM_ORIENTATIONS
        mdp_env_params["robot_type"] = CartesianRobotType.SE2
        mdp_env_params["mode_set_type"] = ModeSetType.OneD

        num_patches = 2
        if OCCUPANCY_LEVEL == 0.0:
            mdp_env_params["original_mdp_obstacles"] = []
        else:
            mdp_env_params["original_mdp_obstacles"] = self._create_rectangular_gw_obstacles(
                width=mdp_env_params["grid_width"],
                height=mdp_env_params["grid_height"],
                num_obstacle_patches=num_patches,
            )

        print ("OBSTACLES", mdp_env_params["original_mdp_obstacles"])
        goal_list = create_random_goals(
            width=mdp_env_params["grid_width"],
            height=mdp_env_params["grid_height"],
            num_goals=NUM_GOALS,
            obstacle_list=mdp_env_params["original_mdp_obstacles"],
        )  # make the list a tuple

        for i, g in enumerate(goal_list):
            g = list(g)
            g.append(np.random.randint(mdp_env_params["num_discrete_orientations"]))
            goal_list[i] = tuple(g)

        print (goal_list)
        mdp_env_params["all_goals"] = goal_list
        mdp_env_params["obstacle_penalty"] = -100
        mdp_env_params["goal_reward"] = 100
        mdp_env_params["step_penalty"] = -10
        mdp_env_params["sparsity_factor"] = SPARSITY_FACTOR
        mdp_env_params["rand_direction_factor"] = RAND_DIRECTION_FACTOR
        mdp_env_params["mdp_obstacles"] = []

        return mdp_env_params

    def _create_rectangular_gw_obstacles(self, width, height, num_obstacle_patches):

        obstacle_list = []
        all_cell_coords = list(itertools.product(range(width), range(height)))
        # pick three random starting points
        obstacle_patch_seeds = random.sample(all_cell_coords, num_obstacle_patches)
        for i, patch_seed in enumerate(obstacle_patch_seeds):

            width_of_obs = np.random.randint(1, 3)
            height_of_obs = np.random.randint(1, 3)

            h_range = list(range(patch_seed[0], min(height - 1, patch_seed[0] + height_of_obs) + 1))
            w_range = list(range(patch_seed[1], min(width - 1, patch_seed[1] + width_of_obs) + 1))
            obstacle_list.extend(list(itertools.product(h_range, w_range)))

        return obstacle_list

    def create_mdp_list(self, mdp_env_params):
        mdp_list = []
        for i, g in enumerate(mdp_env_params["all_goals"]):
            mdp_env_params["mdp_goal_state"] = g
            # 2d goals.
            goals_that_are_obs = [(g_obs[0], g_obs[1]) for g_obs in mdp_env_params["all_goals"] if g_obs != g]
            mdp_env_params["mdp_obstacles"] = copy.deepcopy(mdp_env_params["original_mdp_obstacles"])
            mdp_env_params["mdp_obstacles"].extend(goals_that_are_obs)
            discrete_se2_modes_mdp = MDPDiscreteSE2GridWorldWithModes(copy.deepcopy(mdp_env_params))

            mdp_list.append(discrete_se2_modes_mdp)

        return mdp_list


if __name__ == "__main__":
    subject_id = sys.argv[1]
    assistance_block = sys.argv[2]
    block_id = sys.argv[3]
    training = int(sys.argv[4])
    print type(subject_id), type(block_id), type(training)
    Simulator(subject_id, assistance_block, block_id, training)
    rospy.spin()
