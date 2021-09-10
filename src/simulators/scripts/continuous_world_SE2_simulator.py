#!/usr/bin/env python

import collections
import rospy
import time
from sensor_msgs.msg import Joy
from envs.continuous_world_SE2_env import ContinuousWorldSE2Env
from sim_pfields.msg import CuboidObs
from sim_pfields.srv import CuboidObsList, CuboidObsListRequest,CuboidObsListResponse
from sim_pfields.srv import AttractorPos, AttractorPosRequest, AttractorPosResponse
from sim_pfields.srv import ComputeVelocity, ComputeVelocityRequest, ComputeVelocityResponse

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension, String, Int8
from teleop_nodes.msg import InterfaceSignal
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
from adaptive_assistance_sim_utils import *

GRID_WIDTH = 10
GRID_HEIGHT = 10
NUM_ORIENTATIONS = 8
NUM_GOALS = 1
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
        user_vel = InterfaceSignal()

        self.input_action = {}
        self.input_action["full_control_signal"] = user_vel

        rospy.Subscriber("/user_vel", InterfaceSignal, self.joy_callback)
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

        rospy.loginfo("Waiting for sim_pfields node")
        rospy.wait_for_service("/sim_pfields/init_obstacles")
        rospy.wait_for_service("/sim_pfields/update_ds")
        rospy.wait_for_service("/sim_pfields/compute_velocity")
        rospy.loginfo("sim pfields node services found! ")

        self.init_obstacles_srv = rospy.ServiceProxy('/sim_pfields/init_obstacles', CuboidObsList)
        self.init_obstacles_request = CuboidObsListRequest()

        self.update_attractor_ds_srv = rospy.ServiceProxy('/sim_pfields/update_ds', AttractorPos)
        self.update_attractor_ds_request = AttractorPosRequest()

        self.compute_velocity_srv = rospy.ServiceProxy('/sim_pfields/compute_velocity', ComputeVelocity)
        self.compute_velocity_request = ComputeVelocityRequest()

        self.terminate = False
        self.restart = False
        if self.trial_info_dir_path is not None and os.path.exists(self.trial_info_dir_path) and not self.training:
            pass
        else:
            if not self.training:
                mdp_env_params = self._create_mdp_env_param_dict()
                print('DYN OBS SPECS', mdp_env_params['dynamic_obs_specs'])

                # _init_pfields_obstacles(mdp_env_params['dynamic_obs_specs'])
                mdp_env_params["cell_size"] = collections.OrderedDict()

                # create mdp list here. Select start positoin from valid stats.
                # generate continuous world bounds from width and height and cell size, and offset info
                world_bounds = collections.OrderedDict()
                world_bounds["xrange"] = collections.OrderedDict()
                world_bounds["yrange"] = collections.OrderedDict()
                # bottom left corner in continuous space
                world_bounds["xrange"]["lb"] = 0.05 * VIEWPORT_W / SCALE
                world_bounds["yrange"]["lb"] = 0.05 * VIEWPORT_H / SCALE
                world_bounds["xrange"]["ub"] = 0.75 * VIEWPORT_W / SCALE
                world_bounds["yrange"]["ub"] = 0.9 * VIEWPORT_H / SCALE
                mdp_env_params['cell_size']['x'] = (world_bounds["xrange"]["ub"] - world_bounds["xrange"]["lb"]) / mdp_env_params['grid_width']
                mdp_env_params['cell_size']['y'] = (world_bounds["yrange"]["ub"] - world_bounds["yrange"]["lb"]) / mdp_env_params['grid_height']
                

                self._init_pfield_obs_desc(mdp_env_params['dynamic_obs_specs'], mdp_env_params['cell_size'], world_bounds)

                self.env_params = dict()
                self.env_params["all_mdp_env_params"] = mdp_env_params
                

                self.env_params["world_bounds"] = world_bounds

                mdp_list = self.create_mdp_list(self.env_params["all_mdp_env_params"])
                self.env_params["mdp_list"] = mdp_list
                self.env_params["num_goals"] = NUM_GOALS
                self.env_params["is_visualize_grid"] = True

                print ("GOALS", mdp_env_params["all_goals"])
                discrete_robot_state = mdp_list[0].get_random_valid_state()
                robot_position, robot_orientation, start_mode = self._convert_discrete_state_to_continuous_pose(
                    discrete_robot_state, mdp_env_params["cell_size"], world_bounds
                )
                self.env_params["robot_position"] = robot_position
                self.env_params["robot_orientation"] = robot_orientation
                self.env_params["start_mode"] = start_mode
                self.env_params["robot_type"] = CartesianRobotType.SE2
                self.env_params["mode_set_type"] = ModeSetType.OneD
                self.env_params["mode_transition_type"] = ModeTransitionType.Forward_Backward
                # generate continuous obstacle bounds based
                self.env_params["obstacles"] = []
                for o in mdp_env_params["original_mdp_obstacles"]:
                    obs = collections.OrderedDict()
                    obs["bottom_left"] = (
                        o[0] * mdp_env_params["cell_size"]['x'] + world_bounds["xrange"]["lb"],
                        o[1] * mdp_env_params["cell_size"]['y'] + world_bounds["yrange"]["lb"],
                    )
                    obs["top_right"] = (
                        (o[0] + 1) * mdp_env_params["cell_size"]['x'] + world_bounds["xrange"]["lb"],
                        (o[1] + 1) * mdp_env_params["cell_size"]['y'] + world_bounds["yrange"]["lb"],
                    )
                    self.env_params["obstacles"].append(obs)

                self.env_params["goal_poses"] = self._generate_continuous_goal_poses(
                    mdp_env_params["all_goals"], mdp_env_params["cell_size"], self.env_params["world_bounds"]
                )
                self._init_goal_attractor_pose(self.env_params['goal_poses'][0])
                print('cell size ', mdp_env_params["cell_size"] )
                self.env_params['assistance_type'] = 1
            else:
                pass
        
        # instantiate the environement
        self.env_params["start"] = False
        self.env = ContinuousWorldSE2Env(self.env_params)
        self.env.initialize()
        self.env.initialize_viewer()
        self.env.reset()
        self.env.viewer.window.on_key_press = self.key_press

        rospy.loginfo("Waiting for goal inference node")
        rospy.wait_for_service("/goal_inference/init_belief")
        rospy.loginfo("goal inference node service found! ")

        self.init_belief_srv = rospy.ServiceProxy("/goal_inference/init_belief", InitBelief)
        self.init_belief_request = InitBeliefRequest()
        self.init_belief_request.num_goals = self.env_params["num_goals"]
        status = self.init_belief_srv(self.init_belief_request)
        
        

        #init pfield nodes with 
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
                
                robot_continuous_position = self.env.get_robot_position()
                self.compute_velocity_request.current_pos  = robot_continuous_position
                vel_response = self.compute_velocity_srv(self.compute_velocity_request)
                # self.input_action['full_control_signal'] = vel_response.velocity_final

                # compute mode_conditioned_velocity for self.input_action['human'] directly access the self.robot._mode_conditioned_velocity via env
                # this would 3D vel.
                # Blend the above with vel_response.velocity_final. TODO add rotational vel to the sim_pfield_node
                # Update step function in robot to deal with full 3D vel. 
                print('velocity', vel_response.velocity_final)
                
                #get current robot_pose
                #get current belief. and entropy of belief. compute argmax g
                #get pfield vel for argmax g and current robot pose
                #blend velocity by combining it with user vel.
                # apply blend velocity to robot. 
                # if uservel is Null for 2 seconds, activate disamb mode. 
                # # get current discrete state, compute nearby states.
                # # compute MI. get disamb discrete state
                # # convert to continuous disamb state (centre of disamb discrete state)
                # # change goal for disamb pfield
                # # get disamb pfield vel. Continue in disamb mode, until current robot_pose is eps within contonuious disamb state
                if self.restart:
                    pass

                (
                    robot_continuous_position,
                    robot_continuous_orientation,
                    is_done,
                ) = self.env.step(self.input_action)

                if self.terminate:
                    self.shutdown_hook("Session terminated")
                    break

                self.env.render()
            
            r.sleep()
    
    def _generate_continuous_goal_poses(self, discrete_goal_list, cell_size, world_bounds):
        goal_poses = []
        for dg in discrete_goal_list:
            goal_pose = [0, 0, 0]
            goal_pose[0] = (dg[0] * cell_size['x']) + cell_size['x'] / 2.0 + world_bounds["xrange"]["lb"]
            goal_pose[1] = (dg[1] * cell_size['y']) + cell_size['y'] / 2.0 + world_bounds["yrange"]["lb"]
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
            x_coord * cell_size['x'] + cell_size['x'] / 2.0 + world_bounds["xrange"]["lb"],
            y_coord * cell_size['y'] + cell_size['y'] / 2.0 + world_bounds["yrange"]["lb"],
        ]
        robot_orientation = (theta_coord * 2 * PI) / NUM_ORIENTATIONS
        start_mode = MODE_INDEX_TO_DIM[mode]

        return robot_position, robot_orientation, start_mode

    def joy_callback(self, msg):
        self.input_action['full_control_signal'] = msg
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

    def _create_mdp_env_param_dict(self):
        mdp_env_params = collections.OrderedDict()
        mdp_env_params["rl_algo_type"] = RlAlgoType.ValueIteration
        mdp_env_params["gamma"] = 0.96
        mdp_env_params["grid_width"] = GRID_WIDTH
        mdp_env_params["grid_height"] = GRID_HEIGHT
        mdp_env_params["num_discrete_orientations"] = NUM_ORIENTATIONS
        mdp_env_params["robot_type"] = CartesianRobotType.SE2
        mdp_env_params["mode_set_type"] = ModeSetType.OneD

        num_patches = 2 #two patches of obstacles. 
        if OCCUPANCY_LEVEL == 0.0:
            mdp_env_params["original_mdp_obstacles"] = []
        else:
            mdp_env_params["original_mdp_obstacles"], dynamics_obs_specs = self._create_rectangular_gw_obstacles(
                width=mdp_env_params["grid_width"],
                height=mdp_env_params["grid_height"],
                num_obstacle_patches=num_patches,
            )
            print(mdp_env_params["original_mdp_obstacles"]  )
            # mdp_env_params['original_mdp_obstacles'] = [(0,0), (mdp_env_params["grid_width"] - 1, 0), (0, mdp_env_params["grid_height"]-1)]


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
        mdp_env_params['dynamic_obs_specs'] = dynamics_obs_specs

        return mdp_env_params

    def _init_goal_attractor_pose(self, continuous_goal_pose):
        self.update_attractor_ds_request.attractor_position = continuous_goal_pose[:-1]
        self.update_attractor_ds_srv(self.update_attractor_ds_request)

    def _init_pfield_obs_desc(self, obs_param_dict_list, cell_size, world_bounds):
        cell_size_x = cell_size['x']
        cell_size_y = cell_size['y']
        
        self.init_obstacles_request.num_obstacles = len(obs_param_dict_list)
        self.init_obstacles_request.obs_descs = []

        for obs_param_dict in obs_param_dict_list:

            obs_desc = CuboidObs()

            #TODO add world bounds offset
            bottom_left_cell_x = obs_param_dict['bottom_left_cell_x']
            bottom_left_cell_y = obs_param_dict['bottom_left_cell_y']
            true_width_of_obs_in_cells = obs_param_dict['true_width_of_obs_in_cells']
            true_height_of_obs_in_cells = obs_param_dict['true_height_of_obs_in_cells']

            center_position_x = (bottom_left_cell_x  + true_width_of_obs_in_cells * 0.5) * cell_size_x
            center_position_y = (bottom_left_cell_y  + true_height_of_obs_in_cells * 0.5) * cell_size_y
            axes_0 = true_width_of_obs_in_cells * cell_size_x #assuming axis 0 is the width
            axes_1 = true_height_of_obs_in_cells * cell_size_y #assuming axis 1 is height
            
            # populate the obs desc msg
            obs_desc.center_position = [center_position_x, center_position_y]
            obs_desc.orientation = 0.0
            obs_desc.axes_length = [axes_0, axes_1]
            obs_desc.is_boundary = False

            self.init_obstacles_request.obs_descs.append(obs_desc)
        
        self.init_obstacles_srv(self.init_obstacles_request)


    def _create_rectangular_gw_obstacles(self, width, height, num_obstacle_patches):
        
        obstacle_list = []
        dynamic_obs_specs = []
        all_cell_coords = list(itertools.product(range(width), range(height)))
        # pick three random starting points
        obstacle_patch_seeds = random.sample(all_cell_coords, num_obstacle_patches)
        for i, patch_seed in enumerate(obstacle_patch_seeds):
            
            width_of_obs = np.random.randint(1, 3) #width of obstacle in cells
            height_of_obs = np.random.randint(1, 3) #height of obstacles in cells

            w_range = list(range(patch_seed[0], min(width - 1, patch_seed[0] + width_of_obs - 1) + 1))
            h_range = list(range(patch_seed[1], min(height - 1, patch_seed[1] + height_of_obs -1) + 1))

            bottom_left_cell_x = patch_seed[0]
            bottom_left_cell_y = patch_seed[1]
            right_most_cell_id = min(width - 1, patch_seed[0] + width_of_obs - 1)
            true_width_of_obs_in_cells = right_most_cell_id - patch_seed[0] + 1
            top_most_cell_id = min(height - 1, patch_seed[1] + height_of_obs -1) 
            true_height_of_obs_in_cells = top_most_cell_id - patch_seed[1] + 1

            dyn_obs_desc_dict = collections.OrderedDict()
            dyn_obs_desc_dict['bottom_left_cell_x'] = bottom_left_cell_x
            dyn_obs_desc_dict['bottom_left_cell_y'] = bottom_left_cell_y
            dyn_obs_desc_dict['true_width_of_obs_in_cells'] = true_width_of_obs_in_cells
            dyn_obs_desc_dict['true_height_of_obs_in_cells'] = true_height_of_obs_in_cells

            print('PATCH ', list(itertools.product(w_range, h_range)))
            obstacle_list.extend(list(itertools.product(w_range, h_range)))
            dynamic_obs_specs.append(dyn_obs_desc_dict)

        return obstacle_list, dynamic_obs_specs

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
