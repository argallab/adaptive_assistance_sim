#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import itertools
import random
import os
import pickle
import numpy as np
import collections
from envs.utils import *
from mdp.autonomy_multi_goal_2d_gw_modes_controller import AutonomyMultiGoal2DGridWorldModesController


RAND_DIRECTION_LEVELS = [0.0]
SPARSITY_LEVELS = [0.0]
# OCCUPANCY_LEVELS = [0.05, 0.1]
NUM_GOALS = [3, 4]
OCCUPANCY_LEVELS = [0.05, 0.1, 0.15]

GRID_WIDTH = 25
GRID_HEIGHT = 30
START_DIST_THRESHOLD = 4
INTER_GOAL_THRESHOLD = 4

def create_random_obstacles(width, height, occupancy_measure, num_obstacle_patches, dirichlet_scale_param=10):
    assert occupancy_measure < 1.0 and occupancy_measure >= 0.0
    num_cells = width * height
    num_occupied_cells = int(round(occupancy_measure * num_cells))
    num_cells_for_all_patches = list(np.int32(np.round(num_occupied_cells*np.random.dirichlet(np.ones(num_obstacle_patches)*dirichlet_scale_param))))

    all_cell_coords = list(itertools.product(range(width), range(height)))
    #pick three random starting points
    obstacle_patch_seeds = random.sample(all_cell_coords, num_obstacle_patches)

    def get_random_obstacle_neighbors(obs):
        def check_bounds(state):
            state[0] = max(0, min(state[0], width-1))
            state[1] = max(0, min(state[1], height-1))
            return state

        top_neighbor = tuple(check_bounds(np.array(obs) + (0, 1)))
        bottom_neighbor = tuple(check_bounds(np.array(obs) + (0, -1)))
        left_neighbor = tuple(check_bounds(np.array(obs) + (-1, 0)))
        right_neighbor = tuple(check_bounds(np.array(obs) + (1, 0)))

        all_neighbors = [top_neighbor, bottom_neighbor, left_neighbor, right_neighbor]
        # return all_neighbors
        num_neighbors_to_be_returned = random.randint(1, len(all_neighbors))
        return random.sample(all_neighbors, num_neighbors_to_be_returned)

    obstacle_list = []
    for i, (num_cells_for_patch, patch_seed) in enumerate(zip(num_cells_for_all_patches, obstacle_patch_seeds)):
        # print('Creating obstacle patch ', i)
        obstacles_in_patch = [tuple(patch_seed)]
        while len(obstacles_in_patch) <= num_cells_for_patch:
            new_cells = []
            for obs in obstacles_in_patch:
                new_cells.extend(get_random_obstacle_neighbors(obs))
            obstacles_in_patch.extend(new_cells)
            obstacles_in_patch = list(set(obstacles_in_patch)) #remove duplicates

        obstacle_list.extend(obstacles_in_patch)

    return obstacle_list

def create_random_goals(width, height, num_goals, obstacle_list):
    all_cell_coords = list(itertools.product(range(width), range(height)))
    random_goals = []
    sampled_goal = random.sample(list(set(all_cell_coords) - set(obstacle_list)- set(random_goals)), 1)[0]
    random_goals.append(sampled_goal) #add the first goal into the array.
    print(random_goals)
    while len(random_goals) < num_goals:
        sampled_goal = random.sample(list(set(all_cell_coords) - set(obstacle_list)- set(random_goals)), 1)[0] #tuple
        dist_to_goals = [np.linalg.norm(np.array(sampled_goal) - np.array(g)) for g in random_goals]
        if min(dist_to_goals) > INTER_GOAL_THRESHOLD:
            random_goals.append(sampled_goal)
        else:
            continue
    
    return random_goals


#Modify the following with **kwargs to deal with R2, R2modes, SE2 and SE2Modes
    
def create_random_start_state(width, height, obstacle_list, goal_list, mode_set):
    all_cell_coords = list(itertools.product(range(width), range(height)))
    dist_to_goals = [-1000]
    while min(dist_to_goals) < START_DIST_THRESHOLD: 
        random_start_state = random.sample(list(set(all_cell_coords) - set(goal_list) - set(obstacle_list)), 1)[0]
        dist_to_goals = [np.linalg.norm(np.array(random_start_state) - np.array(g)) for g in goal_list]
    
    random_mode = random.sample(mode_set.keys(), 1) #[m] 
    
    return tuple(list(random_start_state) + random_mode) #a tuple

def generate_experiment_trials(args):
    num_reps_per_condition = args.num_reps_per_condition
    trial_dir = args.trial_dir
    metadata_dir = args.metadata_dir
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
    
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    
    index = 0
    condition_to_pkl_index = collections.defaultdict(list)

    for rand_direction_factor, sparsity_level, occupancy_level, num_goals in itertools.product(RAND_DIRECTION_LEVELS, SPARSITY_LEVELS, OCCUPANCY_LEVELS, NUM_GOALS):
        grid_world_names = ['shared']
        print("                          ")
        for _ in range(num_reps_per_condition):
            trial_info_dict = collections.OrderedDict()
            trial_info_dict['env_params'] = collections.OrderedDict()
            trial_info_dict['mdp_info_dict'] = collections.OrderedDict()
            trial_info_dict['mdp_info_dict']['shared'] = collections.OrderedDict()
            trial_info_dict['mdp_info_dict']['shared']['state_value_function'] = collections.OrderedDict() #one for each goal in the gridworld
            trial_info_dict['mdp_info_dict']['shared']['action_value_function'] = collections.OrderedDict() 
            trial_info_dict['mdp_info_dict']['shared']['policy'] = collections.OrderedDict() #one for each goal in the gridworld. 
            
            # trial_info_dict['mdp_info_dict']['autonomy'] = collections.OrderedDict()
            # trial_info_dict['mdp_info_dict']['autonomy']['value_function'] = collections.OrderedDict()
            # trial_info_dict['mdp_info_dict']['autonomy']['policy'] = collections.OrderedDict()
            
            for _, gw_name in enumerate(grid_world_names):
                print('GW TYPE: ', gw_name)
                ep = collections.OrderedDict()
                ep['grid_boundary_offset'] = GRID_BOUNDS_OFFSET
                ep['grid_world_name'] = gw_name
                ep['grid_width'] = GRID_WIDTH
                ep['grid_height'] = GRID_HEIGHT
                ep['grid_num_obs_patches'] = np.random.randint(2) + 1
                ep['grid_obs_occupancy_measure'] = occupancy_level
                ep['grid_obstacles'] = create_random_obstacles( width=ep['grid_width'],
                                                                height=ep['grid_height'],
                                                                occupancy_measure=ep['grid_obs_occupancy_measure'],
                                                                num_obstacle_patches=ep['grid_num_obs_patches'])
                ep['grid_bounds'] = collections.OrderedDict()
                ep['grid_goal_states'] = create_random_goals(width=ep['grid_width'],
                                                            height=ep['grid_height'],
                                                            num_goals=num_goals,
                                                            obstacle_list=ep['grid_obstacles'])
                # ep['grid_goal_states'] = [(2, 14), (16, 14), (26, 14)]
                ep['intended_goal_index'] = np.random.randint(len(ep['grid_goal_states']))
                ep['robot_type'] = CartesianRobotType.R2 #in for for human teleop
                ep['mode_set_type'] = ModeSetType.OneD
                ep['mode_set'] = CARTESIAN_MODE_SET_OPTIONS[ep['robot_type']][ep['mode_set_type']]
                ep['grid_robot_start_state'] = create_random_start_state(width=ep['grid_width'],
                                                                        height=ep['grid_height'],
                                                                        obstacle_list=ep['grid_obstacles'],
                                                                        goal_list=ep['grid_goal_states'],
                                                                        mode_set=ep['mode_set'])
                # print('Start state', ep['grid_robot_start_state'])
                # ep['grid_robot_start_state'] = (16, 2)
    
                ep['grid_bounds']['xrange'] = (ep['grid_boundary_offset'], VIEWPORT_W - ep['grid_boundary_offset'])
                ep['grid_bounds']['yrange'] = (ep['grid_boundary_offset'], VIEWPORT_H - ep['grid_boundary_offset'])
                ep['sparsity_factor'] = sparsity_level
                ep['rand_direction_factor'] = rand_direction_factor
                
                ep['num_obstacle_types'] = 2
                # ep['grid_obstacles'] = [(2,2), (3,3), (4,4), (4,5)]
                # ep['grid_obstacles'] = self.create_action_dict(ep['num_obstacle_types'])
                ep['robot_radius_scale_factor'] = 0.5
                ep['goal_radius_scale_factor'] = 0.4
                ep['is_visualize_grid'] = False
                ep['rl_algo_type'] = RlAlgoType.ValueIteration

                #compute optimal policy and value function for both grid-world. This way at test time can directly load the policy from file instead of wasting time computing it.
                #Also optimal policy for human is needed for
                trial_info_dict['env_params'][gw_name] = ep
                autonomy_controller = AutonomyMultiGoal2DGridWorldModesController(ep)
                mdp_dict_for_all_goals = autonomy_controller.get_mdp_dict_for_all_goals()
                #TODO potentially add the transition matrix and the reward matrix for each goal as well so that the transition between trials is even faster. 
                for key in mdp_dict_for_all_goals.keys(): #indexed by goal
                    vs = mdp_dict_for_all_goals[key].get_state_value_function()
                    qsa = mdp_dict_for_all_goals[key].get_action_value_function()
                    policy = mdp_dict_for_all_goals[key].get_optimal_policy_for_mdp()
                    trial_info_dict['mdp_info_dict'][gw_name]['state_value_function'][key] = vs
                    trial_info_dict['mdp_info_dict'][gw_name]['action_value_function'][key] = qsa
                    trial_info_dict['mdp_info_dict'][gw_name]['policy'][key] = policy
            
            with open(os.path.join(trial_dir, str(index) + '.pkl'), 'wb') as fp:
                pickle.dump(trial_info_dict, fp)
            condition_to_pkl_index[(rand_direction_factor, sparsity_level, occupancy_level)].append(index)
            index += 1
            print('Trial number ', index)
    
    with open(os.path.join(metadata_dir, 'condition_to_pkl_index.pkl'), 'wb') as fp:
        pickle.dump(condition_to_pkl_index, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_dir', dest='trial_dir',default=os.path.join(os.getcwd(), 'adaptive_trial_dir'), help="The directory where trials will be stored are")
    parser.add_argument('--metadata_dir', dest='metadata_dir',default=os.path.join(os.getcwd(), 'adaptive_metadata_dir'), help="The directory where metadata of trials will be stored")
    parser.add_argument('--num_reps_per_condition', action='store', type=int, default=3, help="number of repetetions for single combination of conditions ")
    args = parser.parse_args()
    generate_experiment_trials(args)