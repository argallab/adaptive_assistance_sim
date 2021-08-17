#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np 
from scipy import special
import collections
import os
import sys
import rospkg
sys.path.append(os.path.join(rospkg.RosPack().get_path('simulators'), 'scripts'))
from generate_adaptive_assistance_trials import create_random_obstacles, create_random_goals, create_random_start_state
from mdp.mdp_discrete_SE2_NH_gridworld_with_modes import MDPDiscrete_SE2_NH_GridWorldWithModes
from envs.utils import *
import matplotlib.pyplot as plt
import copy

NUM_GOALS = 2
OCCUPANCY_LEVEL = 0.0
MAX_PATCHES = 2

GRID_WIDTH = 12
GRID_HEIGHT = 16

def visualize_grid(mdp):
    mdp_params = mdp.get_env_params()
    grid = np.zeros((mdp_params['grid_width'], mdp_params['grid_height']))
    
    for obs in mdp_params['mdp_obstacles']:
        grid[obs[0], obs[1]] = 1
    
    gs = mdp_params['mdp_goal_state']
    grid[gs[0], gs[1]] = 0.5
    grid = grid.T
    grid = np.flipud(grid)
    plt.imshow(grid)
    plt.colorbar()
    plt.show()

def visualize_V_and_policy(mdp):
    mdp_params = mdp.get_env_params()
    nd = mdp_params['num_discrete_orientations']
    nm = mdp_params['num_modes']
    V = np.array(mdp.get_value_function()).reshape((mdp_params['grid_width'],mdp_params['grid_height'], nd, nm))
    fig, ax = plt.subplots(2, nd*nm)
    for i in range(nd):
        for j in range(nm):
            Va = np.flipud(V[:, :, i, j].T)
            vmin = np.percentile(Va, 1)
            vmax = np.percentile(Va, 99)

            im = ax[0, i*nm+ j].imshow(Va, vmin=vmin, vmax=vmax)
            cbar = ax[0, i*nm + j].figure.colorbar(im, ax=ax[0, i*nm + j])
            cbar.ax.set_ylabel("V", rotation=-90, va="bottom")
    
    _, policy_dict = mdp.get_optimal_policy_for_mdp()
    grids = np.zeros((nm, nd, mdp_params['grid_width'], mdp_params['grid_height']))

    for s in policy_dict.keys():
        grids[s[-1]-1, s[-2], s[0], s[1]] = policy_dict[s]
    
    for i in range(nd):
        for j in range(nm):
            grid = grids[j, i, :, :]
            grid = grid.T
            grid = np.flipud(grid)
            im = ax[1, i*nm + j].imshow(grid)
            ax[1, i*nm + j].set_title('Learned policy map')
            cbar = ax[1, i*nm + j].figure.colorbar(im, ax=ax[1, i*nm+j])
            cbar.ax.set_ylabel("cc-c-f-b-t1-t2", rotation=90, va="bottom")
    
    # fig.tight_layout()
    plt.show()

def visualize_trajectory(sas_trajectory, mdp_env_params):
    pass

def create_mdp_env_param_dict():
    mdp_env_params = collections.OrderedDict()
    mdp_env_params['rl_algo_type'] = RlAlgoType.ValueIteration
    mdp_env_params['gamma'] = 0.9
    mdp_env_params['grid_width'] = GRID_WIDTH
    mdp_env_params['grid_height'] = GRID_HEIGHT
    mdp_env_params['num_discrete_orientations'] = 5
    mdp_env_params['robot_type'] = CartesianRobotType.SE2_NH
    mdp_env_params['mode_set_type'] = ModeSetType.OneD
    num_patches = np.random.randint(MAX_PATCHES) + 1
    num_patches = 0
    mdp_env_params['original_mdp_obstacles'] = create_random_obstacles(  width=mdp_env_params['grid_width'],
                                                                        height=mdp_env_params['grid_height'],
                                                                        occupancy_measure=OCCUPANCY_LEVEL,
                                                                        num_obstacle_patches=num_patches)
    
    # g_list = create_random_goals(width=mdp_env_params['grid_width'],
    #                         height=mdp_env_params['grid_height'],
    #                         num_goals=NUM_GOALS,
    #                         obstacle_list=mdp_env_params['original_mdp_obstacles']) #list of tuples
    
    g_list = [(0,0,0), (0, 15, 0)]
    
    for i, g in enumerate(g_list):
        g = list(g)
        g.append(np.random.randint(mdp_env_params['num_discrete_orientations']))
        g_list[i] = tuple(g)
    
    mdp_env_params['all_goals'] = g_list #list of tuples
    # mdp_env_params['mdp_goal_state'] = tuple(g)
    # print(mdp_env_params['mdp_goal_state']) #(3d goal state)
    mdp_env_params['obstacle_penalty'] = -100
    mdp_env_params['goal_reward'] = 100
    mdp_env_params['step_penalty'] = -10
    mdp_env_params['rand_direction_factor'] = 0.3
    mdp_env_params['mdp_obstacles'] = []
    
    return mdp_env_params

def simulate_human_SE2_NH_modes_mdp():
    mdp_env_params = create_mdp_env_param_dict()

    mdp_list = []
    # import IPython; IPython.embed(banner1='check goals')
    for i, g in enumerate(mdp_env_params['all_goals']):
        mdp_env_params['mdp_goal_state'] = g
        goals_that_are_obs = [(g_obs[0], g_obs[1]) for g_obs in mdp_env_params['all_goals'] if g_obs != g]
        mdp_env_params['mdp_obstacles'] = copy.deepcopy(mdp_env_params['original_mdp_obstacles'])
        mdp_env_params['mdp_obstacles'].extend(goals_that_are_obs)
        discrete_SE2_NH_modes_mdp = MDPDiscrete_SE2_NH_GridWorldWithModes(copy.deepcopy(mdp_env_params))
        # visualize_grid(discrete_SE2_NH_modes_mdp)
        # visualize_V_and_policy(discrete_SE2_NH_modes_mdp)
        mdp_list.append(discrete_SE2_NH_modes_mdp)
        
    
    num_rand_states = 1
    for rs in range(num_rand_states):
        random_state = mdp_list[0].get_random_valid_state()
        random_state = (9, 15, 0, 0)
        # is_random_state_valid = False
        # while not is_random_state_valid:
        #     random_state = mdp_list[0].get_random_valid_state()

        print(random_state)
        # rand_traj_from_random_state = discrete_SE2_NH_modes_mdp.get_random_trajectory_from_state(random_state, horizon=20)
        # visualize_trajectory(rand_traj_from_random_state, discrete_SE2_NH_modes_mdp)

        num_trajectories = 10000
        traj_list = collections.defaultdict(list)
        for i in range(num_trajectories):
            sampled_goal_index = np.random.choice(NUM_GOALS, p=[0.3, 0.7])
            mdp_for_sampled_goal = mdp_list[sampled_goal_index]
            opt_traj_from_state = mdp_for_sampled_goal.get_optimal_trajectory_from_state(random_state, horizon=1, return_optimal=False)
            traj_list[sampled_goal_index].append(opt_traj_from_state)
        
        action_to_action_id_map = mdp_for_sampled_goal.get_action_to_action_id_map()

        p_a_g_s0 = collections.defaultdict(list)
        for g in traj_list.keys():
            for traj_g in traj_list[g]:
                a0 = traj_g[0][1]
                p_a_g_s0[g].append(action_to_action_id_map[a0])
        
        p_a_s0 = []
        for k in p_a_g_s0.keys():
            p_a_s0.extend(p_a_g_s0[k])
        
        pa_s = np.array(collections.Counter(p_a_s0).values(), dtype=np.float32)
        pa_s = pa_s/np.sum(pa_s)

        kl_list = []
        for g in p_a_g_s0.keys():
            pa_sg = np.array(collections.Counter(p_a_g_s0[g]).values(), dtype=np.float32)
            pa_sg = pa_sg/np.sum(pa_sg)
            kl = np.sum(special.rel_entr(pa_sg, pa_s)) #log e
            kl_list.append(kl)
        
        print('Average MI for state is ', sum(kl_list)/len(kl_list), random_state)
    
    all_valid_states = mdp_list[0].get_all_state_coords()
    import IPython; IPython.embed(banner1='check state coords')
    visualize_trajectory(opt_traj_from_state, discrete_SE2_NH_modes_mdp)
    

if __name__ == "__main__":
    simulate_human_SE2_NH_modes_mdp()    