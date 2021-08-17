# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape, b2Random, b2CircleShape, b2Vec2, b2Color)
import Box2D
from backends.rendering import Viewer, Transform
import sys
import os
import rospkg
sys.path.append(os.path.join(rospkg.RosPack().get_path('simulators'), 'scripts'))
from corrective_mode_switch_utils import *
import csv
import math
import numpy as np
import collections
import itertools
import rospy
import threading
from envs.srv import OptimalAction, OptimalActionRequest, OptimalActionResponse
import time

class Robot4DEnv(object):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, env_params):
        self.viewer = None

        self.world = Box2D.b2World()
        self.world.gravity = (0,0)
        self.env_params = env_params
        assert self.env_params is not None

        assert 'num_turns' in self.env_params
        assert 'robot_position' in self.env_params
        assert 'goal_position' in self.env_params
        assert 'robot_orientation' in self.env_params
        assert 'goal_orientation' in self.env_params
        assert 'r_to_g_relative_orientation' in self.env_params
        assert 'start_direction' in self.env_params
        assert 'start_mode' in self.env_params
        assert 'location_of_turn' in self.env_params
        assert 'location_of_gripper_action' in self.env_params
        assert 'start_gripper_angle' in self.env_params

        self.num_turns = None
        self.num_locations = None
        self.STATES = None
        self.LOCATIONS = None
        self.ORIENTATIONS = None
        self.DIMENSIONS = None
        self.LOCATIONS_TO_WAYPOINTS_DICT = None
        self.WAYPOINTS_TO_LOCATION_DICT = None
        self.robot = None
        self.waypoints = None
        self.path_points = None
        self.robot_position = None
        self.robot_orientation = None
        self.goal_position = None
        self.goal_orientation = None
        self.start_gripper_angle = None
        self.r_to_g_relative_orientation = None
        self.start_direction = None
        self.start_mode = None
        self.location_of_turn = None
        self.location_of_gripper_action = None
        self.service_initialized = False

        self.ready_for_new_prompt = True
        self.text_display_delay = 2
        self.start_msg_ind = 0
    
    def update_params(self, env_params):
        self.env_params = env_params
        assert self.env_params is not None

        assert 'num_turns' in self.env_params
        assert 'robot_position' in self.env_params
        assert 'goal_position' in self.env_params
        assert 'robot_orientation' in self.env_params
        assert 'goal_orientation' in self.env_params
        assert 'r_to_g_relative_orientation' in self.env_params
        assert 'start_direction' in self.env_params
        assert 'start_mode' in self.env_params
        assert 'location_of_turn' in self.env_params
        assert 'location_of_gripper_action' in self.env_params
        assert 'start_gripper_angle' in self.env_params
    
    def get_optimal_action(self, req):
        #change this into get stochastic policy. 
        pass
    
    def _check_continuous_position_on_line_joining_waypoints(self, start_position, end_position, current_position):
        '''
        code borrowed from https://kite.com/python/answers/how-to-determine-if-a-point-is-on-a-line-segment-in-python
        '''
        start_x, end_x, curr_x = start_position[0], end_position[0], current_position[0]
        start_y, end_y, curr_y = start_position[1], end_position[1], current_position[1]

        if not start_x == end_x: #slope is not infinite #this is parallel to x axis
            m = (end_y - start_y)/(end_x - start_x) #compute finite slope
            curr_on = (curr_y - start_y) == m * (curr_x - start_x) #check if the current point is ON the line
            curr_between = (min(start_x, end_x) <= curr_x <= max(start_x, end_x)) and (min(start_y, end_y) <= curr_y <= max(start_y, end_y)) #check if the current point is between the end points
            # curr_on_and_between = curr_on and curr_between #Logical AND of ON and BETWEEN
            curr_on_and_between = curr_on #removing the need to be BETWEEN the end points so that the movement can be extended beyond the end points. The stabilizing controller will have to become active to bring it back
        else:
            # curr_on_and_between = (min(start_x, end_x) <= curr_x <= max(start_x, end_x))  and (min(start_y, end_y) <= curr_y <= max(start_y,end_y))
            curr_on_and_between = (min(start_x, end_x) <= curr_x <= max(start_x, end_x)) # can go beyond the y limits, but still on the line. 

        print(curr_on_and_between)
        position_on_line = PositionOnLine.NOT_ON_LINE
        if curr_on_and_between:
            # position_on_line = PositionOnLine.BETWEEN
            position_on_line = PositionOnLine.ON_LINE
            if abs(curr_y - end_y) < 0.1 and abs(curr_x - end_x) < 0.1: #TODO (deepak.gopinath) Define these thresholds in utils.py folder.
                position_on_line = PositionOnLine.END
            if abs(curr_y -start_y) < 0.1 and abs(curr_x - start_x) < 0.1:
                position_on_line = PositionOnLine.START

        # assert position_on_line is not PositionOnLine.NOT_ON_LINE
        return position_on_line
    
    def _check_continuous_position_in_relation_to_current_segment(self, start_position, end_position, current_position):
        
        #start and end points of the current line segment. 
        start_x, end_x, curr_x = start_position[0], end_position[0], current_position[0]
        start_y, end_y, curr_y = start_position[1], end_position[1], current_position[1]

        if not start_x == end_x: #slope is not infinite #this is parallel to x axis
            m = (end_y - start_y)/(end_x - start_x) #compute finite slope
            curr_on = (curr_y - start_y) == m * (curr_x - start_x) #check if the current point is ON the line
            curr_between = (min(start_x, end_x) <= curr_x <= max(start_x, end_x)) and (min(start_y, end_y) <= curr_y <= max(start_y, end_y)) #check if the current point is between the end points
            # curr_on_and_between = curr_on and curr_between #Logical AND of ON and BETWEEN
            curr_on_and_between = curr_on #removing the need to be BETWEEN the end points so that the movement can be extended beyond the end points. The stabilizing controller will have to become active to bring it back
        else:
            # curr_on_and_between = (min(start_x, end_x) <= curr_x <= max(start_x, end_x))  and (min(start_y, end_y) <= curr_y <= max(start_y,end_y))
            curr_on_and_between = (min(start_x, end_x) <= curr_x <= max(start_x, end_x)) # can go beyond the y limits, but still on the line. 

    # def _transform_continuous_position_to_discrete_position(self):
    #     pass

    def _transform_continuous_position_to_discrete_position(self):
        '''
        Transforms continuous robot position to a discrete position id.
        The current position is between two locations pM and pN (assuming pM is earlier in the path than pN), then if the robot position is anywhere between the line segment joining
        pM and pN, the discrete position will still be pM. If the robot position is within some predefined threshold of the pN, then the discrete id will be pN.
        The restricted motion will ensure that the robot stays ON the line segment always.

        Returns - discrete location associated with the continuous position, Boolean indicator for ensure that the robot is 'snapped' into position at the end of the line segment
        '''
        current_position = self.robot.get_position()
        current_orientation = self.robot.get_angle()
        start_index = int(self.current_discrete_state[0][-1]) #get the number in the location id. That is, if location is p3, retrieve 3. This number is a proxy for which waypoint is the robot at
        #only consider line segments from the current index onwards.
        for (turn_id, p, pt) in zip(range(start_index, self.num_locations-1), self.waypoints[start_index:-1], self.waypoints[start_index+1:]):
            if turn_id < self.location_of_turn or turn_id > self.location_of_turn:
                position_on_line =  self._check_continuous_position_on_line_joining_waypoints(p, pt, current_position) #check if the current position of the robot is at the START, BETWEEN or END of the line segment or ON the line
                print(position_on_line)
                if position_on_line == PositionOnLine.START or position_on_line == PositionOnLine.BETWEEN or position_on_line == PositionOnLine.ON_LINE:
                    return self.WAYPOINTS_TO_LOCATION_DICT[tuple(p)], False
                elif position_on_line == PositionOnLine.END:
                    return self.WAYPOINTS_TO_LOCATION_DICT[tuple(pt)], True
                elif position_on_line == PositionOnLine.NOT_ON_LINE:
                    import IPython; IPython.embed(banner1='robot not on line segment')
            elif turn_id == self.location_of_turn:
                if abs(current_orientation - self.goal_orientation) > 0.1: #hasn't turned or is turning. Might have to take care of rotation. #TODO Convert this check for goal orientation to a functoin
                    return self.WAYPOINTS_TO_LOCATION_DICT[tuple(p)], False
                elif abs(current_orientation - self.goal_orientation) < 0.1: #finished turning
                    position_on_line =  self._check_continuous_position_on_line_joining_waypoints(p, pt, current_position)
                    if position_on_line == PositionOnLine.START or position_on_line == PositionOnLine.BETWEEN or position_on_line == PositionOnLine.ON_LINE:
                        return self.WAYPOINTS_TO_LOCATION_DICT[tuple(p)], False
                    elif position_on_line == PositionOnLine.END:
                        return self.WAYPOINTS_TO_LOCATION_DICT[tuple(pt)], True
                    elif position_on_line == PositionOnLine.NOT_ON_LINE:
                        import IPython; IPython.embed(banner1='robot not on line segment at turning location')

        return self.WAYPOINTS_TO_LOCATION_DICT[tuple(self.waypoints[-1])], True

    def _transform_continuous_orientation_to_discrete_orientation(self):
        '''
        Retrieves discrete orientation from continuous orientation
        '''
        current_orientation = self.robot.get_angle()
        print(current_orientation)
        if self.goal_orientation == PI/2:
            # assert current_orientation >= 0.0 and current_orientation <= (PI/2 + 0.1) #this is to ensure that the rotation doesn't happen in the "wrong" direction
            if abs(current_orientation - self.goal_orientation) > 0.1:# (todo add wraparound so that reaching goal from the wrong direction is valud)
                return 0.0, False
            else:
                return self.goal_orientation, True
        elif self.goal_orientation == -PI/2:
            # assert current_orientation >= (-PI/2 - 0.1) and current_orientation <= 0.0
            if abs(current_orientation - self.goal_orientation) > 0.1:
                return 0.0, False
            else:
                return self.goal_orientation, True
    
    def _transform_continuous_continuous_gripper_angle_to_discrete_gripper_angle(self):
        '''
        Retrieves discrete gripper angle from continuous gripper angle. 
        Return values: Discrete gripper angle and bool indicating whether goal was reached or not. 
        '''
        current_gripper_angle = self.robot.get_gripper_angle()
        if self.goal_gripper_angle == PI/2:
            assert current_gripper_angle >= PI/8 and current_gripper_angle <= (PI/2 + 0.1)
            if abs(current_gripper_angle - self.goal_gripper_angle) > 0.1:
                return PI/8, False
            else:
                return self.goal_gripper_angle, True
        elif self.goal_gripper_angle == PI/8:
            assert current_gripper_angle >= (PI/8 - 0.1) and current_gripper_angle <= PI/2
            if abs(current_gripper_angle - self.goal_gripper_angle) > 0.1: #hasn't reached goal.
                return PI/2, False
            else:
                return self.goal_gripper_angle, True

    def _retrieve_current_allowed_mode(self):
        '''
        For a given discrete state, retrieves the mode/dimension in which motion is supposed to happen.

        '''
        location = self.current_discrete_state[0]
        orientation =  self.current_discrete_state[1]
        gripper_angle = self.current_discrete_state[2]
        if location !=  self.LOCATIONS[self.location_of_turn] and location != self.LOCATIONS[self.location_of_gripper_action]:
            return self.MODES_MOTION_ALLOWED[location][0] #primary mode in which user is supposed to move, [1] is the secondar mode in which the blending should happen
        elif location != self.LOCATIONS[self.location_of_gripper_action]:
            if orientation == 0.0:
                return self.MODES_MOTION_ALLOWED[location][-1] #'t'
            else: #has reached orientation
                return self.MODES_MOTION_ALLOWED[location][0] #primary mode, second dimension would be [1]
        else:
            if (gripper_angle == PI/8 and self.goal_gripper_angle == PI/2) or (gripper_angle == PI/2 and self.goal_gripper_angle == PI/8): #hasn't reached the goal gripper angle
                return self.MODES_MOTION_ALLOWED[location][-1] #'g'
            else:
                return self.MODES_MOTION_ALLOWED[location][0] #primary mode


    def _init_modes_in_which_motion_allowed_dict(self):
        '''
        Specifies, for each location, the modes in which motion is allowed
        '''
        for i, s in enumerate(self.LOCATIONS[:-1]): #for all locations before the end location
            if self.start_direction == StartDirection.X:
                if i % 2 == 0:
                    self.MODES_MOTION_ALLOWED[s] = ['x', 'y'] # primary mode is the first place. The secondary mode is the one in which the autonomous controller will be active. 
                else:
                    self.MODES_MOTION_ALLOWED[s] = ['y', 'x']
                
                if i == self.location_of_turn:
                    self.MODES_MOTION_ALLOWED[s].append('t')
                if i == self.location_of_gripper_action:
                    self.MODES_MOTION_ALLOWED[s].append('gr')
            
            elif self.start_direction == StartDirection.Y:
                if i % 2 == 0:
                    self.MODES_MOTION_ALLOWED[s] = ['y', 'x']
                else:
                    self.MODES_MOTION_ALLOWED[s] = ['x', 'y']
                if i == self.location_of_turn:
                    self.MODES_MOTION_ALLOWED[s].append('t')
                if i == self.location_of_gripper_action:
                    self.MODES_MOTION_ALLOWED[s].append('gr')
            
        self.MODES_MOTION_ALLOWED[self.LOCATIONS[-1]] = self.MODES_MOTION_ALLOWED[self.LOCATIONS[-2]][0] #for the last location, copy the linear mode in which the motion is allowed for the second last location

    def _create_state_transition_model(self):
        for s in self.STATES: #(cartesian product of x, y, orientation angle and gripper angle)
            self.STATE_TRANSITION_MODEL[s] = collections.OrderedDict()
            for u in LOW_LEVEL_COMMANDS: #[hp, hs, ss, sp]
                self.STATE_TRANSITION_MODEL[s][u] = None

    def _init_state_transition_model(self):
        rgc = self.r_to_g_relative_orientation
        for s in self.STATE_TRANSITION_MODEL.keys(): #for all states in the world
            for u in self.STATE_TRANSITION_MODEL[s].keys(): #for all available low level commands, hp, hs, ss, sp
                if u == 'Hard Puff' or u == 'Hard Sip':
                    self.STATE_TRANSITION_MODEL[s][u] = (s[0], s[1], MODE_SWITCH_TRANSITION_4D[s[2]][u], s[3]) #perform state conditioned mode switch
                if u == 'Soft Puff' or u == 'Soft Sip':
                    allowed_modes_for_motion = self.MODES_MOTION_ALLOWED[s[0]] #this is a list of modes in which the motion is allowed. 
                    self.STATE_TRANSITION_MODEL[s][u] = (s[0], s[1], s[2], s[3]) #by default store the same state as next state. Because if no motion happens, the resultant state is also the same state
                    for i, m in enumerate(allowed_modes_for_motion):
                        #0th is the primary allowed mode
                        #1th is the second allowed mode in which the discrete transition remains the same, because the autonomous controller is supposed to bring it back to original continuous position
                        if i == 0 or i == 2: #Is the primary mode, so discrete transition is possible
                            if m == s[2]: #make sure that the allowed mode matches the mode in the state s. If it doesn't no motion will happen
                                if m != 't' and m != 'gr':
                                    if TRANSITION_FOR_ACTION_4D[rgc][u][m] == 'next':
                                        new_loc_next = self.LOCATIONS[min(self.LOCATIONS.index(s[0]) + 1, len(self.LOCATIONS)-1 )]
                                        self.STATE_TRANSITION_MODEL[s][u] = (new_loc_next, s[1], s[2], s[3])
                                
                                    elif TRANSITION_FOR_ACTION_4D[rgc][u][m] == 'prev':
                                        new_loc_prev = self.LOCATIONS[max(self.LOCATIONS.index(s[0]) - 1, 0 )]
                                        if m == self.MODES_MOTION_ALLOWED[new_loc_prev][0]:
                                            self.STATE_TRANSITION_MODEL[s][u] = (new_loc_prev, s[1], s[2], s[3])
                                
                                elif m == 't' and m != 'gr': #rotation mode
                                    new_theta = s[1]
                                    if TRANSITION_FOR_ACTION_4D[rgc][u][m] == 'next':
                                        new_theta = min(PI/2, s[1] + PI/2) #max angle allowed is PI/2
                                    elif TRANSITION_FOR_ACTION_4D[rgc][u][m] == 'prev':
                                        new_theta = max(-PI/2, s[1] - PI/2) #min angle allowed is 0.0

                                    self.STATE_TRANSITION_MODEL[s][u] = (s[0], new_theta, s[2], s[3])

                                elif m == 'gr': #gripper mode
                                    new_gripper_angle = s[3]
                                    if TRANSITION_FOR_ACTION_4D[rgc][u][m] == 'next': #close the gripper
                                        new_gripper_angle = max(0.0, s[3] - PI/2)
                                    elif TRANSITION_FOR_ACTION_4D[rgc][u][m] == 'prev': #open the gripper
                                        new_gripper_angle = min(PI/2, s[3] + PI/2)
                                    
                                    self.STATE_TRANSITION_MODEL[s][u] = (s[0], s[1], s[2], new_gripper_angle)
                
    def _create_optimal_next_state_dict(self):
        pass

    def _generate_optimal_control_dict(self):
        pass

    def _init_allowed_directions_of_motion(self):
        pass
    def _retrieve_allowed_direction_motion_in_allowed_mode(self):
        pass
    
    def _render_turn_location(self):
        location_of_turn_waypoint = self.waypoints[self.location_of_turn]
        t =  Transform(translation=(location_of_turn_waypoint[0], location_of_turn_waypoint[1]))
        self.viewer.draw_circle(8*WP_RADIUS/SCALE, 4, True, color=TURN_LOCATION_COLOR).add_attr(t) # TODO Look into how to properly render a box instead of a circle with 4 points!

    def _render_gripper_location(self):
        location_of_gripper_action = self.waypoints[self.location_of_gripper_action]
        t =  Transform(translation=(location_of_gripper_action[0], location_of_gripper_action[1]))
        self.viewer.draw_circle(8*WP_RADIUS/SCALE, 4, True, color=GRIPPER_ACTION_LOCATION_COLOR).add_attr(t)


    def _render_goal(self):
        t = Transform(translation=(self.goal_position[0],self.goal_position[1]))
        self.viewer.draw_circle(GOAL_RADIUS/SCALE, 30, True, color=(0.53, 1.0, 0.42)).add_attr(t)
        self.viewer.draw_line(self.goal_position, (self.goal_position[0] + 2*(GOAL_RADIUS/SCALE)*math.cos(self.goal_orientation), self.goal_position[1] + 2*(GOAL_RADIUS/SCALE)*math.sin(self.goal_orientation)), linewidth=3.0)

    def _render_bodies(self):
        for r in [self.robot]:
            if isinstance(r, Robot4D):
                robot = r.robot
            else:
                robot = r
            for f in robot.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = Transform(translation = trans.position)
                    if isinstance(r, Robot4D):
                        if abs(np.linalg.norm(robot.linearVelocity)) > 0 or abs(np.linalg.norm(robot.angularVelocity)) > 0.0:
                            self.viewer.draw_circle(f.shape.radius, 30, color=ROBOT_COLOR_WHEN_MOVING, filled=True).add_attr(t)
                        else:
                            self.viewer.draw_circle(f.shape.radius, 30, color=r.robot_color, filled=True).add_attr(t)
                    else:
                        self.viewer.draw_circle(f.shape.radius, 30, color=(1.0, 1.0, 1.0), filled=True).add_attr(t)
                elif type(f.shape) is b2EdgeShape:
                    self.viewer.draw_line(f.shape.vertices[0], f.shape.vertices[1])
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=(1.0, 0.0, 0.0))
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=(1.0, 0.0, 0.0), linewidth=2)
    
    def _render_robot_gripper(self):
        gripper_handles = self.robot.get_gripper_handle_markers()
        for gripper_handle in gripper_handles:
            self.viewer.draw_line(gripper_handle[0], gripper_handle[1], linewidth=3.0)
    
    def _render_waypoints(self):
        #render the waypoints
        for i in range(len(self.waypoints)):
            t =  Transform(translation=(self.waypoints[i][0], self.waypoints[i][1]))
            robot_position = self.robot.get_position()
            if robot_position[0] == self.waypoints[i][0] and robot_position[1] == self.waypoints[i][1]:
                self.viewer.draw_circle(WP_RADIUS/SCALE, 30, True, color=(0,1,0)).add_attr(t) #light up the waypoint if the robot is EXACTLY at the waypoint. Feedback for the user
            else:
                self.viewer.draw_circle(WP_RADIUS/SCALE, 30, True, color=(0,0,0)).add_attr(t)

    def _render_path(self):
        for i in range(1, len(self.path_points)):
            self.viewer.draw_line(tuple(self.path_points[i-1][0]), tuple(self.path_points[i][0])) #draw left edge
            self.viewer.draw_line(tuple(self.path_points[i-1][1]), tuple(self.path_points[i][1])) #draw right edge

    def _render_mode_display(self):
        for i, d in enumerate(self.DIMENSIONS):
            if d == 'x':
                t = Transform(translation=(MODE_DISPLAY_CIRCLE_START_POSITION_S[0], MODE_DISPLAY_CIRCLE_START_POSITION_S[1]))
            elif d == 'y':
                t = Transform(translation=(MODE_DISPLAY_CIRCLE_START_POSITION_S[0] + 2*MODE_DISPLAY_CIRCLE_X_OFFSET_S, MODE_DISPLAY_CIRCLE_START_POSITION_S[1]))
            elif d == 't':
                t = Transform(translation=(MODE_DISPLAY_CIRCLE_START_POSITION_S[0] + 2*MODE_DISPLAY_CIRCLE_X_OFFSET_S, MODE_DISPLAY_CIRCLE_START_POSITION_S[1] - MODE_DISPLAY_CIRCLE_Y_OFFSET_S))
            elif d == 'gr':
                t = Transform(translation=(MODE_DISPLAY_CIRCLE_START_POSITION_S[0], MODE_DISPLAY_CIRCLE_START_POSITION_S[1] - MODE_DISPLAY_CIRCLE_Y_OFFSET_S))
            
            if d == self.current_mode:
                self.viewer.draw_circle(MODE_DISPLAY_RADIUS/SCALE, 30, True, color=ACTIVE_MODE_COLOR).add_attr(t)
            else:
                self.viewer.draw_circle(MODE_DISPLAY_RADIUS/SCALE, 30, True, color=NONACTIVE_MODE_COLOR).add_attr(t)

    def _render_mode_display_text(self):
        '''
        Note that the coordinates of the text should in real pixels. Which is why here there is a multiplicative factor of SCALE.
        '''
        self.viewer.draw_text("Horizontal", x=MODE_DISPLAY_TEXT_START_POSITION[0], y=MODE_DISPLAY_TEXT_START_POSITION[1] , font_size=MODE_DISPLAY_TEXT_FONTSIZE, color=MODE_DISPLAY_TEXT_COLOR, anchor_y=MODE_DISPLAY_TEXT_Y_ANCHOR)
        self.viewer.draw_text("Vertical", x=MODE_DISPLAY_TEXT_START_POSITION[0] + 2*MODE_DISPLAY_TEXT_X_OFFSET, y=MODE_DISPLAY_TEXT_START_POSITION[1], font_size=MODE_DISPLAY_TEXT_FONTSIZE, color=MODE_DISPLAY_TEXT_COLOR, anchor_y=MODE_DISPLAY_TEXT_Y_ANCHOR)
        self.viewer.draw_text("Rotation", x=MODE_DISPLAY_TEXT_START_POSITION[0] + 2*MODE_DISPLAY_TEXT_X_OFFSET, y=MODE_DISPLAY_TEXT_START_POSITION[1] - MODE_DISPLAY_CIRCLE_Y_OFFSET_S*SCALE, font_size=MODE_DISPLAY_TEXT_FONTSIZE, color=MODE_DISPLAY_TEXT_COLOR, anchor_y=MODE_DISPLAY_TEXT_Y_ANCHOR)
        self.viewer.draw_text("Gripper", x=MODE_DISPLAY_TEXT_START_POSITION[0], y=MODE_DISPLAY_TEXT_START_POSITION[1] - MODE_DISPLAY_CIRCLE_Y_OFFSET_S*SCALE, font_size=MODE_DISPLAY_TEXT_FONTSIZE, color=MODE_DISPLAY_TEXT_COLOR, anchor_y=MODE_DISPLAY_TEXT_Y_ANCHOR)
    
    def _render_timer_text(self):
        if self.current_time < TIMER_WARNING_THRESHOLD:
            self.viewer.draw_text(str(self.current_time), x=TIMER_DISPLAY_POSITION[0], y=TIMER_DISPLAY_POSITION[1], font_size=TIMER_DISPLAY_FONTSIZE, color=TIMER_COLOR_NEUTRAL, anchor_y=TIMER_DISPLAY_TEXT_Y_ANCHOR, bold=True)
        elif self.current_time < TIMER_DANGER_THRESHOLD:
            self.viewer.draw_text(str(self.current_time), x=TIMER_DISPLAY_POSITION[0], y=TIMER_DISPLAY_POSITION[1], font_size=TIMER_DISPLAY_FONTSIZE, color=TIMER_COLOR_WARNING, anchor_y=TIMER_DISPLAY_TEXT_Y_ANCHOR, bold=True)
        else:
            self.viewer.draw_text(str(self.current_time), x=TIMER_DISPLAY_POSITION[0], y=TIMER_DISPLAY_POSITION[1], font_size=TIMER_DISPLAY_FONTSIZE, color=TIMER_COLOR_DANGER, anchor_y=TIMER_DISPLAY_TEXT_Y_ANCHOR, bold=True)

    def _render_text(self, msg, position, font_size=TRIAL_OVER_TEXT_FONTSIZE, color=COMMAND_TEXT_COLOR):
        self.viewer.draw_text(msg, x=position[0], y=position[1], font_size=font_size, color=color, bold=True)
        
    def _render_start_end_text(self):
        if self.r_to_g_relative_orientation == RGOrient.TOP_RIGHT:
            if self.start_direction == StartDirection.X:
                start_text_location = (self.robot_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.robot_position[1]*SCALE)
                goal_text_location = (self.goal_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.goal_position[1]*SCALE)
            elif self.start_direction == StartDirection.Y:
                start_text_location = (self.robot_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.robot_position[1]*SCALE)
                goal_text_location = (self.goal_position[0]* SCALE, self.goal_position[1]*SCALE - START_GOAL_TEXT_DISPLACEMENT)
        elif self.r_to_g_relative_orientation == RGOrient.TOP_LEFT:
            if self.start_direction == StartDirection.X:
                start_text_location = (self.robot_position[0]* SCALE + START_GOAL_TEXT_DISPLACEMENT, self.robot_position[1]*SCALE)
                goal_text_location = (self.goal_position[0]* SCALE + START_GOAL_TEXT_DISPLACEMENT, self.goal_position[1]*SCALE)
            elif self.start_direction == StartDirection.Y:
                start_text_location = (self.robot_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.robot_position[1]*SCALE)
                goal_text_location = (self.goal_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.goal_position[1]*SCALE)
        elif self.r_to_g_relative_orientation == RGOrient.BOTTOM_LEFT:
            if self.start_direction == StartDirection.X:
                start_text_location = (self.robot_position[0]* SCALE, self.robot_position[1]*SCALE - START_GOAL_TEXT_DISPLACEMENT)
                goal_text_location = (self.goal_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.goal_position[1]*SCALE)
            elif self.start_direction == StartDirection.Y:
                start_text_location = (self.robot_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.robot_position[1]*SCALE)
                goal_text_location = (self.goal_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.goal_position[1]*SCALE)
        elif self.r_to_g_relative_orientation == RGOrient.BOTTOM_RIGHT:
            if self.start_direction == StartDirection.X:
                start_text_location = (self.robot_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.robot_position[1]*SCALE)
                goal_text_location = (self.goal_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT, self.goal_position[1]*SCALE)
            elif self.start_direction == StartDirection.Y:
                start_text_location = (self.robot_position[0]* SCALE - START_GOAL_TEXT_DISPLACEMENT , self.robot_position[1]*SCALE)
                goal_text_location = (self.goal_position[0]* SCALE + START_GOAL_TEXT_DISPLACEMENT, self.goal_position[1]*SCALE)

        self.viewer.draw_text('Start', x=start_text_location[0], y=start_text_location[1], font_size=TRIAL_OVER_TEXT_FONTSIZE-6, color=(0,0,0,255), bold=False)
        self.viewer.draw_text('Goal', x=goal_text_location[0], y=goal_text_location[1], font_size=TRIAL_OVER_TEXT_FONTSIZE-6, color=(0,0,0,255), bold=False)

    def _render_trial_over_text(self):
        self.viewer.draw_text("TRIAL OVER", x=TRIAL_OVER_TEXT_DISPLAY_POSITION[0], y=TRIAL_OVER_TEXT_DISPLAY_POSITION[1], font_size=TRIAL_OVER_TEXT_FONTSIZE, color=TRIAL_OVER_TEXT_COLOR, anchor_y=TRIAL_OVER_TEXT_Y_ANCHOR, bold=True)

    def _render_timer(self, period):
        while not rospy.is_shutdown():
            start = rospy.get_rostime()
            self.lock.acquire()
            if self.viewer is not None:
                self.current_time += 1
            else:
                pass
            self.lock.release()
            end = rospy.get_rostime()
            if end-start < period:
                rospy.sleep(period - (end-start))
            else:
                rospy.loginfo('took more time')

    def render_clear(self, msg):
        self.viewer.window.clear()
        self._render_text(msg, COMMAND_DISPLAY_POSITION)
        return self.viewer.render(False)

    def start_countdown(self):
        if self.ready_for_new_prompt:
            self.msg_prompt = EXPERIMENT_START_COUNTDOWN[self.start_msg_ind]
            self.start_msg_ind += 1
            self.ts = time.time()
            self.ready_for_new_prompt = False
        if (time.time() - self.ts) >= self.text_display_delay:
            self.ready_for_new_prompt = True

        self._render_text(self.msg_prompt, COMMAND_DISPLAY_POSITION)
        self.viewer.render(False)

        if self.start_msg_ind == len(EXPERIMENT_START_COUNTDOWN):
            return 1
        else:
            return 0

    def render(self, mode='human'):
        self._render_start_end_text()
        #render location for turning
        self._render_turn_location()
        #render location for gripping()
        self._render_gripper_location()
        #render the goal position
        self._render_goal()
        #render bodies
        self._render_bodies()
        #draw robot direction indicator after the robot has been drawn.
        self._render_robot_gripper()
        #render waypoints
        self._render_waypoints()
        #render path
        self._render_path()
        #render virtual mode display
        self._render_mode_display()
        #render dimension text
        self._render_mode_display_text()
        #render timer
        self._render_timer_text()
        #render assistance block label
        self._render_text('Condition: '+self.assistance_type, LABEL_DISPLAY_POSITION, MODE_DISPLAY_TEXT_FONTSIZE+5)

        if self.current_time >= self.max_time and not self.training: #TODO change this time limit to a param
            self._render_trial_over_text()

        return self.viewer.render(False)

    def _generate_path(self):
        '''
        Function to generate the track boundaries.
        '''
        if self.r_to_g_relative_orientation == RGOrient.TOP_RIGHT or self.r_to_g_relative_orientation == RGOrient.BOTTOM_LEFT:
            left_edge_angle = PI/2 + PI/4.0
            right_edge_angle = -PI/4.0
        elif self.r_to_g_relative_orientation == RGOrient.TOP_LEFT or self.r_to_g_relative_orientation == RGOrient.BOTTOM_RIGHT:
            left_edge_angle = -(PI/2 + PI/4.0)
            right_edge_angle = PI/4.0

        r = math.sqrt(2)*PATH_HALF_WIDTH/SCALE
        for i in range(0, self.num_turns+2):
            self.path_points[i][0] = self.waypoints[i] + (r*math.cos(left_edge_angle), r*math.sin(left_edge_angle))#left edge
            self.path_points[i][1] = self.waypoints[i] + (r*math.cos(right_edge_angle), r*math.sin(right_edge_angle))

    def _generate_way_points(self):
        '''
        Function to generate the waypoints given the params of the path
        '''
        self.waypoints[0] = self.robot_position
        self.waypoints[-1] = self.goal_position
        #for odd num_turns, both directions are divided using the same divisor.
        #for even num_turns, one dimensions uses divisor and the other uses divisor-1
        if self.num_turns % 2 == 0:
            divisor = self.num_turns/2 + 1
            if self.start_direction == StartDirection.X: #move along x direction first
                x_inc = (self.waypoints[-1][0] - self.waypoints[0][0])/divisor
                y_inc = (self.waypoints[-1][1] - self.waypoints[0][1])/(divisor - 1)
            elif self.start_direction == StartDirection.Y: #move along y direction first
                x_inc = (self.waypoints[-1][0] - self.waypoints[0][0])/(divisor - 1)
                y_inc = (self.waypoints[-1][1] - self.waypoints[0][1])/divisor
        else: #odd number of turns
            divisor = (self.num_turns + 1)/2
            x_inc = (self.waypoints[-1][0] - self.waypoints[0][0])/divisor
            y_inc = (self.waypoints[-1][1] - self.waypoints[0][1])/divisor

        if self.start_direction == StartDirection.X:
            for i in range(1, self.num_turns+1):
                if i % 2 == 0:
                    self.waypoints[i][0] = self.waypoints[i-1][0]
                    self.waypoints[i][1] = self.waypoints[i-1][1] + y_inc
                else:
                    self.waypoints[i][0] = self.waypoints[i-1][0] + x_inc
                    self.waypoints[i][1] = self.waypoints[i-1][1]
        elif self.start_direction == StartDirection.Y:
            for i in range(1, self.num_turns+1):
                if i % 2 == 0:
                    self.waypoints[i][0] = self.waypoints[i-1][0] + x_inc
                    self.waypoints[i][1] = self.waypoints[i-1][1]
                else:
                    self.waypoints[i][0] = self.waypoints[i-1][0]
                    self.waypoints[i][1] = self.waypoints[i-1][1] + y_inc

    def start_countdown(self):
        if self.ready_for_new_prompt:
            self.msg_prompt = EXPERIMENT_START_COUNTDOWN[self.start_msg_ind]
            self.start_msg_ind += 1
            self.ts = time.time()
            self.ready_for_new_prompt = False
        if (time.time() - self.ts) >= self.text_display_delay:
            self.ready_for_new_prompt = True

        self._render_text(self.msg_prompt, COMMAND_DISPLAY_POSITION)
        self.viewer.render(False)

        if self.start_msg_ind == len(EXPERIMENT_START_COUNTDOWN):
            return 1
        else:
            return 0

    def initialize_viewer(self):
        if self.viewer is None:
            self.viewer = Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)
            self.viewer.window.set_location(650, 300)
            self.timer_thread.start()
    
    def close_window(self):
        self.viewer.close()
            
    def _destroy(self):
        if self.robot is None: return
        self.world.DestroyBody(self.robot.robot)
        self.robot = None
        self.waypoints = np.zeros((self.num_locations, 2))
        self.LOCATIONS = []
        self.ORIENTATIONS = []
        self.DIMENSIONS = []
        self.DIMENSION_INDICES = []
        self.STATES = []
        self.path_points = np.zeros((self.num_locations, 2, 2))
    
    def initialize(self):
        self.start_session = self.env_params['start']
        self.period = rospy.Duration(1.0)
        self.timer_thread = threading.Thread(target=self._render_timer, args=(self.period,)) #timer for trial time. Run on separate thread
        self.lock = threading.Lock()
        self.current_time = 0
        self.max_time = 50
    
    def reset(self):
        self._destroy()

        #Grab all trial params from the env_params dict
        self.num_turns = self.env_params['num_turns']
        assert self.num_turns > 0
        self.robot_position = self.env_params['robot_position']
        self.robot_orientation = self.env_params['robot_orientation']
        self.goal_position = self.env_params['goal_position']
        self.goal_orientation = self.env_params['goal_orientation']
        self.r_to_g_relative_orientation = self.env_params['r_to_g_relative_orientation'] #top right, top left, bottom, right, bottom left. relative position of the goal with respect to starting position of robot
        self.start_direction = self.env_params['start_direction']
        self.start_mode = self.env_params['start_mode']
        self.start_gripper_angle = self.env_params['start_gripper_angle']
        self.goal_gripper_angle = self.env_params['goal_gripper_angle']
        assert self.start_gripper_angle != self.goal_gripper_angle, "starting gripper angle and goal gripper angles need to be different. "
        self.training = self.env_params['training']
        self.assistance_type = ASSISTANCE_CODE_NAME[self.env_params['assistance_type']] # label assistance type
        self.location_of_turn = self.env_params['location_of_turn'] #corresponds to the index in self.waypoints list
        self.location_of_gripper_action = self.env_params['location_of_gripper_action']
        self.num_locations = self.num_turns + 2 #num turns + start + end point

        self.LOCATIONS = ['p' + str(i) for i in range(self.num_locations)] #create location id. p0, p1, p2....pN
        self.ORIENTATIONS = [0, PI/2, -PI/2] #discrete set of rotations. 0 and pi/2
        self.GRIPPER_ANGLES = [PI/8, PI/2] #open, closed
        self.DIMENSIONS = ['x', 'y', 't', 'gr'] #set of dimensions or modes
        self.DIMENSION_INDICES = np.array([0,1,2,3]) #set of mode indices (needed for look up)
        self.STATES = [s for s in itertools.product(self.LOCATIONS, self.ORIENTATIONS, self.DIMENSIONS, self.GRIPPER_ANGLES)] #list of all discrete states for the given configuration
        self.waypoints = np.zeros((self.num_locations, 2))
        self.path_points = np.zeros((self.num_locations, 2, 2)) #to draw the outline of the path
        self.current_discrete_state = (self.LOCATIONS[0], self.ORIENTATIONS[0], self.start_mode, self.start_gripper_angle) #init current discrete state
        self.current_mode_index = DIM_TO_MODE_INDEX[self.start_mode]
        self.current_mode = self.start_mode
        self.current_gripper_angle = self.start_gripper_angle
        self.mode_display_label = None
        self.dim = len(self.DIMENSIONS)

        #create and initialize the dictionary which contains info regarding which modes/dimension allow motion for each location
        self.MODES_MOTION_ALLOWED = collections.OrderedDict()
        self._init_modes_in_which_motion_allowed_dict()

        #create robot
        self.robot =  Robot4D(self.world, position=self.robot_position, orientation=self.robot_orientation, gripper_angle=self.start_gripper_angle, robot_color=ROBOT_COLOR_WHEN_COMMAND_REQUIRED,  radius=ROBOT_RADIUS/SCALE)
        self._generate_way_points()
        self._generate_path()

        #create STATE_TRANSITION_MODEL
        self.STATE_TRANSITION_MODEL = collections.OrderedDict()
        self._create_state_transition_model()
        self._init_state_transition_model()

        #TODO change the following to incorporate stochastic policy
        #create optimal next state and optimal next action dict
        self.OPTIMAL_NEXT_STATE_DICT = collections.OrderedDict()
        self.OPTIMAL_ACTION_DICT = collections.OrderedDict()
        self._create_optimal_next_state_dict()
        self._generate_optimal_control_dict()


        #create bidrectional mapping between discrete location ids and the waypoints
        self.LOCATIONS_TO_WAYPOINTS_DICT = collections.OrderedDict()
        for i in range(len(self.LOCATIONS)):
            self.LOCATIONS_TO_WAYPOINTS_DICT[self.LOCATIONS[i]] = tuple(self.waypoints[i])

        self.WAYPOINTS_TO_LOCATION_DICT = collections.OrderedDict()
        for i in range(len(self.LOCATIONS)):
            self.WAYPOINTS_TO_LOCATION_DICT[tuple(self.waypoints[i])] = self.LOCATIONS[i]

        self.ALLOWED_DIRECTIONS_OF_MOTION = collections.OrderedDict()
        self._init_allowed_directions_of_motion()

        if not self.service_initialized:
            rospy.Service('/mode_inference_env/get_stochastic_policy', OptimalAction, self.get_optimal_action)
            self.service_initialized = True

        self.period = rospy.Duration(1.0)
        self.timer_thread = threading.Thread(target=self._render_timer, args=(self.period,))
        self.lock = threading.Lock()
        self.current_time = 0
        self.max_time = 100
    
    def step(self, input_action):
        assert 'human' in input_action.keys()
        #Todo potentially replace all of the 'snapping functionality with some sort of stablization controller which gets activated when in the vicinity of the set point?
        current_discrete_position, should_snap = self._transform_continuous_position_to_discrete_position() #transform the continuous robot position to a discrete state representation. Easier for computing the optimal action etc
        if should_snap:
            #if the robot is very close to the 'next' corner, then artificially snap the position of the robot to that corner. So that the linear motion along the segment is properly completed
            self.robot.set_position(self.LOCATIONS_TO_WAYPOINTS_DICT[current_discrete_position]) #directly set the position of the robot
            current_discrete_position, should_snap = self._transform_continuous_position_to_discrete_position() #recompute the discrete position

        current_discrete_orientation, should_snap = self._transform_continuous_orientation_to_discrete_orientation() #compute discrete orientation state from continuous orientation of the robot
        if should_snap:
            self.robot.set_orientation(current_discrete_orientation) #if snap is true then force orientation to be that of the target orientation
            current_discrete_orientation, should_snap = self._transform_continuous_orientation_to_discrete_orientation() #recompute the discrete orientation
        
        current_discrete_gripper_angle, should_snap = self._transform_continuous_continuous_gripper_angle_to_discrete_gripper_angle()
        if should_snap:
            self.robot.set_gripper_angle(current_discrete_gripper_angle)
            current_discrete_gripper_angle, should_snap = self._transform_continuous_continuous_gripper_angle_to_discrete_gripper_angle()

        #restrict the nonzero components of the velocity only to the allowed modes.
        self.current_mode_index = rospy.get_param('mode') #0,1,2,3 #get current mode index
        # print(self.current_mode_index)
        self.current_mode =  MODE_INDEX_TO_DIM[self.current_mode_index] #x,y,t,g #get current mode
        # self.current_mode = 0
        self.current_discrete_state = (current_discrete_position, current_discrete_orientation, self.current_mode, current_discrete_gripper_angle) #update the current discrete state.
        current_allowed_mode = self._retrieve_current_allowed_mode() #x,y,t,gr #for the given location, retrieve what is the allowed mode of motion.
        # current_allowed_mode = 'x'
        current_allowed_mode_index = DIM_TO_MODE_INDEX[current_allowed_mode] #0,1,2 #get the mode index of the allowed mode of motion
        # print(current_allowed_mode_index)
        user_vel = np.array([input_action['human'].velocity.data[0], input_action['human'].velocity.data[1], input_action['human'].velocity.data[2], input_action['human'].velocity.data[3]]) #numpyify the velocity data. note the negative sign on the 3rd component.To account for proper counterclockwise motion
        user_vel[np.setdiff1d(self.DIMENSION_INDICES, current_allowed_mode_index)] = 0.0 #zero out all velocities except the ones for the allowed mode

        #check if the direction of user_vel is correct as well. For each location in the allowed mode/dimension there is proper direction in which the velocity should be.
        # current_allowed_direction_of_motion_in_allowed_mode = self._retrieve_allowed_direction_motion_in_allowed_mode()
        # if get_sign_of_number(user_vel[current_allowed_mode_index]) != current_allowed_direction_of_motion_in_allowed_mode: #check if the direction velocity component in the allowed mode matches the allowed direction of motion in the allowed mode
        #     user_vel[current_allowed_mode_index] = 0.0 #if not, zero the velocity out

        rospy.set_param('current_discrete_state', self.current_discrete_state)
        print(self.current_discrete_state)
        # self.robot.robot.linearVelocity = b2Vec2(user_vel[0], user_vel[1]) #update robot velocity
        # self.robot.robot.angularVelocity = -user_vel[2]
        if self.current_time >= self.max_time and not self.training:
            user_vel = [0.0] * self.dim

        self.robot.update(user_vel)
        self.world.Step(1.0/FPS, VELOCITY_ITERATIONS, POSITION_ITERATIONS)
        is_done = False
        if self.current_discrete_state[0] == self.LOCATIONS[-1]: #reached the last location
            is_done = True

        return self.robot.get_position(), self.robot.get_angle(), self.robot.get_gripper_angle(), [user_vel[0], user_vel[1]], -user_vel[2], self.current_discrete_state, is_done



