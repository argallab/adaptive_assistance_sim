# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape, b2Random, b2CircleShape, b2Vec2, b2Color)
import Box2D
from backends.rendering import Viewer, Transform, LineWidth
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

class PUmGivenAEnv(object):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }
    def __init__(self, env_params):
        self.viewer = None
        self.world = Box2D.b2World()
        self.world.gravity = (0, 0)

        self.env_params = env_params
        assert self.env_params is not None
        assert 'robot_position' in self.env_params      # starting position of robot
        assert 'goal_position' in self.env_params       # goal position
        assert 'robot_orientation' in self.env_params   # starting orientation of robot
        assert 'goal_orientation' in self.env_params    # goal orientation
        assert 'start_direction' in self.env_params     # orientation of path in y or x (even if rotation, it'll be either x or y but will rotate in place)
        assert 'is_rotation' in self.env_params         # whether testing for rotation vs linear motion
        assert 'is_mode_switch' in self.env_params      # boolean indicates whether about changing modes or moving robot, if true, bypass regular rendering

        self.WAYPOINTS_TO_LOCATION_DICT = None
        self.waypoints = None
        self.robot = None
        self.path_points = None
        self.robot_position = None
        self.robot_orientation = None
        self.goal_position = None
        self.goal_orientation = None
        self.start_direction = None

        self.ready_for_new_prompt = True
        self.text_display_delay = 2 # secs
        self.start_msg_ind = 0
        self.orientation_thresh = 0.7
        self.position_thresh = [3.5, 2.5]
        self.TARGET_TEXT = None

    def update_params(self, env_params):
        self.env_params = env_params
        assert self.env_params is not None

        assert 'robot_position' in self.env_params
        assert 'goal_position' in self.env_params
        assert 'robot_orientation' in self.env_params
        assert 'goal_orientation' in self.env_params
        assert 'start_direction' in self.env_params
        assert 'is_rotation' in self.env_params
        assert 'is_mode_switch' in self.env_params

    def _check_continuous_position_on_line_joining_waypoints(self, start_position, end_position, current_position):
        '''
        code borrowed from https://kite.com/python/answers/how-to-determine-if-a-point-is-on-a-line-segment-in-python
        '''
        start_x, end_x, curr_x = start_position[0], end_position[0], current_position[0]
        start_y, end_y, curr_y = start_position[1], end_position[1], current_position[1]

        if not start_x == end_x: #slope is not infinite #parallel to y axis
            m = (end_y - start_y)/(end_x - start_x) #compute finite slope
            curr_on = (curr_y - start_y) == m * (curr_x - start_x) #check if the current point is ON the line
            curr_between = (min(start_x, end_x) <= curr_x <= max(start_x, end_x)) and (min(start_y, end_y) <= curr_y <= max(start_y, end_y)) #check if the current point is between the end points
            curr_on_and_between = curr_on and curr_between #Logical AND of ON and BETWEEN
        else:
            curr_on_and_between =  (min(start_x, end_x) <= curr_x <= max(start_x, end_x))  and (min(start_y, end_y) <= curr_y <= max(start_y,end_y))

        position_on_line = PositionOnLine.NOT_ON_LINE
        if curr_on_and_between:
            position_on_line = PositionOnLine.BETWEEN
            if abs(curr_y - end_y) < 0.1 and abs(curr_x - end_x) < 0.1: #TODO (deepak.gopinath) Define these thresholds in utils.py folder.
                position_on_line = PositionOnLine.END
            if abs(curr_y -start_y) < 0.1 and abs(curr_x - start_x) < 0.1:
                position_on_line = PositionOnLine.START

        assert position_on_line is not PositionOnLine.NOT_ON_LINE
        return position_on_line

    def _transform_continuous_position_to_discrete_position(self):
        '''
        Transforms continuous robot position to a discrete position id.
        The current position is between two locations pM and pN (assuming pM is earlier in the path than pN), then if the robot position is anywhere between (and n the ealso on pM) the line segment joining
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
                position_on_line =  self._check_continuous_position_on_line_joining_waypoints(p, pt, current_position) #check if the current position of the robot is at the START, BETWEEN or END of the line segment
                if position_on_line == PositionOnLine.START or position_on_line == PositionOnLine.BETWEEN:
                    return self.WAYPOINTS_TO_LOCATION_DICT[tuple(p)], False
                elif position_on_line == PositionOnLine.END:
                    return self.WAYPOINTS_TO_LOCATION_DICT[tuple(pt)], True
            elif turn_id == self.location_of_turn:
                if abs(current_orientation - self.goal_orientation) > 0.1: #hasn't turned or is turning
                    return self.WAYPOINTS_TO_LOCATION_DICT[tuple(p)], False
                elif abs(current_orientation - self.goal_orientation) < 0.1: #finished turning
                    position_on_line =  self._check_continuous_position_on_line_joining_waypoints(p, pt, current_position)
                    if position_on_line == PositionOnLine.START or position_on_line == PositionOnLine.BETWEEN:
                        return self.WAYPOINTS_TO_LOCATION_DICT[tuple(p)], False
                    elif position_on_line == PositionOnLine.END:
                        return self.WAYPOINTS_TO_LOCATION_DICT[tuple(pt)], True

        return self.WAYPOINTS_TO_LOCATION_DICT[tuple(self.waypoints[-1])], True

    def _transform_continuous_orientation_to_discrete_orientation(self):
        '''
        Retrieves discrete orientation from continuous orientation
        '''
        current_orientation = self.robot.get_angle()
        if self.goal_orientation == PI/2:
            assert current_orientation >= 0.0 and current_orientation <= (PI/2 + 0.1)
            if abs(current_orientation - self.goal_orientation) > 0.1:
                return 0.0, False
            else:
                return self.goal_orientation, True
        elif self.goal_orientation == -PI/2:
            assert current_orientation >= (-PI/2 - 0.1) and current_orientation <= 0.0
            if abs(current_orientation - self.goal_orientation) > 0.1:
                return 0.0, False
            else:
                return self.goal_orientation, True

    def _render_goal(self):
        t = Transform(translation=(self.goal_position[0],self.goal_position[1]))
        self.viewer.draw_circle(GOAL_RADIUS/SCALE, 30, True, color=(0.53, 1.0, 0.42)).add_attr(t)
        if self.is_rotation:
            self.viewer.draw_line(self.goal_position, (self.goal_position[0] + 2*(GOAL_RADIUS/SCALE)*math.cos(self.goal_orientation), self.goal_position[1] + 2*(GOAL_RADIUS/SCALE)*math.sin(self.goal_orientation)), linewidth=3)

    def _render_bodies(self):
        for r in [self.robot]:
            if isinstance(r, RobotSE2):
                robot = r.robot
            else:
                robot = r
            for f in robot.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = Transform(translation = trans.position)
                    if isinstance(r, RobotSE2):
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

    def _render_robot_direction_indicators(self):
        ep_markers = self.robot.get_direction_marker_end_points()
        self.viewer.draw_line(ep_markers[0], ep_markers[1], linewidth=3)

    def _render_waypoints(self):
        #render the waypoints
        for i in range(len(self.waypoints)):
            t =  Transform(translation=(self.waypoints[i][0], self.waypoints[i][1]))
            self.viewer.draw_circle(WP_RADIUS/SCALE, 30, True, color=(0,0,0)).add_attr(t)

    def _render_path(self):
        for i in range(1, len(self.path_points)):
            self.viewer.draw_line(tuple(self.path_points[i-1][0]), tuple(self.path_points[i][0])) #draw left edge
            self.viewer.draw_line(tuple(self.path_points[i-1][1]), tuple(self.path_points[i][1])) #draw right edge

    def _render_mode_display(self):
        #TODO change location of the circles. Change how target is indicated.
        for i, d in enumerate(self.DIMENSIONS):
            if d == 'y':
                t = Transform(translation=(P_UM_GIVEN_A_MODE_DISPLAY_CIRCLE_START_POSITION_S[0] + i*MODE_DISPLAY_CIRCLE_X_OFFSET_S, P_UM_GIVEN_A_MODE_DISPLAY_CIRCLE_START_POSITION_S[1]))
            else:
                t = Transform(translation=(P_UM_GIVEN_A_MODE_DISPLAY_CIRCLE_START_POSITION_S[0] + i*MODE_DISPLAY_CIRCLE_X_OFFSET_S, P_UM_GIVEN_A_MODE_DISPLAY_CIRCLE_START_POSITION_S[1] - MODE_DISPLAY_CIRCLE_Y_OFFSET_S -10/SCALE))

            if d == self.current_mode:
                self.viewer.draw_circle(MODE_DISPLAY_RADIUS/SCALE, 30, True, color=ACTIVE_MODE_COLOR).add_attr(t) # just so coloring is consistent with motions (red is current, green is goal)
            else:
                self.viewer.draw_circle(MODE_DISPLAY_RADIUS/SCALE, 30, True, color=NONACTIVE_MODE_COLOR).add_attr(t)

    def _render_mode_display_text(self):
        '''
        Note that the coordinates of the text should in real pixels. Which is why here there is a multiplicative factor of SCALE.
        '''
        self.viewer.draw_text("Horizontal", x=P_UM_GIVEN_A_MODE_DISPLAY_TEXT_START_POSITION[0], y=P_UM_GIVEN_A_MODE_DISPLAY_TEXT_START_POSITION[1]-MODE_DISPLAY_CIRCLE_Y_OFFSET_S*SCALE, font_size=MODE_DISPLAY_TEXT_FONTSIZE, color=MODE_DISPLAY_TEXT_COLOR, anchor_y=MODE_DISPLAY_TEXT_Y_ANCHOR)
        self.viewer.draw_text("Vertical", x=P_UM_GIVEN_A_MODE_DISPLAY_TEXT_START_POSITION[0] + MODE_DISPLAY_TEXT_X_OFFSET, y=P_UM_GIVEN_A_MODE_DISPLAY_TEXT_START_POSITION[1]+5, font_size=MODE_DISPLAY_TEXT_FONTSIZE, color=MODE_DISPLAY_TEXT_COLOR, anchor_y=MODE_DISPLAY_TEXT_Y_ANCHOR)
        self.viewer.draw_text("Rotation", x=P_UM_GIVEN_A_MODE_DISPLAY_TEXT_START_POSITION[0] + 2*MODE_DISPLAY_TEXT_X_OFFSET, y=P_UM_GIVEN_A_MODE_DISPLAY_TEXT_START_POSITION[1]-MODE_DISPLAY_CIRCLE_Y_OFFSET_S*SCALE, font_size=MODE_DISPLAY_TEXT_FONTSIZE, color=MODE_DISPLAY_TEXT_COLOR, anchor_y=MODE_DISPLAY_TEXT_Y_ANCHOR)

    def _render_text(self, text_position):
        self.viewer.draw_text(self.msg_prompt, x=text_position[0], y=text_position[1], font_size=TRIAL_OVER_TEXT_FONTSIZE, color=COMMAND_TEXT_COLOR, bold=True)

    def render_clear(self, msg):
        self.viewer.window.clear()
        self.msg_prompt = msg
        self._render_text(COMMAND_DISPLAY_POSITION)
        return self.viewer.render(False)

    def close_window(self):
        self.viewer.close()

    def initialize_viewer(self):
        if self.viewer is None:
            self.viewer = Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)
            self.viewer.window.set_location(650, 300)
            self.timer_thread.start()

    def start_countdown(self):
        if self.ready_for_new_prompt:
            self.msg_prompt = EXPERIMENT_START_COUNTDOWN[self.start_msg_ind]
            self.start_msg_ind += 1
            self.ts = time.time()
            self.ready_for_new_prompt = False
        if (time.time() - self.ts) >= self.text_display_delay:
            self.ready_for_new_prompt = True

        self._render_text(COMMAND_DISPLAY_POSITION)
        self.viewer.render(False)
        if self.start_msg_ind == len(EXPERIMENT_START_COUNTDOWN):
            return 1
        else:
            return 0

    def _render_timer_text(self):
        if self.current_time < TIMER_WARNING_THRESHOLD:
            self.viewer.draw_text(str(self.current_time), x=TIMER_DISPLAY_POSITION[0], y=TIMER_DISPLAY_POSITION[1], font_size=TIMER_DISPLAY_FONTSIZE, color=TIMER_COLOR_NEUTRAL, anchor_y=TIMER_DISPLAY_TEXT_Y_ANCHOR, bold=True)
        elif self.current_time < TIMER_DANGER_THRESHOLD:
            self.viewer.draw_text(str(self.current_time), x=TIMER_DISPLAY_POSITION[0], y=TIMER_DISPLAY_POSITION[1], font_size=TIMER_DISPLAY_FONTSIZE, color=TIMER_COLOR_WARNING, anchor_y=TIMER_DISPLAY_TEXT_Y_ANCHOR, bold=True)
        else:
            self.viewer.draw_text(str(self.current_time), x=TIMER_DISPLAY_POSITION[0], y=TIMER_DISPLAY_POSITION[1], font_size=TIMER_DISPLAY_FONTSIZE, color=TIMER_COLOR_DANGER, anchor_y=TIMER_DISPLAY_TEXT_Y_ANCHOR, bold=True)

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

    def render(self, mode='human'):
        if not self.is_mode_switch:
            #render bodies
            self._render_bodies()
            #draw robot direction indicator after the robot has been drawn.
            if self.is_rotation:
                self._render_robot_direction_indicators()
            #render waypoints
            self._render_waypoints()
            #render path
            self._render_path()
        else:
            self._render_mode_display()
            self._render_mode_display_text()
        self._render_timer_text()
        self._render_text(ACTION_DISPLAY_POSITION)
        return self.viewer.render(False)

    def _init_allowed_directions_of_motion(self):
        #depending on direction of motion and relative position of the goal with respect to the robot make sure only moves in the correct direction
        if self.allowed_mode_index == 't':
            allowed_direction_of_motion = -1*get_sign_of_number(self.goal_orientation) # opposite sign for pyglet positive motion
        else:
            allowed_direction_of_motion = get_sign_of_number(self.goal_position[self.start_direction.value ]-self.robot_position[self.start_direction.value ])

        return allowed_direction_of_motion

    def _is_goal_reached(self):
        # if close enough to goal (based on threshold), then signal that it is done so can move to next trial
        # NOTICE absolute value: assumes direction is already taken care of
        is_done = False
        if self.allowed_mode_index == 't':
            if np.abs(self.goal_orientation - self.robot.get_angle()) < self.orientation_thresh:
                is_done = True
        else:
            if np.abs(self.goal_position[self.start_direction.value]-self.robot.get_position()[self.start_direction.value]) < self.position_thresh[self.start_direction.value]:
                is_done = True

        return is_done

    def _generate_path(self):
        if self.start_direction == StartDirection.X:
            left_edge_angle = PI/2
            right_edge_angle = -PI/2
        elif self.start_direction == StartDirection.Y:
            left_edge_angle = PI
            right_edge_angle = 0.0

        r = math.sqrt(2)*PATH_HALF_WIDTH/SCALE
        for i in range(0, self.num_locations):
            self.path_points[i][0] = self.waypoints[i] + (r*math.cos(left_edge_angle), r*math.sin(left_edge_angle))
            self.path_points[i][1] = self.waypoints[i] + (r*math.cos(right_edge_angle), r*math.sin(right_edge_angle))

    def _generate_way_points(self): # initialize waypoints for each trial (two points, start and end)
        self.waypoints[0] = self.robot_position
        self.waypoints[-1] = self.goal_position

    def _destroy(self):
        if self.robot is None: return
        self.world.DestroyBody(self.robot.robot)
        self.robot = None
        self.waypoints = np.zeros((self.num_locations, 2))

    def initialize(self):
        self.start_session = self.env_params['start']
        self.period = rospy.Duration(1.0)
        self.timer_thread = threading.Thread(target=self._render_timer, args=(self.period,))
        self.lock = threading.Lock()
        self.current_time = 0
        self.max_time = 50

    def reset(self):
        self._destroy()
        self.TARGET_TEXT = {'x': 'Horizontal', 'y': 'Vertical', 't': 'Rotation'}
        self.robot_position = self.env_params['robot_position'] # starting position
        self.robot_orientation = self.env_params['robot_orientation'] # starting orientation
        self.goal_position = self.env_params['goal_position']
        self.goal_orientation = self.env_params['goal_orientation']
        self.is_rotation = self.env_params['is_rotation'] # testing for rotation vs linear motion
        self.is_mode_switch = self.env_params['is_mode_switch'] # boolean indicates whether about changing modes or moving robot, if true, bypass regular rendering
        self.start_direction = self.env_params['start_direction'] # orientation of path in y or x (even if rotation, it'll be either x or y but will rotate in place)
        self.allowed_mode_index = self.env_params['allowed_mode_index'] # (whcih mode is active) if have vertical segment, only mode in which motion is allowed is y, etc.
        self.mode_config = self.env_params['mode_config'] # only useful if trial is testing for mode switch
        self.action = self.env_params['action']
        # generate command prompt text:
        if self.is_rotation:
            self.msg_prompt = 'Turn ' + self.action
        else:
            self.msg_prompt = 'Move ' + self.action
        if self.is_mode_switch: # if mode switch trial, specify start and target mode
            self.start_mode = self.mode_config['start_mode']
            self.current_mode = self.start_mode
            self.target_mode = self.mode_config['target_mode']
            self.msg_prompt = 'Switch Mode To '+ self.TARGET_TEXT[self.target_mode]
        else:
            self.start_mode = None
            self.current_mode = None
            self.target_mode = None

        self.num_locations = 2 # one segment of path, location refers to start and end of segment
        self.waypoints = np.zeros((self.num_locations, 2)) # (start and end position of segment ) above and below the robot or right and left of the robot
        self.path_points = np.zeros((self.num_locations,2,2)) # (outer border of path, just for visualization purposes) points to draw the outline of the path.

        self.LOCATIONS = ['p' +str(i) for i in range(self.num_locations)] #create location id. p0, p1, p2....pN
        self.DIMENSIONS = ['x', 'y', 't'] #set of dimensions or modes
        self.DIMENSION_INDICES = np.array([0,1,2]) #set of mode indices (needed for look up)

        if not self.is_mode_switch:
            self._generate_way_points() # generate waypoints (start and edn)
            self._generate_path() # generate outer border
            self.allowed_direction_of_motion = self._init_allowed_directions_of_motion() # for each trial mode/dimension there is a proper direction in which the velocity should be
        self.robot = RobotSE2(self.world, position=self.robot_position, orientation=self.robot_orientation, robot_color=ROBOT_COLOR_WHEN_COMMAND_REQUIRED, radius=ROBOT_RADIUS/SCALE)
        self.current_time = 0

    def step(self, input_action):
        assert 'human' in input_action.keys()
        is_done = False
        if self.is_mode_switch:
            self.current_mode_index = rospy.get_param('mode')
            self.current_mode = MODE_INDEX_TO_DIM[self.current_mode_index]
            if self.current_mode == self.target_mode:
                is_done = True
        else:
            user_vel = np.array([input_action['human'].velocity.data[0], input_action['human'].velocity.data[1], input_action['human'].velocity.data[2]]) #numpyify the velocity data.
            allowed_mode_index = DIM_TO_MODE_INDEX[self.allowed_mode_index] #0,1,2 #get the mode index of the allowed mode of motion
            user_vel[np.setdiff1d(self.DIMENSION_INDICES, allowed_mode_index)] = 0.0 #zero out all velocities except the ones for the allowed mode
            # allowed_direction_of_motion = self._init_allowed_directions_of_motion()
            # only allowing velocities in allowed mode. now also check if the direction of user_vel is correct as well.
            if get_sign_of_number(user_vel[allowed_mode_index]) != self.allowed_direction_of_motion:
                user_vel[allowed_mode_index] = 0.0  # if not, zero that velocity
            # if close enough to the goal position/orientation, then move onto next trial.
            is_done = self._is_goal_reached()
            self.robot.robot.linearVelocity = b2Vec2(user_vel[0], user_vel[1]) #update robot velocity
            self.robot.robot.angularVelocity = -user_vel[2] # note the negative sign on the 3rd component to account for proper counterclockwise motion

        self.world.Step(1.0/FPS, VELOCITY_ITERATIONS, POSITION_ITERATIONS) #call box2D step function
        return is_done
