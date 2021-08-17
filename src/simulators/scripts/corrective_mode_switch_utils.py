# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

from Box2D import b2CircleShape, b2EdgeShape, b2Vec2
import math
from enum import Enum
import collections

# TODO move defines to a separate file
FPS = 100
VELOCITY_ITERATIONS = 180
POSITION_ITERATIONS = 60
SCALE = 20.0

VIEWPORT_W = 600
PI = math.pi
VIEWPORT_H = 400

# COLORS
HUMAN_ROBOT_COLOR = (0.0, 0.0, 0.0)
AUTONOMY_ROBOT_COLOR = (0.5, 0.5, 0.5)
# TURN_LOCATION_COLOR = (173/255.0, 216/255.0, 230/255.0)
TURN_LOCATION_COLOR = (0.7, 0.52, 0.75)
GRIPPER_ACTION_LOCATION_COLOR = (0.2, 0.3, 0.5)
ACTIVE_MODE_COLOR = (0, 1.0, 0.0)
NONACTIVE_MODE_COLOR = (0.8, 0.8, 0.8)
TARGET_MODE_COLOR = (0.8, 0.8, 0.8)
ACTIVE_MODE_COLOR_WARNING = (1.0, 1.0, 0)
ACTIVE_MODE_COLOR_ERROR = (1.0, 0, 0)
MODE_DISPLAY_TEXT_COLOR = (0, 0, 0, 255)
TRIAL_OVER_TEXT_COLOR = (0, 0, 0, 255)

ROBOT_COLOR_WHEN_COMMAND_REQUIRED = (1.0, 105.0 / 255, 80 / 255.0)
ROBOT_COLOR_WHEN_MOVING = (1.0, 0.0, 0.0)
COMMAND_TEXT_COLOR = (82, 179, 217, 255)  # blue (shakespeare)
OPTIONS_TEXT_COLOR = (0, 0, 0, 255)
TIMER_COLOR_NEUTRAL = (3, 166, 120, 255)  # green (free speech aquamarine)
TIMER_COLOR_DANGER = (240, 52, 52, 255)  # red (pomegranate)
TIMER_COLOR_WARNING = (255, 197, 97, 255)  # yellow (candy corn)

# GEOMETRY
MODE_DISPLAY_RADIUS = 15
ROBOT_RADIUS = 14
GOAL_RADIUS = 14
TRIANGLE_L = 6
GOAL_COLORS = {"circle": (1.0, 0.0, 0.0), "triangle": (0.0, 1.0, 0.0), "rect": (0.0, 0.0, 1.0)}
GOAL_SHAPES = {0: "circle", 1: "triangle", 2: "rect"}
WP_RADIUS = 4
INFLATION_FACTOR = 1.2
PATH_HALF_WIDTH = INFLATION_FACTOR * ROBOT_RADIUS

VIEWPORT_WS = VIEWPORT_W / SCALE
VIEWPORT_HS = VIEWPORT_H / SCALE
ROBOT_RADIUS_S = ROBOT_RADIUS / SCALE
GOAL_RADIUS_S = GOAL_RADIUS / SCALE


MODE_DISPLAY_CIRCLE_START_POSITION_S = [(3 * VIEWPORT_WS) / 4 + 2 * ROBOT_RADIUS_S, VIEWPORT_HS - 2 * ROBOT_RADIUS_S]
MODE_DISPLAY_CIRCLE_X_OFFSET_S = 3 * ROBOT_RADIUS_S
MODE_DISPLAY_CIRCLE_Y_OFFSET_S = 60 / SCALE
P_UM_GIVEN_A_MODE_DISPLAY_CIRCLE_START_POSITION_S = [
    VIEWPORT_WS / 2 - MODE_DISPLAY_CIRCLE_X_OFFSET_S,
    2 * VIEWPORT_HS / 3,
]

MODE_DISPLAY_TEXT_START_POSITION = [(3 * VIEWPORT_W) / 4 + 2 * ROBOT_RADIUS, VIEWPORT_H - 3.5 * ROBOT_RADIUS]
P_UM_GIVEN_A_MODE_DISPLAY_TEXT_START_POSITION = [
    VIEWPORT_W / 2 - MODE_DISPLAY_CIRCLE_X_OFFSET_S * SCALE,
    2 * VIEWPORT_H / 3 - 2 * ROBOT_RADIUS,
]
MODE_DISPLAY_TEXT_X_OFFSET = 3 * ROBOT_RADIUS
MODE_DISPLAY_TEXT_FONTSIZE = 8
MODE_DISPLAY_TEXT_Y_ANCHOR = "top"

TIMER_DISPLAY_POSITION = [(VIEWPORT_W) / 2, VIEWPORT_H - ROBOT_RADIUS]
TIMER_DISPLAY_FONTSIZE = 28
TIMER_DISPLAY_TEXT_Y_ANCHOR = "top"

START_GOAL_TEXT_DISPLACEMENT = 40

TRIAL_OVER_TEXT_DISPLAY_POSITION = [VIEWPORT_W / 2, 35]
TRIAL_OVER_TEXT_FONTSIZE = 20
TRIAL_OVER_TEXT_Y_ANCHOR = "top"

COMMAND_DISPLAY_POSITION = [VIEWPORT_W / 2, VIEWPORT_H / 2]
ACTION_DISPLAY_POSITION = [VIEWPORT_W / 2, 20]
LABEL_DISPLAY_POSITION = [1 * VIEWPORT_W / 6, 9 * VIEWPORT_H / 10]
OPTION_DISPLAY_POSITION = [VIEWPORT_W / 5, VIEWPORT_H / 5]
OPTION_TEXT_DISPLAY_POSITION = [VIEWPORT_W / 5, VIEWPORT_H / 7]

OPTION_DISPLAY_OFFSET = VIEWPORT_W / 5
OPTION_TEXT_DISPLAY_OFFSET = VIEWPORT_W / 5

COMMAND_DISPLAY_FONTSIZE = 36
OPTION_DISPLAY_FONTSIZE = 20
OPTION_TEXT_DISPLAY_FONTSIZE = 14

TIMER_WARNING_THRESHOLD = 20
TIMER_DANGER_THRESHOLD = 30

# DICTIONARIES
DIM_TO_MODE_INDEX = {"x": 0, "y": 1, "t": 2, "gr": 3}
MODE_INDEX_TO_DIM = {v: k for k, v in DIM_TO_MODE_INDEX.items()}

ASSISTANCE_CODE_NAME = {0: "Filtas", 1: "Coras", 2: "Noas"}

# LISTS
# low level commands issued by the snp interface. hp = hard puff, hs= hard sip, sp = soft puff, ss = soft sip. Also the domain for ui and um
LOW_LEVEL_COMMANDS = ["Hard Puff", "Hard Sip", "Soft Puff", "Soft Sip"]
EXPERIMENT_START_COUNTDOWN = ["Get Ready!", "3", "2", "1"]

# high level actions, move_p = move in positive direction, move_n = move in negative direction, mode_r = switch mode to right, mode_l = switch mode to left. positive and negative is conditioned on mode
HIGH_LEVEL_ACTIONS = ["move_p", "move_n", "mode_r", "mode_l"]
# true mapping of a to u
TRUE_ACTION_TO_COMMAND = collections.OrderedDict(
    {
        "x": collections.OrderedDict(
            {"move_p": "Soft Puff", "move_n": "Soft Sip", "mode_r": "Hard Puff", "mode_l": "Hard Sip"}
        ),
        "y": collections.OrderedDict(
            {"move_p": "Soft Puff", "move_n": "Soft Sip", "mode_r": "Hard Puff", "mode_l": "Hard Sip"}
        ),
        "t": collections.OrderedDict(
            {"move_p": "Soft Sip", "move_n": "Soft Puff", "mode_r": "Hard Puff", "mode_l": "Hard Sip"}
        ),
        "gr": collections.OrderedDict(
            {"move_p": "Soft Puff", "move_n": "Soft Sip", "mode_r": "Hard Puff", "mode_l": "Hard Sip"}
        ),
    }
)  # for gripper mode, move_p refers to closing the gripper and move_n refers to opening the gripper
# true inverse mapping of u to a
TRUE_COMMAND_TO_ACTION = collections.OrderedDict()
for k in TRUE_ACTION_TO_COMMAND.keys():
    TRUE_COMMAND_TO_ACTION[k] = collections.OrderedDict({v: k for k, v in TRUE_ACTION_TO_COMMAND[k].items()})

# transition function for mode switches.
MODE_SWITCH_TRANSITION = {
    "x": {"Hard Puff": "y", "Hard Sip": "t", "Soft Puff": "x", "Soft Sip": "x"},
    "y": {"Hard Puff": "t", "Hard Sip": "x", "Soft Puff": "y", "Soft Sip": "y"},
    "t": {"Hard Puff": "x", "Hard Sip": "y", "Soft Puff": "t", "Soft Sip": "t"},
}

MODE_SWITCH_TRANSITION_4D = {
    "x": {"Hard Puff": "y", "Hard Sip": "gr", "Soft Puff": "x", "Soft Sip": "x"},
    "y": {"Hard Puff": "t", "Hard Sip": "x", "Soft Puff": "y", "Soft Sip": "y"},
    "t": {"Hard Puff": "gr", "Hard Sip": "y", "Soft Puff": "t", "Soft Sip": "t"},
    "gr": {"Hard Puff": "x", "Hard Sip": "t", "Soft Puff": "gr", "Soft Sip": "gr"},
}
# Depending on the configuration of the initial robot position and goal position, the motion commands will result in either moving towards the
# next location or the previous location
# Enum defintions


# low level commands issued by the snp interface. hp = hard puff, hs= hard sip, sp = soft puff, ss = soft sip. Also the domain for ui and um
INTERFACE_LEVEL_ACTIONS = ["Hard Puff", "Hard Sip", "Soft Puff", "Soft Sip"]
# high level actions, move_p = move in positive direction, move_n = move in negative direction, mode_r = switch mode to right, mode_l = switch mode to left. positive and negative is conditioned on mode
TASK_LEVEL_ACTIONS = ["move_p", "move_n", "to_mode_r", "to_mode_l"]
# true mapping of a to phi
TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP = collections.OrderedDict(
    {"move_p": "Soft Puff", "move_n": "Soft Sip", "to_mode_r": "Hard Puff", "to_mode_l": "Hard Sip"}
)
# true inverse mapping of phi to a
TRUE_INTERFACE_ACTION_TO_TASK_ACTION_MAP = collections.OrderedDict(
    {v: k for k, v in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.items()}
)


class AssistanceType(Enum):
    Filter = 0
    Corrective = 1
    No_Assistance = 2


class PositionOnLine(Enum):
    START = 0
    BETWEEN = 1
    END = 2
    NOT_ON_LINE = 3
    ON_LINE = 4  # could be beyond the end points


class CartesianRobotType(Enum):
    R2 = 2  # (x,y)
    SE2_NH = 2  # (v, w)
    R3 = 3  # (x, y, z)
    SE2 = 3  # (x,y,theta)
    SE3 = 6  # (x,y,z,r,p,yaw)


class ModeSetType(Enum):
    OneD = 1
    TwoD = 2
    ThreeD = 3


class ModeTransitionType(Enum):
    Direct = 0
    Forward = 1
    Backward = 2
    Forward_Backward = 3


class OrientationOnArc(Enum):
    START = 0
    BETWEEN = 1
    END = 2
    ON_EXTENDED_ARC = 3


class StartDirection(Enum):
    X = 0
    Y = 1


class RGOrient(Enum):
    """
	Relative position of the goal with respect to the robot. This information is used for constructing properly shared paths in ModeInferenceEnv
	"""

    TOP_RIGHT = 0
    TOP_LEFT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


TRANSITION_FOR_ACTION = {
    RGOrient.TOP_RIGHT: {
        "Soft Puff": {"x": "next", "y": "next", "t": "prev"},
        "Soft Sip": {"x": "prev", "y": "prev", "t": "next"},
    },
    RGOrient.TOP_LEFT: {
        "Soft Puff": {"x": "prev", "y": "next", "t": "prev"},
        "Soft Sip": {"x": "next", "y": "prev", "t": "next"},
    },
    RGOrient.BOTTOM_RIGHT: {
        "Soft Puff": {"x": "next", "y": "prev", "t": "prev"},
        "Soft Sip": {"x": "prev", "y": "next", "t": "next"},
    },
    RGOrient.BOTTOM_LEFT: {
        "Soft Puff": {"x": "prev", "y": "prev", "t": "prev"},
        "Soft Sip": {"x": "next", "y": "next", "t": "next"},
    },
}

TRANSITION_FOR_ACTION_4D = {
    RGOrient.TOP_RIGHT: {
        "Soft Puff": {"x": "next", "y": "next", "t": "prev", "gr": "next"},
        "Soft Sip": {"x": "prev", "y": "prev", "t": "next", "gr": "prev"},
    },
    RGOrient.TOP_LEFT: {
        "Soft Puff": {"x": "prev", "y": "next", "t": "prev", "gr": "next"},
        "Soft Sip": {"x": "next", "y": "prev", "t": "next", "gr": "prev"},
    },
    RGOrient.BOTTOM_RIGHT: {
        "Soft Puff": {"x": "next", "y": "prev", "t": "prev", "gr": "next"},
        "Soft Sip": {"x": "prev", "y": "next", "t": "next", "gr": "prev"},
    },
    RGOrient.BOTTOM_LEFT: {
        "Soft Puff": {"x": "prev", "y": "prev", "t": "prev", "gr": "next"},
        "Soft Sip": {"x": "next", "y": "next", "t": "next", "gr": "prev"},
    },
}
# utility functions
def get_sign_of_number(x):
    """
	Utility function for getting the sign of a scalar. +1 for positive, -1 for negative
	"""
    if int(x >= 0):
        return 1.0
    else:
        return -1.0


# class definitions
class Robot2D(object):
    def __init__(self, world, position, robot_color=(1.0, 0.0, 0.0), radius=3, type="kinematic"):
        self.robot = world.CreateKinematicBody(position=position, shapes=[b2CircleShape(pos=position, radius=radius)])
        self.robot_color = robot_color


class RobotSE2(object):
    def __init__(self, world, position, orientation, robot_color=(1.0, 0.0, 0.0), radius=3, type="kinematic"):
        self.robot = world.CreateKinematicBody(
            position=position, angle=orientation, shapes=[b2CircleShape(pos=position, radius=radius)]
        )
        self.robot_color = robot_color
        self.radius = radius

    def set_robot_color(self, robot_color):
        self.robot_color = robot_color

    # GETTERS
    def get_direction_marker_end_points(self):
        return (
            (self.robot.position[0], self.robot.position[1]),
            (
                self.robot.position[0] + 2 * self.radius * math.cos(self.robot.angle),
                self.robot.position[1] + 2 * self.radius * math.sin(self.robot.angle),
            ),
        )

    def get_position(self):
        return [self.robot.position[0], self.robot.position[1]]

    def get_angle(self):
        return self.robot.angle

    def get_linear_velocity(self):
        return [self.robot.linearVelocity[0], self.robot.linearVelocity[1]]

    def get_angular_velocity(self):
        return self.robot.angularVelocity

    # Setters
    def set_position(self, position):
        self.robot.position = position

    def set_orientation(self, orientation):
        self.robot.angle = orientation


class Robot4D(object):
    def __init__(
        self,
        world,
        position,
        orientation,
        gripper_angle=PI / 2,
        robot_color=(1.0, 0.0, 0.0),
        radius=3,
        type="kinematic",
    ):
        self.robot = world.CreateKinematicBody(
            position=position, angle=orientation, shapes=[b2CircleShape(pos=position, radius=radius)]
        )
        self.robot_color = robot_color
        self.radius = radius
        self.gripper_angle = gripper_angle

    def set_robot_color(self, robot_color):
        self.robot_color = robot_color

    # GETTERS
    def get_direction_marker_end_points(self):
        return (
            (self.robot.position[0], self.robot.position[1]),
            (
                self.robot.position[0] + 2 * self.radius * math.cos(self.robot.angle),
                self.robot.position[1] + 2 * self.radius * math.sin(self.robot.angle),
            ),
        )

    def get_gripper_handle_markers(self):
        handle_1 = (
            (self.robot.position[0], self.robot.position[1]),
            (
                self.robot.position[0] + 2 * self.radius * math.cos(self.robot.angle + self.gripper_angle / 2.0),
                self.robot.position[1] + 2 * self.radius * math.sin(self.robot.angle + self.gripper_angle / 2.0),
            ),
        )
        handle_2 = (
            (self.robot.position[0], self.robot.position[1]),
            (
                self.robot.position[0] + 2 * self.radius * math.cos(self.robot.angle - self.gripper_angle / 2.0),
                self.robot.position[1] + 2 * self.radius * math.sin(self.robot.angle - self.gripper_angle / 2.0),
            ),
        )
        return [handle_1, handle_2]

    def get_position(self):
        return [self.robot.position[0], self.robot.position[1]]

    def get_angle(self):
        return self.robot.angle

    def get_gripper_angle(self):
        return self.gripper_angle

    def get_linear_velocity(self):
        return [self.robot.linearVelocity[0], self.robot.linearVelocity[1]]

    def get_angular_velocity(self):
        return self.robot.angularVelocity

    # Setters
    def set_position(self, position):
        self.robot.position = position

    def set_orientation(self, orientation):
        self.robot.angle = orientation

    def set_gripper_angle(self, gripper_angle):
        self.gripper_angle = gripper_angle

    def gripper_angle_update(self, gripper_vel):
        if gripper_vel > 0.0:
            self.gripper_angle = max(0.0, self.gripper_angle - 0.05)
        elif gripper_vel < 0.0:
            self.gripper_angle = min(PI / 2, self.gripper_angle + 0.05)

    def update(self, velocity):
        self.robot.linearVelocity = b2Vec2(velocity[0], velocity[1])
        self.robot.angularVelocity = velocity[2]
        self.gripper_angle_update(velocity[3])


class Goal(object):
    """docstring forGoal."""

    def __init__(self, arg):
        pass
