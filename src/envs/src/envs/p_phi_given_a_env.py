#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import os
from backends.rendering import Viewer
import sys
import os
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from adaptive_assistance_sim_utils import *
from simulators.msg import Command
from std_msgs.msg import String
import pyglet
import collections
import rospy
import time
from pyglet.window import key
import random
import threading

# TODO: (mahdeih) Too many if statments and booleans, can do a better job of the state machine, clean in up!!


class PPhiGivenAEnv(object):
    def __init__(self, env_params):

        self.viewer = None
        self.action_msg = Command()
        self.sim_state_msg = String()

        self.env_params = env_params
        assert self.env_params is not None
        assert "file_dir" in self.env_params
        assert "blocks" in self.env_params

        self.file_dir = self.env_params["file_dir"]
        self.img_prompt = self.env_params["img_prompt"]

        self.ts = time.time()
        self.te = time.time()
        self.prompt_ind = 0
        self.msg_prompt = ""
        self.start_prompt = False
        self.clear_for_next_prompt = False
        self.bold = True
        self.action_timing_bound = 7  # seconds
        self.text_timing_bound = 2

        self.period = rospy.Duration(1.0)
        self.timer_thread = threading.Thread(target=self._render_timer, args=(self.period,))
        self.lock = threading.Lock()
        self.current_time = 0
        self.start_timer = False
        self.start_training = False
        self.next = False
        self.back = False
        self.current_block = 0
        self.display_timer = False
        self.ready_for_user = False

    def initialize_viewer(self):
        if self.viewer is None:
            self.viewer = Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, (VIEWPORT_W) / SCALE, 0, (VIEWPORT_H) / SCALE)
            self.viewer.window.set_location(650, 300)
            self.timer_thread.start()

    def initialize_publishers(self, rostopic):
        # TODO: clean publishers:
        self.sim_state = rospy.Publisher("sim_state", String, queue_size=1)

    def publish_action(self, msg):
        self.action_msg.header.stamp = rospy.Time.now()
        self.action_msg.command = msg
        self.action_pub.publish(self.action_msg)

    def publish_sim_state(self, msg):
        self.sim_state.publish(msg)

    def _set_image_path(self):
        self.file_path = os.path.abspath(os.path.join(self.file_dir, self.img_prompt + ".png"))

    def _render_sprite(self, x, y, scale):
        self.viewer.draw_sprite(self.file_path, x=x, y=y, scale=scale)

    def _render_text(self):
        self.viewer.draw_text(
            self.msg_prompt,
            x=COMMAND_DISPLAY_POSITION[0],
            y=COMMAND_DISPLAY_POSITION[1],
            font_size=COMMAND_DISPLAY_FONTSIZE,
            color=COMMAND_TEXT_COLOR,
            bold=self.bold,
        )

    def _render_options(self):
        self.viewer.draw_text(
            "1",
            x=OPTION_DISPLAY_POSITION[0],
            y=OPTION_DISPLAY_POSITION[1],
            font_size=OPTION_DISPLAY_FONTSIZE,
            color=OPTIONS_TEXT_COLOR,
            bold=self.bold,
        )
        self.viewer.draw_text(
            "2",
            x=OPTION_DISPLAY_POSITION[0] + OPTION_DISPLAY_OFFSET,
            y=OPTION_DISPLAY_POSITION[1],
            font_size=OPTION_DISPLAY_FONTSIZE,
            color=OPTIONS_TEXT_COLOR,
            bold=self.bold,
        )
        self.viewer.draw_text(
            "3",
            x=OPTION_DISPLAY_POSITION[0] + 2 * OPTION_DISPLAY_OFFSET,
            y=OPTION_DISPLAY_POSITION[1],
            font_size=OPTION_DISPLAY_FONTSIZE,
            color=OPTIONS_TEXT_COLOR,
            bold=self.bold,
        )
        self.viewer.draw_text(
            "4",
            x=OPTION_DISPLAY_POSITION[0] + 3 * OPTION_DISPLAY_OFFSET,
            y=OPTION_DISPLAY_POSITION[1],
            font_size=OPTION_DISPLAY_FONTSIZE,
            color=OPTIONS_TEXT_COLOR,
            bold=self.bold,
        )

        self.viewer.draw_text(
            "Hard Puff",
            x=OPTION_TEXT_DISPLAY_POSITION[0],
            y=OPTION_TEXT_DISPLAY_POSITION[1],
            font_size=OPTION_TEXT_DISPLAY_FONTSIZE,
            color=OPTIONS_TEXT_COLOR,
            bold=self.bold,
        )
        self.viewer.draw_text(
            "Hard Sip",
            x=OPTION_TEXT_DISPLAY_POSITION[0] + OPTION_TEXT_DISPLAY_OFFSET,
            y=OPTION_TEXT_DISPLAY_POSITION[1],
            font_size=OPTION_TEXT_DISPLAY_FONTSIZE,
            color=OPTIONS_TEXT_COLOR,
            bold=self.bold,
        )
        self.viewer.draw_text(
            "Soft Puff",
            x=OPTION_TEXT_DISPLAY_POSITION[0] + 2 * OPTION_TEXT_DISPLAY_OFFSET,
            y=OPTION_TEXT_DISPLAY_POSITION[1],
            font_size=OPTION_TEXT_DISPLAY_FONTSIZE,
            color=OPTIONS_TEXT_COLOR,
            bold=self.bold,
        )
        self.viewer.draw_text(
            "Soft Sip",
            x=OPTION_TEXT_DISPLAY_POSITION[0] + 3 * OPTION_TEXT_DISPLAY_OFFSET,
            y=OPTION_TEXT_DISPLAY_POSITION[1],
            font_size=OPTION_TEXT_DISPLAY_FONTSIZE,
            color=OPTIONS_TEXT_COLOR,
            bold=self.bold,
        )

    def _render_timer_text(self):
        self.viewer.draw_text(
            str(self.current_time),
            x=TIMER_DISPLAY_POSITION[0] + VIEWPORT_W / 2 - 40,
            y=TIMER_DISPLAY_POSITION[1],
            font_size=TIMER_DISPLAY_FONTSIZE,
            color=TIMER_COLOR_NEUTRAL,
            anchor_y=TIMER_DISPLAY_TEXT_Y_ANCHOR,
            bold=True,
        )

    def _render_timer(self, period):
        while not rospy.is_shutdown():
            start = rospy.get_rostime()
            self.lock.acquire()
            if self.start_timer:
                if self.current_time > 0:
                    self.current_time -= 1
            else:
                pass
            self.lock.release()
            end = rospy.get_rostime()
            if end - start < period:
                rospy.sleep(period - (end - start))
            else:
                rospy.loginfo("took more time")

    def render(self):
        self.viewer.window.clear()
        if self.img_prompt != "":
            if not self.start_training:
                self._render_options()
            if self.img_prompt in self.motion_actions:
                self._render_sprite(x=VIEWPORT_W / 4, y=VIEWPORT_H / 4, scale=0.2)
            if self.img_prompt in self.mode_actions:
                self._render_sprite(x=VIEWPORT_W / 4, y=VIEWPORT_H / 4, scale=0.05)
            if self.img_prompt in self.training_prompts:
                self._render_sprite(x=VIEWPORT_W / 8, y=VIEWPORT_H / 8, scale=0.7)
        if self.display_timer:
            self._render_timer_text()

        self._render_text()
        return self.viewer.render(False)

    def training_refresher(self):
        self.img_prompt = self.training_prompts[self.msg_ind]
        self._set_image_path()

    def step(self):
        bool_publish = False
        if self.start_prompt:
            if self.display_start_countdown:
                self.display_timer = False
                if self.start_msg_ind < len(EXPERIMENT_START_COUNTDOWN):
                    if self.ready_for_new_prompt:
                        self.msg_prompt = EXPERIMENT_START_COUNTDOWN[self.start_msg_ind]
                        self.start_msg_ind += 1
                        self.ts = time.time()
                        self.ready_for_new_prompt = False
                    if (time.time() - self.ts) >= self.text_timing_bound:
                        self.ready_for_new_prompt = True
                else:
                    self.msg_prompt = ""
                    self.display_start_countdown = False
                    self.clear_for_next_prompt = False
                    self.ready_for_new_prompt = True
                    self.publish_sim_state("start_test")
            else:
                if self.ready_for_new_prompt:
                    self.img_prompt = self.action_prompts[self.prompt_ind]
                    bool_publish = True
                    self._set_image_path()
                    self.ready_for_new_prompt = False
                    self.ready_for_user = True
                    self.display_timer = True
                    self.ts = time.time()
                    self.current_time = self.action_timing_bound

                if self.clear_for_next_prompt:
                    # TODO: make  not hardcoded
                    # delay for a little bit so user has some break in between trainsitions (not immediate)
                    if time.time() - self.te >= 0.5:
                        self.prompt_ind += 1
                        self.ready_for_new_prompt = True
                        self.clear_for_next_prompt = False
                        self.img_prompt = ""
                        bool_publish = True
                        # if reached end of prompt list, if more batches left, go to training, else show end message
                        if self.prompt_ind >= len(self.action_prompts):
                            self.img_prompt = ""
                            self.start_prompt = False
                            if int(self.current_block) < int(self.blocks):
                                self.start_training = True
                                self.publish_sim_state("start_training")
                            else:
                                self.msg_prompt = "End of Test"

                elif (time.time() - self.ts) >= self.action_timing_bound:
                    # if no response from user and set time limit reached, clear for next prompt
                    self.clear_for_next_prompt = True
                    self.ready_for_user = False
                    self.te = time.time()
                    self.msg_prompt = ""
                    self.img_prompt = ""
                    self.display_timer = False

        if self.start_training:
            self.msg_prompt = ""
            self.display_timer = False
            self.start_timer = False
            self.training_refresher()

        return (bool_publish, self.img_prompt)

    def reset(self):
        if "action_prompts" in self.env_params.keys():
            self.action_prompts = self.env_params["action_prompts"]

        if "actions" in self.env_params.keys():
            if "motion_actions" in self.env_params["actions"]:
                self.motion_actions = self.env_params["actions"]["motion_actions"]
            if "mode_actions" in self.env_params["actions"]:
                self.mode_actions = self.env_params["actions"]["mode_actions"]

        if "start_prompt" in self.env_params.keys():
            if self.start_prompt == False:
                self.start_prompt = self.env_params["start_prompt"]
                self.display_start_countdown = True
                self.ready_for_new_prompt = True
                self.clear_for_next_prompt = False
                self.start_training = False
                self.prompt_ind = 0
                self.start_msg_ind = 0
                self.start_timer = True
                random.shuffle(self.action_prompts)
                self.publish_sim_state("start_countdown")

        if "training_prompts" in self.env_params.keys():
            self.training_prompts = self.env_params["training_prompts"]
            self.msg_ind = 0

        if "blocks" in self.env_params.keys():
            self.blocks = self.env_params["blocks"]
            self.blocks = int(self.blocks) - 1

    def _get_user_input(self):
        # TODO: Clean this up, divide to separate functions for next prompt and (next+back)
        # works only during start_prompt, if user responds with 1,2,3,4 key, will show next prompt
        if "next_prompt" in self.env_params.keys():
            if self.ready_for_user:
                self.clear_for_next_prompt = self.env_params["next_prompt"]
                self.env_params["next_prompt"] = False  # reset
                self.ready_for_user = False
                self.te = time.time()
                self.msg_prompt = ""
                self.img_prompt = ""
                self.display_timer = False
                return True
        # works only during training, if user presses -> button, will go to next image
        if "next" in self.env_params.keys():
            if self.start_training:
                self.next = self.env_params["next"]
                if self.next:
                    if self.msg_ind < len(self.training_prompts) - 1:
                        self.msg_ind += 1
                    elif (
                        self.msg_ind == len(self.training_prompts) - 1
                    ):  # if at last training image and press next, start next batch again
                        self.env_params["start_prompt"] = True
                        self.current_block += 1  # increment batch/block
                        self.start_training = False
                        self.img_prompt = ""
                        self.msg_prompt = ""
                        self.reset()
                    self.env_params["next"] = False

        if "back" in self.env_params.keys():  # works only during training, if user presses <- goes to previous image
            if self.start_training:
                self.back = self.env_params["back"]
                if self.back:
                    if self.msg_ind > 0:
                        self.msg_ind -= 1
                    self.env_params["back"] = False

        return False


if __name__ == "__main__":
    PPhiGivenAEnv()
