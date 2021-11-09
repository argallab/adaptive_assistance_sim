#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

from backends.rendering import Viewer
import sys
import os
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from adaptive_assistance_sim_utils import *


class PUmGivenUIEnv(object):
    def __init__(self, env_params):
        self.viewer = None
        self.env_params = env_params
        assert self.env_params is not None
        assert "text" in self.env_params
        self.text = ""
        self.bold = True

    def _render_text(self):
        self.viewer.draw_text(
            self.text,
            x=COMMAND_DISPLAY_POSITION[0],
            y=COMMAND_DISPLAY_POSITION[1],
            font_size=COMMAND_DISPLAY_FONTSIZE,
            color=COMMAND_TEXT_COLOR,
            bold=self.bold,
        )

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)
            self.viewer.window.set_location(650, 300)

        self._render_text()
        return self.viewer.render(False)

    def reset(self):
        # TODO(mahdieh) Definitely better ways of doing this...
        if "x" in self.env_params.keys():
            self.x = self.env_params["x"]
        if "y" in self.env_params.keys():
            self.y = self.env_params["y"]
        if "text" in self.env_params.keys():
            self.text = self.env_params["text"]
        if "font_name" in self.env_params.keys():
            self.font_name = self.env_params["font_name"]
        if "font_size" in self.env_params.keys():
            self.font_size = self.env_params["font_size"]
        if "anchor_x" in self.env_params.keys():
            self.anchor_x = self.env_params["anchor_x"]
        if "anchor_y" in self.env_params.keys():
            self.anchor_y = self.env_params["anchor_y"]
        if "color" in self.env_params.keys():
            self.color = self.env_params["color"]
        if "bold" in self.env_params.keys():
            self.bold = self.env_params["bold"]
