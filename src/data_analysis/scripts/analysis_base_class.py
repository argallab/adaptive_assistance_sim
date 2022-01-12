#!/usr/bin/env python

import os
import csv
import sys
import argparse
import pandas as pd
import pickle
from data_filter import DataFilter
import collections
import numpy as np
from scipy.stats import entropy
import copy


class AnalysisBase(object):
    def __init__(self, args):
        self.data_filter_obj = DataFilter(args)
        import IPython

        IPython.embed(banner1="check analysis")
