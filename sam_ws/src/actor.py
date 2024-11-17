import pybullet as p
import numpy as np
import sys
import os
import json
import time
import open3d as o3d
import matplotlib.pyplot as plt

parent_dir = "/home/user/sam_pybullet_env"
sys.path.append(parent_dir)
from utils.utils import *
from utils.planner import GraspPlanner
from env.ycb_scene import SimulatedYCBEnv
from env.lidar import LIDAR
from utils.grasp_checker import ValidGraspChecker

class Actor(object):
    def __init__(self):

        file = os.path.join(parent_dir, "object_index", "ycb_large.json")
        with open(file) as f: file_dir = json.load(f)
        file_dir = file_dir["test"]
        file_dir = [f[:-5] for f in file_dir]
        test_file_dir = list(set(file_dir))

        self.env = SimulatedYCBEnv()
        lidar = LIDAR()
        self.env._load_index_objs(test_file_dir)
        self.env.reset()

        return

if __name__ == '__main__':
    actor = Actor()