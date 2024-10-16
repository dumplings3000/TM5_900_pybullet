import pybullet as p
import numpy as np
import sys
import os
import json
import time
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append("../")
from utils.utils import *
from utils.planner import GraspPlanner
from env.ycb_scene import SimulatedYCBEnv
from env.lidar import LIDAR

CONTROL_MODES = "velocity"
'''
get data file name in json file and load mesh in pybullet
then reset robot and object position
'''

path_points = [[-0.5, 3.5]]
target_angle = -3.14

max_wheel_velocity = 15  # 最大車輪速度


file = os.path.join("../object_index", 'ycb_large.json')
with open(file) as f: file_dir = json.load(f)
file_dir = file_dir['test']
file_dir = [f[:-5] for f in file_dir]
test_file_dir = list(set(file_dir))

env = SimulatedYCBEnv()
lidar = LIDAR()
env._load_index_objs(test_file_dir)
state = env.reset(save=False, enforce_face_target=True)
state = env.reset(save=False, reset_free=True)
# state = env.step(action=np.array([0, 0, 0.1, 0, 0, 0]))
# base_pose = list(p.getBasePositionAndOrientation(env._panda.pandaUid)[0])
# pcd = lidar._get_observation(pose = base_pose,vis=True)
env._panda.reset()

env._panda.Mobilebase_control(path_points, target_angle)

# env._panda.reset()
while True:


    # base_pose = p.getBasePositionAndOrientation(env._panda.pandaUid)[0]
    # print(base_pose)
    # env._panda.AMR_control(CONTROL_MODES, right_wheel_value, left_wheel_value)
    # p.stepSimulation()
    pass




