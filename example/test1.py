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
from utils.grasp_checker import ValidGraspChecker

CONTROL_MODES = "velocity"

def expert_plan(goal_pose, world=False, visual=False):
    if world:
        pos, orn = env._get_ef_pose()
        ef_pose_list = [*pos, *orn]
    else:
        ef_pose_list = [0, 0, 0, 0, 0, 0, 1]
    goal_pos = [*goal_pose[:3], *ros_quat(goal_pose[3:])]

    solver = planner.plan(ef_pose_list, goal_pos)
    path = solver.getSolutionPath().getStates()
    planer_path = []
    for i in range(len(path)):
        waypoint = path[i]
        rot = waypoint.rotation()
        action = [waypoint.getX(), waypoint.getY(), waypoint.getZ(), rot.w, rot.x, rot.y, rot.z]
        planer_path.append(action)

    return planer_path

'''
get data file name in json file and load mesh in pybullet
then reset robot and object position
'''

path_points = [[0, -1, -0],[1, -1, -0],[1, -3.5, -3.14]]

max_wheel_velocity = 15  # 最大車輪速度


file = os.path.join("../object_index", 'ycb_large.json')
with open(file) as f: file_dir = json.load(f)
file_dir = file_dir['test']
file_dir = [f[:-5] for f in file_dir]
test_file_dir = list(set(file_dir))

env = SimulatedYCBEnv()
lidar = LIDAR()
env._load_index_objs(test_file_dir)
state = env.reset(save=False)
for i in range(20):
    state = env.reset(save=False, reset_free=True, enforce_face_target=True)
name = env.target_name
print(name)

init_m = env._get_ef_pose(mat=True)
init_pose = pack_pose(init_m)

# planner = GraspPlanner()

# # pcd = lidar._get_observation(pose = base_pose,vis=True)
# grasp_checker = ValidGraspChecker(env)
# scale_str_num = len(f"_{env.object_scales[env.target_idx]}") * (-1)
# obj_name = env.obj_path[env.target_idx].split('/')[-2]
# print(obj_name)
# current_dir = os.path.abspath('')
# data_dir = current_dir.replace("example", "data/grasps/simulated")
# tr = np.load(f'{data_dir}/{obj_name}.npy',
#             allow_pickle=True,
#             fix_imports=True,
#             encoding="bytes")
# grasp = tr.item()[b'transforms']

# obj_pos = p.getBasePositionAndOrientation(env._objectUids[env.target_idx])
# obj_pos_mat = unpack_pose([*obj_pos[0] , *tf_quat(obj_pos[1])])
# grasp_candidate = obj_pos_mat.dot(grasp.T)
# grasp_candidate = np.transpose(grasp_candidate, axes=[2, 0, 1])

# obj_pos = p.getBasePositionAndOrientation(env._objectUids[env.target_idx])
# obj_pos_mat = unpack_pose([*obj_pos[0] , *tf_quat(obj_pos[1])])
# grasp_candidate = obj_pos_mat.dot(grasp.T)
# grasp_candidate = np.transpose(grasp_candidate, axes=[2, 0, 1])

# grasp_array, grasp_index = grasp_checker.extract_grasp(grasp_candidate,
#                                                        drawback_distance=0.015,
#                                                        visual=False,
#                                                        filter_elbow=True)
# target = pack_pose(grasp_array[0])
# target[2] += 0.2
# plan = expert_plan(target, world=True)
# for i in range(len(plan)):
#     # Set Target position with world frame based coordinate
#     next_pos = plan[i]
#     jointPoses = env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
#     jointPoses[6] = 0
#     jointPoses = jointPoses[:7].copy()
#     obs = env.step(jointPoses, config=True, repeat=200)[0]

# target = pack_pose(grasp_array[0])
# plan = expert_plan(target, world=True)
# for i in range(len(plan)):
#     # Set Target position with world frame based coordinate
#     next_pos = plan[i]
#     jointPoses = env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
#     jointPoses[6] = 0
#     jointPoses = jointPoses[:7].copy()
#     obs = env.step(jointPoses, config=True, repeat=200)[0]

# env.retract()

# plan = expert_plan(init_pose, world=True)
# for i in range(len(plan)):
#     # Set Target position with world frame based coordinate
#     next_pos = plan[i]
#     jointPoses = env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
#     jointPoses[6] = 0.9
#     jointPoses = jointPoses[:7].copy()
#     obs = env.step(jointPoses, config=True, repeat=200)[0]

# env._panda.Mobilebase_control(path_points)

# init_m = env._get_ef_pose(mat=True)
# init_pose = pack_pose(init_m)

# target[:3] = [0.17, -3.5, 0.65]
# target[6] *= -1
# plan = expert_plan(target, world=True)
# for i in range(len(plan)):
#     # Set Target position with world frame based coordinate
#     next_pos = plan[i]
#     jointPoses = env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
#     jointPoses[6] = 0.9
#     jointPoses = jointPoses[:7].copy()
#     obs = env.step(jointPoses, config=True, repeat=200)[0]

# plan = expert_plan(init_pose, world=True)
# for i in range(len(plan)):
#     # Set Target position with world frame based coordinate
#     next_pos = plan[i]
#     jointPoses = env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
#     jointPoses[6] = 0
#     jointPoses = jointPoses[:7].copy()
#     obs = env.step(jointPoses, config=True, repeat=200)[0]

# path_points = [[1, -1, -0],[0, -1, -0],[0, 0, 0]]
# env._panda.Mobilebase_control(path_points)



# state = env.reset(save=False, reset_free=True, enforce_face_target=True)
# env._panda.reset()
while True:


    # base_pose = p.getBasePositionAndOrientation(env._panda.pandaUid)[0]
    # print(base_pose)
    # env._panda.AMR_control(CONTROL_MODES, right_wheel_value, left_wheel_value)
    # p.stepSimulation()
    pass



