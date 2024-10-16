import pybullet as p
import numpy as np
import json
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.append("../")
from env.ycb_scene import SimulatedYCBEnv
from utils.utils import *
from utils.planner import GraspPlanner

file = os.path.join("../object_index", 'ycb_large.json')
with open(file) as f: file_dir = json.load(f)
file_dir = file_dir['test']
file_dir = [f[:-5] for f in file_dir]
test_file_dir = list(set(file_dir))

env = SimulatedYCBEnv()
env._load_index_objs(test_file_dir)
state = env.reset(save=False, enforce_face_target=True)

env._panda.reset()

# Get current status of end effector, specify "mat" argument to switch data expression
robot_pos = env._get_ef_pose()
robot_pos_mat = env._get_ef_pose(mat=True)
print(f"Position with quaternions:\n {robot_pos}\nPosition with SE(3) matrix:\n {robot_pos_mat}")
planner = GraspPlanner()

def expert_plan(goal_pose, world=False, visual=False):
    if world:
        pos, orn = env._get_ef_pose()
        ef_pose_list = [*pos, *orn]
    else:
        ef_pose_list = [0, 0, 0, 0, 0, 0, 1]
    goal_pos = [*goal_pose[:3], *ros_quat(goal_pose[3:])]

    solver = planner.plan(ef_pose_list, goal_pos)
    if visual:
        path_visulization(solver)
    path = solver.getSolutionPath().getStates()
    planer_path = []
    for i in range(len(path)):
        waypoint = path[i]
        rot = waypoint.rotation()
        action = [waypoint.getX(), waypoint.getY(), waypoint.getZ(), rot.w, rot.x, rot.y, rot.z]
        planer_path.append(action)

    return planer_path

def path_visulization(ss):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = []
    y = []
    z = []
    for i in range(len(ss.getSolutionPath().getStates())):
        state = ss.getSolutionPath().getStates()[i]
        x.append(state.getX())
        y.append(state.getY())
        z.append(state.getZ())
    ax.plot(x, y, z, color='gray', label='Curve')

    ax.scatter(x, y, z, c=z, cmap='jet', label='Points')
    plt.show()

env._panda.reset()

# A position in SE(3) relative to camera frame
desired_pt = np.array([[ 0.85913459, 0.50408285, 0.08825102, -0.09968021],
                      [-0.43048553, 0.61863272, 0.65724863, -0.13205695],
                      [ 0.27671279, -0.60265582, 0.74848914, 0.41399397],
                      [ 0. , 0. , 0. , 1. ]])

# Convert SE(3) to postion with quaternions [x, y, z, w, x, y, z]
target = pack_pose(desired_pt)

# Transform position from camera frame to flange
action = env.transform_pose_from_camera(target)
# SE(3) matrix relative to world frame
action_world = env._get_ef_pose(mat=True).dot(unpack_pose(action))

print(f"world frame:\n {action_world}\nrelative to ef: \n {action}")

env._panda.reset()

# Get path list by GraspPlanner
plan = expert_plan(action)
# Get ef pose at beginning as initial pose
init_pos = env._get_ef_pose(mat=True)
for i in range(len(plan)):
    # Get current ef pose
    ef_pos = env._get_ef_pose(mat=True)
    # Transform path waypoint from initial pose based to current pose based
    next_pos_mat = np.dot(se3_inverse(ef_pos), init_pos.dot(unpack_pose(plan[i])))
    next_pos = pack_pose(next_pos_mat)
    env.step([*next_pos[:3], *quat2euler(next_pos[3:])], repeat=200)

env._panda.reset()
# Get path list by GraspPlanner
plan = expert_plan(pack_pose(action_world), world=True, visual=True)
for i in range(len(plan)):
    # Set Target position with world frame based coordinate
    next_pos = plan[i]
    jointPoses = env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
    jointPoses[6] = 0
    jointPoses = jointPoses[:7].copy()
    obs = env.step(jointPoses, config=True, repeat=200)[0]