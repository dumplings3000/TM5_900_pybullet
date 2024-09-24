import pybullet as p
import numpy as np
import sys
import os
import json
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append("../")
from utils.utils import *
from utils.planner import GraspPlanner
from env.ycb_scene import SimulatedYCBEnv

CONTROL_MODES = "velocity"
'''
get data file name in json file and load mesh in pybullet
then reset robot and object position
'''

file = os.path.join("../object_index", 'ycb_large.json')
with open(file) as f: file_dir = json.load(f)
file_dir = file_dir['test']
file_dir = [f[:-5] for f in file_dir]
test_file_dir = list(set(file_dir))

env = SimulatedYCBEnv()
env._load_index_objs(test_file_dir)
state = env.reset(save=False, enforce_face_target=True)
# # state = env.reset(save=False, reset_free=True)
# # state = env.step(action=np.array([0, 0, 0.1, 0, 0, 0]))

# # env._panda.reset()


# LiDAR 位置
lidar_position = [0, 0, 0.8]

# 設置 30° 視野角
fov = 30
aspect = 1.0
near = 0.1
far = 10
raw_data = False
vis = False

# 定義點雲儲存
point_cloud_all = []

# 模擬水平 360° 旋轉
for yaw in np.linspace(0, 360, 13):  # 每次旋轉 20°
    lidar_orientation = p.getQuaternionFromEuler([0, 0 , np.radians(yaw) ])
    
    # 計算視圖矩陣
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=lidar_position,
        cameraTargetPosition=[lidar_position[0] + np.cos(np.radians(yaw)),
                              lidar_position[1] + np.sin(np.radians(yaw)),
                              lidar_position[2]],
        cameraUpVector=[0, 0, 1]
    )

    # 計算投影矩陣 (30° FOV)
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov, aspect=aspect, nearVal=near, farVal=far
    )

    # 獲取影像
    width, height, rgb, depth, mask = p.getCameraImage(
        width=224, height=224, viewMatrix=view_matrix, projectionMatrix=projection_matrix
    )
    
    depth = (far * near / (far - (far - near) * depth) * 5000).astype(np.uint16)  # transform depth from NDC to actual depth
    intrinsic_matrix = projection_to_intrinsics(projection_matrix, width, height)

    mask[mask <= 2] = -1
    mask[mask > 0] -= 3
    obs = np.concatenate([rgb[..., :3], depth[..., None], mask[..., None]], axis=-1)
    obs = env.process_image(obs[..., :3], obs[..., [3]], obs[..., [4]], tuple(env._resize_img_size), if_raw=True)
    point_state = backproject_camera_target(obs[3].T, intrinsic_matrix, None)  # obs[4].T
    point_state_copy = point_state.copy()
    point_state[0] = -point_state_copy[2]
    point_state[1] = point_state_copy[0]
    point_state[2] = point_state_copy[1]
    obs = (point_state, obs)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obs[0][:3].T)
    transform = np.eye(4)
    rotation_matrix = np.array(p.getMatrixFromQuaternion(lidar_orientation)).reshape(3, 3)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = lidar_position
    print(transform)
    pcd.transform(transform)
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10000, origin=lidar_position)
    # o3d.visualization.draw_geometries([pcd]+ [axis_pcd])
    point_cloud_all.append(pcd)
    
merged_point_cloud = o3d.geometry.PointCloud()
for pcd in point_cloud_all:
    merged_point_cloud += pcd
points = np.asarray(merged_point_cloud.points)

# 設置距離閾值
distance_threshold = 40000.0  # 例如，3米
height_threshold = -3900

# 計算每個點到原點的距離
distances = np.linalg.norm(points, axis=1)

# 過濾點雲，保留距離在閾值內的點
filtered_points = points[distances <= distance_threshold]
# filtered_points = filtered_points[filtered_points[:, 1] > height_threshold]

# 創建一個新的點雲對象
filtered_point_cloud = o3d.geometry.PointCloud()
filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

# 可視化過濾後的點雲
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5000, origin=lidar_position)
o3d.visualization.draw_geometries([filtered_point_cloud]+ [axis_pcd])

while True:


#     # base_pose = p.getBasePositionAndOrientation(env._panda.pandaUid)[0]
#     # env._panda.AMR_control(CONTROL_MODES, right_wheel_value, left_wheel_value)
#     # p.stepSimulation()
    pass




