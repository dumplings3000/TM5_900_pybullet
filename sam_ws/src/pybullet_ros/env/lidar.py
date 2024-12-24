import pybullet as p
import numpy as np
import sys
import os
import json
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

sys.path.append("../")
from utils.utils import *

class LIDAR():
    def __init__(self,
                 fov = 30,
                 aspect = 1.0,
                 near = 0.1,
                 far = 10,
                 raw_data = False,
                 vis = True,
                 distance_threshold = 40,
                 height_threshold = -3.9,
                 lidar_position = [0, 0, 0.8],
                 regularize_pc_point_count=True,
                 uniform_num_pts=1024,  
                 _resize_img_size = (224, 224)          
                 ):
        self.lidar_position = lidar_position
        self.fov = fov
        self.aspect = aspect
        self.near = near        
        self.far = far
        self.raw_data = raw_data
        self.vis = vis
        self.distance_threshold = distance_threshold
        self.height_threshold = height_threshold
        self._regularize_pc_point_count = regularize_pc_point_count
        self.uniform_num_pts = uniform_num_pts
        self._resize_img_size = _resize_img_size

    def _get_observation(self, pose=None, vis=False, raw_data=False):
        """
        Get observation from the environment
        """
        if pose is None:
            pose = [0, 0, 0]
        pose[2] = self.lidar_position[2]
        points = self.get_point_cloud(rotation=20, lidar_position=pose)
        pcd = self.pcd_filter(points, if_plane=False)
        self.show_result(pcd, mode='pcd',vis=vis, axis=True)
        return pcd
    
    def get_rgbd_image(self, view_matrix, projection_matrix):
        """
        Get RGBD image from the environment
        """
        _, _, rgb, depth, mask = p.getCameraImage(
                width=224, height=224, viewMatrix=view_matrix, projectionMatrix=projection_matrix
            )
            
        depth = (self.far * self.near / (self.far - (self.far - self.near) * depth) * 5000).astype(np.uint16)  # transform depth from NDC to actual depth
        return rgb, depth, mask
    
    def get_point_cloud(self, rotation=20, lidar_position=None):
        point_cloud_list = []

        rotation_times = 360//rotation

        for yaw in np.linspace(0, 360, rotation_times):  # 每次旋轉 20°
            lidar_orientation = p.getQuaternionFromEuler([0, 0 , np.radians(yaw) ])

            projection_matrix, view_matrix, intrinsic_matrix = self.get_camera_info(lidar_position, yaw)

            rgb, depth, mask = self.get_rgbd_image(view_matrix=view_matrix, projection_matrix=projection_matrix)

            mask[mask == 0] = -1
            obs = np.concatenate([rgb[..., :3], depth[..., None], mask[..., None]], axis=-1)
            obs = self.process_image(obs[..., :3], obs[..., [3]], obs[..., [4]], tuple(self._resize_img_size), if_raw=True)
            point_state = backproject_camera_target(obs[3].T, intrinsic_matrix, None)  # obs[4].T
            point_state_copy = point_state.copy()
            point_state[0] = -point_state_copy[2]
            point_state[1] = point_state_copy[0]
            point_state[2] = point_state_copy[1]
            obs = (point_state, obs)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obs[0][:3].T)
            self.show_result(pcd, mode='pcd',vis=False, axis=True)

            transform = self.get_transform(lidar_position, lidar_orientation, vis=False)
            pcd.transform(transform)
            point_cloud_list.append(pcd)

        merged_point_cloud = o3d.geometry.PointCloud()
        for pcd in point_cloud_list:
            merged_point_cloud += pcd
        points = np.asarray(merged_point_cloud.points)
        return points

    def get_camera_info(self,lidar_position, rotation, width=224, height=224):
        """
        Get camera intrinsics and extrinsics
        """
        view_matrix = p.computeViewMatrix(
                cameraEyePosition=lidar_position,
                cameraTargetPosition=[lidar_position[0] + np.cos(np.radians(rotation)),
                                    lidar_position[1] + np.sin(np.radians(rotation)),
                                    lidar_position[2]],
                cameraUpVector=[0, 0, 1]
            )

        projection_matrix = p.computeProjectionMatrixFOV(
                self.fov, self.aspect, self.near, self.far
            )
        
        intrinsic_matrix = projection_to_intrinsics(projection_matrix, width, height)
        
        return projection_matrix, view_matrix, intrinsic_matrix

    def process_image(self, color, depth, mask, size=None, if_raw=False):
        """
        Normalize RGBDM
        """
        if not if_raw:
            color = color.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)
            depth = depth.astype(np.float32) / 5000
        if size is not None:
            color = cv2.resize(color, size)
            mask = cv2.resize(mask, size)
            depth = cv2.resize(depth, size)
        obs = np.concatenate([color, depth[..., None], mask[..., None]], axis=-1)
        obs = obs.transpose([2, 1, 0])
        return obs

    def process_pointcloud(self, point_state, vis, use_farthest_point=False):
        """
        Process point cloud input
        Downsmaple or oversample the point cloud
        """
        if self._regularize_pc_point_count and point_state.shape[1] > 0:
            point_state = regularize_pc_point_count(point_state.T, self.uniform_num_pts, use_farthest_point).T

        if vis:
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(point_state.T[:, :3])
            o3d.visualization.draw_geometries([pred_pcd])

        return point_state
    
    def pcd_filter(self, points, if_plane = False):
        """
        Filter the point cloud
        """
        distance_threshold = self.distance_threshold *1000  
        height_threshold = self.height_threshold *1000  
        distances = np.linalg.norm(points, axis=1)

        filtered_points = points[distances <= distance_threshold]
        if if_plane:
            filtered_points = filtered_points[filtered_points[:, 2] > height_threshold]
        
        filtered_points_o3d = o3d.geometry.PointCloud()
        filtered_points_o3d.points = o3d.utility.Vector3dVector(filtered_points)
        self.show_result(filtered_points_o3d, mode='pcd',vis=False)

        return filtered_points_o3d


    
    def show_result(self, pcd=None, rgbd=None ,vis=False, axis=False, mode=None, position=[0, 0, 0.8]):
        """
        Show the result of the environment
        """
        if mode == 'pcd':
            if vis:
                if axis:
                    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10000, origin=position)
                    o3d.visualization.draw_geometries([pcd]+ [axis_pcd])
                else:
                    o3d.visualization.draw_geometries([pcd])
            else:
                return None
        elif mode == 'rgbd':
            if vis:
                cv2.imshow('RGBD', rgbd)
                cv2.waitKey(0)
            else:
                return None
        else:
            return None
        
    def get_transform(self, position, orientation, vis=False):
        transform = np.eye(4)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = position
        if vis:
            print(transform)
        return transform
    
    def process_image(self, color, depth, mask, size=None, if_raw=False):
        """
        Normalize RGBDM
        """
        if not if_raw:
            color = color.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)
            depth = depth.astype(np.float32) / 5000
        if size is not None:
            color = cv2.resize(color, size)
            mask = cv2.resize(mask, size)
            depth = cv2.resize(depth, size)
        obs = np.concatenate([color, depth[..., None], mask[..., None]], axis=-1)
        obs = obs.transpose([2, 1, 0])
        return obs
        
    
if __name__ == '__main__':
    pass





