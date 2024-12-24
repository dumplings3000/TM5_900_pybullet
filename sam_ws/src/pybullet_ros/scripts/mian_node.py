#!/usr/bin/env python3
import sys
import os
from copy import deepcopy
import numpy as np

import cv2
import open3d as o3d

import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from
from sensor_msgs.point_cloud2 import create_cloud_xyz32

from actor import Actor

class ros_main_node(object):
    def __init__(self):
        self.parent_directory = os.path.join(os.path.dirname(__file__), '..')
        self.env = None
        self.planner = Grasppalnner
        rospy.init_node = rospy.init_node('ros_main_node')
        rospy.log.info("start to initialize ros_main_node")
        self.bridge = CvBridge()
        # sensor information
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.camera_pcd_sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self.camera_pcd_callback)
        self.lidar_pcd_sub = rospy.Subscriber('/lidar/points', PointCloud2, self.lidar_pcd_callback)
        # transition information
        self.start_sub = rospy.Subscriber('/start', Float64, self.start_callback)
        self.contact_client = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)
        self.alghorithm_client = rospy.ServiceProxy('alghorithm', path_planning)
        
        # observation variables
        self.img = None
        self.pcd_camera = None
        self.pcd_lidar = None
        self.lidar_pcd_subdepth_img = None
        # state variables
        self.joint_states = [None] * 6
        self.ee_pose = [None] * 6
        self.ee_open = True
        self._base_pose = [None] * 6
        
        # alogorithm variables
        # for object classification 
        self.name = None
        self.place_pose = [None] * 6
        self.goal_shelf_num = -1
        self.region = [0, 0]
        self.goal_base_pos = [None] * 6
        # for contact grasp net
        self.grasp_pose_set = []
        self.goal_pose = [None] * 6

        # other operation variables
        self.flag = False

    def image_callback(self, data):
        return
    
    def camera_pcd_callback(self, data):
        return
    
    def lidar_pcd_callback(self, data):
        return

    def start_callback(self, data):
        return

if __name__ == '__main__':
    main_node = ros_main_node()
    rospy.spin()