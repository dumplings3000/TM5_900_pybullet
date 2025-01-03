#!/usr/bin/env python3

import pybullet as p
import numpy as np
import sys
import os
import json
import open3d as o3d
import matplotlib.pyplot as plt
import time
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.ycb_scene import SimulatedYCBEnv
from utils.grasp_checker import ValidGraspChecker
from utils.utils import *
from utils.planner import GraspPlanner
import utils.tcp_utils as tcp_utils
import rospy
from pybullet_ros.srv import GraspGroup, GraspGroupRequest
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from utils.my_utils import *
from utils.planner import GraspPlanner
from cv_bridge import CvBridge
import pyscreenshot as ImageGrab

    
class PlacingNode:
    def __init__(self):
        self.parent_dir = os.path.join(os.path.dirname(__file__), '..')
        self.bridge = CvBridge()
        self.env = None
        self.planner = GraspPlanner()
        self.execute = rospy.get_param('~execute', False)
        self.visual_simulation = rospy.get_param('~visual_simulation', False)
        self.vis_draw_coordinate = rospy.get_param('~vis_draw_coordinate', True)
        self.target_place_name = None
        self.path_length = 10
        self.renders = True

    def initial(self):
        
        '放置物體相關 data'
        self.placed_name = None
        self.placed_obj = None
        self.target_object = None # 會在find_closet_target中被賦值
        self.poseEstimate = None
        self.target_pose_world = None
        self.result_z_rotation_angle = 0
        self.final_place_grasp_pose = None
        self.final_grasp_pose = None

        '場景設置相關 data'
        # stage2 是最上面的貨價
        self.placing_stage = 1
        self.num_object = 0
        self.single_release =False
        self.if_stack = False
        self.cabinet_pose_world = None
        self.place_pose_world = None

        '多物體擺放策略'
        self.placing_location = [0.2, 0, -0.2]

        'jointpose 相關data'
        self.success_joint_grasp_list = []
        self.success_joint_mid_list = []
        self.success_joint_place_list = []

    def load_environment(self):
        file = os.path.join(self.parent_dir, "object_index", "ycb_large.json")
        with open(file) as f: file_dir = json.load(f)
        file_dir = file_dir["test"]
        file_dir = [f[:-5] for f in file_dir]
        test_file_dir = list(set(file_dir))
        self.env = SimulatedYCBEnv()
        self.env._load_index_objs(test_file_dir)
        state = self.env.reset(save=False, enforce_face_target=True)

    def expert_plan(self, goal_pose, world=False, visual=False, path_length = None):
        if world:
            pos, orn = self.env._get_ef_pose()
            ef_pose_list = [*pos, *orn]
        else:
            ef_pose_list = [0, 0, 0, 0, 0, 0, 1]
        goal_pos = [*goal_pose[:3], *ros_quat(goal_pose[3:])]

        solver = self.planner.plan(ef_pose_list, goal_pos, path_length=path_length)
        if visual:
            self.path_visulization(solver)
        path = solver.getSolutionPath().getStates()
        planer_path = []
        for i in range(len(path)):
            waypoint = path[i]
            rot = waypoint.rotation()
            action = [waypoint.getX(), waypoint.getY(), waypoint.getZ(), rot.w, rot.x, rot.y, rot.z]
            planer_path.append(action)

        return planer_path

    def path_visulization(self, ss):
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
        
    def execute_motion_plan(self, plan, execute=False, gripper_set="close", repeat=100, mode='grasping'):
        """
        Executes a series of movements in a robot environment based on the provided plan.

        Parameters:
        - env: The current robot environment, providing inverse kinematics solving and stepping capabilities.
        - plan: A plan containing target positions, each position is a list with coordinates and orientation.
        - execute: A boolean indicating whether to execute the actions. If False, only resets the robot's joint positions without stepping through the environment.
        """

        '''
        plan中的每一個step都會做碰撞檢查, 如果有碰撞就會停止並回傳flase
        '''


        for i in range(len(plan)):
            # Set target position using world frame based coordinates
            next_pos = plan[i]
            jointPoses = self.env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
            if gripper_set == "close":
                jointPoses[6] = 0.85
            else:
                jointPoses[6] = 0.0
            jointPoses = jointPoses[:7].copy()  # Consider only the first 7 joint positions

            if execute:
                # Execute the action and obtain the observation
                obs = self.env.step(jointPoses, config=True, repeat=repeat)[0]
                # print("JointPoses = ", jointPoses)
            else:
                # Only reset the robot's joint positions
                self.env._panda.reset(joints=jointPoses)
                # 在path length中每一步都檢查是否會發生碰撞
                if(self.env._panda.check_for_collisions() == True):
                    print("Collision detected in the path")
                    # 有碰撞表示此grasp pose不行, 初始化joint list
                    if(mode == 'grasping'):
                        self.success_joint_grasp_list = []
                    elif(mode == 'placing_mid'):
                        self.success_joint_mid_list = []
                    elif(mode == 'placing'):
                        self.success_joint_place_list = []
                    return False
                else:
                    if(mode == 'grasping'):
                        self.success_joint_grasp_list.append(jointPoses[:6])
                    elif(mode == 'placing_mid'):
                        self.success_joint_mid_list.append(jointPoses[:6])
                    elif(mode == 'placing'):
                        self.success_joint_place_list.append(jointPoses[:6])
        return True
    
    def check_scene_placing(self):
        state = self.env.reset(save=False, reset_free=True, num_object=self.num_object, if_stack=self.if_stack, single_release=self.single_release)

        placed_obj = {}
        placed_idx = np.where(np.array(self.env.placed_objects))[0]
        placed_name = np.array(self.env.obj_indexes)[np.where(np.array(self.env.placed_objects))]
        for i in range(self.num_object):
            placed_obj[placed_idx[i]] = placed_name[i]
        print(placed_obj)
    
    # relative pose of z axis rotation
    def get_rotation_matrix_z_4x4(self, input_value):
        # 將輸入值轉換為旋轉角度（度）
        rotation_angle_degrees = 90 * input_value
        # 將旋轉角度轉換為弧度
        rotation_angle_radians = np.radians(rotation_angle_degrees)
        
        # 計算 4x4 旋轉矩陣
        rotation_matrix = np.array([
            [np.cos(rotation_angle_radians), -np.sin(rotation_angle_radians), 0, 0],
            [np.sin(rotation_angle_radians), np.cos(rotation_angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return rotation_matrix
    
    def get_grasp_place_pose(self):
        self.new_pred_grasps_cam_place = np.load(os.path.join(self.parent_dir, 'results', 'new_pred_grasps_cam_place.npy'), allow_pickle=True)
        print(self.new_pred_grasps_cam_place.shape)
        self.scores = np.load(os.path.join(self.parent_dir, 'results', 'scores.npy'), allow_pickle=True)
        print(self.scores.shape)

    def placing_object_on_self(self):
        state = self.env.reset(save=False, reset_free=True, num_object=1, 
        if_stack=self.if_stack, single_release=self.single_release, place_target_matrix=self.final_place_target_matrix)
        if self.vis_draw_coordinate:
            self.env.draw_ef_coordinate(self.place_pose_world, 5)

    def grasp_pose_checker(self, time=0.05, grasp_poses=None, only_vis_grasp=False):
        if not only_vis_grasp:
            self.grasp_array, self.grasp_index = self.grasp_checker.extract_grasp(self.new_pred_grasps_cam_place,
                                                       drawback_distance=0.03,
                                                       visual=self.visual_simulation,
                                                       filter_elbow=True,
                                                       time=time, distance=0.0001)
        else:
            _, _ = self.grasp_checker.extract_grasp(grasp_poses,
                                                drawback_distance=0.03,
                                                visual=True, time=time)

    def refine_grasp_place_pose(self):
        '''
        將best_grasp_pose的y軸和[1, 0, 0]做內積, 若<0則代表要旋轉180度
        僅遍歷 grasp_index 中包含的索引, 且grasp index依照分數高低排好
        '''
        self.grasp_index = np.array(self.grasp_index)
        if len(self.grasp_index) == 0:
            return
        self.grasp_index = self.grasp_index[np.argsort(-self.scores[self.grasp_index])]
        print("grasp_index = ", self.grasp_index)
        for i in self.grasp_index:
            self.grasp_pose = self.new_pred_grasps_cam_place[i]
            
            if np.dot(self.grasp_pose[:3, 1], np.array([0, 0, 1])) < 0:
                # print(f"Grasp pose {i}：Rotate 180 degree")
                self.new_pred_grasps_cam_place[i] = np.dot(self.grasp_pose, rotZ(np.pi))

    def execute_plan_with_check(self, pose, execute=False, mode='grasping', path_length=None):
        if self.vis_draw_coordinate:
            self.env.draw_ef_coordinate(pose, 1)
        plan = self.expert_plan(pack_pose(pose), world=True, visual=False, path_length=path_length)
        # checker true代表對的plan及pose
        plan_checker = self.execute_motion_plan(plan, gripper_set="open", mode=mode)
        checker = check_pose_difference(self.env._get_ef_pose(mat=True), pose, tolerance=0.04)
        return plan_checker, checker
    
    def execute_placing_checker(self, execute=False):
        count = 0
        count_in_1 = 0
        count_in_1_collision = 0
        count_in_1_checker = 0
        count_in_2 = 0
        count_in_2_collision = 0
        count_in_2_checker = 0
        count_in_3 = 0
        count_in_3_collision = 0
        count_in_3_checker = 0
        # start time
        start_time = time.time()
        for i in self.grasp_index:
            self.env._panda.reset()
            
            grasp_pose = self.new_pred_grasps_cam_place[i]
            relative_grasp_transform = np.linalg.inv(self.place_pose_world) @ grasp_pose
            final_grasp_pose = self.target_pose_world @ relative_grasp_transform
            
            # 第一次執行計劃並檢查
            pose_z_bias = adjust_pose_with_bias(final_grasp_pose, -0.1)
            plan_checker, checker = self.execute_plan_with_check(pose_z_bias, execute, mode='grasping', path_length=self.path_length)
            count += 1
            print(f"第 {count} / {len(self.grasp_index)} 個夾取姿態。")
            print("=====================================================")
            if self.visual_simulation:
                time.sleep(3)

            if not plan_checker or not checker:
                count_in_1 += 1
                if not plan_checker and not checker:
                    count_in_1_checker += 1
                    count_in_1_collision += 1
                    print("No.1存在碰撞且檢查失敗。")
                elif not plan_checker:
                    count_in_1_collision += 1
                    print("No.1路徑存在碰撞。")
                elif not checker:
                    count_in_1_checker += 1
                    print("No.1姿態錯誤。")
                continue

            # 第二次執行計劃並檢查
            if self.placing_stage == 1:
                mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.5)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)
            elif self.placing_stage == 2:
                mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.85)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)
            plan_checker, checker = self.execute_plan_with_check(mid_retract_pose, execute, mode='placing_mid', path_length=30)
            print("=====================================================")
            if self.visual_simulation:
                time.sleep(0.2)
            if not plan_checker or not checker:
                count_in_2 += 1
                if not plan_checker and not checker:
                    count_in_2_checker += 1
                    count_in_2_collision += 1
                    print("No.2存在碰撞且檢查失敗。")
                elif not plan_checker:
                    count_in_2_collision += 1
                    print("No.2路徑存在碰撞。")
                elif not checker:
                    count_in_2_checker += 1
                    print("No.2姿態錯誤。")
                continue
                
            # 第三次執行計劃並檢查
            if self.placing_stage == 1:
                print("*****additional condition angle!!!!!!!*****")
                    # 印出角度degree
                print(np.degrees(np.arccos(np.dot(grasp_pose[:3, 2], np.array([1, 0, 0])))))
                # # 檢查grasp_pose的z軸是否和world的x軸小於30度, 若大於10度則continue
                # if np.degrees(np.arccos(np.dot(grasp_pose[:3, 2], np.array([1, 0, 0])))) > 20:
                #     print("*****additional condition*****")
                #     # 印出角度degree
                #     print(np.degrees(np.arccos(np.dot(grasp_pose[:3, 2], np.array([1, 0, 0])))))
                #     continue
            pose_z_bias = adjust_pose_with_bias(grasp_pose, 0.015)
            plan_checker, checker = self.execute_plan_with_check(pose_z_bias, execute, mode='placing', path_length=self.path_length)
            print("=====================================================")

            if not plan_checker or not checker:
                count_in_3 += 1
                time.sleep(1)
                if self.visual_simulation:
                    time.sleep(3)
                if not plan_checker and not checker:
                    count_in_3_checker += 1
                    count_in_3_collision += 1
                    print("No.3存在碰撞且檢查失敗。")
                elif not plan_checker:
                    count_in_3_collision += 1
                    print("No.3路徑存在碰撞。")
                elif not checker:
                    count_in_3_checker += 1
                    print("No.3姿態錯誤。")
                self.success_joint_grasp_list = []
                self.success_joint_mid_list = []
                self.success_joint_place_list = []
                continue


            end_time = time.time()
            # save in the txt file
            with open(os.path.join(self.parent_dir, 'results', 'ros_placing_final.txt'), 'a') as f:
                f.write(f"target_place_name: {self.target_place_name}\n")
                f.write(f"第 {count} / {len(self.grasp_index)} 個有效的夾取姿態。\n")
                f.write(f"第一次失敗次數：{count_in_1}, collision: {count_in_1_collision}, checker: {count_in_1_checker}\n")
                f.write(f"第二次失敗次數：{count_in_2}, collision: {count_in_2_collision}, checker: {count_in_2_checker}\n")
                f.write(f"第三次失敗次數：{count_in_3}, collision: {count_in_3_collision}, checker: {count_in_3_checker}\n")
                f.write(f"!!!!!!!!!!!!!!!Time elapsed: {end_time - start_time} seconds!!!!!!!!!!!!!!!!!!!!!!!!\n")
                f.write("=====================================================\n")
            
            print("grasp_pose = ", grasp_pose)
            # print("success_joint_grasp_list = ", self.success_joint_grasp_list)
            # print("success_joint_mid_list = ", self.success_joint_mid_list)
            # print("success_joint_place_list = ", self.success_joint_place_list)
            self.grasp_pose_checker(time=5, grasp_poses=np.expand_dims(grasp_pose, axis=0), only_vis_grasp=True)
            return grasp_pose
        # if all grasp pose failed return np.eye(4)
        if len(self.grasp_index) == 0:
            with open(os.path.join(self.parent_dir, 'results', 'ros_placing_final.txt'), 'a') as f:
                f.write(f"target_place_name: {self.target_place_name}\n")
                f.write(f"第 {count} / {len(self.grasp_index)} 個有效的夾取姿態。\n")
                f.write(f"第一次失敗次數：{count_in_1}, collision: {count_in_1_collision}, checker: {count_in_1_checker}\n")
                f.write(f"第二次失敗次數：{count_in_2}, collision: {count_in_2_collision}, checker: {count_in_2_checker}\n")
                f.write(f"第三次失敗次數：{count_in_3}, collision: {count_in_3_collision}, checker: {count_in_3_checker}\n")
                f.write(f"!!!!!!!!!!!!!!!Time elapsed:NO GRASP POSE IN CONTACT GRASPNET!!!!!!!!!!!!!!!!!!!!!!!!\n")
                f.write("=====================================================\n")
            return np.eye(4)
        return np.eye(4)


    def run(self):
        while not rospy.is_shutdown():
            self.target_place_name = tcp_utils.start_server_target_name('127.0.0.1', 19111)
            self.load_environment()
            self.initial()
            self.check_scene_placing()   # 和ros execution的不同
            self.grasp_checker = ValidGraspChecker(self.env)
            checker_message = tcp_utils.start_server_checker('127.0.0.1', 12346)
            if checker_message:
                print(f"成功接收到：{checker_message}")
            self.get_grasp_place_pose()
            # tcp client
            self.final_place_target_matrix = tcp_utils.start_server('127.0.0.1', 33333)            
            self.place_pose_world = tcp_utils.start_server('127.0.0.1', 55557)
            # if self.place_pose_world.5則代表是放在第一層

            if self.place_pose_world[2, 3] < 20:
                self.placing_stage = 1
            else:
                self.placing_stage = 2
            self.target_pose_world = tcp_utils.start_server('127.0.0.1', 56471)
            self.placing_object_on_self()
            self.grasp_pose_checker(only_vis_grasp=False)
            self.grasp_pose_checker(grasp_poses=self.new_pred_grasps_cam_place, only_vis_grasp=True)  # vis the non-collide grasp
            self.refine_grasp_place_pose()
            success_grasp_pose = self.execute_placing_checker()
            tcp_utils.send_matrix('127.0.0.1', 44412, success_grasp_pose)
            img = ImageGrab.grab()
            random = np.random.randint(0, 1000)
            img.save(os.path.join(self.parent_dir, f'results/scene_placing_checker_{random}.png'))
            # 結束馬上關閉node
            rospy.signal_shutdown("Finish placing object")

if __name__ == '__main__':
    rospy.init_node('robot_placing_node', anonymous=True)
    robot = PlacingNode()
    robot.run()
    rospy.spin()

            

            
    