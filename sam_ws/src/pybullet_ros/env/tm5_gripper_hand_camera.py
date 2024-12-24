# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import pybullet as p
import numpy as np
import IPython
import os
import math


class TM5:
    def __init__(self, stepsize=1e-3, realtime=0, init_joints=None, base_shift=[0, 0, 0], other_object=None):
        '''
        1.初始化一些成员变量，比如时间 t、步长 stepsize、实时模式 realtime 和关节初始化 init_joints。
        2.定义了控制模式为“位置控制”。
        3.设置了比例和微分控制增益以及最大扭矩。
        4.连接到PyBullet仿真环境并加载机器人的URDF模型（统一机器人描述格式），设定基础位置及加载相关参数。
        '''
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime
        self.control_mode = "position"

        self.position_control_gain_p = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.position_control_gain_d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        f_max = 250
        self.max_torque = [f_max, f_max, f_max, f_max, f_max, f_max, 100, 100, 100]

        # connect pybullet
        p.setRealTimeSimulation(self.realtime)

        # load models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        p.setAdditionalSearchPath(current_dir + "/models")
        print(current_dir + "/models")
        self.robot = p.loadURDF("mobile_manipulator.urdf",
                                 useFixedBase=False,
                                 flags=p.URDF_USE_SELF_COLLISION)
        self._base_position = [0.0 - base_shift[0], 0.0 - base_shift[1], 0.0 - base_shift[2]]
        self.pandaUid = self.robot

        # robot parameters
        self.dof = p.getNumJoints(self.robot)

        print("DOF of the robot: ", self.dof)
        for joint_index in range(self.dof):
            joint_info = p.getJointInfo(self.pandaUid, joint_index)
            
            joint_name = joint_info[1].decode('utf-8')  # 關節名稱
            joint_type = joint_info[2]                   # 關節類型
            joint_state = p.getJointState(self.pandaUid, joint_index)  # 當前關節狀態
            
            position = joint_state[0]                    # 當前位置
            velocity = joint_state[1]                    # 當前速度
            torque = joint_state[3]                      # 當前力矩
            
            print(f"Joint {joint_index}:")
            print(f"  Name: {joint_name}")
            print(f"  Type: {joint_type}")
            print(f"  Position: {position}")
            print(f"  Velocity: {velocity}")
            print(f"  Torque: {torque}")
            print("----------------------------")

        # # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        mimic_parent_id = []
        mimic_child_multiplier = {}
        for i in range(p.getNumJoints(self.robot)):
            inf = p.getJointInfo(self.robot, i)
            name = inf[1].decode('utf-8')
            if name == mimic_parent_name:
                mimic_parent_id.append(inf[0])
            if name in mimic_children_names:
                mimic_child_multiplier[inf[0]] = mimic_children_names[name]
            if inf[2] != p.JOINT_FIXED:
                p.setJointMotorControl2(self.robot, inf[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        self.mimic_parent_id = mimic_parent_id[0]

        for joint_id, multiplier in mimic_child_multiplier.items():
            c = p.createConstraint(self.robot, self.mimic_parent_id,
                                   self.robot, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

        p.setCollisionFilterPair(self.robot, self.robot, 11, 13, 0)
        p.setCollisionFilterPair(self.robot, self.robot, 16, 18, 0)

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.pandaEndEffectorIndex = 14
        self.target_torque = []
        self._joint_min_limit = np.array([-4.712385, -3.14159, -3.14159, -3.14159, -3.14159, -4.712385, 0, 0, 0])
        self._joint_max_limit = np.array([4.712385, 3.14159, 3.14159,  3.14159,  3.14159,  4.712385, 0, 0, 0.8])
        self.gripper_range = [0, 0.085]
        self.wheel_distance = 0.46
        self.wheel_radius = 0.05
        self.max_wheel_velocity = 20

        for j in range(self.dof):
            if j >=7:
                p.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            else:
                p.changeDynamics(self.robot, j, lateralFriction=0.3)
            joint_info = p.getJointInfo(self.robot, j)
            if j in range(8, 17):
                self.joints.append(j)
                self.q_min.append(joint_info[8])
                self.q_max.append(joint_info[9])
                self.target_pos.append((self.q_min[j-8] + self.q_max[j-8])/2.0)
                self.target_torque.append(0.)
        self.reset(init_joints)

        self.cabinet = other_object

    def reset(self, joints=None):
        '''
        重置机器人的状态和关节位置，可以传入指定的关节位置
        '''
        self.t = 0.0
        self.control_mode = "position"
        p.resetBasePositionAndOrientation(self.pandaUid, self._base_position,
                                          [0.000000, 0.000000, 0.000000, 1.000000])

        if joints is None:
            self.target_pos = [
                    0.0, -1, 2, 0, 1.571, 0.0, 0.0, 0.0, 0.0]

            self.target_pos = self.standardize(self.target_pos)
            for j in range(8, 17):
                self.target_torque[j-8] = 0.
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j-8])

        else:
            joints = self.standardize(joints)
            for j in range(8, 17):
                self.target_pos[j-8] = joints[j-8]
                self.target_torque[j-8] = 0.
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j-8])
        self.resetController()
        self.setTargetPositions(self.target_pos)

    def step(self):
        '''
        用于在仿真中执行一次步进，以更新机器人的状态。
        '''
        self.t += self.stepsize
        p.stepSimulation()

    def resetController(self):
        """
        重置机器人控制器，设定所有关节的控制模式为速度控制并将力设置为零。
        """
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for i in range(8, 17)])

    def standardize(self, target_pos):
        '''
        确保目标关节位置在定义的最小和最大限制之内，并在需要时扩展目标位置数组。
        '''
        if len(target_pos) == 7:
            if type(target_pos) == list:
                target_pos[6:6] = [0, 0]
            else:
                target_pos = np.insert(target_pos, 6, [0, 0])

        target_pos = np.array(target_pos)

        target_pos = np.minimum(np.maximum(target_pos, self._joint_min_limit), self._joint_max_limit)
        return target_pos

    def setTargetPositions(self, target_pos):
        '''
        设定机器人的目标位置，并将目标位置转化为PyBullet的控制指令。
        '''
        self.target_pos = self.standardize(target_pos)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)

    def getJointStates(self):
        '''
        获取并返回当前关节的位置和速度状态。
        '''
        joint_states = p.getJointStates(self.robot, self.joints)

        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]

        del joint_pos[6:8]
        del joint_vel[6:8]

        return joint_pos, joint_vel

    def solveInverseKinematics(self, pos, ori):
        '''
        计算给定位置和方向的逆运动学解，返回相应的关节角度。
        '''
        jointPoses = list(p.calculateInverseKinematics(self.robot,
                                  14, pos, ori,
                                  maxNumIterations=500,
                                  residualThreshold=1e-8))
        jointPoses[12] = 0.0
        action = jointPoses[6:13]
        return action
        
    def move_gripper(self, open_length):
        '''
        控制机械手的开合，接收一个开合长度并更新抓手的目标角度。

        '''
        open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.robot, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=100)
        
    def wheel_control(self, mode, value_l,value_r):
        '''
        接收两个控制信号并执行相应的动作。

        '''
        if mode == "position":
            p.setJointMotorControl2(self.pandaUid, jointIndex=1, controlMode=p.POSITION_CONTROL, targetPosition=value_r, force=10)
            p.setJointMotorControl2(self.pandaUid, jointIndex=2, controlMode=p.POSITION_CONTROL, targetPosition=value_l, force=10)
            
        elif mode == "velocity":
            # print(value_l,value_r)
            p.setJointMotorControl2(self.pandaUid, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=value_r, force=20)
            p.setJointMotorControl2(self.pandaUid, jointIndex=2, controlMode=p.VELOCITY_CONTROL, targetVelocity=value_l, force=20)

    def compute_wheel_velocities(self, v, omega):
        """
        計算給定線速度和角速度時的車輪速度
        :param v: 車輛的線速度 (m/s)
        :param omega: 車輛的角速度 (rad/s)
        :return: 左右車輪速度 (rad/s)
        """
        v_left = (2 * v + omega * self.wheel_distance) / (2 * self.wheel_radius)
        v_right = (2 * v - omega * self.wheel_distance) / (2 * self.wheel_radius)
        print(v_left, v_right)
        return v_left, v_right

    def go_to_point(self, car_pos, car_angle, target_pos):
        """
        控制車輛前往目標點
        :param car_pos: 當前車輛位置 (x, y)
        :param car_angle: 當前車輛角度 (朝向)
        :param target_pos: 目標位置 (x, y)
        """
        dx = target_pos[0] - car_pos[0]
        dy = target_pos[1] - car_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # 計算目標方向的角度
        target_angle = np.arctan2(dy, dx)
        angle_diff = target_angle - car_angle
        
        # 限制角度差在 [-pi, pi] 之內
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        # 設置車輛的速度
        if np.abs(angle_diff) > 0.1:
            v = 0 
            omega = 30 * angle_diff  
            if omega > 0:
                omega = max(omega, 10)
            else:
                omega = min(omega, -10)
        else:
            if distance < 0.5:
                v = 1
                omega = 0
            else:
                v = 5.0 * min(10, distance)  
                v = max(v, 3)
                omega = 0 

        v_left, v_right = self.compute_wheel_velocities(v, omega)
        # v_left = np.clip(v_left, -max_wheel_velocity, max_wheel_velocity)
        # v_right = np.clip(v_right, -max_wheel_velocity, max_wheel_velocity)
        
        return v_left, v_right

    def go_to_pose(self, car_angle, target_angle):

        angle_diff = target_angle - car_angle
        
        # 限制角度差在 [-pi, pi] 之內
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        v = 0 
        omega = 20 * angle_diff  
        if np.abs(angle_diff) < 0.2:  # 當接近目標角度時減小轉向速度
            omega = angle_diff * 5  # 減少轉向速度

        v_left, v_right = self.compute_wheel_velocities(v, omega)
        return v_left, v_right

    def Mobilebase_control(self, path_points):
        control_mode = "velocity"
        print(path_points)
        for target_pose in path_points:
            target_reached = False
            target_angle = target_pose[2]
            target_pos = [target_pose[0], target_pose[1]]
            while not target_reached:
                print("do :", target_pose)
                car_state = p.getBasePositionAndOrientation(self.pandaUid)
                car_pos = [car_state[0][0], car_state[0][1]]
                car_angle = p.getEulerFromQuaternion(car_state[1])[2]
                distance_to_target = np.linalg.norm(np.array(car_pos) - np.array(target_pos))
                angle_diff = target_angle - car_angle
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                angle_diff = abs(angle_diff)

                left_wheel_value,right_wheel_value  = self.go_to_point(car_pos, car_angle, target_pos)

                # 當接近目標點時停止
                if target_pose == path_points[-1]:
                    if distance_to_target < 0.07:
                        print("distance_to_target:", distance_to_target)
                        left_wheel_value,right_wheel_value = self.go_to_pose(car_angle, target_angle)
                        if angle_diff < 0.01:
                            left_wheel_value,right_wheel_value = 0, 0
                            self.wheel_control(control_mode, right_wheel_value, left_wheel_value)
                            p.stepSimulation()
                            target_reached = True
                            break
                else:
                    if distance_to_target < 0.2:
                        print("distance_to_target:", distance_to_target)
                        target_reached = True

                # 進行一步模擬
                self.wheel_control(control_mode, right_wheel_value, left_wheel_value)
                p.stepSimulation()

    def check_for_collisions(self):
        # 定義需要檢查碰撞的連結名稱
        check_links = [
            "shoulder_1_link",
            "arm_1_link",
            "arm_2_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "flange_link"
        ]

        # 建立一個空字典來存儲這些連結的ID
        link_ids = {name: self.find_link_id(name) for name in check_links}

        # 定義應該排除的連結對，即這些連結對之間的碰撞檢查將被忽略
        exclude_pairs = {
            ("shoulder_1_link", "arm_1_link"),
            ("arm_1_link", "arm_2_link"),
            ("arm_2_link", "wrist_1_link"),
            ("wrist_1_link", "wrist_2_link"),
            ("wrist_2_link", "wrist_3_link"),
            ("wrist_3_link", "flange_link"),
        }
        
        threshold = -0.03  # 定義檢查碰撞的距離閾值
        cabinet_threshold = 0.0  # 定義檢查與 cabinet 碰撞的距離閾值

        # 定義額外的 bounding boxes，以 (xmin, xmax, ymin, ymax, zmin, zmax) 的形式
        extra_bboxes = [
            (0.68, 1.0, -0.5, 0.5, 0.57, 0.598),
            (0.68, 1.0, -0.5, 0.5, 0.22, 0.248)
        ]

        # 检查 link 和 link 之间的碰撞
        for name_i in check_links:
            for name_j in check_links:
                if (name_i, name_j) in exclude_pairs or (name_j, name_i) in exclude_pairs:
                    continue  # 如果當前的連結對在排除列表中，則跳過不檢查

                link_id_i = link_ids[name_i]
                link_id_j = link_ids[name_j]
                if link_id_i == link_id_j:
                    continue  # 忽略自身的檢查

                closest_points = p.getClosestPoints(bodyA=self.robot, bodyB=self.robot, distance=threshold, linkIndexA=link_id_i, linkIndexB=link_id_j)
                if closest_points:
                    # 如果發現任何連結對的最近點小於閾值，認為發生了碰撞
                    print(f"連結 {name_i} 和連結 {name_j} 的最近距離小於 {threshold} 米")
                    for point in closest_points:
                        print(f"最近點距離：{point[8]} 米")
                    return True

        # # 與 cabinet 進行碰撞檢查
        # for name in check_links:
        #     link_id = link_ids[name]
            
        #     # 定义需要检查的 cabinet 的 link index 列表
        #     cabinet_link_indices = [2, 3]
            
        #     for cabinet_link_id in cabinet_link_indices:
        #         closest_points = p.getClosestPoints(bodyA=self.robot, bodyB=self.cabinet, distance=cabinet_threshold, linkIndexA=link_id, linkIndexB=cabinet_link_id)
        #         if closest_points:
        #             # 如果發現與 cabinet 的最近點小於閾值，認為發生了碰撞
        #             print(f"連結 {name} 和 cabinet 的 link {cabinet_link_id} 的最近距離小於 {cabinet_threshold} 米")
        #             # 印出碰撞的最近點距離
        #             for point in closest_points:
        #                 print(f"最近點距離：{point[8]} 米")
        #             return True
        return False
    
    def find_link_id(self, link_name):
        # print each link name
        for i in range(p.getNumJoints(self.robot)):
            # print(p.getJointInfo(self.robot, i)[12].decode('utf-8'))
            if p.getJointInfo(self.robot, i)[12].decode('utf-8') == link_name:
                # print("找到了", link_name, "的索引：", i)
                return i
        print("未找到", link_name, "的索引")
        return None
    
    def is_point_in_bbox(self, point, bbox):
        """
        檢查點是否在包圍盒內
        :param point: 點的座標 (x, y, z)
        :param bbox: 包圍盒的範圍 (xmin, xmax, ymin, ymax, zmin, zmax)
        :return: 如果點在包圍盒內則返回 True，否則返回 False
        """
        x, y, z = point
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax

if __name__ == "__main__":
    robot = TM5(realtime=1)
    while True:
        pass
