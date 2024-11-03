# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import random
import os
import time
import sys

import pybullet as p
import numpy as np
import IPython

from env.tm5_gripper_hand_camera import TM5
from transforms3d.quaternions import *
import scipy.io as sio
from utils.utils import *
import json
from itertools import product
import math

class SimulatedYCBEnv():
    def __init__(self,
                 renders=True,
                 blockRandom=0.5,
                 cameraRandom=0,
                 use_hand_finger_point=False,
                 data_type='RGB',
                 filter_objects=[],
                 img_resize=(224, 224),
                 regularize_pc_point_count=True,
                 egl_render=False,
                 width=224,
                 height=224,
                 uniform_num_pts=1024,
                 change_dynamics=False,
                 initial_near=0.2,
                 initial_far=0.5,
                 disable_unnece_collision=False,
                 use_acronym=False):
        self._timeStep = 1. / 2000.
        self._observation = []
        self._renders = renders
        self._resize_img_size = img_resize

        self._p = p
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._use_hand_finger_point = use_hand_finger_point
        self._data_type = data_type
        self._egl_render = egl_render
        self._disable_unnece_collision = disable_unnece_collision

        self._change_dynamics = change_dynamics
        self._initial_near = initial_near
        self._initial_far = initial_far
        self._filter_objects = filter_objects
        self._regularize_pc_point_count = regularize_pc_point_count
        self._uniform_num_pts = uniform_num_pts
        self.observation_dim = (self._window_width, self._window_height, 3)
        self._use_acronym = use_acronym

        self.shelf_num = 8

        self.init_constant()
        self.connect()

    def init_constant(self):
        self._shift = [0.0, 0.0, 0.0]  # to work without axis in DIRECT mode
        self.root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        self._standoff_dist = 0.08

        self.cam_offset = np.eye(4)
        self.cam_offset[:3, 3] = (np.array([0., 0.1186, -0.0191344123493]))
        self.cam_offset[:3, :3] = euler2mat(0, 0, np.pi)

        self.target_idx = 0
        self.objects_loaded = False
        self.connected = False
        self.stack_success = True

    def connect(self):
        """
        Connect pybullet.
        """
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)

            p.resetDebugVisualizerCamera(1.3, 180.0, -41.0, [-0.35, -0.58, -0.88])

        else:
            self.cid = p.connect(p.DIRECT)

        if self._egl_render:
            import pkgutil
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.connected = True

    def disconnect(self):
        """
        Disconnect pybullet.
        """
        p.disconnect()
        self.connected = False

    def reset(self, save=False, init_joints=None, num_object=1, if_stack=True,
              cam_random=0, reset_free=False, enforce_face_target=False):
        """
        Environment reset called at the beginning of an episode.
        """
        self.retracted = False
        if reset_free:
            return self.cache_reset(init_joints, enforce_face_target, num_object=num_object, if_stack=if_stack)

        self.disconnect()
        self.connect()

        self._light_position = np.array([-1.0, 0, 2.5])

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)

        p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        # Set table and plane
        # plane
        plane_file = os.path.join(self.root_dir,  'data/objects/floor/model_normalized.urdf')  # _white
        self.plane_id = p.loadURDF(plane_file, [0 - self._shift[0], 0 - self._shift[1], 0 - self._shift[2]], useFixedBase=True)
        # table
        table_file = os.path.join(self.root_dir,  'data/objects/table/models/model_normalized.urdf')
        self.table_pos = np.array([0.75 - self._shift[0], 0.0 - self._shift[1], 0.05 - self._shift[2]])
        self.table_id = p.loadURDF(table_file, [self.table_pos[0], self.table_pos[1], self.table_pos[2]],
                                   [0.707, 0., 0., 0.707], useFixedBase=True)
        object_bbox = p.getAABB(self.table_id)
        self.length_weight = (object_bbox[1][0] - object_bbox[0][0]) / 2
        self.width_weight = (object_bbox[1][1] - object_bbox[0][1]) / 2
        self.height_weight = (object_bbox[1][2] - object_bbox[0][2]) / 2
        self.table_radius = min(self.length_weight, self.width_weight)  # 或者使用合适的半径值

        # Set the camera  .
        look = [0.9 - self._shift[0], 0.0 - self._shift[1], 0 - self._shift[2]]
        distance = 2.5
        pitch = -56
        yaw = 245
        roll = 0.
        fov = 20.
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
                                   
        # Load shelf
        shelf_file = os.path.join(self.root_dir,  'data/objects/shelf/model.urdf')  
        shelf_files = [shelf_file]
        self.load_shelf(shelf_files)
        
        # Intialize robot and objects
        if init_joints is None:
            self._panda = TM5(stepsize=self._timeStep, base_shift=self._shift)

        else:
            self._panda = TM5(stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift)
            for _ in range(1000):
                p.stepSimulation()

        p.setCollisionFilterPair(self._panda.robot, self.plane_id, -1, -1, 1)

        if not self.objects_loaded:
            self._objectUids = self.cache_objects()

        self.collided = False
        self.collided_before = False
        return None  # observation

    def load_shelf(self, shelf_files):
        self.shelf_ids = [None] * self.shelf_num
        # 0 (0, 0, 0, 1),90 (0, 0, 0.707, 0.707),180 (0, 0, 1, 0),270 (0, 0, 0.707, -0.707)
        # selected_shelf_file = random.choice(shelf_files)
        selected_shelf_file = shelf_files[0]
        shelf_num = int(self.shelf_num/4)
        x, y ,x_shift, y_shift = self.check_which_shelf(shelf_files, selected_shelf_file)
        for i in range(shelf_num):
            self.shelf_ids[i*4] = p.loadURDF(selected_shelf_file, [(i * x + x_shift) - self._shift[0], -(2 + y_shift) - self._shift[1], 0 - self._shift[2]], 
                    [0, 0, 0, 1], useFixedBase=True)
            self.shelf_ids[i*4 +1] = p.loadURDF(selected_shelf_file, [-(i * x + x_shift) - self._shift[0], -(2 + y_shift) - self._shift[1], 0 - self._shift[2]], 
                    [0, 0, 0, 1], useFixedBase=True)
            self.shelf_ids[i*4 +2] = p.loadURDF(selected_shelf_file, [((i+1) * x - x_shift) - self._shift[0], -(2 + y_shift) - self._shift[1], 0 - self._shift[2]], 
                    [0, 0, 0, 1], useFixedBase=True)
            self.shelf_ids[i*4 +3] = p.loadURDF(selected_shelf_file, [-((i+1) * x - x_shift) - self._shift[0], -(2 + y_shift) - self._shift[1], 0 - self._shift[2]], 
                    [0, 0, 0, 1], useFixedBase=True)

    def check_which_shelf(self, shelf_set, shelf_id):
        if shelf_id == shelf_set[0]:
            x = 2
            y = 2.7
            x_shift = 0.2
            y_shift = 1.35
        return x, y, x_shift, y_shift
    



    def cache_reset(self, init_joints, enforce_face_target, num_object=1, if_stack=True):
        """
        Hack to move the loaded objects around to avoid loading multiple times
        """
        for idx in self._objectUids:
            if idx is not None:
                p.removeBody(idx)

        self._panda.reset(init_joints)
        self._randomly_place_objects_pack(self._get_random_object(num_object), scale=1, if_stack=if_stack)

        self.retracted = False
        self.collided = False
        self.collided_before = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]

        observation = self.enforce_face_target() if enforce_face_target else self._get_observation()
        return observation
    
    def cache_objects(self):
        """
        get all object urdf files , if placed , heights and scales list
        """
        self.object_heights = []
        self.object_scales = []

        obj_path = os.path.join(self.root_dir, 'data/objects/')
        objects = self.obj_indexes
        print("self.obj_indexes:", self.obj_indexes)
        obj_path = [obj_path + objects[i] for i in self._all_obj]
        objects_paths = [p_.strip() + '/' for p_ in obj_path]

        for i, name in enumerate(objects_paths):
            mesh_scale = name.split('_')[-1][:-1]
            self.object_scales.append(float(mesh_scale))
            name = name.replace(f"_{mesh_scale}/", "/")
            point_z = np.loadtxt(os.path.join(self.root_dir, name, 'model_normalized.extent.txt'))
            half_height = float(point_z.max()) / 2 if len(point_z) > 0 else 0.01
            self.object_heights.append(half_height)
            objects_paths[i] = objects_paths[i].replace(f"_{mesh_scale}/", "/")
        self.obj_path = objects_paths
        print("self.obj_path:", self.obj_path)

        for i, name in enumerate(self.obj_indexes):
            mesh_scale = name.split('_')[-1]
            self.obj_indexes[i] = self.obj_indexes[i].replace(f"_{mesh_scale}", "")

        self.objects_loaded = True
        self.placed_objects = [False] * len(self.obj_path)

        return [None] * len(self.obj_path)

    def _randomly_place_objects_pack(self, urdfList, scale, if_stack=True, poses=None):
        '''
        Input:
            urdfList: File path of mesh urdf, support single and multiple object list
            scale: mesh scale
            if_stack (bool): scene setup with uniform position or stack with collision

        Func:
            For object in urdfList do:
                (1) find Uid of pybullet body by indexing object
                (1) reset position of pybullet body in urdfList
                (2) set self.placed_objects[idx] = True
        '''
        self.stack_success = True

        if len(urdfList) == 1:
            return self._randomly_place_objects(urdfList=urdfList, scale=scale, poses=poses)
        else:
            print("the length of urdfList:", len(urdfList))
            
    def _randomly_place_objects(self, urdfList, scale, poses=None):
        """
        Randomize positions of each object urdf.
        """
        # for plane
        # xpos = self.table_pos[0] + length_weight * 2 * (self._blockRandom * random.random() - 0.5) - self._shift[0]
        # ypos = self.table_pos[1] + width_weight * 2* (self._blockRandom * random.random() - 0.5) - self._shift[0]
        # for circle
        obj_path = '/'.join(urdfList[0].split('/')[:-1]) + '/'
        self.target_idx = self.obj_path.index(os.path.join(self.root_dir, obj_path)) 
        self.placed_objects[self.target_idx] = True
        self.target_name = urdfList[0].split('/')[-2]

        angle = random.uniform(0, 2 * math.pi)
        radius = 0.5 * random.uniform(0, self.table_radius)     

        xpos = self.table_pos[0] + 0.8 * radius * math.cos(angle) - self._shift[0]
        ypos = self.table_pos[1] + 0.8 * radius * math.sin(angle) - self._shift[0]
        height_weight = self.object_heights[self.target_idx]
        z_init = 0.4 + 1 * height_weight
        orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(-np.pi, np.pi)])
        mesh_scale = self.object_scales[self.target_idx]

        uid = self._add_mesh(os.path.join(self.root_dir, obj_path, 'model_normalized.urdf'),
                            [xpos, ypos,  z_init - self._shift[2]], 
                            [orn[0], orn[1], orn[2], orn[3]], 
                            scale=float(mesh_scale))
        self._objectUids[self.target_idx] = uid

        if self._use_acronym:
            object_bbox = p.getAABB(uid)
            height_weight = (object_bbox[1][2] - object_bbox[0][2]) / 2
            z_init = 0.4 + 1 * height_weight
            p.resetBasePositionAndOrientation(uid, [xpos, ypos,  z_init - self._shift[2]], [orn[0], orn[1], orn[2], orn[3]])

        p.resetBaseVelocity(uid, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for _ in range(2000):
            p.stepSimulation()

        pos, new_orn = p.getBasePositionAndOrientation(uid)  # to target
        ang = np.arccos(2 * np.power(np.dot(tf_quat(orn), tf_quat(new_orn)), 2) - 1) * 180.0 / np.pi

        if (self.target_name in self._filter_objects or ang > 50) and not self._use_acronym:  # self.target_name.startswith('0') and
            self.target_name = 'noexists'
        print('>>>> target name: {}'.format(self.target_name))
        return []
    
    def _add_mesh(self, obj_file, trans, quat, scale=1):
        """
        Add a mesh with URDF file.
        """
        bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        return bid

    def _get_random_object(self, num_objects):
        """
        Randomly choose an object urdf from the selected objects
        """
        target_obj = np.random.choice(np.arange(0, len(self.obj_indexes)), size=num_objects, replace=False)
        selected_objects = target_obj           
        selected_objects_filenames = [os.path.join('data/objects/', self.obj_indexes[int(selected_objects[i])],
                                      'model_normalized.urdf') for i in range(num_objects)]
        print('selected objects:', selected_objects_filenames)
        return selected_objects_filenames
    
    def get_env_info(self, scene_file=None):
        """
        Return object names and poses of the current scene
        """
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        obj_dir = []

        for idx, uid in enumerate(self._objectUids):
            if self.placed_objects[idx]:
                pos, orn = p.getBasePositionAndOrientation(uid)  # center offset of base
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(inv_relative_pose(obj_pose, base_pose))
                obj_dir.append('/'.join(self.obj_path[idx].split('/')[:-1]).strip())  # .encode("utf-8")

        return obj_dir, poses

    def enforce_face_target(self):
        """
        Move the gripper to face the target
        """
        target_forward = self._get_target_relative_pose('ef')[:3, 3]
        target_forward = target_forward / np.linalg.norm(target_forward)
        r = a2e(target_forward)
        action = np.hstack([np.zeros(3), r])
        return self.step(action, repeat=200, vis=False)[0]
    
    def step(self, action ,delta=False, obs=True, repeat=150, config=False, vis=False):
        """
        Environment step.
        """
        action = self.process_action(action, delta, config)
        if not config:
            action[6] = 0.85
        self._panda.setTargetPositions(action)
        for _ in range(int(repeat)):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

        observation = self._get_observation(vis=vis)
        joint = p.getJointStates(self._panda.pandaUid, self._panda.joints)

        return observation, joint, self._get_ef_pose(mat=True)

    def _get_observation(self, pose=None, vis=False, raw_data=False):
        """
        Get observation
        """

        object_pose = self._get_target_relative_pose('ef')  # self._get_relative_ef_pose()
        ef_pose = self._get_ef_pose('mat')

        joint_pos, joint_vel = self._panda.getJointStates()
        near, far = self.near, self.far
        view_matrix, proj_matrix = self._view_matrix, self._proj_matrix
        camera_info = tuple(view_matrix) + tuple(proj_matrix)
        hand_cam_view_matrix, hand_proj_matrix, lightDistance, lightColor, lightDirection, near, far = self._get_hand_camera_view(pose)
        camera_info += tuple(hand_cam_view_matrix.flatten()) + tuple(hand_proj_matrix)
        _, _, rgba, depth, mask = p.getCameraImage(width=self._window_width,
                                                   height=self._window_height,
                                                   viewMatrix=tuple(hand_cam_view_matrix.flatten()),
                                                   projectionMatrix=hand_proj_matrix,
                                                   physicsClientId=self.cid,
                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)

        depth = (far * near / (far - (far - near) * depth) * 5000).astype(np.uint16)  # transform depth from NDC to actual depth
        intrinsic_matrix = projection_to_intrinsics(hand_proj_matrix, self._window_width, self._window_height)
        if raw_data:
            mask[mask <= 2] = -1
            mask[mask > 0] -= 3
            obs = np.concatenate([rgba[..., :3], depth[..., None], mask[..., None]], axis=-1)
            obs = self.process_image(obs[..., :3], obs[..., [3]], obs[..., [4]], tuple(self._resize_img_size), if_raw=True)
            point_state = backproject_camera_target(obs[3].T, intrinsic_matrix, None)  # obs[4].T
            point_state[1] *= -1
            obs = (point_state, obs)
        else:
            mask[mask >= 0] += 1  # transform mask to have target id 0
            target_idx = self.target_idx + 4

            mask[mask == target_idx] = 0
            mask[mask == -1] = 50
            mask[mask != 0] = 1

            obs = np.concatenate([rgba[..., :3], depth[..., None], mask[..., None]], axis=-1)
            obs = self.process_image(obs[..., :3], obs[..., [3]], obs[..., [4]], tuple(self._resize_img_size))
            point_state = backproject_camera_target(obs[3].T, intrinsic_matrix, obs[4].T)  # obs[4].T

            point_state[1] *= -1
            point_state = self.process_pointcloud(point_state, vis)
            obs = (point_state, obs)
        pose_info = (object_pose, ef_pose)
        return [obs, joint_pos, camera_info, pose_info]


    def reset_joint(self, init_joints):
        if init_joints is not None:
            self._panda.reset(np.array(init_joints).flatten())

    def process_action(self, action, delta=False, config=False):
        """
        Process different action types
        para action: relative action to be processed
        """
        # transform to local coordinate
        if config:
            if delta:
                cur_joint = np.array(self._panda.getJointStates()[0])
                action = cur_joint + action
        else:
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]

            pose = np.eye(4)
            pose[:3, :3] = quat2mat(tf_quat(orn))
            pose[:3, 3] = pos

            pose_delta = np.eye(4)
            pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
            pose_delta[:3, 3] = action[:3]

            new_pose = pose.dot(pose_delta)
            orn = ros_quat(mat2quat(new_pose[:3, :3]))
            pos = new_pose[:3, 3]

            jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                  self._panda.pandaEndEffectorIndex, pos, orn,
                                  maxNumIterations=500,
                                  residualThreshold=1e-8))
            jointPoses[12] = 0.0
            action = jointPoses[6:13]
        return action

    def _get_hand_camera_view(self, cam_pose=None):
        """
        Get hand camera view
        """
        if cam_pose is None:
            pos, orn = p.getLinkState(self._panda.pandaUid, 26)[4:6]
            cam_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        cam_pose_mat = unpack_pose(cam_pose)

        fov = 90
        aspect = float(self._window_width) / (self._window_height)
        hand_near = 0.035
        hand_far = 2
        hand_proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, hand_near, hand_far)
        hand_cam_view_matrix = se3_inverse(cam_pose_mat.dot(rotX(np.pi/2).dot(rotY(-np.pi/2)))).T  # z backward

        lightDistance = 2.0
        lightDirection = self.table_pos - self._light_position
        lightColor = np.array([1., 1., 1.])
        light_center = np.array([-1.0, 0, 2.5])
        return hand_cam_view_matrix, hand_proj_matrix, lightDistance, lightColor, lightDirection, hand_near, hand_far

    def target_lifted(self):
        """
        Check if target has been lifted
        """
        end_height = self._get_target_relative_pose()[2, 3]
        if end_height - self.init_target_height > 0.08:
            return True
        return False

    def _load_index_objs(self, file_dir):

        self._target_objs = range(len(file_dir))
        self._all_obj = range(len(file_dir))
        self.obj_indexes = file_dir


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
        """
        if self._regularize_pc_point_count and point_state.shape[1] > 0:
            point_state = regularize_pc_point_count(point_state.T, self._uniform_num_pts, use_farthest_point).T

        if vis:
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(point_state.T[:, :3])
            o3d.visualization.draw_geometries([pred_pcd])

        return point_state

    def transform_pose_from_camera(self, pose, mat=False):
        """
        Input: pose with length 7 [pos_x, pos_y, pos_z, orn_w, orn_x, orn_y, orn_z]
        Transform from 'pose relative to camera' to 'pose relative to ef'
        """
        mat_camera = unpack_pose(list(pose[:3]) + list(pose[3:]))
        if mat:
            return self.cam_offset.dot(mat_camera)
        else:
            return pack_pose(self.cam_offset.dot(mat_camera))

    def _get_ef_pose(self, mat=False): 
        """
        end effector pose in world frame
        """
        if not mat:
            return p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]
        else:
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]
            return unpack_pose(list(pos) + [orn[3], orn[0], orn[1], orn[2]]) #list to matrix

    def _get_target_relative_pose(self, option='base'):
        """
        Get target obejct poses with respect to the different frame.
        """
        if option == 'world':
            pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
            # pos, orn = p.getLinkState(self._panda.pandaUid, 7)[4:6]
        elif option == 'base':
            pos, orn = p.getLinkState(self._panda.pandaUid, 7)[4:6]
        elif option == 'ef':
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]
        elif option == 'tcp':
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]
            rot = quat2mat(tf_quat(orn))
            tcp_offset = rot.dot(np.array([0, 0, 0.13]))
            pos = np.array(pos) + tcp_offset

        pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        uid = self._objectUids[self.target_idx]
        pos, orn = p.getBasePositionAndOrientation(uid)  # to target
        obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        # poses = []
        # for idx, uid in enumerate(self._objectUids):
        #     if self.placed_objects[idx]:
        #         pos, orn = p.getBasePositionAndOrientation(uid)  # to target
        #         obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        #         poses.append(inv_relative_pose(obj_pose, pose))
        return inv_relative_pose(obj_pose, pose)
    
    def retract(self):
        """
        Move the arm to lift the object.
        """

        cur_joint = np.array(self._panda.getJointStates()[0])
        cur_joint[-1] = 0  # close finger
        observations = [self.step(cur_joint, repeat=300, config=True, vis=False)[0]]
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]

        for i in range(10):
            pos = (pos[0], pos[1], pos[2] + 0.03)
            jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                                               self._panda.pandaEndEffectorIndex, pos,
                                                               maxNumIterations=500,
                                                               residualThreshold=1e-8))
            jointPoses[12] = 0.9
            jointPoses = jointPoses[6:13].copy()
            obs = self.step(jointPoses, config=True)[0]

        self.retracted = True

if __name__ == '__main__':
    pass
