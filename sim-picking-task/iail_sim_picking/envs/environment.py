"""Environment class."""

import os
import tempfile
import time
import cv2
import imageio

import gym
import numpy as np
from iail_sim_picking.tasks import cameras
from iail_sim_picking.utils import pybullet_utils
from iail_sim_picking.utils import utils
from numpy import linalg as LA

import pybullet as p

PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005

UR5_URDF_PATH = 'ur5/ur5.urdf'
UR5_WORKSPACE_URDF_PATH = 'ur5/workspace.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'


class Environment(gym.Env):
    """OpenAI Gym-style environment class."""

    def __init__(self,
                 assets_root,
                 task=None,
                 disp=False,
                 shared_memory=False,
                 hz=240,
                 record_cfg=None):
        """Creates OpenAI Gym-style environment with PyBullet.

        Args:
          assets_root: root directory of assets.
          task: the task to use. If None, the user must call set_task for the
            environment to work properly.
          disp: show environment with PyBullet's built-in display viewer.
          shared_memory: run with shared memory.
          hz: PyBullet physics simulation step speed. Set to 480 for deformables.

        Raises:
          RuntimeError: if pybullet cannot load fileIOPlugin.
        """
        self.pix_size = 0.003125
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        self.agent_cams = cameras.RealSenseD415.CONFIG
        self.record_cfg = record_cfg
        self.save_video = False
        self.step_counter = 0
        self._reset_in_progress = False
        self.live_display = False
        self.live_display_window = 'IAIL Sim Picking'
        self.live_display_interval = 20

        self.assets_root = assets_root

        color_tuple = [
            gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
            for config in self.agent_cams
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
            for config in self.agent_cams
        ]
        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
        })
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5, 0.], dtype=np.float32),
            high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
            shape=(3,),
            dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        })

        disp_option = p.DIRECT
        if disp:
            disp_option = p.GUI
            if shared_memory:
                disp_option = p.SHARED_MEMORY
        client = p.connect(disp_option)
        file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
        if file_io < 0:
            raise RuntimeError('pybullet: cannot load FileIO!')
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(assets_root)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1. / hz)

        if disp:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target)

        if task:
            self.set_task(task)

    def __del__(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(p.getBaseVelocity(i)[0])
             for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)

    def add_object(self, urdf, pose, category='rigid'):
        """List of (fixed, rigid, or deformable) objects in env."""
        fixed_base = 1 if category == 'fixed' else 0
        obj_id = pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, urdf),
            pose[0],
            pose[1],
            useFixedBase=fixed_base)
        if not obj_id is None:
            self.obj_ids[category].append(obj_id)
        return obj_id

    # Standard Gym functions.

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def reset(self):
        """Performs common reset functionality for all supported tasks."""
        if not self.task:
            raise ValueError('environment task must be set. Call set_task or pass '
                             'the task arg in the environment constructor.')
        self._reset_in_progress = True
        try:
            self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
            p.setGravity(0, 0, -9.8)

            # Disable rendering temporarily to speed up scene construction.
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

            pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
                                     [0, 0, -0.001])
            pybullet_utils.load_urdf(
                p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])

            # Load the UR5 arm with the task-defined end effector.
            self.ur5 = pybullet_utils.load_urdf(
                p, os.path.join(self.assets_root, UR5_URDF_PATH))
            self.ee = self.task.ee(self.assets_root, self.ur5, 9, self.obj_ids)
            self.ee_tip = 10  # Link ID of suction cup.

            # Collect revolute joints and reset them to the home configuration.
            n_joints = p.getNumJoints(self.ur5)
            joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
            self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
            for i in range(len(self.joints)):
                p.resetJointState(self.ur5, self.joints[i], self.homej[i])

            self.homepose = ((0.3, 0.0, 0.5), (0, 0, 0, 1))
            timeout = self.movep(self.homepose)

            self.ee.release()

            self.task.reset(self)

            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

            obs, _, _, info = self.step()
            if self.live_display:
                self.show_live_frame()
            return obs, info
        finally:
            self._reset_in_progress = False

    def compute_trajectory(self, action):
        ee_state = p.getLinkState(self.ur5, self.ee_tip)[0]
        # Interpolate from the current pose to a pre-pick waypoint.
        prepick_ee_state = action['pose0'][0] + np.array([0,0,0.35])
        dist_prepick = LA.norm(ee_state-prepick_ee_state)
        step_num = int(dist_prepick/0.15) + 1
        trajectory_prepick = np.linspace(ee_state, prepick_ee_state, step_num)

        # Continue from the pre-pick waypoint to the target pose.
        pick_ee_state = action['pose0'][0]
        dist_pick = LA.norm(prepick_ee_state-pick_ee_state)
        step_num = int(dist_pick/0.15) + 1
        trajectory_pick = np.linspace(prepick_ee_state, pick_ee_state, step_num)

        trajectory = np.concatenate((trajectory_prepick, trajectory_pick[1:]), axis=0)

        print('trajectory',trajectory)
        return trajectory

    def step(self, action=None):
        """Execute action with specified primitive.

        Args:
          action: action to execute.

        Returns:
          (obs, reward, done, info) tuple containing MDP step data.
        """
        if action is not None:
            action = self._normalize_action(action)
            timeout, pre_pick_ee, pre_pick_joint = self.task.primitive(
                self.movej, self.movep, self.ee, action['pose0'], action.get('pose1'))

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = {'color': (), 'depth': ()}
                for config in self.agent_cams:
                    color, depth, _ = self.render_camera(config)
                    obs['color'] += (color,)
                    obs['depth'] += (depth,)

                print('------ time out ------')
                obs['ee_state'] = pre_pick_ee      
                obs['joint_state'] = pre_pick_joint                                              
                return obs, 0.0, True, self.info

        # Step the simulator until all rigid bodies come to rest.
        while not self.is_static:
            self.step_simulation()

        reward, info = self.task.reward() if action is not None else (0, {})
        done = self.task.done()

        info.update(self.info)

        obs = self._get_obs()
        if action is not None:
            obs['ee_state'] = pre_pick_ee
            obs['joint_state'] = pre_pick_joint
        return obs, reward, done, info

    def step_simulation(self):
        p.stepSimulation()
        self.step_counter += 1

        if not self._reset_in_progress and self.save_video and self.step_counter % 5 == 0:
            self.add_video_frame()
        if (not self._reset_in_progress and self.live_display and
                self.step_counter % self.live_display_interval == 0):
            self.show_live_frame()

    def enable_live_display(self, window_name='IAIL Sim Picking', interval=1):
        """Enable OpenCV visualization using the recording camera path."""
        self.live_display = True
        self.live_display_window = window_name
        self.live_display_interval = max(1, int(interval))

    def disable_live_display(self):
        """Disable OpenCV visualization and close the display window."""
        self.live_display = False
        try:
            cv2.destroyWindow(self.live_display_window)
        except cv2.error:
            pass

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise NotImplementedError('Only rgb_array implemented')
        color, _, _ = self.render_camera(self.agent_cams[0])
        return color

    def render_camera(self, config, image_size=None, shadow=1):
        """Render RGB-D image with specified camera configuration."""
        if not image_size:
            image_size = config['image_size']

        # Configure the OpenGL camera from the task camera parameters.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # The field of view is vertical, and the aspect ratio must be a float.
        aspect_ratio = image_size[1] / image_size[0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        _, _, color, depth, segm = p.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=shadow,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config['noise']:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, image_size))
            color = np.uint8(np.clip(color, 0, 255))

        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += self._random.normal(0, 0.003, depth_image_size)

        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                dim = p.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)

        info['lang_goal'] = self.get_lang_goal()
        info.update(self.get_task_metadata())
        return info

    def set_task(self, task):
        task.set_assets_root(self.assets_root)
        self.task = task

    def get_task_metadata(self):
        return {
            'all_obj': list(getattr(self.task, 'obj_names', [])),
            'target_obj': list(getattr(self.task, 'target_obj_names', [])),
            'task_name': getattr(self.task, 'task_name', ''),
            'picked_obj': self.get_picked_object_name(),
            'picked_obj_id': self.get_picked_object_id(),
        }

    def check_picked_object_name(self, height_threshold=0.1):
        object_names = list(getattr(self.task, 'obj_names', []))
        for index, obj_id in enumerate(self.obj_ids.get('rigid', [])):
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            if pos[2] > height_threshold:
                if index < len(object_names):
                    return object_names[index]
                return str(obj_id)
        return None

    def get_picked_object_id(self):
        if not hasattr(self, 'ee') or getattr(self.ee, 'contact_constraint', None) is None:
            return None
        try:
            return p.getConstraintInfo(self.ee.contact_constraint)[2]
        except Exception:
            return None

    def get_picked_object_name(self):
        lifted_object = self.check_picked_object_name()
        if lifted_object is not None:
            return lifted_object
        picked_obj_id = self.get_picked_object_id()
        if picked_obj_id is None:
            return 'none'
        object_id_to_name = getattr(self.task, 'object_id_to_name', {})
        return object_id_to_name.get(picked_obj_id, str(picked_obj_id))

    def _normalize_action(self, action):
        if 'pose0' not in action:
            raise KeyError("Action must contain 'pose0'.")
        return {'pose0': action['pose0'], 'pose1': action.get('pose1')}

    def get_lang_goal(self):
        if self.task:
            return self.task.get_lang_goal()
        else:
            raise Exception("No task for was set")

    # Robot movement functions.

    def movej(self, targj, speed=0.01, timeout=5, return_ee_info=False):
        """Move UR5 to target joint configuration."""
        if self.save_video:
            timeout = timeout * 50

        targj[-1] = 0

        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                if return_ee_info:
                    ee_xyz = p.getLinkState(self.ur5, self.ee_tip)[0]
                    ee_joint = [p.getJointState(self.ur5, i)[0] for i in self.joints]
                    return False, ee_xyz, ee_joint
                else:         
                    return False

            # Move with a constant joint-space step.
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self.step_simulation()

        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')

        if return_ee_info:
            ee_xyz = p.getLinkState(self.ur5, self.ee_tip)[0]
            ee_joint = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            return True, ee_xyz, ee_joint
        else:
            return True

    def start_rec(self, video_filename):
        assert self.record_cfg

        if not os.path.exists(self.record_cfg['save_video_path']):
            os.makedirs(self.record_cfg['save_video_path'])

        if hasattr(self, 'video_writer'):
            self.video_writer.close()

        self.video_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                                            f"{video_filename}.mp4"),
                                               fps=self.record_cfg['fps'],
                                               format='FFMPEG',
                                               codec='h264',)
        p.setRealTimeSimulation(False)
        self.save_video = True

    def end_rec(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

        p.setRealTimeSimulation(True)
        self.save_video = False

    def _get_record_camera_config(self):
        """Return the camera config used for task videos and live display."""
        if self.task.task_name == 'seen':
            return self.agent_cams[0]
        if self.task.task_name == 'unseen':
            return self.agent_cams[1]
        if self.task.task_name == 'extra':
            return self.agent_cams[2]
        if self.task.task_name == 'top':
            return self.agent_cams[3]
        return self.agent_cams[0]

    def _render_record_frame(self):
        """Render the task-aware frame shared by video recording and display."""
        config = self._get_record_camera_config()
        image_size = (self.record_cfg['video_height'], self.record_cfg['video_width'])
        color, _, _ = self.render_camera(config, image_size, shadow=0)
        color = np.array(color)

        if self.record_cfg['add_text']:
            lang_goal = self.get_lang_goal()

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.65
            font_thickness = 1

            lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
            lang_textX = (image_size[1] - lang_textsize[0]) // 2

            color = cv2.putText(color, lang_goal, org=(lang_textX, 600),
                                fontScale=font_scale,
                                fontFace=font,
                                color=(0, 0, 0),
                                thickness=font_thickness, lineType=cv2.LINE_AA)
            color = np.array(color)

        return color

    def add_video_frame(self):
        color = self._render_record_frame()
        self.video_writer.append_data(color)

    def show_live_frame(self):
        """Display the task-aware recording frame in an OpenCV window."""
        color = self._render_record_frame()
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.live_display_window, color)
        cv2.waitKey(1)

    def movep(self, pose, speed=0.01, return_ee_info=False):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed, return_ee_info=return_ee_info)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def _get_obs(self):
        obs = {'color': (), 'depth': (),}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)

        ee_state = p.getLinkState(self.ur5, self.ee_tip)[0]
        obs['ee_state'] = ee_state
        currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
        obs['joint_state'] = currj

        return obs


class EnvironmentNoRotationsWithHeightmap(Environment):
    """Environment that disables any rotations and always passes [0, 0, 0, 1]."""

    def __init__(self,
                 assets_root,
                 task=None,
                 disp=False,
                 shared_memory=False,
                 hz=240):
        super(EnvironmentNoRotationsWithHeightmap,
              self).__init__(assets_root, task, disp, shared_memory, hz)

        heightmap_tuple = [
            gym.spaces.Box(0.0, 20.0, (320, 160, 3), dtype=np.float32),
            gym.spaces.Box(0.0, 20.0, (320, 160), dtype=np.float32),
        ]
        self.observation_space = gym.spaces.Dict({
            'heightmap': gym.spaces.Tuple(heightmap_tuple),
        })
        self.action_space = gym.spaces.Dict({
            'pose0': gym.spaces.Tuple((self.position_bounds,))
        })

    def step(self, action=None):
        """Execute action with specified primitive.

        Args:
          action: action to execute.

        Returns:
          (obs, reward, done, info) tuple containing MDP step data.
        """
        if action is not None:
            action = {
                'pose0': (action['pose0'][0], [0., 0., 0., 1.]),
            }
        return super(EnvironmentNoRotationsWithHeightmap, self).step(action)

    def _get_obs(self):
        obs = {}

        color_depth_obs = {'color': (), 'depth': ()}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            color_depth_obs['color'] += (color,)
            color_depth_obs['depth'] += (depth,)
        cmap, hmap = utils.get_fused_heightmap(color_depth_obs, self.agent_cams,
                                               self.task.bounds, pix_size=0.003125)
        obs['heightmap'] = (cmap, hmap)
        return obs
