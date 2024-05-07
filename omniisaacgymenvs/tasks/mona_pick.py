import math
import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import tensor_clamp, torch_rand_float, unscale
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.humanoid import Humanoid
from omniisaacgymenvs.tasks.shared.locomotion import LocomotionTask
from pxr import PhysxSchema
from omni.isaac.core.robots import Robot
import omni

# from omni.importer.urdf import _urdf
from omni.isaac.core.utils.nucleus import get_assets_root_path
import carb
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.prims import RigidPrimView

import wandb

class MonaPickTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._max_episode_length = 2000
        # these must be defined in the task class
        self._num_observations = 18
        self._num_actions = self._num_observations//2
        # call the parent class constructor to initialize key RL variables
        RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        # extract task config from main config dictionary
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        # parse task config parameters
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._robot_positions = torch.tensor([0.0, 0.0, 2.0])
        # reset and actions related variables
        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self.max_torque = self._task_cfg["env"]["maxEffort"]

    def set_up_scene(self, scene) -> None:
        # self.get_mona()
        RLTask.set_up_scene(self, scene)
        # urdf_interface = _urdf.acquire_urdf_interface()
        # Set the settings in the import config
        # import_config = _urdf.ImportConfig()
        # import_config.merge_fixed_joints = False
        # import_config.convex_decomp = False
        # import_config.fix_base = True
        # import_config.make_default_prim = True
        # import_config.self_collision = False
        # import_config.create_physics_scene = True
        # import_config.import_inertia_tensor = False
        # import_config.default_drive_strength = 1047.19751
        # import_config.default_position_drive_damping = 52.35988
        # import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        # import_config.distance_scale = 1
        # import_config.density = 0.0
        # # Finally import the robot
        # result, prim_path = omni.kit.commands.execute( "URDFParseAndImportFile", urdf_path="example-robot-data/robots/panda_description/urdf/panda.urdf",
        #                                                 import_config=import_config, get_articulation_root=True,)

        ## Generic USD loading
        
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            # Use carb to log warnings, errors and infos in your application (shown on terminal)
            carb.log_error("Could not find nucleus server with /Isaac folder")
        asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        # asset_path = '/home/albericlajarte/Desktop/mona_torso_description/urdf/mona_torso/mona_torso.usd'
        # This will create a new XFormPrim and point it to the usd file as a reference
        # Similar to how pointers work in memory
        prim_path=self.default_zero_env_path+"/Mona"
        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        
        # Create robot
        robot = Robot(prim_path=prim_path, name="Mona")
        self._sim_config.apply_articulation_settings(
            "Mona", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("Mona")
        )

        # call the parent class to clone the single environment
        # super().set_up_scene(scene)

        # View list of robots
        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/Mona", name="mona_view", reset_xform_properties=False)
        self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/Mona/panda_link7", name="hands_view", reset_xform_properties=False)
        
        # Add to scene
        scene.add(self._robots)
        scene.add(self._hands)
        return
    
    def post_reset(self):
        # retrieve cart and pole joint indices

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = 0.2 * torch.randn(num_resets, self._num_actions, device=self._device)

        # randomize DOF velocities
        dof_vel = 0. * torch.randn(num_resets, self._num_actions, device=self._device)

        # apply randomized joint positions and velocities to environments
        indices = env_ids.to(dtype=torch.int32)
        self._robots.set_joint_positions(dof_pos, indices=indices)
        self._robots.set_joint_velocities(dof_vel, indices=indices)
        self._robots.set_gains(torch.tensor([0.]), torch.tensor([0.]))

        # reset the reset buffer and progress buffer after applying reset
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        # make sure simulation has not been stopped from the UI
        if not self._env._world.is_playing():
            return

        # extract environment indices that need reset and reset them
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        ## Baseline classic controller
        # position_error = self._hands.get_local_poses()[0] - torch.tensor([0.5, 0, 0.5], device=self._device)

        # jacobians = self._robots.get_jacobians()[:, -1, :3, :]

        # joint_velocities = (jacobians.transpose(1,2)@jacobians +1e-7*torch.eye(9, device=self._device)) @jacobians.transpose(1,2)@position_error.unsqueeze(-1)
        # joint_velocities[:, -2:] = 0
        # actions = -300*joint_velocities.squeeze(-1)
        
        # self._robots.set_joint_velocities(actions, indices=indices)
        ## 

        # RL controller
        actions = self.max_torque*actions.to(self._device)
        self._robots.set_joint_efforts(actions, indices=indices)


    def get_observations(self) -> dict:
        # retrieve joint positions and velocities
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        # populate the observations buffer
        self.obs_buf = torch.concatenate([dof_pos, dof_vel], dim=1)

        # construct the observations dictionary and return
        observations = {self._robots.name: {"obs_buf": self.obs_buf}}
        return observations
    
    def calculate_metrics(self) -> None:
        # use states from the observation buffer to compute reward
        joint_velocities = self.obs_buf[:, -self._num_actions:]

        # define the reward function based on pole angle and robot velocities
        # reward = 1.0 - torch.sum(joint_velocities**2, axis=1)
        
        position_error = (( self._hands.get_local_poses()[0] 
                            - torch.tensor([0.5, 0, 0.5], device=self._device) ).abs()).mean(axis=1)
        
        if wandb.run: wandb.log({"position_error": position_error.mean().item()})
        
        reward_reached = torch.where(position_error<0.05, 10, 0)
        
        reward = 1-position_error + reward_reached

        # assign rewards to the reward buffer
        self.rew_buf[:] = reward

    def is_done(self) -> None:
        joint_velocities = self.obs_buf[:, -self._num_actions:]

        # check for which conditions are met and mark the environments that satisfy the conditions
        resets = torch.where((torch.abs(joint_velocities)>10).any(axis=1), 1, 0)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        
        if resets.any():
            position_error = (( self._hands.get_local_poses()[0] 
                                - torch.tensor([0.5, 0, 0.5], device=self._device) ).abs()).mean(axis=1)
            if wandb.run: wandb.log({"final_position_error": position_error[resets.to(bool)].mean().item()})

        # assign the resets to the reset buffer
        self.reset_buf[:] = resets