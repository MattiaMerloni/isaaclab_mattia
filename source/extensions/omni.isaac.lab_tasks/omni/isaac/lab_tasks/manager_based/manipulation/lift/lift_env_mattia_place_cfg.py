# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import CameraCfg



from . import mdp

#The . indicates that you're importing something from the current package or module directory. In this case, you're 
# importing the module mdp from the same package or directory as the script you're working on.

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # target object: will be populated by agent env cfg
    # object_1: RigidObjectCfg = MISSING
    object_2: RigidObjectCfg = MISSING
    object_3: RigidObjectCfg = MISSING
    # object_4: RigidObjectCfg = MISSING
   

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0), rot=(0.707, 0, 0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(1.65, 1.80, 1.0)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),  
        spawn=sim_utils.GroundPlaneCfg(),  # GREEN GROUDPLANE color=(0.0, 1.0, 0.0)
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # # distant light (light that reproduces a window's light) - intensity increased at 30000
    # distant_light = AssetBaseCfg(
    #     prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=30000.0, color=(0.75, 0.75, 0.75), angle=0.53), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, -0.35, -0.15, 0.0)),
    # )

    # camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/front_cam",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24,   
            focus_distance=400.0, 
            horizontal_aperture=20.955,  
            clipping_range=(0.1, 1.0e5)
        ),
        # offset=CameraCfg.OffsetCfg(
        #     pos=(1.27, 0.0, 0.8),  
        #     rot=(1.0, 0.2, 0.2, 0.7),  
        #     convention="opengl",
        # ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.8, 0.1, 0.8),  
            rot=(1.0, 0.4, 0.45, 0.7), 
            convention="opengl",
        ),
    )

    camera_hand = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/hand_cam",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0), rot=(0.0, 1.0, 1.0, 0), convention="opengl"
        ),
    )

    camera_side = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Table/side_cam",
        update_period=0.1,
        height=224,    #480
        width=224,    #640
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.3, 0.3, 0.5),  
            rot=(0.5, 0.3, -0.6, -1.0),
            convention="opengl",
        ),
    )



def camera_rgb_front(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    rgb = asset.data.output["rgb"]  # (num_env, 480, 640, 4), rgb
    return rgb

def camera_rgb_hand(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera_hand")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    rgb = asset.data.output["rgb"]  # (num_env, 480, 640, 4), rgb
    return rgb

def camera_rgb_side(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera_side")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    rgb = asset.data.output["rgb"]  # (num_env, 480, 640, 4), rgb
    return rgb

# the trajectory pos and rot have been defined in frame_transformer_data.py which is located in source/extensions/omni.isaac.lab/omni/isaac/lab/sensors/frame_transformer

def trajectory_pos_data(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # traj_pos = asset.data.target_pos_w
    traj_pos= asset.data.target_pos_source     # Rgiht reference frame: because if I increase the environment nothing changes in terms of values (the reference remains the center of the robot)
    return traj_pos

def trajectory_rot_data(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # traj_rot= asset.data.target_quat_w
    traj_rot= asset.data.target_quat_source  # Right reference frame: because if I increase the environment nothing changes in terms of values (the reference remains the center of the robot)
    return traj_rot





##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,                                 # changing this to True will show the reference frame of the object
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            # pos_x=(0.4, 1.0), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # @configclass
    # class PolicyCfg(ObsGroup):
    #     """Observations for policy group."""

    #     joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    #     joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    #     object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
    #     target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
    #     actions = ObsTerm(func=mdp.last_action)

    #     def __post_init__(self):
    #         self.enable_corruption = True
    #         self.concatenate_terms = True

    # # observation groups
    # policy: PolicyCfg = PolicyCfg()

    @configclass
    class TRAJCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        traj_pos = ObsTerm(func=trajectory_pos_data)
        traj_rot= ObsTerm(func=trajectory_rot_data)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBDCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        rgb_front = ObsTerm(func=camera_rgb_front)

        # observation terms (order preserved)
        rgb_hand = ObsTerm(func=camera_rgb_hand)

        # observation terms (order preserved)
        rgb_side = ObsTerm(func=camera_rgb_side)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False
              
    
    # observation groups for the camera sensor
    trajb: TRAJCfg = TRAJCfg()

    # observation groups for the camera sensor
    rgbd: RGBDCfg = RGBDCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 10.0)},   # range di valori attorno all'initial state impostato nll'altro script
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object_1"),
    #     },
    # )

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object_2"),
    #     },
    # )

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object_3"),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping_1 = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_1")}
    # )

    # object_dropping_2 = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_2")}
    # )

    # object_dropping_3 = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_3")}
    # )

    # object_dropping_4 = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_4")}
    # )





@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
