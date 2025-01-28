# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer import OffsetCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from . import mdp

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##


@configclass
class CabinetSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,  #False
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0, 0.4),    # my position: (1.0, 0, 0.4)    # validation dataset cabinet position: (1.1, 0.0, 0.4)
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,   
                velocity_limit=100.0,  
                stiffness=10.0,   
                damping=1.0,     
                friction=0.0,    # HO ANNULLATO L'ATTRITO IN MODO DA POTER APRIRE FACILMENTE I CASSETTI CON IL GRIPPER DEL ROBOT
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # Frame definitions for the cabinet.   #qui definisco rot e pos del nuovo frame di riferimento del cabinet
    cabinet_frame = FrameTransformerCfg(            # FrameTransformerCfg è una classe che permette di trasformare i frame di riferimento di un oggetto in un altro frame di riferimento
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=False,                             #metti a False per non vedere i frame di riferimento
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",   #drawer_handle_top drawer_handle_bottom
                name="drawer_handle_top",       #drawer_handle_top   drawer_handle_bottom
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),   
                    rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                ),
            ),
            # FrameTransformerCfg.FrameCfg(
            #     prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_bottom",   #drawer_handle_top drawer_handle_bottom
            #     name="drawer_handle_bottom",       #drawer_handle_top   drawer_handle_bottom
            #     offset=OffsetCfg(
            #         pos=(0.305, 0.0, 0.01),   
            #         rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
            #     ),
            # ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/front_cam",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 1.1),  
            rot=(1.0, 0.0, 0.0, -0.715), 
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
        prim_path="{ENV_REGEX_NS}/Robot/side_cam",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -1.5, 0.4),  
            rot=(1.0, 0.0, 0.0, 0.715),
            convention="world",
        ),
    )

    #cubo
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cubo",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, -0.25, 1.0)),  #cubo training dataset: pos=(0.8, -0.25, 1.0)  #cubo validation dataset: pos=(0.95, -0.25, 1.0)
        spawn=UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cube.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
            scale=(0.0006, 0.0006, 0.0006),            #(0.0006, 0.0006, 0.0006)  #(1.1, 1.1, 1.1)         #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
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

# le trajectory pos e rot sono state definite in frame_transformer_data.py che si trova in source/extensions/omni.isaac.lab/omni/isaac/lab/sensors/frame_transformer

def trajectory_pos_data(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # traj_pos = asset.data.target_pos_w
    traj_pos= asset.data.target_pos_source     # SISTEMA DI RIFERIMENTO GIUSTO: perchè se aumento gli environment non cambia nulla in termini di valori (il riferimento rimane sempre il centro del robot) 
    return traj_pos

def trajectory_rot_data(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # traj_rot= asset.data.target_quat_w
    traj_rot= asset.data.target_quat_source  # SISTEMA DI RIFERIMENTO GIUSTO: perchè se aumento gli environment non cambia nulla in termini di valori (il riferimento rimane sempre il centro del robot)
    return traj_rot

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    null_command = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

    #     joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    #     joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    #     cabinet_joint_pos = ObsTerm(
    #         func=mdp.joint_pos_rel,
    #         params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    #     )
    #     cabinet_joint_vel = ObsTerm(
    #         func=mdp.joint_vel_rel,
    #         params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    #     )
    #     # cabinet_joint_pos = ObsTerm(
    #     #     func=mdp.joint_pos_rel,
    #     #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
    #     # )
    #     # cabinet_joint_vel = ObsTerm(
    #     #     func=mdp.joint_vel_rel,
    #     #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
    #     # )
    #     rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)

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
    # """Configuration for events."""

    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 1.25),
    #         "dynamic_friction_range": (0.8, 1.25),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )

    # cabinet_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
    #         "static_friction_range": (1.0, 1.25),
    #         "dynamic_friction_range": (1.25, 1.5),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )


# @configclass
# class RewardsCfg:
    # """Reward terms for the MDP."""

    # # 1. Approach the handle
    # approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    # align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # # 2. Grasp the handle
    # approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    # align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    # grasp_handle = RewTerm(
    #     func=mdp.grasp_handle,
    #     weight=0.5,
    #     params={
    #         "threshold": 0.03,
    #         "open_joint_pos": MISSING,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
    #     },
    # )

    # # 3. Open the drawer
    # open_drawer_bonus = RewTerm(
    #     func=mdp.open_drawer_bonus,
    #     weight=7.5,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    # )
    # multi_stage_open_drawer = RewTerm(
    #     func=mdp.multi_stage_open_drawer,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    # )

    # # 4. Penalize actions for cosmetic reasons
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class CabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    # Scene settings
    scene: CabinetSceneCfg = CabinetSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    # rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
