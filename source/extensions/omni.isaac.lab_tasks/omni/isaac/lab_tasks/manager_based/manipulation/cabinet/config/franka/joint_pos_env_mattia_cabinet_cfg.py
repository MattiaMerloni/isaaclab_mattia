# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.manipulation.cabinet import mdp

from omni.isaac.lab_tasks.manager_based.manipulation.cabinet.cabinet_env_mattia_cfg import (  # QUI RICHIAMO LO SCRIPT CHE HO CREATO IO DEL CABINET
    FRAME_MARKER_SMALL_CFG,
    CabinetEnvCfg,
)


# # robot  #COME DI SPAWNA NEL DIRECT WORKFLOW  
# robot = ArticulationCfg(
#         prim_path="/World/envs/env_.*/Robot",
#         spawn=sim_utils.UsdFileCfg(
#             usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
#             activate_contact_sensors=False,
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                 disable_gravity=True,
#                 max_depenetration_velocity=5.0,
#             ),
#             articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#                 enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
#             ),
#         ),
#         init_state=ArticulationCfg.InitialStateCfg(
#             joint_pos={
#                 "panda_joint1": -1.0763,
#                 "panda_joint2": -0.4690,
#                 "panda_joint3": 1.0088,
#                 "panda_joint4": -2.2224,
#                 "panda_joint5": 2.8825,
#                 "panda_joint6": 2.6945,
#                 "panda_joint7": 1.4151, #+-2.897
#                 "panda_finger_joint.*": 0.04,
#             },
#             pos=(0.0, 0.0, 0),
#             rot=(0.0, 0.0, 0.0, 0.0),
#         ),
#         actuators={
#             "panda_shoulder": ImplicitActuatorCfg(
#                 joint_names_expr=["panda_joint[1-4]"],
#                 effort_limit=870.0,
#                 velocity_limit=2.175,
#                 stiffness=800.0,
#                 damping=80.0,
#             ),
#             "panda_forearm": ImplicitActuatorCfg(
#                 joint_names_expr=["panda_joint[5-7]"],
#                 effort_limit=120.0,
#                 velocity_limit=2.61,
#                 stiffness=800.0,
#                 damping=80.0,
#             ),
#             "panda_hand": ImplicitActuatorCfg(
#                 joint_names_expr=["panda_finger_joint.*"],
#                 effort_limit=200.0,
#                 velocity_limit=0.2,
#                 stiffness=2e3,
#                 damping=1e2,
#             ),
#         },
#     )




##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg



@configclass
class FrankaCabinetEnvCfg(CabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        #IL ROBOT E' STATO SPAWNATO NELLO SCRIPT ik_rel_env_cfg.py OPPORTUNAMENTE RUOTATO

        # # Set franka as robot
        # self.scene.robot = FRANKA_PANDA_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot",
        #     init_state=ArticulationCfg.InitialStateCfg(
        #         joint_pos={
        #             "panda_joint1": -1.0763,
        #             "panda_joint2": -0.4690,
        #             "panda_joint3": 1.0088,
        #             "panda_joint4": -2.2224,
        #             "panda_joint5": 2.8825,
        #             "panda_joint6": 2.6945,
        #             "panda_joint7": 2.897, #+-2.897
        #             "panda_finger_joint.*": 0.04,
        #         },
        # ),
        # )

        # Set Actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,  #True
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,                                #metti a False per non vedere i sistemi di riferimento di gripper e dita
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),  #(0.0, 0.0, 0.1034)
                    ),
                ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                #     name="tool_leftfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                #     name="tool_rightfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
            ],
        )

        # # override rewards                                               #COMMMENTATI PERCHE' HO ESCLUSO I REWARDS
        # self.rewards.approach_gripper_handle.params["offset"] = 0.04
        # self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        # self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger_.*"]


@configclass
class FrankaCabinetEnvCfg_PLAY(FrankaCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
