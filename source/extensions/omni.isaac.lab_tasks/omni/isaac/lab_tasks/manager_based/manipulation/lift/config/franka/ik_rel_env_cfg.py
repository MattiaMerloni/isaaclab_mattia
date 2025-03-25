# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg

# from . import joint_pos_env_cfg
from . import joint_pos_env_mattia_pick_cfg    #PICK THE CUBOID
# from . import joint_pos_env_mattia_place_cfg  #PLACE THE COKE CAN IN THE PAN


from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(joint_pos_env_mattia_pick_cfg.FrankaCubeLiftEnvCfg):    #Change here in accrodance with the task
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",   
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # # camera
        # self.scene.camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/front_cam",
        #     update_period=0.1,
        #     height=480,
        #     width=640,
        #     data_types=["rgb", "distance_to_image_plane"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24, 
        #         focus_distance=400.0, 
        #         horizontal_aperture=30,  # Increase the aperture to get a wider field of view
        #         clipping_range=(0.1, 1.0e5)
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(1.3, 0.0, 0.7),  # Define the position of the camera
        #         rot=(1.0, 0.4, 0.4, 0.65), 
        #         convention="opengl",
        #     ),
        # )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
