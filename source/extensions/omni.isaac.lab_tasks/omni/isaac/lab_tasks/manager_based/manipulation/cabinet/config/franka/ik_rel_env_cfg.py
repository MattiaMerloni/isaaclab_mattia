# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass

# from . import joint_pos_env_cfg
from . import joint_pos_env_mattia_cabinet_cfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg

# SE VOGLIO CONTROLLARE LE POSIZIONI DEI JOINTS DEL ROBOT DEVO ANDARE A MODIFICARE QUESTO SCRIPT PERCHE' QUESTO E' L'ULTIMO CHE VIENE RICHIAMATO/SOVRASCRITTO


@configclass
class FrankaCabinetEnvCfg(joint_pos_env_mattia_cabinet_cfg.FrankaCabinetEnvCfg):   #ATTENTO CHE DEVI CAMBIARE QUI!
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": -1.0763,
                    "panda_joint2": -0.4690,
                    "panda_joint3": 1.0088,
                    "panda_joint4": -2.2224,
                    "panda_joint5": 2.8825,
                    "panda_joint6": 2.6945,
                    "panda_joint7": 2.897, #+-2.897
                    "panda_finger_joint.*": 0.04,
                },
            ),
        )

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),     
        )

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
