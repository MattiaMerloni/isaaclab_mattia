
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, CameraCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_mattia_stack_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # # Set Franka as robot
        # self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True   
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"    


        # #container
        # self.scene.object_1 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Container",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/container.usd", 
        #         scale=(0.01, 0.01, 0.013),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        #         # visual_material=sim_utils.PreviewSurfaceCfg(),   #diffuse_color=(0.8, 0.75, 0.75)
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.22, 0.1), rot=(0.0, 0.0, 0.0, 1.0)),   #(0.6, 0.22, 0.1) 
        # )


        # #blue cuboid
        # self.scene.object_2 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/BlueCuboid",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, 0.22, 0.3), rot=(0.0, 1.0, 0.0, 1.0)),
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cube.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        #         scale=(0.001, 0.0006, 0.0006),     #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        #SE TOLGO IL PREVIEW SURFACE DAGLI OGGETTI CHE SCARICO DA SKETCHFAB ESSI MI SPAWNANO CON LA LORO LIVREA BASE 

        # #red cube
        # self.scene.object_2 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/RedCube",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, -0.22, 0.3)),
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cube.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        #         scale=(0.0004, 0.0004, 0.0004),     #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        # #banana test 
        # self.scene.object_1 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Banana",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.342, -0.18, 0.1), rot=(0.0, 0.0, 1.0, 1.0)),
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/banana.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        #         scale=(0.0006, 0.0006, 0.0006),     #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        # #apple test 
        # self.scene.object_4 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Apple",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, -0.22, 0.1)),
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/apple_2.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        #         scale=(0.0002, 0.0002, 0.0002),     #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )


        # #tovaglietta 1 test
        # self.scene.object_1 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Tovaglietta",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/parallelepipedo.usd", 
        #         scale=(1.15, 1.5, 0.0006),  #(0.62, 1.5, 0.0006) per la mezza piastra
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, -0.01, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),  #(0.75, -0.01, 0.05) per la mezza piastra
        # )

        # #tovaglietta 2 test
        # self.scene.object_1 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Tovaglietta",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/placemat.usd", 
        #         scale=(0.0004, 0.0004, 0.0004), 
        #         # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, -0.01, 0.05), rot=(0.0, 0.0, 1.0, 1.0)),
        # )


        #coke can 2 test 
        self.scene.object_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CokeCan",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.36242, 0.311, 0.1),rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/jonatha/IsaacLab/usd_files_mattia/coke_can.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
                scale=(0.0004, 0.0004, 0.0004),     #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        #firing pan 2 test 
        self.scene.object_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/FiringPan1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, -0.22, 0.4), rot=(0.0, 0.0, 1.0, 1.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/jonatha/IsaacLab/usd_files_mattia/firing_pan_2.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
                scale=(0.00017, 0.00017, 0.00017),     #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )


        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",  
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),    # ATTRAVERSO QUESTO COMANDO POSSO CAMBIARE IL PUNTO DA CUI PRENDO I VALORI DELLA TRAIETTORIA NEL GRIPPER # [0.0, 0.0, 0.1034]
                        # DEVI DETERMINARE IL VALORE SPERIMENTALE CHE FA COMBACIARE I DUE SCRIPT (per ora va bene [0, -0.028, 0.01]) - era il valore quando il cubo era spawnato male 
                    ),
                ),
            ],
        )


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