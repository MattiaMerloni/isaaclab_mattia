
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, CameraCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_mattia_pick_cfg import LiftEnvCfg

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

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

        # #glass
        # self.scene.object_1 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Glass",
        #     spawn=UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/glass.usd", 
        #         scale=(0.01, 0.01, 0.01),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.4, 1), rot=(0.0, 0.0, 0, 1.0)),
        # )

        # #cuboide
        # self.scene.object_3 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Cuboide",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.332, 0.281, 0.1)),    
        #     spawn=UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cube.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        #         scale=(0.0006, 0.0006, 0.001),            #(0.0006, 0.0006, 0.001)  #vogliamo un parallepipedo        #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
            
        # )

        # # A 0.8 DI x IL CUBO NON E' PIU' RAGGIUNGIBILE DAL BRACCIO DEL FRANKA
        # # A 0.5 DI y IL CUBO NON E' PIU' RAGGIUNGIBILE DAL BRACCIO DEL FRANKA
        # # BISOGNA STARE TRA 0.7 E 0.3 DI x E TRA +- 0.25 DI y PER ESSERE SICURI DI POTER SEMPRE RAGGIUNGERE IL CUBO CON IL FRANKA 

        # #ATTENTO A COME SALVI I FILE USD PERCHE' SE LI TRASLI PRIMA DI SALVARLI OTTERRAI UNA POSIZIONE DIFFERENTE TRA LO SCRIPT DELLA TELEOPERATION E QUELLO DELLA INVERSE KINEMATIC 
        # #CONTROLLA TUTTI GLI ALTRI FILE SALVATI ALL'INTERNO DELLA CARTELLA "usd_files_mattia" PER ESSERE SICURO CHE SIANO SALVATI CORRETTAMENTE

        #banana test 
        self.scene.object_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Banana",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.2, 0.1), rot=(0.0, 0.0, 1.0, 1.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/jonatha/IsaacLab/usd_files_mattia/banana.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
                scale=(0.0006, 0.0006, 0.0006),     #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
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


        #coke can 2 test 
        self.scene.object_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CokeCan",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.442, 0.25, 0.1),rot=(0.0, 0.0, 0.0, 1.0)),
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




        #TENTATIVI PASSATI

        # #tovaglietta
        # self.scene.object_4 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Tovaglietta",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/parallelepipedo.usd", 
        #         scale=(0.35, 0.35, 0.0015),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.35, 0.1), rot=(0.0, 0.0, 0.0, 1.0)),
        # )

        # #Forchetta
        # self.scene.object_5 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Forchetta",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/fork.usd", 
        #         scale=(0.01, 0.01, 0.01),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.647, 0.165, 0.165)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.2), rot=(0.0, 0.707, -0.707, 0.0)),    #manca da ruotare di 90 gradi la forchetta 
        # )

        # #Coltello
        # self.scene.object_6 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Coltello",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/coltello.usd", 
        #         scale=(0.01, 0.01, 0.01),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, -0.35, 0.2), rot=(0.0, 0.0, -0.707, 0.0)),    #manca da ruotare di 90 gradi la forchetta 
        # )

        # #porta_forchetta
        # self.scene.object_7 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Porta_forchetta",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cylinder.usd", 
        #         scale=(0.15, 0.15, 0.01),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.1)),
        # )

        # #Piatto_1
        # self.scene.object_8 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Piatto_1",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/plate.usd", 
        #         scale=(0.008, 0.008, 0.008),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.3), rot=(0.0, 0.0, 0.0, 1.0)),
        # )

        ## QUESTA CAMERA NON SERVE PERCHE' NON LA USIAMO
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
        #         horizontal_aperture=30,  # Aumentare l'apertura orizzontale per un campo visivo più ampio
        #         clipping_range=(0.1, 1.0e5)
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(1.3, 0.0, 0.5),  # Regolare la posizione per una migliore visione del tavolo
        #         rot=(1.0, 0.4, 0.4, 0.65), 
        #         convention="opengl",
        #     ),
        # )


        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",  
            debug_vis=False,    #CAMBIARE IN TRUE PER VEDERE IL FRAME DEL ROBOT
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