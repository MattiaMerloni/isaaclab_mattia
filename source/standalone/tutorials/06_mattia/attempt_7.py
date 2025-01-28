
    # Usage
'''    ./isaaclab.sh -p source/standalone/tutorials/06_mattia/attempt_7.py  --num_envs 1  '''


#in questo script il robot si inizilizza ad ogni cambiamento del ee_goal ma io voglio che esso faccia un movimento continuo e non si resetta ad ogni cambio di goal

#per ridurre il costo computazionale devo ottimizzare lo script 


"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import os
import json

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import CameraCfg, FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip



# questa funzione mi servirà quando gli darò in input le azioni predette da RT1
# def euler_to_quaternion(roll, pitch, yaw):   #DA DECOMMENTARE
#     """
#     Convert Euler angles (roll, pitch, yaw) to a quaternion using PyTorch.
#     """

#     # Convert input angles to tensors
#     roll = roll.clone().detach().float()      #devo usare clone().detach() perchè roll,pitch e yaw sono tensori di PyTorch e non posso fare operazioni direttamente su di essi
#     pitch = pitch.clone().detach().float()
#     yaw = yaw.clone().detach().float()

#     # roll= x, pitch = y, yaw = z

#     cy = torch.cos(yaw * 0.5)
#     sy = torch.sin(yaw * 0.5)
#     cp = torch.cos(pitch * 0.5)
#     sp = torch.sin(pitch * 0.5)
#     cr = torch.cos(roll * 0.5)
#     sr = torch.sin(roll * 0.5)

#     w = cr * cp * cy + sr * sp * sy
#     x = sr * cp * cy - cr * sp * sy
#     y = cr * sp * cy + sr * cp * sy
#     z = cr * cp * sy - sr * sp * cy

#     return torch.tensor([w, x, y, z], dtype=torch.float32)


def euler_to_quaternion(roll, pitch, yaw):    #DA DECOMMENTARE
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion using PyTorch.
    """

    # Convert input angles to tensors
    roll = torch.tensor(roll)     #devo usare clone().detach() perchè roll,pitch e yaw sono tensori di PyTorch e non posso fare operazioni direttamente su di essi
    pitch = torch.tensor(pitch)
    yaw = torch.tensor(yaw)

    # roll= x, pitch = y, yaw = z

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.tensor([w, x, y, z], dtype=torch.float32)


def quaternion_to_euler(quaternion):
    """
    Convert a quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) using PyTorch.
    """
    w,x,y,z = quaternion

    # Calcola il roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Calcola il pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Per evitare valori fuori dal dominio di asin, bisogna limitare sinp tra -1 e 1
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # Calcola il yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)


def convert_goals_to_quaternion(ee_goals_eul):
    """
    Convert a list of goals from Euler angles to quaternion representation.
    Excludes the gripper state, resulting in 7 values per goal.
    """
    ee_goals_quat = []
    for goal in ee_goals_eul:
        pos = goal[:3]  # x, y, z position
        rot = euler_to_quaternion(goal[3], goal[4], goal[5])  # convert Euler angles to quaternion
        # coord= torch.cat((pos,rot))    #DA DECOMMENTARE

        # Exclude the gripper state and only append position and quaternion
        ee_goals_quat.append(pos + rot.tolist())  #DA DECOMMENTARE
        # ee_goals_quat.append(coord.tolist())   #DA DECOMMENTARE
        
    return ee_goals_quat



@configclass
class TableTopSceneCfg(InteractiveSceneCfg):


    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0.0, -1.05))
    )
 

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(1.65, 1.80, 1)
        ),
    )

    # #container
    # container = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Container",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/container.usd", 
    #         scale=(0.01, 0.01, 0.01),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.3, 0.1), rot=(0.0, 0.0, 1.0, 1.0)),
    # )



    #glass
    glass = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Glass",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/glass.usd", 
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.4, 1), rot=(0.0, 0.0, 1.0, 1.0)),
    )

    #cylinder
    cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cylinder.usd", 
            scale=(0.1, 0.1, 0.1),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.3, 0.4)),
    )


    #cubo
    cubo = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cubo",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cube.usd", 
            scale=(0.0006, 0.0006, 0.0006),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.25, 0.1)),
    )

    #tovaglietta
    tovaglietta = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tovaglietta",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/parallelepipedo.usd", 
            scale=(0.35, 0.35, 0.0015),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.647, 0.165, 0.165)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.35, 0.1), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    #Forchetta
    forchetta = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Forchetta",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/fork.usd", 
            scale=(0.01, 0.01, 0.01),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.2), rot=(0.707, 0.707, 0.707, 0.707)),    #manca da ruotare di 90 gradi la forchetta 
    )

    #Coltello
    coltello = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Coltello",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/coltello.usd", 
            scale=(0.01, 0.01, 0.01),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, -0.35, 0.2), rot=(0.0, 0.0, -0.707, 0.707)),    #manca da ruotare di 90 gradi la forchetta 
    )

    #porta_forchetta
    porta_forchetta = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Porta_forchetta",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cylinder.usd", 
            scale=(0.15, 0.15, 0.01),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.1)),
    )

    #Piatto_1
    piatto_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Piatto_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/plate.usd", 
            scale=(0.008, 0.008, 0.008),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.647, 0.165, 0.165)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.3), rot=(0.0, 0.0, 1.0, 1.0)),
    )


    #METTERE A POSTO GLI USDZ FILES 

    # #martello
    # hammer = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/martello",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="/home/jonatha/IsaacLab/usd_files_mattia/hammer.usd", 
    #         scale=(0.008, 0.008, 0.008),    
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, -0.4, 0.1), rot=(0.0, 0.0, 1.0, 1.0)),
    # )

    # Franka Panda robot
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24, 
            focus_distance=400.0, 
            horizontal_aperture=30,  # Aumentare l'apertura orizzontale per un campo visivo più ampio
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.27, 0.0, 0.8),  
            rot=(1.0, 0.2, 0.2, 0.7), 
            convention="opengl",
        ),
    )

    camera_hand = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/hand_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0), rot=(0.0, 1.0, 1.0, 0), convention="opengl"
        ),
    )

    camera_side_bridge = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Table/side_cam_Bridge",
        update_period=0.1,
        height=480,
        width=640,
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

# la posizione della camera di DROID deve essere randomized quindi la possiamo decidere noi (in teoria però rimane sempre una side camera)
#     camera_side_DROID = CameraCfg(
#     prim_path="{ENV_REGEX_NS}/Table/side_cam_DROID",
#     update_period=0.1,
#     height=480,
#     width=640,
#     data_types=["rgb", "distance_to_image_plane"],
#     spawn=sim_utils.PinholeCameraCfg(
#         focal_length=15.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
#     ),
#     offset=CameraCfg.OffsetCfg(
#         pos=(-0.3, 0.3, 0.5),  
#         rot=(0.5, 0.3, -0.6, -1.0),
#         convention="opengl",
#     ),
# )

def get_gripper_joint_ids(robot, gripper_joint_names):
    """
    Retrieve joint IDs for the gripper joints. 
    :param robot: The robot entity.
    :param gripper_joint_names: List of gripper joint names.
    :return: List of joint IDs.
    """
    gripper_joint_ids = []
    joint_names = robot.joint_names  # Ottenere i nomi dei giunti
    for joint_name in gripper_joint_names:
        if joint_name in joint_names:
            gripper_joint_ids.append(joint_names.index(joint_name))
        else:
            raise ValueError(f"Joint name '{joint_name}' not found in the robot's joints.")
    return gripper_joint_ids

def set_gripper_state(robot, gripper_state):
    """
    Set the state of the gripper using joint positions.
    """
    gripper_joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
    gripper_joint_ids = get_gripper_joint_ids(robot, gripper_joint_names)

    # Mappa lo stato del gripper a valori di comando per i giunti
    # Approxima il gripper_state all'intero più vicino
    gripper_command = round(gripper_state)  
    gripper_command = 1 if gripper_command == 0 else 0  # 1 per chiuso, 0 per aperto

    gripper_commands = [gripper_command] * len(gripper_joint_ids)

       # Converti i comandi in un tensor di PyTorch e trasferiscilo sulla GPU se necessario
    gripper_commands_tensor = torch.tensor(gripper_commands, dtype=torch.float32, device='cuda')

    # Imposta i target di posizione dei giunti del gripper
    robot.set_joint_position_target(gripper_commands_tensor, joint_ids=gripper_joint_ids)


# Specifica la directory di salvataggio
save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

def save_image(rgb, depth, index):
    rgb_path = os.path.join(save_dir, f"rgb_image_{index}.png")
    depth_path = os.path.join(save_dir, f"depth_image_{index}.png")
    torch.save(rgb, rgb_path)
    torch.save(depth, depth_path)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    # camera=scene["camera"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # # Listens to the required transforms
    # frame_marker_cfg = FRAME_MARKER_CFG.copy()
    # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    # ee_marker = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
    #     debug_vis=False,
    #     visualizer_cfg=frame_marker_cfg,
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
    #             name="end_effector",
    #             offset=OffsetCfg(
    #                 pos=[0.0, 0.0, 0.1034],
    #             ),
    #         ),
    #     ],
    # )


    # Define goals for the arm  #qui devo mettere le prdicted actiions di RT1
    # ee_goals = [
    #     [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0.0],
    #     [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
    #     [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    #     ]

    # vettore dei goal rappresentato nella forma corretta per essere utilizzato con RT1-X 
    # ee_goals_eul = [
    #     [0.4, 0.3, 0.7, torch.pi / 2, 0, torch.pi / 2, 1.0],
    #     [0.4, -0.3, 0.6, torch.pi / 2, torch.pi / 2, 0.0, 1.0],
    #     [0.7, 0, 0.5, 0, torch.pi / 2, 0.0, -1.0],
    # ]

    ee_goals_eul = [
        [0.3100048005580902, 0.13334313035011292, 0.4775620698928833, 3.092672824859619, 0.1832459717988968, 0.024089191108942032, 0.0],
        [0.31000643968582153, 0.13334643840789795, 0.47755908966064453, 3.092668294906616, 0.18324477970600128, 0.024097593501210213, 0.0],
        [0.31001150608062744, 0.1333305984735489, 0.4775649607181549, 3.0926945209503174, 0.1832312047481537, 0.024061404168605804, 0.0],
        [0.3100097179412842, 0.13333722949028015, 0.4775623381137848, 3.092683792114258, 0.18323105573654175, 0.024082111194729805, 0.0],
        [0.31000855565071106, 0.133342444896698, 0.47755998373031616, 3.0926754474639893, 0.18323121964931488, 0.024093234911561012, 0.0],
        [0.3100103437900543, 0.1333380788564682, 0.47756248712539673, 3.0926833152770996, 0.18322472274303436, 0.024080190807580948, 0.0],
        [0.31001099944114685, 0.1333388090133667, 0.47756457328796387, 3.092684507369995, 0.18321767449378967, 0.024085376411676407, 0.0],
        [0.31000977754592896, 0.13334263861179352, 0.47756028175354004, 3.0926806926727295, 0.18322865664958954, 0.02409367635846138, 0.0],
        [0.3100105822086334, 0.1333416998386383, 0.4775596857070923, 3.0926783084869385, 0.18322917819023132, 0.024087535217404366, 0.0],
        [0.31276145577430725, 0.1320100873708725, 0.4774886667728424, 3.0796499252319336, 0.1605641096830368, 0.009576672688126564, 0.0],
        [0.3169015645980835, 0.1289527416229248, 0.4779689908027649, 3.0562222003936768, 0.12956853210926056, -0.005411612801253796, 0.0],
        [0.3209681212902069, 0.12565156817436218, 0.47934049367904663, 3.0424487590789795, 0.10797173529863358, -0.01584535837173462, 0.0],
        [0.3263462483882904, 0.12118298560380936, 0.4817211925983429, 3.037262439727783, 0.08601397275924683, -0.011770612560212612, 0.0],
        [0.3313096761703491, 0.11753028631210327, 0.48367568850517273, 3.021634101867676, 0.05919913202524185, -0.014218701049685478, 0.0],
        [0.33722931146621704, 0.11264827847480774, 0.4868740439414978, 3.010546922683716, 0.03538520634174347, -0.021821582689881325, 0.0],
        [0.3423182964324951, 0.10797218978404999, 0.48914390802383423, 3.0020768642425537, 0.013133657164871693, -0.021095413714647293, 0.0],
        [0.34823670983314514, 0.10110566765069962, 0.4919849932193756, 2.995420455932617, -0.007796803489327431, -0.027850156649947166, 0.0],
        [0.3542804718017578, 0.09371642023324966, 0.4942355155944824, 2.9852685928344727, -0.029066849499940872, -0.03529035672545433, 0.0],
        [0.36043089628219604, 0.08505373448133469, 0.4967541992664337, 2.97948956489563, -0.050162483006715775, -0.044157449156045914, 0.0],
        [0.3655969500541687, 0.07717354595661163, 0.4984458088874817, 2.9702672958374023, -0.06779228150844574, -0.05466680973768234, 0.0],
        [0.3712698817253113, 0.06790981441736221, 0.5004864931106567, 2.9743151664733887, -0.08124768733978271, -0.05512203276157379, 0.0],
        [0.37743937969207764, 0.05834852531552315, 0.501460611820221, 2.9727249145507812, -0.0983738899230957, -0.06803131103515625, 0.0],
        [0.3841532766819, 0.04844911769032478, 0.5022048950195312, 2.978360176086426, -0.11352664977312088, -0.08270356804132462, 0.0],
        [0.39131075143814087, 0.03784647956490517, 0.5020055174827576, 2.977612257003784, -0.13338620960712433, -0.10052745789289474, 0.0],
        [0.3976762294769287, 0.028672190383076668, 0.5011340379714966, 2.978020668029785, -0.14812694489955902, -0.11526982486248016, 0.0],
        [0.4049408435821533, 0.017849642783403397, 0.5001022219657898, 2.97829270362854, -0.15785004198551178, -0.12917983531951904, 0.0],
        [0.41142112016677856, 0.008751202374696732, 0.49862614274024963, 2.9736533164978027, -0.1750916838645935, -0.14228308200836182, 0.0],
        [0.4184703826904297, -0.002108923392370343, 0.49716806411743164, 2.9791347980499268, -0.18589423596858978, -0.15556350350379944, 0.0],
        [0.4252324104309082, -0.011029869318008423, 0.4951932728290558, 2.992999792098999, -0.21373587846755981, -0.16727448999881744, 0.0],
        [0.43202510476112366, -0.0214454997330904, 0.4927476644515991, 2.988034248352051, -0.22049926221370697, -0.1724090874195099, 0.0],
        [0.4381217062473297, -0.02989453449845314, 0.4900759756565094, 2.9923462867736816, -0.2324809432029724, -0.17481648921966553, 0.0],
        [0.44472694396972656, -0.03933015093207359, 0.48754504323005676, 2.996218681335449, -0.23476794362068176, -0.17641595005989075, 0.0],
        [0.4509095251560211, -0.04763421788811684, 0.4838343858718872, 2.994924306869507, -0.23591633141040802, -0.18117362260818481, 0.0],
        [0.4578514099121094, -0.05650176480412483, 0.47887054085731506, 2.9957032203674316, -0.23097674548625946, -0.18647632002830505, 0.0],
        [0.4643342196941376, -0.06501942873001099, 0.4744715392589569, 2.988314628601074, -0.22159279882907867, -0.1867896169424057, 0.0],
        [0.4713609516620636, -0.07339755445718765, 0.468830406665802, 2.991049289703369, -0.21270060539245605, -0.18060044944286346, 0.0],
        [0.47743159532546997, -0.08000214397907257, 0.4642840623855591, 2.9892547130584717, -0.21053734421730042, -0.17555707693099976, 0.0],
        [0.4846748113632202, -0.08641824126243591, 0.4583149254322052, 2.9918212890625, -0.20334100723266602, -0.16376511752605438, 0.0],
        [0.49099302291870117, -0.09228197485208511, 0.4533364474773407, 2.9863739013671875, -0.20622502267360687, -0.15645061433315277, 0.0],
        [0.49828436970710754, -0.09844823181629181, 0.44645175337791443, 2.985419511795044, -0.21429407596588135, -0.14902472496032715, 0.0],
        [0.5051765441894531, -0.10400528460741043, 0.44040554761886597, 2.9879112243652344, -0.2169559895992279, -0.1458301693201065, 0.0],
        [0.5114229321479797, -0.10886314511299133, 0.4346213936805725, 2.9916484355926514, -0.22139304876327515, -0.1411457061767578, 0.0],
        [0.5176029205322266, -0.11399342119693756, 0.4274561107158661, 2.9943039417266846, -0.2240489274263382, -0.13479258120059967, 0.0],
        [0.5229924917221069, -0.11860179901123047, 0.42056146264076233, 2.9901010990142822, -0.22700746357440948, -0.12690506875514984, 0.0],
        [0.5282594561576843, -0.12330886721611023, 0.4128792881965637, 2.9950156211853027, -0.22480419278144836, -0.12279076129198074, 0.0],
        [0.533558189868927, -0.12815527617931366, 0.40444350242614746, 2.990309238433838, -0.2218938171863556, -0.11761936545372009, 0.0],
        [0.53822922706604, -0.13300412893295288, 0.3973034620285034, 2.9869539737701416, -0.21527457237243652, -0.11527513712644577, 0.0],
        [0.5423539876937866, -0.1380465030670166, 0.3899900019168854, 2.9862990379333496, -0.21081817150115967, -0.11482725292444229, 0.0],
        [0.547006368637085, -0.14402703940868378, 0.3827948570251465, 2.984245538711548, -0.20760640501976013, -0.11978348344564438, 0.0],
        [0.5507500171661377, -0.14945530891418457, 0.37698492407798767, 2.9803307056427, -0.20518584549427032, -0.11919612437486649, 0.0],
        [0.5557356476783752, -0.15564535558223724, 0.3710666298866272, 2.982144594192505, -0.20631204545497894, -0.12602652609348297, 0.0],
        [0.5602179765701294, -0.1609363704919815, 0.3654908835887909, 2.9797792434692383, -0.20644594728946686, -0.1316431760787964, 0.0],
        [0.5650935769081116, -0.16683505475521088, 0.36011314392089844, 2.9792938232421875, -0.2054651826620102, -0.1358424574136734, 0.0],
        [0.570378303527832, -0.1719937026500702, 0.35439252853393555, 2.9791159629821777, -0.20351776480674744, -0.13807590305805206, 0.0],
        [0.5755625367164612, -0.17659156024456024, 0.349371999502182, 2.9753165245056152, -0.204753115773201, -0.13901205360889435, 0.0],
        [0.5808338522911072, -0.18108990788459778, 0.34450235962867737, 2.9739227294921875, -0.2062150537967682, -0.13997282087802887, 0.0],
        [0.5851735472679138, -0.18449538946151733, 0.33948302268981934, 2.970447540283203, -0.2047969549894333, -0.1404234915971756, 0.0],
        [0.5897679924964905, -0.1879255175590515, 0.333668053150177, 2.9703991413116455, -0.20392361283302307, -0.1396697759628296, 0.0],
        [0.5940585732460022, -0.1909300535917282, 0.3269619345664978, 2.9712984561920166, -0.19842053949832916, -0.14046618342399597, 0.0],
        [0.5971166491508484, -0.19327248632907867, 0.32044872641563416, 2.9743857383728027, -0.19202232360839844, -0.1419709324836731, 0.0],
        [0.5992733240127563, -0.1955927461385727, 0.3130303621292114, 2.9816982746124268, -0.18132925033569336, -0.14129158854484558, 0.0],
        [0.6018201112747192, -0.19788599014282227, 0.30594781041145325, 2.9857420921325684, -0.17430485785007477, -0.1380748152732849, 0.0],
        [0.603425145149231, -0.1999027580022812, 0.2992859184741974, 2.993441581726074, -0.16380608081817627, -0.13579772412776947, 0.0],
        [0.6056430339813232, -0.2020750343799591, 0.292116641998291, 3.003960132598877, -0.152372345328331, -0.13651129603385925, 0.0],
        [0.6083778738975525, -0.2050708532333374, 0.2849832773208618, 3.013723134994507, -0.1447562724351883, -0.1370081603527069, 0.0],
        [0.6112242937088013, -0.20821881294250488, 0.27931442856788635, 3.019716501235962, -0.1414279192686081, -0.1379333734512329, 0.0],
        [0.6134258508682251, -0.21047526597976685, 0.2743559181690216, 3.0265471935272217, -0.13557754456996918, -0.13814447820186615, 0.0],
        [0.6163296699523926, -0.21345876157283783, 0.2685686945915222, 3.0282154083251953, -0.13176265358924866, -0.13997820019721985, 0.0],
        [0.6188324689865112, -0.215897336602211, 0.26329830288887024, 3.03406023979187, -0.12702898681163788, -0.14188720285892487, 0.0],
        [0.6210062503814697, -0.21890418231487274, 0.25803160667419434, 3.032855749130249, -0.12361440807580948, -0.14565029740333557, 0.0],
        [0.6227162480354309, -0.22000746428966522, 0.25257158279418945, 3.036508798599243, -0.11906405538320541, -0.14485502243041992, 0.0],
        [0.6250520348548889, -0.22005340456962585, 0.24601253867149353, 3.038646936416626, -0.11543042212724686, -0.14870229363441467, 0.0],
        [0.6269227266311646, -0.22083613276481628, 0.24093139171600342, 3.0399763584136963, -0.11007855832576752, -0.15123014152050018, 0.0],
        [0.6290947794914246, -0.22163552045822144, 0.23570610582828522, 3.0406992435455322, -0.10571074485778809, -0.15221412479877472, 0.0],
        [0.6305789351463318, -0.2223985344171524, 0.22993195056915283, 3.039783477783203, -0.0994090810418129, -0.15190662443637848, 0.0],
        [0.6321239471435547, -0.22289413213729858, 0.22445757687091827, 3.040010690689087, -0.09245611727237701, -0.15156660974025726, 0.0],
        [0.633631706237793, -0.22405312955379486, 0.21847911179065704, 3.0366854667663574, -0.08700605481863022, -0.15168294310569763, 0.0],
        [0.6355512738227844, -0.22556523978710175, 0.21232275664806366, 3.035776376724243, -0.08032149076461792, -0.15242266654968262, 0.0],
        [0.6376243233680725, -0.22747458517551422, 0.2071063369512558, 3.0349156856536865, -0.0754028782248497, -0.15351712703704834, 0.0],
        [0.6391931176185608, -0.22940593957901, 0.20205473899841309, 3.0345332622528076, -0.07066384702920914, -0.15412256121635437, 0.0],
        [0.6410733461380005, -0.23117880523204803, 0.19766250252723694, 3.0360445976257324, -0.067532017827034, -0.15532329678535461, 0.0],
        [0.642501711845398, -0.2331150770187378, 0.193049818277359, 3.036579132080078, -0.06431560963392258, -0.15623965859413147, 0.0],
        [0.6447285413742065, -0.2354179322719574, 0.1888032853603363, 3.036372184753418, -0.06481178849935532, -0.1594979614019394, 0.0],
        [0.6462242007255554, -0.23785966634750366, 0.1850360482931137, 3.0339698791503906, -0.06317964196205139, -0.16237282752990723, 0.0],
        [0.6476269960403442, -0.23999987542629242, 0.18148314952850342, 3.03659987449646, -0.06058317422866821, -0.16466638445854187, 0.0],
        [0.6486977338790894, -0.24188125133514404, 0.1783585399389267, 3.0358598232269287, -0.0593927726149559, -0.16551773250102997, 0.0],
        [0.6502438187599182, -0.24393515288829803, 0.1752055436372757, 3.036480188369751, -0.05820241943001747, -0.16783453524112701, 0.0],
        [0.6516575813293457, -0.24551181495189667, 0.17234250903129578, 3.039430618286133, -0.05641740560531616, -0.16958770155906677, 0.0],
        [0.6532878875732422, -0.24697065353393555, 0.16937251389026642, 3.039825201034546, -0.05657270550727844, -0.17007644474506378, 0.0],
        [0.6549857258796692, -0.2482098788022995, 0.16614912450313568, 3.0411412715911865, -0.054846931248903275, -0.1708725094795227, 0.0],
        [0.6560325026512146, -0.24914605915546417, 0.16360853612422943, 3.041653633117676, -0.05418017879128456, -0.1725539267063141, 0.0],
        [0.6573247909545898, -0.24955853819847107, 0.16070181131362915, 3.043330192565918, -0.052277278155088425, -0.17329692840576172, 0.0],
        [0.6582075953483582, -0.2500719428062439, 0.15808209776878357, 3.044457197189331, -0.050700750201940536, -0.1742904782295227, 0.0],
        [0.6590666770935059, -0.25045865774154663, 0.15588971972465515, 3.045203447341919, -0.04949051886796951, -0.17467065155506134, 0.0],
        [0.659686267375946, -0.2510400712490082, 0.15388473868370056, 3.0468223094940186, -0.04809477552771568, -0.17580564320087433, 0.0],
        [0.660065233707428, -0.2517811059951782, 0.1519346833229065, 3.0484135150909424, -0.04659513384103775, -0.17641882598400116, 0.12783899903297424],
        [0.6604788899421692, -0.25220730900764465, 0.15070798993110657, 3.050417184829712, -0.045027248561382294, -0.1757480502128601, 0.27643176913261414],
        [0.6606247425079346, -0.2524890601634979, 0.14986512064933777, 3.050161838531494, -0.04357996582984924, -0.17683668434619904, 0.3381057381629944],
        [0.6610081791877747, -0.25270116329193115, 0.14910560846328735, 3.051166534423828, -0.04338550567626953, -0.1777249276638031, 0.43061670660972595],
        [0.6613457202911377, -0.2531164884567261, 0.1486985981464386, 3.0495047569274902, -0.04381424933671951, -0.17922407388687134, 0.5011013150215149],
        [0.6617559790611267, -0.25321677327156067, 0.14885179698467255, 3.0498974323272705, -0.04397675022482872, -0.17985215783119202, 0.5715858936309814],
        [0.6623371243476868, -0.25373154878616333, 0.1491907685995102, 3.049659490585327, -0.045089054852724075, -0.18060912191867828, 0.6552863121032715],
        [0.6628320217132568, -0.2541565001010895, 0.14956429600715637, 3.0498623847961426, -0.04565190151333809, -0.18147216737270355, 0.7257709503173828],
        [0.6633761525154114, -0.2546186149120331, 0.15002772212028503, 3.050126314163208, -0.047150325030088425, -0.18297643959522247, 0.8094713687896729],
        [0.6639317274093628, -0.2548523545265198, 0.15032228827476501, 3.0509116649627686, -0.04823010414838791, -0.1839735507965088, 0.8975771069526672],
        [0.664201557636261, -0.25514739751815796, 0.15041156113147736, 3.0508179664611816, -0.04848242551088333, -0.184333935379982, 0.9636564254760742],
        [0.6644056439399719, -0.25520867109298706, 0.15047389268875122, 3.0506365299224854, -0.04832190275192261, -0.18437206745147705, 0.9944933652877808],
        [0.6642432808876038, -0.2552213668823242, 0.1507929265499115, 3.051179885864258, -0.0478060208261013, -0.18539993464946747, 0.9669603705406189],
        [0.6630988717079163, -0.2546718418598175, 0.15246015787124634, 3.05035138130188, -0.048588335514068604, -0.18781742453575134, 0.9537444710731506],
        [0.6601584553718567, -0.2524237036705017, 0.1578371822834015, 3.0484871864318848, -0.06225576996803284, -0.18984490633010864, 0.9537444710731506],
        [0.6560112237930298, -0.24896195530891418, 0.16622881591320038, 3.050628185272217, -0.07658452540636063, -0.19120259582996368, 0.9537444710731506],
        [0.6534085869789124, -0.2467574030160904, 0.1738162487745285, 3.050872564315796, -0.08669192343950272, -0.19329716265201569, 0.9537444710731506],
        [0.6505246162414551, -0.24382908642292023, 0.18438835442066193, 3.0520904064178467, -0.09602563828229904, -0.19655632972717285, 0.9537444710731506],
        [0.6472153067588806, -0.24033765494823456, 0.1974458247423172, 3.053802967071533, -0.10595374554395676, -0.19746892154216766, 0.9537444710731506],
        [0.6454494595527649, -0.2375233918428421, 0.2083573043346405, 3.0567963123321533, -0.12126075476408005, -0.1957356333732605, 0.9537444710731506],
        [0.6413580775260925, -0.23353950679302216, 0.2218756377696991, 3.0547447204589844, -0.13117292523384094, -0.20081135630607605, 0.9537444710731506],
        [0.6384565234184265, -0.229857936501503, 0.23312212526798248, 3.0566883087158203, -0.14313898980617523, -0.19911670684814453, 0.9537444710731506],
        [0.6329392194747925, -0.2241862267255783, 0.24626421928405762, 3.061046600341797, -0.15463097393512726, -0.19807280600070953, 0.9537444710731506],
        [0.629016101360321, -0.2192070335149765, 0.25751954317092896, 3.0609183311462402, -0.16399399936199188, -0.19930940866470337, 0.9537444710731506],
        [0.6226998567581177, -0.21165500581264496, 0.26907315850257874, 3.062305212020874, -0.17063716053962708, -0.19396962225437164, 0.9537444710731506],
        [0.6157895922660828, -0.2026446908712387, 0.2791285514831543, 3.0591299533843994, -0.17557579278945923, -0.1826568990945816, 0.9537444710731506],
        [0.6089655160903931, -0.19360984861850739, 0.28665515780448914, 3.059124708175659, -0.1816074103116989, -0.16952963173389435, 0.9537444710731506],
        [0.6024953722953796, -0.18480364978313446, 0.29372456669807434, 3.0566041469573975, -0.18477551639080048, -0.16381476819515228, 0.9537444710731506],
        [0.5950162410736084, -0.17426913976669312, 0.2996208071708679, 3.0594961643218994, -0.18855832517147064, -0.15720126032829285, 0.9537444710731506],
        [0.5888389348983765, -0.16479182243347168, 0.3037143647670746, 3.0655550956726074, -0.19266141951084137, -0.1493758112192154, 0.9537444710731506],
        [0.5812180042266846, -0.15495665371418, 0.30629414319992065, 3.059297561645508, -0.1917443722486496, -0.13960906863212585, 0.9537444710731506],
        [0.5736398696899414, -0.1459481567144394, 0.30881088972091675, 3.068737506866455, -0.19260632991790771, -0.1308281421661377, 0.9537444710731506],
        [0.5667398571968079, -0.13667047023773193, 0.30656349658966064, 3.071105718612671, -0.19191999733448029, -0.12836109101772308, 0.9537444710731506],
        [0.5586244463920593, -0.12965059280395508, 0.3055744767189026, 3.0752358436584473, -0.19223102927207947, -0.12545986473560333, 0.9537444710731506],
        [0.5518192052841187, -0.12344314903020859, 0.3022530972957611, 3.078881025314331, -0.1915312558412552, -0.12957194447517395, 0.9537444710731506],
        [0.5452375411987305, -0.1179569661617279, 0.29731544852256775, 3.082453489303589, -0.19317923486232758, -0.1329352855682373, 0.9537444710731506],
        [0.5399477481842041, -0.11333213001489639, 0.29195916652679443, 3.086010694503784, -0.1932031810283661, -0.14511406421661377, 0.9537444710731506],
        [0.5353534817695618, -0.10906776785850525, 0.2863641083240509, 3.0892817974090576, -0.19225910305976868, -0.15125904977321625, 0.9537444710731506],
        [0.5310835838317871, -0.10668598860502243, 0.27659356594085693, 3.094953775405884, -0.1891016960144043, -0.16520647704601288, 0.9537444710731506],
        [0.5280290842056274, -0.10363856703042984, 0.26753801107406616, 3.0967519283294678, -0.18375840783119202, -0.17900727689266205, 0.9537444710731506],
        [0.5249689221382141, -0.10129650682210922, 0.25772085785865784, 3.0994951725006104, -0.17843173444271088, -0.18978559970855713, 0.9537444710731506],
        [0.52147376537323, -0.09977802634239197, 0.2491113543510437, 3.099097490310669, -0.17368145287036896, -0.20161178708076477, 0.9537444710731506],
        [0.5190767645835876, -0.09815730899572372, 0.2404744029045105, 3.103264570236206, -0.1666388213634491, -0.21611541509628296, 0.9537444710731506],
        [0.5177916884422302, -0.09737857431173325, 0.23100881278514862, 3.104506492614746, -0.16035012900829315, -0.23154045641422272, 0.9537444710731506],
        [0.5163038969039917, -0.0964406281709671, 0.22406692802906036, 3.103450059890747, -0.15566876530647278, -0.24605663120746613, 0.9537444710731506],
        [0.5161080360412598, -0.09606048464775085, 0.21686242520809174, 3.1053991317749023, -0.14979496598243713, -0.2582364082336426, 0.9537444710731506],
        [0.5162104368209839, -0.09630334377288818, 0.209710493683815, 3.1075093746185303, -0.14416201412677765, -0.2699953019618988, 0.9537444710731506],
        [0.5173990726470947, -0.0973447933793068, 0.2019437998533249, 3.1108479499816895, -0.13827839493751526, -0.2897561490535736, 0.9537444710731506],
        [0.517899751663208, -0.09834528714418411, 0.19476832449436188, 3.1156506538391113, -0.1329910010099411, -0.3055858910083771, 0.9537444710731506],
        [0.5170758366584778, -0.0992167517542839, 0.18817740678787231, 3.1166532039642334, -0.12756243348121643, -0.318264365196228, 0.9537444710731506],
        [0.5167443752288818, -0.10024110972881317, 0.1803903430700302, 3.1170012950897217, -0.12066489458084106, -0.33347615599632263, 0.9537444710731506],
        [0.5162721872329712, -0.10093490779399872, 0.1749715507030487, 3.117300510406494, -0.11553125083446503, -0.3428833484649658, 0.9537444710731506],
        [0.5164443254470825, -0.10160376876592636, 0.16924045979976654, 3.119527578353882, -0.11232073605060577, -0.3594757318496704, 0.9537444710731506],
        [0.5166327953338623, -0.10241943597793579, 0.16358615458011627, 3.1191928386688232, -0.1109776422381401, -0.371369332075119, 0.9537444710731506],
        [0.5165793895721436, -0.10259129852056503, 0.1600058227777481, 3.119636297225952, -0.10858455300331116, -0.37755492329597473, 0.9537444710731506],
        [0.5164688229560852, -0.10244841873645782, 0.15695630013942719, 3.120208740234375, -0.10740698128938675, -0.3850211799144745, 0.9537444710731506],
        [0.516010582447052, -0.10167863965034485, 0.1551060676574707, 3.1174561977386475, -0.1103605329990387, -0.39125922322273254, 0.5649780035018921],
        [0.5157163143157959, -0.10127996653318405, 0.15447406470775604, 3.1212692260742188, -0.11218912154436111, -0.39340105652809143, 0.5561674237251282],
        [0.5149586200714111, -0.1004059687256813, 0.15619468688964844, 3.120227813720703, -0.12186729907989502, -0.395293653011322, 0.4856828451156616],
        [0.5138930678367615, -0.09947017580270767, 0.15976069867610931, 3.118664503097534, -0.13336315751075745, -0.3959435224533081, 0.4240088164806366],
        [0.5123452544212341, -0.09819462895393372, 0.16499614715576172, 3.117380380630493, -0.14853480458259583, -0.39391854405403137, 0.3359030783176422],
        [0.5105926394462585, -0.09649042785167694, 0.17135587334632874, 3.1145682334899902, -0.16809135675430298, -0.3914516270160675, 0.2698237895965576],
        [0.5082041621208191, -0.09428591281175613, 0.17972800135612488, 3.112643003463745, -0.19309209287166595, -0.39598575234413147, 0.18171806633472443],
        [0.5068457722663879, -0.09218119084835052, 0.18692128360271454, 3.113301992416382, -0.21370327472686768, -0.398972749710083, 0.11563877016305923],
        [0.5036536455154419, -0.08920064568519592, 0.19707049429416656, 3.1135051250457764, -0.23786796629428864, -0.40110546350479126, 0.031938336789608],
        [0.5006608366966248, -0.08623727411031723, 0.20532716810703278, 3.1131250858306885, -0.2571621239185333, -0.4008334279060364, 0.004405254498124123],
        [0.4956406354904175, -0.08168114721775055, 0.2156914472579956, 3.1088104248046875, -0.272838294506073, -0.4034700393676758, 0.0],
        [0.49142205715179443, -0.07805795967578888, 0.22386975586414337, 3.1141369342803955, -0.2911309003829956, -0.40063172578811646, 0.0],
        [0.48444390296936035, -0.07310563325881958, 0.23426590859889984, 3.1121037006378174, -0.30553650856018066, -0.4015388488769531, 0.0],
        [0.47876864671707153, -0.06896137446165085, 0.24282090365886688, 3.1148569583892822, -0.3085072636604309, -0.40738967061042786, 0.0],
        [0.4720492362976074, -0.0660848468542099, 0.25187456607818604, 3.116760730743408, -0.31395575404167175, -0.42036765813827515, 0.0],
        [0.4648926854133606, -0.06348781287670135, 0.26146721839904785, 3.1196281909942627, -0.3237822651863098, -0.42986392974853516, 0.0],
        [0.4576927423477173, -0.06113438308238983, 0.27154988050460815, 3.1183204650878906, -0.33428651094436646, -0.43724584579467773, 0.0],
        [0.4515475630760193, -0.05922863632440567, 0.2810378074645996, 3.11971378326416, -0.34679919481277466, -0.4440920650959015, 0.0],
        [0.44625160098075867, -0.05800655484199524, 0.28911730647087097, 3.121019124984741, -0.3646736741065979, -0.44841668009757996, 0.0],
        [0.44202107191085815, -0.055883485823869705, 0.2977857291698456, 3.121103525161743, -0.38227367401123047, -0.45197317004203796, 0.0],
        [0.43874412775039673, -0.0550818145275116, 0.3055341839790344, 3.124391794204712, -0.3959878385066986, -0.45038390159606934, 0.0],
        [0.43632417917251587, -0.05437527596950531, 0.3131062686443329, 3.129603624343872, -0.41403892636299133, -0.4548290967941284, 0.0],
        [0.4348672032356262, -0.05407063663005829, 0.31810998916625977, 3.132972240447998, -0.42948901653289795, -0.4556586444377899, 0.0],
        [0.46611401438713074, -0.07337160408496857, 0.2902693748474121, 3.1190617084503174, -0.4148690104484558, -0.4332413673400879, 0.0],
        [0.46491724252700806, -0.07324298471212387, 0.2963334023952484, 3.1287786960601807, -0.4253203868865967, -0.4412371814250946, 0.0],
        [0.46482527256011963, -0.07383675128221512, 0.29706570506095886, 3.1304030418395996, -0.42612069845199585, -0.4425404667854309, 0.0],
        [0.46476876735687256, -0.07381464540958405, 0.2949455976486206, 3.126411199569702, -0.422546923160553, -0.44110292196273804, 0.0],
        [0.46462756395339966, -0.07339896261692047, 0.29217809438705444, 3.121084690093994, -0.41788142919540405, -0.43804511427879333, 0.0],
        [0.4645143747329712, -0.07328019291162491, 0.2905879020690918, 3.1179683208465576, -0.4151988923549652, -0.4364487826824188, 0.0],
        [0.46455830335617065, -0.07337307184934616, 0.28968754410743713, 3.1160948276519775, -0.4138264060020447, -0.4358503520488739, 0.0],
    ]

    # with open('trajectory_proof/franka_mio_gt_observation_target.json', 'r') as file:
    #     data = json.load(file)
    #     traj = data['trajectory']
    
    # ee_goals_eul=torch.tensor(traj, dtype=torch.float32)


   

    ##

    # Convert goals to quaternion representation
    ee_goals = convert_goals_to_quaternion(ee_goals_eul)
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]
    
    # Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])


    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)

    # Obtain the frame index of the end-effector
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    joint_pos = robot.data.default_joint_pos.clone() 
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


    # Simulation loop
    while simulation_app.is_running():
        # reset                          
        if count % 150 == 0:
            # reset time
            count = 0
            # reset joint state
            # joint_pos = robot.data.default_joint_pos.clone() 
            # joint_vel = robot.data.default_joint_vel.clone()
            # robot.write_joint_state_to_sim(joint_pos, joint_vel)


            # robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            # diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

            # Recuperare i dati dalla telecamera
            rgb_image = scene["camera"].data.output["rgb"].clone().detach()
            depth_image = scene["camera"].data.output["distance_to_image_plane"].clone().detach()

            # # Salva l'immagine quando il target è raggiunto
            # save_image(rgb_image.cpu(), depth_image.cpu(), current_goal_idx)
            

            
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            """ Sezione che serve solo a capire che tipologia di coordinate usa lo script """

            # print("1",ee_pos_b)  #questo è il reference frame relativo dell'end-effector rispetto alla base del robot e siccome il robot è nell'origine assoluta non ci sarà differenza tra ee_pose_w e ee_pos_b
            # print("2",ee_pose_w[:, 0:3]) #questo è il reference frame assoluto dell'end-effector cioè rispetto al punto (0,0,0) (cioè root_pose_w)
            # print("3",root_pose_w[:, 0:3])  # questo è il reference frame assoluto dell'origine e cioè (0,0,0)
            # # i due tensori sono uguali in quanto la posizione assoluta e quella relativa dell'end-effector coincidono poichè il robot è posizionato esattamente in (0,0,0) 
            # print("4",ee_quat_b[:,1]) 
            # print("5",ee_quat_b[0])

            # ee_euler_angle_b=quaternion_to_euler(ee_quat_b[0])
            # print("5",ee_euler_angle_b)

            # #FACCIO TUTTO DENTRO ALL'ELSE PERCHE' TANTO MI INTERESSANO SOLO LE COORDINATE DI PARTENZA DELL'END-EFFECTOR
         

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # Apply gripper command based on the current goal
        gripper_state = ee_goals_eul[current_goal_idx][-1]  # Gripper state from the goal coming from RT1
        set_gripper_state(robot, gripper_state)

        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

        # print("the current pose (in quaternions) in the world reference frame is  ", ee_pose_w)

        # # print information from the sensors
        # print("-------------------------------")
        # print(scene["camera"])
        # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        # print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        # print("-------------------------------")
    



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()












