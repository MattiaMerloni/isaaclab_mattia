'''    ./isaaclab.sh -p source/standalone/tutorials/06_mattia/attempt_7_openvla_check_open_drawer.py  --num_envs 1  '''

""" SCRIPT TO CHECK THE OPENVLA MODEL """

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

from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG  # FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg

# This function will be useful when I input actions predicted by RT1
def euler_to_quaternion(roll, pitch, yaw):    # TO USE WHEN USING A .json FILE  #UNCOMMENT
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion using PyTorch.
    """

    # Convert input angles to tensors
    roll = roll.clone().detach().float()      #I need to use clone().detach() because roll,pitch and yaw are PyTorch tensors and I can't perform operations directly on them
    pitch = pitch.clone().detach().float()
    yaw = yaw.clone().detach().float()

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

    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # To avoid values outside the domain of asin, we need to limit sinp between -1 and 1
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # Calculate yaw (z-axis rotation)
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
        coord= torch.cat((pos,rot))   #UNCOMMENT   #TO USE WITH .json FILE

        # Exclude the gripper state and only append position and quaternion
        # ee_goals_quat.append(pos + rot.tolist())  #UNCOMMENT   # TO USE WITH SCRIPT
        ee_goals_quat.append(coord.tolist())   #UNCOMMENT  #TO USE WITH .json FILE
        
    return ee_goals_quat

@configclass
class CabinetSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0.0, 0.0))    #(0, 0.0, -1.05)   #I had set the ground plane -1.05 below the robot on the z-axis
    )
 
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,  #False
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0, 0.4),    # (0.8, 0, 0.4)
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
                friction=0.0,    # I REMOVED FRICTION TO EASILY OPEN DRAWERS WITH THE ROBOT GRIPPER
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

    # Franka Panda robot
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(   #I THINK IT'S STILL BETTER TO START FROM THE BASE POSITION IF WE WANT TO USE THIS SCRIPT FOR CHECKING
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

    #cuboid
    cubo = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cubo",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, -0.25, 1.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cube.usd",    # "/home/jonatha/IsaacLab/usd_files_mattia/cube.usd"  #f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
            scale=(0.0006, 0.0006, 0.0006),     #remember that the scale command is relative to the object's reference system, so if we rotate it, x,y and z will be inverted accordingly 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
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
            horizontal_aperture=30,  # Increase horizontal aperture for wider field of view
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.4, 0.0, 0.8),    #(1.27, 0.0, 0.8)
            rot=(1.0, 0.2, 0.2, 0.7), 
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

def get_gripper_joint_ids(robot, gripper_joint_names):
    """
    Retrieve joint IDs for the gripper joints. 
    :param robot: The robot entity.
    :param gripper_joint_names: List of gripper joint names.
    :return: List of joint IDs.
    """
    gripper_joint_ids = []
    joint_names = robot.joint_names  # Get joint names
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

    # Round gripper_state to nearest integer
    gripper_command = round(gripper_state.item())  
    gripper_command = 1 if gripper_command == 1 else 0  # gripper_command is set to 1 (gripper open) if gripper_state is -1, otherwise closed

    gripper_commands = [gripper_command] * len(gripper_joint_ids)

    # Convert commands to PyTorch tensor and transfer to GPU if needed
    gripper_commands_tensor = torch.tensor(gripper_commands, dtype=torch.float32, device='cuda')

    # Set gripper joint position targets
    robot.set_joint_position_target(gripper_commands_tensor, joint_ids=gripper_joint_ids)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")    #'dls' stands for damped least squares
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # TEST TO CHECK OPENVLA TRAJECTORY 
    with open('trajectory_drawer_IN.json', 'r') as file:
        data = json.load(file)
        traj = data['trajectory_0']
    
    ee_goals_eul=torch.tensor(traj, dtype=torch.float32)

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
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]      #here we define the end-effector position in cartesian space
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()   #here we define joint control (7 joints for Franka Panda)

            # reset controller
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1