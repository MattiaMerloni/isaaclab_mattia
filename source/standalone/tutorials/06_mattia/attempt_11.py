# Usage
'''   ./isaaclab.sh -p source/standalone/tutorials/06_mattia/attempt_11.py  --num_envs 1   '''

# qui provo a mettere insieme i due script senza usare ROS2 


"""Initialize the RT1-X model"""
import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from PIL import Image
import numpy as np
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts
from IPython import display
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import torch

# Load TF model checkpoint
# Replace saved_model_path with path to the parent folder of
# the folder rt_1_x_tf_trained_for_002272480_step.

saved_model_path='/home/jonatha/IsaacLab/open_x_embodiment/colabs/rt_1_x_tf_trained_for_002272480_step'

tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    model_path=saved_model_path,
    load_specs_from_pbtxt=True,
    use_tf_function=True)

def resize(image):
  image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
  image = tf.cast(image, tf.uint8)
  return image

# 2. Carica il modello USE e genera l'embedding per l'input testuale
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')

# Supponiamo che tu abbia un comando testuale
episode_natural_language_instruction = "Pick up the glass."

# Funzione per normalizzare il nome del task
def normalize_task_name(task_name):
    replaced = task_name.replace('_', ' ').replace('1f', ' ').replace(
        '4f', ' ').replace('-', ' ').replace('50', ' ').replace('55',
                                                                 ' ').replace('56', ' ')
    return replaced.lstrip(' ').rstrip(' ')

# Genera l'embedding dell'input testuale
natural_language_embedding = embed([normalize_task_name(episode_natural_language_instruction)])[0]


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
torch.backends.cuda.preferred_linalg_library("cusolver")
import os

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
from omni.isaac.lab.sensors import CameraCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


def process_and_infer(image_path, episode_natural_language_instruction,natural_language_embedding):
    # Carica e processa l'immagine
    image = torch.load(image_path)
    image = image.numpy()
    image = tf.squeeze(image, axis=0)  
    image = image[:, :, :3]  
    image = resize(image)

    # Creare l'osservazione combinata
    observation = {
        'image': image,
        'natural_language_instruction': episode_natural_language_instruction,
        'natural_language_embedding': natural_language_embedding
    }

    # Creare il time step per TFA e fare inferenza con il modello RT1-X
    tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))
    policy_state = tfa_policy.get_initial_state(batch_size=1)
    action = tfa_policy.action(tfa_time_step, policy_state)

    # Estrarre le componenti specifiche dall'azione
    rotation_delta = action.action['rotation_delta']
    world_vector = action.action['world_vector']
    gripper_closedness_action = action.action['gripper_closedness_action']

    #  Concatenare gli array in un unico array
    ee_goals = np.concatenate([world_vector, rotation_delta, gripper_closedness_action])

    # Inizializza la matrice se non è stata passata
    if ee_goals_matrix is None:
        ee_goals_matrix = np.array([ee_goals])
    else:
        ee_goals_matrix = np.vstack([ee_goals_matrix, ee_goals])

    return ee_goals_matrix

# questa funzione mi servirà quando gli darò in input le azioni predette da RT1
def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion using PyTorch.
    """

    # Convert input angles to tensors
    roll = torch.tensor(roll, dtype=torch.float32)
    pitch = torch.tensor(pitch, dtype=torch.float32)
    yaw = torch.tensor(yaw, dtype=torch.float32)

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

    return torch.tensor([x, y, z, w], dtype=torch.float32)

def convert_goals_to_quaternion(ee_goals_eul):
    """
    Convert a list of goals from Euler angles to quaternion representation.
    Excludes the gripper state, resulting in 7 values per goal.
    """
    ee_goals_quat = []
    for goal in ee_goals_eul:
        pos = goal[:3]  # x, y, z position
        rot = euler_to_quaternion(goal[3], goal[4], goal[5])  # convert Euler angles to quaternion
        # Exclude the gripper state and only append position and quaternion
        ee_goals_quat.append(pos + rot.tolist())  
    return ee_goals_quat




@configclass
class TableTopSceneCfg(InteractiveSceneCfg):


    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(1.65, 1.80, 1)
        ),
    )


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
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.4, 1), rot=(0.0, 0.0, 1.0, 1.0)),
    )

    #cylinder
    cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cylinder.usd", 
            scale=(0.1, 0.1, 0.1),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.5, 0.1)),
    )

    #parallelepipedo
    parallelepipedo = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Parallelepipedo",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/parallelepipedo.usd", 
            scale=(0.1, 0.1, 0.1),    #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.6, 0.1)),
    )

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
            pos=(1.3, 0.0, 0.7),  # Regolare la posizione per una migliore visione del tavolo
            rot=(1.0, 0.4, 0.4, 0.65), 
            convention="opengl",
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
    :param robot: The robot entity.
    :param gripper_state: 1.0 for open, 0.0 for close.
    """
    gripper_joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
    gripper_joint_ids = get_gripper_joint_ids(robot, gripper_joint_names)

    # Mappa lo stato del gripper a valori di comando per i giunti
    gripper_command = 1.0 if gripper_state == 1.0 else 0.0
    gripper_commands = [gripper_command] * len(gripper_joint_ids)

    # Converti i comandi in un tensor di PyTorch e trasferiscilo sulla GPU se necessario
    gripper_commands_tensor = torch.tensor(gripper_commands, dtype=torch.float32, device='cuda')

    # Imposta i target di posizione dei giunti del gripper
    robot.set_joint_position_target(gripper_commands_tensor, joint_ids=gripper_joint_ids)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    camera=scene["camera"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm  #qui devo mettere le prdicted actiions di RT1
    # ee_goals = [
    #     [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0.0],
    #     [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
    #     [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    #     ]

    #vettore dei goal rappresentato nella forma corretta per essere utilizzato con RT1-X 
    # ee_goals_eul = [
    #     [0.4, 0.3, 0.7, torch.pi / 2, 0, torch.pi / 2, 0.0],
    #     [0.4, -0.3, 0.6, torch.pi / 2, torch.pi / 2, 0.0, 0.0],
    #     [0.7, 0, 0.5, 0, torch.pi / 2, 0.0, 1.0],
    # ]

    # Track the given command
    current_goal_idx = 0

    ee_goals_eul = process_and_infer("/home/jonatha/IsaacLab/saved_images/rgb_image_{current_goal_idx}.pt", episode_natural_language_instruction)

    # Convert goals to quaternion representation
    ee_goals = convert_goals_to_quaternion(ee_goals_eul)
    ee_goals = torch.tensor(ee_goals, device=sim.device)

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

    # Specifica la directory di salvataggio
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)

    def save_image(rgb, depth, index):
        rgb_path = os.path.join(save_dir, f"rgb_image_{index}.pt")
        depth_path = os.path.join(save_dir, f"depth_image_{index}.pt")
        torch.save(rgb, rgb_path)
        torch.save(depth, depth_path)


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


            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

            # Recuperare i dati dalla telecamera
            rgb_image = scene["camera"].data.output["rgb"].clone().detach()
            depth_image = scene["camera"].data.output["distance_to_image_plane"].clone().detach()

            # Salva l'immagine quando il target è raggiunto
            save_image(rgb_image.cpu(), depth_image.cpu(), current_goal_idx)

            ee_goals_eul = process_and_infer("/home/jonatha/IsaacLab/saved_images/rgb_image_{current_goal_idx}.pt", episode_natural_language_instruction)

            # Convert goals to quaternion representation
            ee_goals = convert_goals_to_quaternion(ee_goals_eul)
            ee_goals = torch.tensor(ee_goals, device=sim.device)
                    

            
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

        # print information from the sensors
        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
    



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
