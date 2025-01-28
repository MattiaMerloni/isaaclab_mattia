


'''    ./isaaclab.sh -p source/standalone/tutorials/06_mattia/attempt_7_openvla_check_open_drawer.py  --num_envs 1  '''

""" SCRIPT PER IL CHECK DEL MODELLO OPENVLA"""

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



# questa funzione mi servirà quando gli darò in input le azioni predette da RT1
def euler_to_quaternion(roll, pitch, yaw):    # DA USARE QUANDO USI UN FILE.json  #DA DECOMMENTARE
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion using PyTorch.
    """

    # Convert input angles to tensors
    roll = roll.clone().detach().float()      #devo usare clone().detach() perchè roll,pitch e yaw sono tensori di PyTorch e non posso fare operazioni direttamente su di essi
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


# def euler_to_quaternion(roll, pitch, yaw):   # DA USARE SE COPI ED INCOLLI IL TENSORE DIRETTAMENTE SULLO SCRIPT  #DA DECOMMENTARE
#     """
#     Convert Euler angles (roll, pitch, yaw) to a quaternion using PyTorch.
#     """

#     # Convert input angles to tensors
#     roll = torch.tensor(roll)     #devo usare clone().detach() perchè roll,pitch e yaw sono tensori di PyTorch e non posso fare operazioni direttamente su di essi
#     pitch = torch.tensor(pitch)
#     yaw = torch.tensor(yaw)

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
        coord= torch.cat((pos,rot))   #DA DECOMMENTARE   #DA USARE CON FILE.json

        # Exclude the gripper state and only append position and quaternion
        # ee_goals_quat.append(pos + rot.tolist())  #DA DECOMMENTARE   # DA USARE CON SCRIPT
        ee_goals_quat.append(coord.tolist())   #DA DECOMMENTARE  #DA USARE CON FILE.json
        
    return ee_goals_quat



@configclass
class CabinetSceneCfg(InteractiveSceneCfg):


    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0.0, 0.0))    #(0, 0.0, -1.05)   #avevo settato il groud plane -1.05 sotto al rotto sull'asse z
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

    # Franka Panda robot
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(   #MI SA CHE CONVIENE COMUNQUE PARTIRE DALLA BASE POSITION SE VOGLIAMO UTILIZZARE QUESTO SCRIPT PER FARE IL CHECK
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
            scale=(0.0006, 0.0006, 0.0006),     #ricorda che il comando scale è preso rispetto al sistema di riferimento relativo dell'oggetto quindi se lo ruotiamo x,y e z si invertiranno di coseguenza 
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
            horizontal_aperture=30,  # Aumentare l'apertura orizzontale per un campo visivo più ampio
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

    camera_side_bridge = CameraCfg(
        prim_path="{ENV_REGEX_NS}/side_cam_Bridge",
        update_period=0.1,
        height=224,
        width=224,
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

    # Approxima il gripper_state all'intero più vicino
    gripper_command = round(gripper_state.item())  
    gripper_command = 1 if gripper_command == 1 else 0  # il gripper_commnand è settato ad 1 e quindi il gripper è aperto se il gripper_state è -1, altrimenti è chiuso

    gripper_commands = [gripper_command] * len(gripper_joint_ids)

    # Converti i comandi in un tensor di PyTorch e trasferiscilo sulla GPU se necessario
    gripper_commands_tensor = torch.tensor(gripper_commands, dtype=torch.float32, device='cuda')

    # Imposta i target di posizione dei giunti del gripper
    robot.set_joint_position_target(gripper_commands_tensor, joint_ids=gripper_joint_ids)



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    robot = scene["robot"]


    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")    #'dls' sta per dumped least square che descrive il metodo ai minimi quadrati smorzati
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))





    # # vettore dei goal rappresentato nella forma corretta per essere utilizzato con RT1-X 
    # ee_goals_eul = [
    #     [0.4, 0.3, 0.7, torch.pi / 2, 0, torch.pi / 2, 1.0],
    #     [0.4, -0.3, 0.6, torch.pi / 2, torch.pi / 2, 0.0, 1.0],
    #     [0.7, 0, 0.5, 0, torch.pi / 2, 0.0, -1.0],
    # ]


    # # TEST PER VERIFICARE LA TRAIETTORIA DEL TELEOP
    # with open('trajectory/trajectory.json', 'r') as file:
    #     data = json.load(file)
    #     traj = data['trajectory_0']


    # TEST PER VERIFICARE LA TRAIETTORIA DI OPENVLA 
    with open('trajectory_test_open_drawer.json', 'r') as file:
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
            # reset joint state
            # joint_pos = robot.data.default_joint_pos.clone() 
            # joint_vel = robot.data.default_joint_vel.clone()
            # robot.write_joint_state_to_sim(joint_pos, joint_vel)


            # robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]      #qui invece definiamo la posizione dell'end-effector nel cartesian space
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()   #qui definiamo il controllo dei joint (che in questo caso sono 7 per il Franka Panda)

            # reset controller
            # diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

            

            
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
        gripper_state = ee_goals_eul[current_goal_idx][-1]  # Gripper state from the goal 
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

        ## per stampare la pose dell'end-effector
        # print(f"ee_pose_w: {ee_pose_w}")
        # print("---------------------------------")

        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

    



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])  
    # Design scene
    scene_cfg = CabinetSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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












