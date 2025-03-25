"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

# Franka pick the red cuboid / place the coke can in the pan
"""  ./isaaclab.sh -p source/standalone/tutorials/06_mattia/attempt_12_dataset.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --device keyboard --sensitivity 10  --random_factor 0.0 """

# Franka open the drawer
""" ./isaaclab.sh -p source/standalone/tutorials/06_mattia/attempt_12_dataset.py --task Isaac-Open-Drawer-Franka-IK-Rel-v0 --num_envs 1 --device keyboard --sensitivity 10  --random_factor 0.0 """

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard/Mouse/Gamepad teleoperation for Isaac Lab environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--random_factor", type=float, default=0.05, help="Amount of randomization to apply to inputs.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import os
import numpy as np
import json
from PIL import Image

import carb

from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.scene import InteractiveScene

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

def quaternion_to_euler(quaternion):
    """
    Convert a quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) using PyTorch.
    """
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]

    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # To avoid values outside the domain of asin, limit sinp between -1 and 1
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)

def build_trajectory_tensor_with_euler(trajectory_pos, trajectory_rot, gripper_command, device='cpu'):
    # Ensure that trajectory_pos and trajectory_rot are on the same device
    trajectory_pos = trajectory_pos.to(device)
    trajectory_rot = trajectory_rot.to(device)
    gripper_command = gripper_command.to(device)

    gripper_command = torch.tensor([gripper_command], dtype=torch.float32)

    # Iterate through each step of the trajectory
    for pos, rot_quaternion in zip(trajectory_pos, trajectory_rot):
        # Convert the quaternion into Euler angles
        euler_angles = quaternion_to_euler(rot_quaternion)

        # Ensure that euler_angles is on the same device
        euler_angles = euler_angles.to(device)

        # Combine displacement and rotation (in Euler angles) into a single array
        ee_goal_step = torch.cat([pos, euler_angles, gripper_command])

    return ee_goal_step

def save_trajectory_to_json(trajectory, save_dir, num_envs, file_name="trajectory.json"):
    """
    Save each trajectory in a JSON file, with a separate dictionary for each environment.

    Parameters:
        - trajectory: list of trajectories for each environment (list of lists for each environment)
        - save_dir: directory where to save the file
        - num_envs: number of environments (integer)
        - file_name: name of the file in which to save the trajectory (default: "trajectory.json")
    """

    # Complete path for the save file
    file_path = os.path.join(save_dir, file_name)

    # Create a dictionary where each key represents a specific environment
    trajectory_dict = {}

    # Iterate over each environment
    for i in num_envs:
        if i < len(trajectory):  # Check that there is a trajectory for this environment
            trajectory_data = trajectory[i]
            if isinstance(trajectory_data, list):
                # If it's a list of tensors, convert each tensor into a list
                trajectory_dict[f"trajectory_{i}"] = [
                    t.cpu().numpy().tolist() if isinstance(t, torch.Tensor) else t
                    for t in trajectory_data
                ]
            else:
                # If it's a single tensor
                trajectory_dict[f"trajectory_{i + 1}"] = trajectory_data.cpu().numpy().tolist() if isinstance(trajectory_data, torch.Tensor) else trajectory_data

    # Save the trajectory dictionary in JSON format with indentation
    with open(file_path, 'w') as f:
        json.dump(trajectory_dict, f, indent=4)  # 4-space indentation for readability

def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)
    
def clear_directory(directory):
    """Delete all files in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete the file
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        carb.log_warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.01 * args_cli.sensitivity
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # Specify the save directory
    save_dir_front = "camera_view_front"
    os.makedirs(save_dir_front, exist_ok=True)
    clear_directory(save_dir_front)

    # save_dir_hand = "camera_view_hand"
    # os.makedirs(save_dir_hand, exist_ok=True)
    # clear_directory(save_dir_hand)

    save_dir_side = "camera_view_side"
    os.makedirs(save_dir_side, exist_ok=True)
    clear_directory(save_dir_side)

    save_dir_trajectory = "trajectory"
    os.makedirs(save_dir_trajectory, exist_ok=True)
    clear_directory(save_dir_trajectory)

    num_envs = list(range(args_cli.num_envs))     #Remember that num_envs is a list of numbers from 0 to num_envs-1 and therefore is NOT an integer

    def save_image(rgb, save_dir, index, num_envs):
        """Function to correctly save images in PNG format"""
        # Convert the PyTorch tensor into a NumPy array
        rgb_np = rgb.cpu().numpy()  # Ensure it is on CPU and in NumPy format

        # If the tensor has float values in [0, 1], convert them to [0, 255]
        if rgb_np.dtype == np.float32 or rgb_np.dtype == np.float64:
            rgb_np = (rgb_np * 255).astype(np.uint8)

        # Create the PIL image
        image = Image.fromarray(rgb_np)

        # Create the save path
        rgb_path = os.path.join(save_dir, f"rgb_image_{num_envs}_{index}.png")

        # Save the image in PNG format
        image.save(rgb_path)

    count = 0
    index = 0

    # Initialize the empty list to accumulate trajectory states
    trajectory = [[] for _ in num_envs]  

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
                
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")

            # Add separate noise for each environment (here the noise is different for each environment)
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device)
            delta_pose = delta_pose + torch.tensor(
                np.random.uniform(-args_cli.random_factor, args_cli.random_factor, (env.unwrapped.num_envs, delta_pose.shape[0])),
                device=delta_pose.device
            )

            # EXPLANATION: device=env.unwrapped.device: This part specifies on which device to perform the computation, for example CPU or GPU. It uses env.unwrapped.device, which in this case could point to a GPU (cuda:0, for example) or to the CPU. In other words, the command transfers the tensor to GPU or CPU, depending on where the environment is allocated
            # NOTE: still in delta_pose we don't have the gripper commands, so it's fine to take it all

            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)
            
            # apply actions
            env.step(actions)   #This is where we get the camera data and generally all the environment data

            if count % 25 == 0:    #it was at 25 (2.4 Hz) before 
                index += 1
                count = 0

                # Build the overall tensor for the trajectory, converting rotations to Euler angles
                for i in num_envs:
                    # Get camera data
                    rgb_image_front =  env.step(actions)[0]["rgbd"]["rgb_front"][i]
                    rgb_image_side=  env.step(actions)[0]["rgbd"]["rgb_side"][i]
                            
                    trajectory_pos = env.step(actions)[0]["trajb"]["traj_pos"][i]  # Positions
                    trajectory_rot = env.step(actions)[0]["trajb"]["traj_rot"][i]  # Quaternions
                    gripper_cmd=actions[0,-1]
                    trajectory_tensor = build_trajectory_tensor_with_euler(trajectory_pos, trajectory_rot, gripper_cmd)
                    trajectory[i].append(trajectory_tensor)

                    # Save the image when the target is reached
                    save_image(rgb_image_front.cpu(),save_dir_front,index,i)
                    save_image(rgb_image_side.cpu(),save_dir_side,index,i)

            count += 1

            # Save the trajectory to a file in the specified directory
            save_trajectory_to_json(trajectory, save_dir_trajectory, num_envs, file_name="trajectory.json")

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()