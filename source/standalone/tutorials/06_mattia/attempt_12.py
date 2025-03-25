"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""  ./isaaclab.sh -p source/standalone/tutorials/06_mattia/attempt_12.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --device keyboard --sensitivity 10  --random_factor 0.0 """

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
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
    # To avoid values outside the asin domain, clamp sinp between -1 and 1
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)

def build_trajectory_tensor_with_euler(trajectory_pos, trajectory_rot, gripper_command, ee_goals_eul, device='cpu'):
    """
    Builds a comprehensive tensor combining displacements (trajectory_pos) and rotations (trajectory_rot)
    converted to Euler angles and adds them to an existing list ee_goals_eul.

    Parameters:
        - trajectory_pos: array of trajectory positions (displacements)
        - trajectory_rot: array of trajectory rotations in quaternions
        - ee_goals_eul: current list with accumulated trajectory states (displacements + rotations in Euler)
        - device: the device to perform calculations on ('cpu' or 'cuda')
        
    Returns:
        - ee_goals_eul: updated list with new displacement and rotation values (in Euler)
    """
    # Ensure trajectory_pos and trajectory_rot are on the same device
    trajectory_pos = trajectory_pos.to(device)
    trajectory_rot = trajectory_rot.to(device)
    gripper_command = gripper_command.to(device)

    gripper_command = torch.tensor([gripper_command], dtype=torch.float32)

    # Iterate through each trajectory step
    for pos, rot_quaternion in zip(trajectory_pos, trajectory_rot):
        # Convert quaternion to Euler angles
        euler_angles = quaternion_to_euler(rot_quaternion)

        # Ensure euler_angles is on the same device
        euler_angles = euler_angles.to(device)

        # Combine displacement and rotation (in Euler) into a single array
        ee_goal_step = torch.cat([pos, euler_angles, gripper_command])

        # Add this step to the ee_goals_eul list
        ee_goals_eul.append(ee_goal_step.tolist())

    return ee_goals_eul

def save_trajectory_to_json(trajectory, save_dir, file_name="trajectory.json"):
    """
    Save the trajectory to a JSON file in a specified directory.

    Parameters:
        - trajectory: formatted trajectory list (list of lists for each trajectory step)
        - save_dir: directory to save the file
        - file_name: name of the file to save the trajectory (default "trajectory.json")
    """
    # Full path for the save file
    file_path = os.path.join(save_dir, file_name)

    # Encapsulate trajectory in a dictionary with the "actions" key
    trajectory_dict = {"trajectory": trajectory}

    # Save trajectory in JSON format with indentation
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
                os.unlink(file_path)  # delete the file
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

    # check environment name (for reach, we don't allow the gripper)
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
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
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

    # # Specify save directory
    # save_dir_front = "camera_view_front"
    # os.makedirs(save_dir_front, exist_ok=True)
    # clear_directory(save_dir_front)

    # save_dir_hand = "camera_view_hand"
    # os.makedirs(save_dir_hand, exist_ok=True)
    # clear_directory(save_dir_hand)

    # save_dir_side = "camera_view_side"
    # os.makedirs(save_dir_side, exist_ok=True)
    # clear_directory(save_dir_side)

    # save_dir_trajectory = "trajectory"
    # os.makedirs(save_dir_trajectory, exist_ok=True)
    # clear_directory(save_dir_trajectory)

    num_envs = list(range(args_cli.num_envs))

    # Correct function to save images in PNG format
    def save_image(rgb, save_dir, index, num_envs):
        # Convert PyTorch tensor to NumPy array
        rgb_np = rgb.cpu().numpy()  # Ensure it's on CPU and in NumPy format
        
        # If tensor has float values in [0, 1], convert to [0, 255]
        if rgb_np.dtype == np.float32 or rgb_np.dtype == np.float64:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        
        # Create PIL image
        image = Image.fromarray(rgb_np)
        
        # Create save path
        rgb_path = os.path.join(save_dir, f"rgb_image_{num_envs}_{index}.png")
        
        # Save image in PNG format
        image.save(rgb_path)

    count = 0
    index = 0

    # # Initialize empty list to accumulate trajectory states
    # trajectory = []

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            # print("delta_pose",delta_pose)
            # print("gripper_command",gripper_command)

            # # Input randomization (here noise is applied equally to all environments)
            # noise = np.random.uniform(-args_cli.random_factor, args_cli.random_factor, delta_pose.shape)
            # delta_pose += noise
            # # convert to torch
            # delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            
            # Add separate noise for each environment (here noise is different for each environment)
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device)
            delta_pose = delta_pose + torch.tensor(
                np.random.uniform(-args_cli.random_factor, args_cli.random_factor, (env.unwrapped.num_envs, delta_pose.shape[0])),
                device=delta_pose.device
            )

            #EXPLANATION: device=env.unwrapped.device: This part specifies which device to perform calculations on, such as CPU or GPU. It uses env.unwrapped.device, which in this case could point to a GPU (cuda:0, for example) or CPU. In other words, the command transfers the tensor to GPU or CPU, depending on where the environment is allocated.
            #NOTE: The delta_pose still doesn't have gripper commands, so it's fine to take it all

            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)
            # print(actions)
            # print("gripper command",actions[0,-1])    #the .item() function is used to extract the value from a 1x1 tensor
            # trajectory_pos = env.step(actions)[0]["trajb"]["traj_pos"]
            # print("4",trajectory_pos)
            
            # apply actions
            env.step(actions)  #This is where camera data and general environment data comes from

            ## Explanation of what's inside rgb_image_front
            # rgb_image_front = env.step(actions)[0]["rgbd"]["rgb_front"][num_envs]
            # print("total",rgb_image_front)
            # print("-------------------------------")
            # rgb_image_front = env.step(actions)[0]["rgbd"]["rgb_front"][num_envs[0]]
            # print("0",rgb_image_front)
            # print("-------------------------------")
            # rgb_image_front = env.step(actions)[0]["rgbd"]["rgb_front"][num_envs[1]]
            # print("1",rgb_image_front)
            # print("-------------------------------")
            # rgb_image_front = env.step(actions)[0]["rgbd"]["rgb_front"][num_envs[2]]
            # print("2",rgb_image_front)
            # print("-------------------------------")
            # trajectory_pos = env.step(actions)[0]["trajb"]["traj_pos"][0]
            # print("1",trajectory_pos)
            # print("-------------------------------")
            # trajectory_rot = env.step(actions)[0]["trajb"]["traj_rot"][0]
            # print("2",trajectory_rot)
            # print("-------------------------------")
            # trajectory_pos = env.step(actions)[0]["trajb"]["traj_pos"]
            # print("3",trajectory_pos)
            # print("-------------------------------")
            # trajectory_rot = env.step(actions)[0]["trajb"]["traj_rot"]
            # print("4",trajectory_rot)

            if count % 25 == 0:
                index += 1
                count = 0

                # print("-------------------------------")
                # print(env.step(actions)[0]["rgbd"]["rgb_front"].shape)
                # print("-------------------------------")
                # print(env.step(actions)[0]["rgbd"]["rgb_hand"].shape)
                # print("-------------------------------")

                # TRAJECTORY SIMULATION ON A SINGLE ENVIRONMENT

                # # Simulate an action and get new trajectory coordinates (positions and quaternions)
                # trajectory_pos = env.step(actions)[0]["trajb"]["traj_pos"][1]  # Positions
                # trajectory_rot = env.step(actions)[0]["trajb"]["traj_rot"][0]  # Quaternions
                # gripper_cmd = actions[0,-1]    #this is the action and not the gripper observation

                # print(gripper_cmd)

                # Build comprehensive tensor for trajectory, converting rotations to Euler angles
                # trajectory = build_trajectory_tensor_with_euler(trajectory_pos, trajectory_rot, gripper_cmd, trajectory)

                # print("tensor",trajectory)

            #     for i in num_envs:
            #     # Get camera data
            #         rgb_image_front = env.step(actions)[0]["rgbd"]["rgb_front"][i]
            #         rgb_image_hand = env.step(actions)[0]["rgbd"]["rgb_hand"][i]
            #         rgb_image_side = env.step(actions)[0]["rgbd"]["rgb_side"][i]

            #         # Save image when target is reached
            #         save_image(rgb_image_front.cpu(), save_dir_front, index, i)
            #         save_image(rgb_image_hand.cpu(), save_dir_hand, index, i)
            #         save_image(rgb_image_side.cpu(), save_dir_side, index, i)

            # count += 1
            # # Save trajectory to file in specified directory
            # save_trajectory_to_json(trajectory, save_dir_trajectory, file_name="trajectory.json")

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()