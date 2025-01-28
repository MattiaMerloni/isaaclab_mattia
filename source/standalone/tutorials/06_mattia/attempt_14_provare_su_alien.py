

"""  ./isaaclab.sh -p source/standalone/tutorials/06_mattia/attempt_14_provare_su_alien.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --random_factor 0.0 """

# qui provo a mettere insieme i due script senza usare ROS2 

# assumiamo di prendere num_envs=1 (sempre e solo un environment) - NUM_ENVS=1


"""Initialize the RT1-X model"""
import tensorflow as tf
import tensorflow_datasets as tfds
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts
from IPython import display
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow_hub as hub

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

import argparse

from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# parser.add_argument("--device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
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


import carb

from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.scene import InteractiveScene


import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg


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

def quaternion_to_euler(quaternion):
    """
    Convert a quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) using PyTorch.
    """
    w= quaternion[0]
    x= quaternion[1]
    y= quaternion[2]
    z= quaternion[3]

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
    """Cancella tutti i file nella directory specificata."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # cancella il file
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



    # reset environment
    env.reset()

    # Specifica la directory di salvataggio
    save_dir_front = "camera_view_front"
    os.makedirs(save_dir_front, exist_ok=True)
    clear_directory(save_dir_front)


    num_envs = list(range(args_cli.num_envs))

    def save_image(rgb, save_dir, index, num_envs):
        rgb_path = os.path.join(save_dir, f"rgb_image_{num_envs}_{index}.png")
        torch.save(rgb, rgb_path)


    count = 0
    index = 0



    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            action_tensor=process_and_infer("/home/jonatha/IsaacLab/camera_view_front/rgb_image_0_{index}.pt", episode_natural_language_instruction, natural_language_embedding)

            delta_pose = action_tensor[:6]
            gripper_command = action_tensor[6]

            
            # Aggiungi rumore separato per ciascun environment (qui il noise viene dato diverso per ogni environment)
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device)
            delta_pose = delta_pose + torch.tensor(
                np.random.uniform(-args_cli.random_factor, args_cli.random_factor, (env.unwrapped.num_envs, delta_pose.shape[0])),
                device=delta_pose.device
            )

            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)

            
            # apply actions
            env.step(actions)   #è da qui che tiro fuori i dati della telecamera ed in generale di tutto l'ambiente


            if count % 25 == 0:

                index += 1
                count = 0

                # Recuperare i dati dalla telecamera
                rgb_image_front =  env.step(actions)[0]["rgbd"]["rgb_front"][0]

                # Salva l'immagine quando il target è raggiunto
                save_image(rgb_image_front.cpu(),save_dir_front,index,0)

            count += 1
         

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()