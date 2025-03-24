
    # Usage
'''    ./isaaclab.sh -p source/standalone/tutorials/06_mattia/prova_stampa_immagini.py  --num_envs 1  '''


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
from PIL import Image
import numpy as np
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

def clear_directory(directory):
    """Cancella tutti i file nella directory specificata."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # cancella il file
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]), #rot=[0.707, 0, 0, 0.707]
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(1.65, 1.80, 1)
        ),
    )

    # # Franka Panda robot
    # robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Robot"
    # )

    # camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,  # Aumentare l'apertura orizzontale per un campo visivo più ampio
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.8, 0.3, 0.8),      #pos=(1.8, 0.3, 0.8)world, 
            rot=(1.0, 0.4, 0.45, 0.7),   #  rot=(1.0, 0.4, 0.45, 0.7)opengl rot=(1.0, -0.25, 0.1, 0.9)world
            convention="opengl",
        ),
    )


    # camera_side_bridge = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Table/side_cam_Bridge",
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



# Specifica la directory di salvataggio
save_dir = "camera_images"
os.makedirs(save_dir, exist_ok=True)
clear_directory(save_dir)



#Funzione corretta per salvare le immagini in formato PNG
def save_image(rgb, save_dir, count):
    # Converti il tensore PyTorch in un array NumPy
    rgb_np = rgb.cpu().numpy()  # Assicurati che sia su CPU e in formato NumPy
    
    # Se il tensore ha valori float in [0, 1], convertili a [0, 255]
    if rgb_np.dtype == np.float32 or rgb_np.dtype == np.float64:
        rgb_np = (rgb_np * 255).astype(np.uint8)
    
    # Crea l'immagine PIL
    image = Image.fromarray(rgb_np)
    
    # Crea il percorso di salvataggio
    rgb_path = os.path.join(save_dir, f"rgb_image_{count}.png")
    
    # Salva l'immagine in formato PNG
    image.save(rgb_path)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    # robot = scene["robot"]
    # camera=scene["camera"]


    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    current_goal_idx = 0
    count=0


    # Simulation loop
    while simulation_app.is_running():
        # reset                          
        if count % 25 == 0:
            count=0

            # Recuperare i dati dalla telecamera
            rgb_image = scene["camera"].data.output["rgb"].clone().detach()[0]
            # depth_image = scene["camera"].data.output["distance_to_image_plane"].clone().detach()

            # # Salva l'immagine quando il target è raggiunto
            save_image(rgb_image.cpu(), save_dir, current_goal_idx)
            current_goal_idx += 1
            
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)



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












