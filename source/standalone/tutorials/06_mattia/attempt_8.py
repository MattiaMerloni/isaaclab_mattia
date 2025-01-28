    # Usage
'''   ./isaaclab.sh -p source/standalone/tutorials/06_mattia/attempt_8.py  --num_envs 1   '''


import argparse
from omni.isaac.lab.app import AppLauncher
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(1.65, 1.80, 1)
        ),
    )
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.5, 1), rot=(0.0, 0.0, 1.0, 1.0)),
    )
    cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/cylinder.usd", 
            scale=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.75, 0.75), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.6, 0.1)),
    )
    parallelepipedo = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Parallelepipedo",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonatha/IsaacLab/usd_files_mattia/parallelepipedo.usd", 
            scale=(0.1, 0.1, 0.1),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.75, 0.1)),
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

class EEGoalsSubscriber(Node):
    def __init__(self):
        super().__init__('ee_goals_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'ee_goals',
            self.listener_callback,
            10)
        self.ee_goals = torch.zeros((1, 7))

    def listener_callback(self, msg):
        self.ee_goals = torch.tensor(msg.data, device=sim.device).view(1, -1)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, ee_goals_subscriber: EEGoalsSubscriber):
    robot = scene["robot"]
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    current_goal_idx = 0
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals_subscriber.ee_goals

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()
    count = 0

    joint_pos = robot.data.default_joint_pos.clone() 
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    while simulation_app.is_running():
        if count % 150 == 0:
            count = 0
            robot.reset()
            ik_commands[:] = ee_goals_subscriber.ee_goals
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
        else:
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        
        rclpy.spin_once(ee_goals_subscriber, timeout_sec=0.01)

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    
    rclpy.init(args=None)
    ee_goals_subscriber = EEGoalsSubscriber()

    try:
        run_simulator(sim, scene, ee_goals_subscriber)
    finally:
        ee_goals_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()