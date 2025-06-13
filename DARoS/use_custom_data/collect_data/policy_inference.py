import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="take polciy and create lerobot data")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_camera_env_cfg import CartpoleRGBCameraEnvCfg
from isaaclab_tasks.manager_based.DARoS.multidoorman.multidoorman_env_cfg import MultidoormanEnvCfg_PLAY, MultidoormanCameraEnvCfg_PLAY

from DataRecoder import DataRecoder

def main():
    #works with rsl_rl given that it saves via torch.jit.load
    # rsl_rl makes working with camrea pain.... does this mean we need to work with rl games???
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    #env_cfg = MultidoormanEnvCfg_PLAY()
    env_cfg = MultidoormanCameraEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.sim.device = args_cli.device

    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    env = ManagerBasedRLEnv(cfg=env_cfg)

    num_episodes = 1000
    curr_episode = 0

    image_size = 24*24*3

    dt = env.physics_dt

    data_recorder = DataRecoder(dt=dt)

    # For Debug
    # obs, _ = env.reset()
    # print(obs["policy"][0].detach().cpu().numpy()[:-image_size].shape) # type: ignore
    # with torch.inference_mode():
    #     while simulation_app.is_running():
    #         print(type(obs["policy"]))
    #         action = policy(obs["policy"])
    #         #print("Action:", action)
    #         obs, rew, term, _, _ = env.step(action)
    #         #print("Observation: ", obs)

    # MDP
    obs, _ = env.reset() #s_0
    res = False
    with torch.inference_mode():
        while simulation_app.is_running():
            print(f"Collecting {curr_episode+1}/{num_episodes}")
            while curr_episode < num_episodes:
                if not res:
                    action = policy(obs["policy"]) #a_0
                    #print("Action: ", action)
                    new_obs, rew, term, res, _ = env.step(action) #s_t+1, r_t+1
                    
                    #print("Observation: ", obs)
                    #tem = env.episode_length_buf.cpu().item() >= env.max_episode_length 
                    # TODO: need to figure out how timetep work
                    data_recorder.write_data_to_buffer(observation=obs, action=action, reward=rew, 
                                                       termination_flag=res, cam_data=None, 
                                                       debug_stuff=[env.max_episode_length, env.episode_length_buf.cpu().item()],
                                                       image_size=image_size)
                    obs = new_obs #s_t <- s_t+1
                else:
                    data_recorder.dump_buffer_data()
                    obs, _ = env.reset()
                    data_recorder.reset()
                    curr_episode +=1
                    res = False
                    print(f"Collecting {curr_episode+1}/{num_episodes}")

if __name__ == "__main__":
    main()
    simulation_app.close()
