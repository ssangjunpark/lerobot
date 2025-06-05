from pathlib import Path

import gym_aloha  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

DATASET_PATH = "lerobot/aloha_sim_transfer_cube_human_image"

output_directory = Path("outputs/eval/example_aloha_transfer_box_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

device = "cuda"

pretrained_policy_path = "outputs/train/example_aloha_transfer_box_diffusion"

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

env = gym.make(
    "gym_aloha/AlohaTransferCube-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)

policy.reset()
numpy_observation, info = env.reset(seed=42)

rewards = []
frames = []

frames.append(env.render())

step = 0
done = False
while not done:
    state = torch.from_numpy(numpy_observation["agent_pos"])
    print(state.shape)
    image = torch.from_numpy(numpy_observation["pixels"]["top"])
    print(image.shape)

    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    observation = {
        "observation.state": [state * 4],
        "observation.image": [image * 4],
    }

    with torch.inference_mode():
        action = policy.select_action(observation)

    numpy_action = action.squeeze(0).to('cpu').numpy()

    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    rewards.append(reward)
    frames.append(env.render())

    done = terminated | truncated | done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")

fps = env.metadata["render_fps"]

video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")