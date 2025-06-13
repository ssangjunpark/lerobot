from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType


import os
import datetime
SAVE_DIR = os.getcwd() + "/PolicyFromIsaac/outputs/train/" + str(datetime.datetime.now())
DATA_DIR = "/home/isaac/Documents/Github/lerobot/DARoS/use_custom_data/LeRobotData"

def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path(SAVE_DIR)
    
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 5000
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    
    dataset_metadata = LeRobotDatasetMetadata(repo_id="lerobot/libero_goal_image")
    #dataset_metadata = LeRobotDatasetMetadata(repo_id="lerobot/my_pusht", root=DATA_DIR)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    #print(output_features)
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    #print(input_features)
    # cfg = DiffusionConfig(input_features=input_features, output_features=output_features, crop_shape=None)
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    print(input_features)

    # We can now instantiate our policy with this config and the dataset stats.
    #print(dataset_metadata.stats)
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    delta_timestamps = {
        "observation.images.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        #"observation.images.wrist_image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.images.image": [-0.1, 0.0],
        #"observation.images.wrist_image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # dataset = LeRobotDataset(repo_id="lerobot/my_pusht", root=DATA_DIR, delta_timestamps=delta_timestamps)
    dataset = LeRobotDataset(repo_id="lerobot/libero_goal_image", delta_timestamps=delta_timestamps)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)
    print(SAVE_DIR)
if __name__ == "__main__":
    main()