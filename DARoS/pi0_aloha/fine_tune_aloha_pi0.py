from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType

DATASET_PATH = "lerobot/aloha_sim_transfer_cube_human_image"

def train():
    output_directory = Path("outputs/train/example_aloha_transfer_box_pi0")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')
    #device = torch.device('cpu')

    training_steps = 500
    log_freq = 1

    # this is something that we need to change for our own robot.
    dataset_metadata = LeRobotDatasetMetadata(DATASET_PATH)
    # print(type(dataset_metadata))
    # exit()
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = PI0Config(input_features=input_features, output_features=output_features, n_obs_steps=4)

    policy = PI0Policy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    delta_timestamps = {
        "observation.images.top": [-0.3, -0.2, -0.1, 0.0],
        "observation.state": [-0.3, -0.2, -0.1, 0.0],

        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    dataset = LeRobotDataset(DATASET_PATH, delta_timestamps=delta_timestamps)
    

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

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

    policy.save_pretrained(output_directory)

if __name__ == "__main__":
    train()
