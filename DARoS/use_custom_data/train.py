from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from lerobot.configs.types import FeatureType


import os
import datetime
SAVE_DIR = os.getcwd() + "/PolicyFromIsaac/outputs/train/" + str(datetime.datetime.now())

def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path(SAVE_DIR)
    
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 500
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    
    #dataset_metadata = LeRobotDatasetMetadata(repo_id="lerobot/libero_goal_image")
    dataset_metadata = LeRobotDatasetMetadata(repo_id="ssangjunpark/daros29_0719")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    #print(output_features)
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(input_features)
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)


    # cfg = DiffusionConfig(
    #     input_features=input_features, 
    #     output_features=output_features,

    #     )



    #cfg = SmolVLAConfig(input_features=input_features, output_features=output_features)
    #cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    #print(input_features)

    
    print(dataset_metadata.stats)
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # delta_timestamps = {
    #     "observation.images.top": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "observation.images.hand1": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "observation.images.hand2": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],

    #     #"observation.images.wrist_image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    # }

    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.images.top_camera": [-0.1, 0.0],
        "observation.images.left_camera": [-0.1, 0.0],
        "observation.images.right_camera": [-0.1, 0.0],
        #"observation.images.wrist_image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    #dataset = LeRobotDataset(repo_id="lerobot/libero_goal_image", delta_timestamps=delta_timestamps)
    dataset = LeRobotDataset(repo_id="ssangjunpark/daros29_0719", delta_timestamps=delta_timestamps)

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

    #debug
    # import matplotlib.pyplot as plt
    # for batch in dataloader:
    #     print(batch["observation.images.top_camera"][0])
    #     print(type(batch["observation.images.top_camera"][0].numpy()[0]))

    #     plt.imshow(batch["observation.images.top_camera"][0].numpy()[0].transpose(1,2,0))
    #     plt.show()

    #     plt.imshow(batch["observation.images.left_camera"][0].numpy()[0].transpose(1,2,0))
    #     plt.show()

    #     plt.imshow(batch["observation.images.right_camera"][0].numpy()[0].transpose(1,2,0))
    #     plt.show()
    #     exit()

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


"""

How Image data saving is differnt 
Mine: {'observation.images.image': [{'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x18\x00\x00\x00\x18\x08\x02\x00\x00\x00o\x15\xaa\xaf\x00\x00\x02\x1dIDATx\x9c\xb5\x93[O\xe30\x10\x85\xc7w;\x0eMU\x01\xa5<\x81\x84\xf8\xff\x7f\x88\xa7J\x80*hEIH\x93\xd83\xfb0KwYn)\xd2\x9e\'+\xce|>s<\x16\xd7\xd7\xd7B\x08\xf8KDt{{[\x14EY\x96Z\xeb\x7fv?\x14\x11i!\x84\x10\x02\x11\xf7\x05\x88\xd8\xf7\xbd1\x86?\x8e\x01\x01\x80\xe6\xca\xe5ryrr\xa2\xb5\xae\xebz\xb5Z\x01@\xdf\xf7\x9b\xcd\xe6\xbd\xdfOA)%\x00`\n[\x03\x00\xa5\xd4b\xb1p\xce\x8d\xa4\x00\x80\xbc\xb9\xb9\xa9\xeb\xda\x18#\xa5\x14B\x18c\xb4\xd6UUYk\x95R\x07\x80\x00 \xe7LD}\xdf7Mc\x8c9;;\xdbn\xb7///)%":\x00\xd44\xcd0\x0c\xcf\xcf\xcfwww\x0cM)I)\xa5\x94#)\x00\xf0\xa9\xf3\xab\xab+\xad\xf5H\n\x11\x1dp\xe6\xd7\xfa\x0f\xa0\x10\xc2l6\xe3\xf5|>GDD\x1c\x1f\xf6\x9f\x14\x8c1\xde\xfb\xc9d\xa2\x942\xc6\xa4\x94r\xce\xd6Z\x1e\x8boAo\xfe\x98\xcdf1F"zxxp\xce\x85\x10\x00\xa0(\nk\xed\xd7\x14"zs/\xeb\xf5\xdaZ\x8b\x88\xbbWu]wzz:\x9dN\x85\x10\xdc\xe6\x87\xee\x10q\xd4\xe0\x9e\x9f\x9f#\xe2\xfd\xfd}\x08!\xc6\xe8\xbd\x7f}LB\x08h\xdb\xf6\xf1\xf1Q\x02\x80\x94r\xb1Xp#\xac\xf9|>\x9dNy\xed\xbd\xcf9#\xe2\xbeG6;\x0cC\xce)\xe7\xfc\xf4\xf4\x04\x1c\xf6d2q\xce\xc5\x18\xdb\xb6\xe5J\xe7\x9cRj\xbb\xdd\x86\x10\x9cs\xdcQ\x8c\x91\x1bl\x9a\x86\x88\x8a\xa2\xe0\x93\xf8\xd9\xffn\xad,\xcb\xba\xae\xf7\x8e\xbc\xf7\xbb\xdd\x0e\x00B\x08\xde{DJi\xe8\xba\x8ekXUU\xf1\x0b\x1d\x86!\xe7\xfcM:R\xca\x18\xa3R\xea\xc3\xad1\xf9~C\x7f\xffQ\\\\\\ \xe2f\xb39::RJ\xadV\xab\xae\xeb~B\xb7\xd6:\xe7\xaa\xaa\xe28\x8e\x8f\x8f\x01\xa0,\xcb}\x96cAm\xdb\xb6m\xbb^\xaf\x01@\x08Q\x14\xc5\xe5\xe5%O\n\xdf\xd7H\xe9\xe5r\xc9+D\xe4P\xa5\x94<\xdc\x079\xfa\x05Lc\x0f\'\xd9p\x1a\x07\x00\x00\x00\x00IEND\xaeB`\x82', 'path': 'frame_000130.png'}], 'observation.state': [[-0.386884480714798, -0.22256365418434143, 0.4942839741706848, -0.03101007267832756, 0.05616336315870285, 0.019354507327079773, -0.27371159195899963, -0.05544153228402138, 0.7744953036308289, -0.1722692996263504, 0.6471217274665833, -0.23016709089279175, 0.39692604541778564, 0.0020466807764023542, 0.09844641387462616, -0.053093068301677704, -1.0391792058944702, 0.12128237634897232, -0.66517174243927, -0.11359284818172455, 0.0, 0.0, 0.0, -1.8967430293059806e-08, -1.2402049165416429e-08, -5.986487749964908e-09, 0.9999999403953552, 0.0, 0.0, 0.0, 1.8547537949942239e-09, 4.644485632665651e-24, -0.8792862892150879, -0.5521697402000427, 1.1285834312438965, 0.08377248793840408, 0.36169594526290894, -0.056505873799324036, 0.9185618162155151, 0.483196496963501, -0.09024648368358612, -1.909463882446289, 0.5149480700492859, -1.3637309074401855, -0.47714582085609436]], 'action': [[-1.174290418624878, -0.48887377977371216, 1.3119597434997559, 0.20554468035697937, -0.056134603917598724, -0.0692051500082016, 0.6996654272079468, -0.0019170045852661133, 0.38763028383255005, -2.631547689437866, 0.6172835230827332, -1.284921646118164, 0.194821298122406]], 'timestamp': [1.300000000000001], 'episode_index': [8], 'frame_index': [130], 'index': [2130], 'next.reward': [-0.5800090432167053], 'next.done': [False], 'task_index': [0], '__index_level_0__': [130]}
Theirs: {'observation.images.wrist_image': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x256 at 0x7E4A9477E9E0>, <PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x256 at 0x7E4A9477E980>]}
{'observation.state': [[-0.11203701049089432, 0.041501954197883606, 0.9388449192047119, 3.063246488571167, -0.047417834401130676, -0.2274056077003479, 0.002295437967404723, -0.01092884037643671], [-0.111441969871521, 0.04200008884072304, 0.9469561576843262, 3.062631368637085, -0.05313443765044212, -0.2294313758611679, 0.002358700381591916, -0.010011539794504642]]}
{'observation.images.image': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x256 at 0x7E4A9477D9C0>, <PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x256 at 0x7E4A9477D960>]}
{'observation.state': [[0.069951631128788, 0.21117092669010162, 0.9116916060447693, 3.1422224044799805, 0.0064414385706186295, 0.28978243470191956, 0.040860049426555634, -0.03899764642119408], [0.07028815150260925, 0.21361015737056732, 0.9117487668991089, 3.1425318717956543, 0.006153956986963749, 0.2889854907989502, 0.04085233062505722, -0.039039451628923416]]}
{'action': [[0.02946428582072258, 0.02410714328289032, 0.6455357074737549, 0.0, 0.0, -0.028928572311997414, 1.0], [0.09107142686843872, 0.0053571430034935474, 0.7419642806053162, 0.0, 0.0, -0.027857143431901932, 1.0], [0.1205357164144516, 0.008035714738070965, 0.8651785850524902, 0.0, -0.0053571430034935474, -0.029999999329447746, 1.0], [0.12589286267757416, 0.02142857201397419, 0.9375, 0.0, -0.013928571715950966, -0.029999999329447746, 1.0], [0.12321428209543228, 0.01607142947614193, 0.9375, 0.0, -0.023571427911520004, -0.024642856791615486, 1.0], [0.11785714328289032, 0.0, 0.9375, 0.0, -0.024642856791615486, -0.019285714253783226, 1.0], [0.0803571417927742, 0.0, 0.9375, 0.0, -0.003214285708963871, -0.014999999664723873, 1.0], [0.07767856866121292, -0.0053571430034935474, 0.9375, 0.0, 0.0, -0.02142857201397419, 1.0], [0.08839285373687744, -0.07767856866121292, 0.9375, 0.0, 0.0, -0.024642856791615486, 1.0], [0.1875, -0.21696428954601288, 0.9375, 0.0, 0.0, -0.040714286267757416, 1.0], [0.22232143580913544, -0.3187499940395355, 0.9375, 0.0, 0.0, -0.04714285582304001, 1.0], [0.2142857164144516, -0.43660715222358704, 0.9375, 0.0, 0.0, -0.04500000178813934, 1.0], [0.20357142388820648, -0.6000000238418579, 0.9375, 0.0, 0.0, -0.04821428656578064, 1.0], [0.2008928507566452, -0.6321428418159485, 0.9375, 0.0, 0.0, -0.0503571443259716, 1.0], [0.19285714626312256, -0.6401785612106323, 0.9160714149475098, -0.007499999832361937, 0.0, -0.0503571443259716, 1.0], [0.19821429252624512, -0.6508928537368774, 0.8785714507102966, -0.012857142835855484, 0.0010714285308495164, -0.04928571358323097, 1.0]]}
{'observation.images.image': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x256 at 0x7E4A9477C700>, <PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x256 at 0x7E4A9477C6A0>]}
{'observation.images.image': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x256 at 0x7E4A94783190>], 'observation.images.wrist_image': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x256 at 0x7E4A94783310>], 'observation.state': [[-0.09577780961990356, -0.02547818422317505, 0.9576629400253296, 3.1391305923461914, -0.1070365458726883, 0.020229879766702652, 0.021589290350675583, -0.020944859832525253]], 'action': [[0.31339284777641296, 0.14464285969734192, 0.20357142388820648, -0.018214285373687744, 0.1542857140302658, -0.007499999832361937, -1.0]], 'timestamp': [7.699999809265137], 'frame_index': [77], 'episode_index': [369], 'index': [45005], 'task_index': [3]}


"""