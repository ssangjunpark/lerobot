import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
from PIL import Image

SAVE_DIR = os.getcwd() + "/LeRobotData/"

class DataRecoder:
    def __init__(self, dt):
        self.log_dir = SAVE_DIR + "data/chunk_000/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # episode index
        self.episode_index = 0
        self.index = 0

        self.dt = dt

        self.reset()

    def reset(self):
        self.df = pd.DataFrame(columns=['observation.images.image', 'observation.state', 'action', 'timestamp', 'episode_index', 'frame_index', 'index', 'next.reward', 'next.done', 'task_index'])
        
        self.timestamp = 0
        self.frame_index = 0

        # this is for data logging 
        self.column_index = 0

    def write_data_to_buffer(self, observation, action, reward, termination_flag, cam_data, debug_stuff, image_size):
        if debug_stuff[1] % (debug_stuff[0]//5) == 0:
            print(f"Write Data: {(debug_stuff[1]/debug_stuff[0]) * 100:.3f}%")

        obs_numerical = observation['policy'].cpu().numpy()[0][:-image_size]
        obs_image_top = observation['policy'].cpu().numpy()[0][-image_size:]

    
        obbs_image_top_transformed = np.asarray(obs_image_top).reshape((24,24,3))
        #print(obbs_image_top_transformed)
        #exit()

        obbs_image_top_transformed = np.clip(obbs_image_top_transformed, 0.0, 1.0)
        obbs_image_top_transformed = (obbs_image_top_transformed * 255).astype(np.uint8)
        pil_img_top_trans = Image.fromarray(obbs_image_top_transformed, mode="RGB")

        img_buf = io.BytesIO()
        pil_img_top_trans.save(img_buf, format="PNG")
        img_binary = img_buf.getvalue()

        if self.frame_index <= 9:
            img_file_name = 'frame_00000' + str(self.frame_index) + '.png'
        elif 9 < self.frame_index <= 99:
            img_file_name = 'frame_0000' + str(self.frame_index) + '.png'
        elif 99 < self.frame_index <= 999:
            img_file_name = 'frame_000' + str(self.frame_index) + '.png'
        elif 999 < self.frame_index <= 9999:
            img_file_name = 'frame_00' + str(self.frame_index) + '.png'
        else:
            img_file_name = 'frame_0' + str(self.frame_index) + '.png'

        img_dict = {
            'bytes' : img_binary,
            'frame' : img_file_name
        }

        #print(termination_flag)
        # save it into local memory 

        # https://docs.phospho.ai/learn/lerobot-dataset
        # LeRobot want their .parquet to have:
        # observation.state, action, timestamp, episode_index, frame_index, index, next.done(optional), task_index(optional)
        # we can also include next.reward and next.done it seems like
        #observation['policy'].cpu().numpy()[0]
        self.df.loc[self.column_index] = [img_dict, obs_numerical, action.cpu().numpy()[0], self.timestamp, 
                                          self.episode_index, self.frame_index, self.index, reward.cpu().item(), 
                                          termination_flag.cpu().item(), 0]
        self.column_index += 1
        self.timestamp += self.dt
        self.frame_index += 1
        self.index += 1

        #print(self.df)
        # exit()

    def dump_buffer_data(self):
        print("Start Writing Data")
        # exit()
        # dump all the data into corect dir :(
        if self.episode_index <= 9:
            data_file_name = 'episode_00000' + str(self.episode_index) + '.parquet'
        elif 9 < self.episode_index <= 99:
            data_file_name = 'episode_0000' + str(self.episode_index) + '.parquet'
        elif 99 < self.episode_index <= 999:
            data_file_name = 'episode_000' + str(self.episode_index) + '.parquet'
        elif 999 < self.episode_index <= 9999:
            data_file_name = 'episode_00' + str(self.episode_index) + '.parquet'
        else:
            data_file_name = 'episode_0' + str(self.episode_index) + '.parquet'

        table = pa.Table.from_pandas(self.df)
        pq.write_table(table, self.log_dir + data_file_name)

        self.episode_index += 1
        print(f"Complete Writing Data. Saved to {self.log_dir + data_file_name}")