import numpy as np
import pandas as pd
import os
import json

import io
from PIL import Image

from os import listdir
from os.path import isfile, join

class MetaRecorder:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.task = "Grab the door handle"
    
    def generate_episodes_jsonl(self):
        all_files = [f for f in listdir(self.data_folder_path) if isfile(join(self.data_folder_path, f))]
        all_files.sort()

        jsonl_data = []

        for file in all_files:
            df = pd.read_parquet(self.data_folder_path + '/' + file)
            
            dump_dict = {}

            # print(type(df['episode_index'][0]))
            # print(df['episode_index'][0])
            # print(type(df['episode_index']))
            # exit()

            dump_dict['episode_index'] = int(df['episode_index'][0])
            dump_dict['tasks'] = [self.task]
            dump_dict['length'] = len(df)

            # print(dump_dict)
            # exit()

            jsonl_data.append(dump_dict)

        self._write_data(jsonl_data, "episodes.jsonl")

    def generate_info_json(self):
        all_files = [f for f in listdir(self.data_folder_path) if isfile(join(self.data_folder_path, f))]
        all_files.sort()
        total_episodes = len(all_files)
        total_frames = 0
        unique_task_indices = set()
        fps = 0.0
        total_tasks = 0
        sample_features = {}

        data_path = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        video_path = None

        # just take on df for features since everything is the same
        inference_df = pd.read_parquet(self.data_folder_path + '/' + all_files[0])
        for col in inference_df.columns:
            col_pointer = inference_df[col]
            # print(col_sample)
            # print(type(col_sample))
            sample = col_pointer.iloc[0]
            # print(sample)
            # exit()
            if isinstance(sample, (np.ndarray, list, tuple)):
                arr = np.array(sample)
                data_type = str(arr.dtype)
                data_shape = [str(arr.shape).split(',')[0].split('(')[1]]
            else:
                data_type = str(inference_df[col].dtype)
                data_shape = [1]
            
            sample_features[col] = {
                'dtype' : data_type,
                'shape' : data_shape,
                'names' : 'TODO' # I think it is easier to manually fill this out since getting the joint name via script will be a pain...
            }
        
        # print(sample_features) 
        # {'observation.images.image': {'dtype': 'object', 'shape': [1], 'names': 'TODO'}, 'observation.state': 
        # {'dtype': 'float32', 'shape': ['45'], 'names': 'TODO'}, 'action': {'dtype': 'float32', 'shape': ['13'], 'names': 'TODO'}, 
        # 'timestamp': {'dtype': 'float64', 'shape': [1], 'names': 'TODO'}, 'episode_index': {'dtype': 'int64', 'shape': [1], 'names': 'TODO'}, 
        # 'frame_index': {'dtype': 'int64', 'shape': [1], 'names': 'TODO'}, 'index': {'dtype': 'int64', 'shape': [1], 'names': 'TODO'}, 
        # 'next.reward': {'dtype': 'float64', 'shape': [1], 'names': 'TODO'}, 'next.done': {'dtype': 'bool', 'shape': [1], 'names': 'TODO'}, 
        # 'task_index': {'dtype': 'int64', 'shape': [1], 'names': 'TODO'}}
        # exit()

        fps_l = []
        for file in all_files:
            df = pd.read_parquet(self.data_folder_path + '/' + file)
            total_frames += len(df)
            unique_task_indices.update(df['task_index'].unique().tolist())
            ts = df['timestamp'].values
            # print(type(ts))
            diffs = np.diff(ts) # type: ignore
            # print(diffs)
            # exit()
            fps_est = 1.0 / np.mean(diffs)
            fps_l.append(fps_est)

        fps = float(np.mean(fps_l))
        total_tasks = len(unique_task_indices)

        splits = {"train":f"0:{total_episodes}"}

        info = {
            "codebase_version": "v1.0",
            "robot_type": "realmandoor",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": fps,
            "splits" : splits,
            "data_path" : data_path,
            "video_path" : video_path, 
            "features": sample_features
        }

        out_f_name = 'info.json'
        with open(out_f_name, 'w') as f:
            json.dump(info, f, indent=4)
    
        print(f"Successfully generated {out_f_name} to {os.getcwd()}")
        print("WARNING (info.json): \ninfo.json 1) Need to mannually write shapes for image data (REQUIRED)\ninfo.json 2) add joint/motor names (if needed)")

    def generate_stats_json(self):
        all_files = [f for f in listdir(self.data_folder_path) if isfile(join(self.data_folder_path, f))]
        all_files.sort()

        preprocessed_stats = {}

        for file in all_files:
            df = pd.read_parquet(self.data_folder_path + '/' + file)

            for col in df.columns:
                #first create place holder with sample on top
                if col not in preprocessed_stats:
                    col_pointer = df[col]
                    sample = col_pointer.iloc[0]
                    # print(sample)
                    # exit()
                    # when we see image in bytes 
                    if isinstance(sample, dict) and 'bytes' in sample:
                        img = Image.open(io.BytesIO(sample['bytes']))
                        img = img.convert("RGB")
                        # i think we should normalize since hugging face sample do that - prob becasuse of how fed into NN by mdp.image
                        arr = np.asarray(img).astype(np.float32) / 255.0
                        C = arr.shape[2]
                        # create place holder
                        preprocessed_stats[col] = {
                            "dt" : "img",
                            "sum" : np.zeros((C,), dtype=np.float64),
                            "sum_sq": np.zeros((C,), dtype=np.float64),
                            "pixel_count" : 0,
                            "min" : np.full((C,), np.inf, dtype=np.float64),
                            "max" : np.full((C,), -np.inf, dtype=np.float64)
                        }
                    # when we see other numerical data
                    else:
                        if isinstance(sample, (list, tuple, np.ndarray)):
                            arr = np.asarray(sample, dtype=np.float64)
                        else:
                            arr = np.array([sample], dtype=np.float64)
                        #print(np.zeros(arr.flatten().shape, dtype=np.float64).shape)
                        #exit()
                        preprocessed_stats[col] = {
                            "dt" : "num",
                            "sum" : np.zeros(arr.flatten().shape, dtype=np.float64),
                            "sum_sq": np.zeros(arr.flatten().shape, dtype=np.float64),
                            "count" : 0,
                            "min" : np.full(arr.flatten().shape, np.inf, dtype=np.float64),
                            "max" : np.full(arr.flatten().shape, -np.inf, dtype=np.float64)
                        }
                
                col_dict_point = preprocessed_stats.get(col)
                #print(col_dict_point)
                
                df_rows = df[col]

                if col_dict_point['dt'] == "img": # type: ignore
                    for row in df_rows:
                        #print(row)
                        b = row.get('bytes', None) if isinstance(row, dict) else None
                        img = Image.open(io.BytesIO(b)) # type: ignore
                        img = img.convert("RGB")
                        arr = np.asarray(img).astype(np.float32) / 255.0

                        H, W, C = arr.shape
                        flat = arr.reshape(-1, C)
                        sum = flat.sum(axis=0)
                        sumsq = (flat * flat).sum(axis=0)
                        col_dict_point["sum"] += sum # type: ignore
                        col_dict_point["sum_sq"] += sumsq # type: ignore
                        col_dict_point["pixel_count"] += (H*W) # type: ignore

                        col_dict_point['min'] = np.minimum(flat.min(axis=0), col_dict_point["min"]) # type: ignore
                        col_dict_point['max'] = np.maximum(flat.max(axis=0), col_dict_point["max"]) # type: ignore

                        #print(col_dict_point)
                else:
                    for row in df_rows:
                        if isinstance(row, (list, tuple, np.ndarray)):
                            arr = np.asarray(row, dtype=np.float64).flatten()
                        else:
                            arr = np.array([row], dtype=np.float64)

                        col_dict_point["sum"] += arr # type: ignore
                        col_dict_point["sum_sq"] += arr * arr # type: ignore
                        col_dict_point["count"] += 1 # type: ignore

                        col_dict_point['min'] = np.minimum(arr, col_dict_point["min"]) # type: ignore
                        col_dict_point['max'] = np.maximum(arr, col_dict_point["max"]) # type: ignore

        dump_dict = {}
        # print(preprocessed_stats)
        
        for key, value in preprocessed_stats.items():
            if value['dt'] == 'img':
                pix_count = value["pixel_count"]
                mean = (value["sum"] / pix_count).tolist()
                var = (value["sum_sq"] / pix_count) - np.square(value["sum"] / pix_count)
                var = np.clip(var, a_min=0.0, a_max=None)
                min = value["min"].tolist()
                max = value["max"].tolist()

                dump_dict[key] = {
                    "mean" : self._format_for_RGB(mean),
                    "std" : self._format_for_RGB(np.sqrt(var).tolist()),
                    "min" : self._format_for_RGB(min),
                    "max" : self._format_for_RGB(max),
                }
                # print(dump_dict)
                
            else:
                count = value['count']
                mean = (value["sum"] / count).tolist()
                var = (value["sum_sq"] / count) - np.square(value["sum"] / count)
                var = np.clip(var, a_min=0.0, a_max=None)
                min = value["min"].tolist()
                max = value["max"].tolist()

                dump_dict[key] = {
                    "mean" : mean,
                    "std" : np.sqrt(var).tolist(),
                    "min" : min,
                    "max" : max,
                }
        
        out_f_name = 'stats.json'
        with open(out_f_name, 'w') as f:
            json.dump(dump_dict, f, indent=4)

        print("WARNING (stats.json): \nstats.json 1) Need to mannually convert bool min max (REQUIRED)")
        

    def generate_tasks_jsonl(self, tasks):
        jsonl_data = []

        if tasks is None:
            tasks = [self.task]

        for task in tasks:
            dump_dict = {}
            
            dump_dict["task_index"] = 0
            dump_dict["task"] = task

            # print(dump_dict)
            # exit()
            
            jsonl_data.append(dump_dict)

        self._write_data(jsonl_data, 'tasks.jsonl')


    def _write_data(self, data, f_name):
        with open(f_name, 'w') as f:
            for l in data:
                f.writelines([json.dumps(l)])
                f.writelines('\n')

        print(f"Successfully generated {f_name} to {os.getcwd()}")


    def _format_for_RGB(self, lst):
        return [[[v]] for v in lst]
