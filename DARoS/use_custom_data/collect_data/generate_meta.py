from MetaRecorder import MetaRecorder

def main():
    meta_recorder = MetaRecorder(data_folder_path="/home/isaac/Documents/Github/IsaacLab/LeRobotData/data/chunk_000")

    meta_recorder.generate_episodes_jsonl()
    meta_recorder.generate_info_json()
    meta_recorder.generate_stats_json()
    # it is bad desgin but non means hard coded task lol
    meta_recorder.generate_tasks_jsonl(tasks=None)

if __name__ == "__main__":
    main()