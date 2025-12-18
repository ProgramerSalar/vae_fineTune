import os, glob, json 

def convert_into_json(video_folder_path, video_json_file):
    

    if os.path.exists(video_folder_path):
        print('working...')
    else:
        print('not working...')

    video_file_list = []
    for root, dirs, files in os.walk(video_folder_path):
        
        if not root == video_folder_path:
            for file in files:
                if file != 'metadata.csv':
                    filter_video = os.path.join(root, file)

                    json_format = {
                        "video": filter_video
                    }
                    # print(filter_video)
                    video_file_list.append(json_format)

    if os.path.exists(video_json_file):
        print('working...')
    else:
        print('not working...')

    with open(video_json_file, 'w', encoding='utf-8') as f:
        json.dump(video_file_list, f, indent=1)
    print("succesfuly dumps")






if __name__ == "__main__":

    # <------------------------ video data to json format -----------------------------> 
    # video_folder_path = "./Data"
    video_json_file = "./annotation/video_dataset.json"
    # convert_into_json(video_folder_path, video_json_file)
    # ----------------------------------------------------------------------------------------
    # <------------------------- Testing video json --------------------------->
    # video_folder_path = "../Data"
    # video_json_file = "./annotation/sample_video_dataset.jsonl"
    # convert_into_json(video_folder_path, video_json_file)
    # ------------------------------------------------------------------------------
    # <--------------------- json to jsonl ---------------------------->
    

    # # 1. Setup paths
    input_path = video_json_file
    # input_path = "/home/manish/Desktop/projects/FineTune/annotation/annotation_with_text/video_text_json.json" # Replace this
    output_path = "annotation//video_data_files_path.jsonl"    # This will be the fixed file

    print(f"Reading {input_path}...")

    # 2. Read the standard JSON array
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Found {len(data)} videos. Converting to JSONL...")

    # 3. Write out to JSONL (one object per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            # Write the dictionary as a single line string
            f.write(json.dumps(entry) + "\n")

    print(f"Success! Use this file for training: {output_path}")

    

    

    

