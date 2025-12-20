import os
import json
import jsonlines
import torch
import math
import random
import cv2

from tqdm import tqdm
from collections import OrderedDict

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode



class VideoFrameProcessor:
    # load a video and transform
    def __init__(self, resolution=256, num_frames=24, add_normalize=True, sample_fps=24):
    
        image_size = resolution

        transform_list = [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
        ]
        
        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        print(f"Transform List is {transform_list}")
        self.num_frames = num_frames
        self.transform = transforms.Compose(transform_list)
        self.sample_fps = sample_fps

    def __call__(self, video_path):
        try:
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frames = []

            while True:
                flag, frame = video_capture.read()
                if not flag:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)

            video_capture.release()
            sample_fps = self.sample_fps
            interval = max(int(fps / sample_fps), 1)
            frames = frames[::interval]

            if len(frames) < self.num_frames:
                num_frame_to_pack = self.num_frames - len(frames)
                recurrent_num = num_frame_to_pack // len(frames)
                frames = frames + recurrent_num * frames + frames[:(num_frame_to_pack % len(frames))]
                assert len(frames) >= self.num_frames, f'{len(frames)}'

            start_indexs = list(range(0, max(0, len(frames) - self.num_frames + 1)))            
            start_index = random.choice(start_indexs)

            filtered_frames = frames[start_index : start_index+self.num_frames]
            assert len(filtered_frames) == self.num_frames, f"The sampled frames should equals to {self.num_frames}"

            filtered_frames = torch.stack(filtered_frames).float() / 255
            filtered_frames = self.transform(filtered_frames)
            filtered_frames = filtered_frames.permute(1, 0, 2, 3)

            return filtered_frames, None
            
        except Exception as e:
            print(f"Load video: {video_path} Error, Exception {e}")
            return None, None


class VideoDataset(Dataset):
    def __init__(self, anno_file, resolution=256, max_frames=9, add_normalize=True):
        super().__init__()

        self.video_annos = []
        self.max_frames = max_frames

        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        print(f"The training video clip frame number is {max_frames} ")

        for anno_file_ in anno_file:
            print(f"Load annotation file from {anno_file_}")

            with jsonlines.open(anno_file_, 'r') as reader:
                for item in tqdm(reader):
                    self.video_annos.append(item)
        
        print(f"Totally Remained {len(self.video_annos)} videos")
        
        self.video_processor = VideoFrameProcessor(resolution, max_frames, add_normalize)        

    def __len__(self):
        return len(self.video_annos)

    def __getitem__(self, index):
        video_anno = self.video_annos[index]
        video_path = video_anno['video']

        try:
            video_tensors, video_frames = self.video_processor(video_path)

            assert video_tensors.shape[1] == self.max_frames
            
            return {
                "video": video_tensors,
                "identifier": 'video',
            }

        except Exception as e:
            print('Loading Video Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))


if __name__ == "__main__":

    anno_file = "/home/manish/Desktop/projects/vae_fineTune/annotation/testing.jsonl"
    video_dataset = VideoDataset(anno_file)

    # print(video_dataset)
    video_dataloader = DataLoader(video_dataset, 
                                  batch_size=2)
    # print(video_dataloader)

    for data in video_dataloader:
        print(data['video'].shape)
