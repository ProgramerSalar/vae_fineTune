import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # MUST BE FIRST
import json
import cv2
import torch
import numpy as np
import PIL
from PIL import Image
from einops import rearrange
from video_vae import CausalVideoVAELossWrapper
from torchvision import transforms as pth_transforms
from torchvision.transforms.functional import InterpolationMode


model_path = "/content/vae_fineTune/PATH/vae_ckpt"   # The video-vae checkpoint dir
model_dtype = 'bf16'



model = CausalVideoVAELossWrapper(
    model_path,
    model_dtype,
    interpolate=False, 
    add_discriminator=False,
)
model = model.to("cuda")

if model_dtype == "bf16":
    torch_dtype = torch.bfloat16 
elif model_dtype == "fp16":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

def image_transform(images, resize_width, resize_height):
    transform_list = pth_transforms.Compose([
        pth_transforms.Resize((resize_height, resize_width), InterpolationMode.BICUBIC, antialias=True),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return torch.stack([transform_list(image) for image in images])


def get_transform(width, height, new_width=None, new_height=None, resize=False,):
    transform_list = []

    if resize:
        if new_width is None:
            new_width = width // 8 * 8
        if new_height is None:
            new_height = height // 8 * 8
        transform_list.append(pth_transforms.Resize((new_height, new_width), InterpolationMode.BICUBIC, antialias=True))
    
    transform_list.extend([
        pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_list = pth_transforms.Compose(transform_list)

    return transform_list


def load_video_and_transform(video_path, frame_number, new_width=None, new_height=None, max_frames=600, sample_fps=24, resize=False):
    try:
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frames = []
        pil_frames = []
        while True:
            flag, frame = video_capture.read()
            if not flag:
                break
    
            pil_frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            if len(frames) >= max_frames:
                break

        video_capture.release()
        interval = max(int(fps / sample_fps), 1)
        pil_frames = pil_frames[::interval][:frame_number]
        frames = frames[::interval][:frame_number]
        frames = torch.stack(frames).float() / 255
        width = frames.shape[-1]
        height = frames.shape[-2]
        video_transform = get_transform(width, height, new_width, new_height, resize=resize)
        frames = video_transform(frames)
        pil_frames = [Image.fromarray(frame).convert("RGB") for frame in pil_frames]

        if resize:
            if new_width is None:
                new_width = width // 32 * 32
            if new_height is None:
                new_height = height // 32 * 32
            pil_frames = [frame.resize((new_width or width, new_height or height), PIL.Image.BICUBIC) for frame in pil_frames]
        return frames, pil_frames
    except Exception:
        return None





if __name__ == "__main__":

  torch.cuda.empty_cache()
  video_path = '/content/vae_fineTune/Data/Algebra Basics： Graphing On The Coordinate Plane - Math Antics [9Uc62CuQjc4]/videos/clip_0018.mp4'

  frame_number = 57   # x*8 + 1
  width = 640
  height = 384

  video_frames_tensor, pil_video_frames = load_video_and_transform(video_path, frame_number, new_width=width, new_height=height, resize=True)
  del pil_video_frames
  video_frames_tensor = video_frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)
  print(video_frames_tensor.shape)
  # video_frames_tensor = video_frames_tensor.to("cuda", dtype=torch.bfloat16)
  

  with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
      latent = model.encode_latent(video_frames_tensor.to("cuda"), sample=False, window_size=8, temporal_chunk=True)
      print(latent)

  del video_frames_tensor
  torch.cuda.empty_cache()


