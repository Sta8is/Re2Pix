# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import traceback
import warnings
from typing import Any
import random
from glob import glob
from PIL import Image

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorVideo
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import log
from cosmos_predict2.auxiliary.dinov2_extractor import DINOv2FeatureExtractor
import matplotlib.pyplot as plt

class CityscapesDataset(Dataset):
    def __init__(self, data_root="/opt/dlami/nvme/cityscapes/dpt_pred_seg_sequence",
                 target_height=432, target_width=768, ):
        if not os.path.exists(data_root):
            raise ValueError(f"Cannot find dataset {data_root}")
        self.data_root = data_root
        self.target_height = target_height
        self.target_width = target_width
        self.samples = []
        self.sequences = set()
        self.subset = "train"  # Default subset, can be changed to 'val' or 'test' if needed
        sequences = set()
        city_folders = glob(os.path.join(self.data_root, self.subset, '*'))
        for city_folder in city_folders:
            city_name = os.path.basename(city_folder)
            frames_in_city = glob(os.path.join(city_folder, '*'))
            city_seqs = set([f"{city_name}_{os.path.basename(frame).split('_')[1]}" for frame in frames_in_city])
            if len(city_seqs)<10: # Note that in some cities very few, though very long sequences were recorded
                for seq in city_seqs:
                    sub_seqs = sorted(glob(os.path.join(self.data_root, self.subset, city_name, seq+'*.png')))
                    sub_seq_startframe_ids = [os.path.basename(sub_seqs[i])[:-16] for i in range(len(sub_seqs)) if i%30==0]
                    self.sequences.update(sub_seq_startframe_ids)
            else:
                self.sequences.update(city_seqs)
                continue
        self.sequences = sorted(list(self.sequences))
        for sequence_name in self.sequences:
            splits = sequence_name.split("_")
            if len(splits)==2: # Sample from Short sequences
                frames_filepaths = sorted(glob(os.path.join(self.data_root,self.subset, splits[0], sequence_name+'*.png'))) # Sequence of 30 frames
            elif len(splits)==3: # Sample from Long sequences
                frames_filepaths = [os.path.join(self.data_root,self.subset,splits[0], splits[0]+"_"+splits[1]+"_"+'{:06d}'.format(int(splits[2])+i)+'_leftImg8bit.png') for i in range(30)]
            sample_dict = {"frames": frames_filepaths}
            self.samples.append(sample_dict)
        
    def get_image_path(self, sample_dict, current_index):
        frame_list = sample_dict["frames"]  # 30 frames
        return frame_list[current_index]

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        # image = image.resize((self.target_width, self.target_height), resample=Image.LANCZOS)
        image = image.resize((self.target_width, self.target_height), resample=Image.NEAREST)
        if not image.mode == "L":
            image = image.convert("L")
        # PIL to Tensor (0-255)
        image = T.PILToTensor()(image)
        return image

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        sample_dict = self.samples[index]

        image_seq = list()
        for i in range(len(sample_dict["frames"])):
            img_path = self.get_image_path(sample_dict, i)
            image = self.preprocess_image(img_path)
            image_seq.append(image)
        image_seq = torch.stack(image_seq)
        return image_seq

"""
Test the dataset with the following command:
python -m cosmos_predict2.data.dataset_video
"""


class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        num_frames,
        video_size,
        single_t5_embedding=True,
        modality: Any = "dino"
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames
            single_t5_embedding (bool): Whether to use a single T5 embedding for all videos
            modality (str): Modality to use for the dataset (e.g., None, "dino", "depth")

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.modality = modality
        video_dir = os.path.join(self.dataset_dir, "videos")
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")

        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.video_paths = sorted(self.video_paths)
        self.single_t5_embedding = single_t5_embedding
        # remove video paths that does not have t5_embedding
        if not single_t5_embedding:
            self.video_paths = [
                path
                for path in self.video_paths
                if os.path.exists(os.path.join(self.t5_dir, os.path.basename(path).replace(".mp4", ".pickle")))
            ]

        log.info(f"{len(self.video_paths)} videos in total")

        self.wrong_number = 0
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])
        if self.modality == "dino":
            self.preprocess_dino = T.Compose(
                [
                    ToTensorVideo(),
                    Resize_Preprocess((448, 896)),  # Resize to DINOv2(Foresight) input size
                    T.Normalize([0.45, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif self.modality == "depth":
            # Only Resize with nearest for segmentation
            self.preprocess_depth = T.Compose(
                [
                    ToTensorVideo(),
                    Resize_Preprocess((432, 768), interp_method=T.InterpolationMode.NEAREST),
                ]
            )

        elif self.modality == "segm":
            # Only Resize with nearest for segmentation
            self.segm_dataset = CityscapesDataset(
                data_root="/opt/dlami/nvme/cityscapes/dpt_pred_seg_sequence",
                target_height=432,
                target_width=768,
            )
             
        

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, video_path, start_frame=None) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            # If there are not enough frames, let it fail
            warnings.warn(  # noqa: B028
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )
            raise ValueError(f"Video {video_path} has insufficient frames.")

        # randomly sample a sequence of frames
        if start_frame is None:
            max_start_idx = total_frames - self.sequence_length
            start_frame = np.random.randint(0, max_start_idx)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps, start_frame

    def _get_frames(self, video_path: str, index: int) -> tuple[torch.Tensor, float]:
        frames, fps, start_frame = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        frames_img = self.preprocess(frames)
        frames_img = torch.clamp(frames_img * 255.0, 0, 255).to(torch.uint8)
        if self.modality == "dino":
            frames_modal = frames[::4]
            # frames_dino = frames[:self.sequence_length]
            frames_modal = self.preprocess_dino(frames_modal)
        if self.modality == "segm":
            # Get segmentation frames from the preprocessed dataset
            frames_modal = self.segm_dataset[index]
            frames_modal = frames_modal[start_frame:start_frame+self.sequence_length:4]  # Sample every 4th frame
        elif self.modality == "depth":
            modal_video_path = video_path.replace("cityscapes", "cityscapes_depth_dpt")
            frames_modal, _, _ = self._load_video(modal_video_path, start_frame=start_frame)
            frames_modal = frames_modal[::4]
            frames_modal = frames_modal.astype(np.uint8)
            frames_modal = torch.from_numpy(frames_modal).permute(0, 3, 1, 2)
            frames_modal = self.preprocess_depth(frames_modal)
            # From [0, 1] to [-1, 1]
            # frames_modal = frames_modal * 2.0 - 1.0
            frames_modal = frames_modal[:,0].unsqueeze(1)  # Make it (T, 1, H, W)
        return frames_img, frames_modal, fps

    def __getitem__(self, index):
        try:
            data = dict()
            if self.modality == "dino":
                video, dino_frames, fps = self._get_frames(self.video_paths[index], index)
                data["dino_frames"] = dino_frames
            if self.modality == "depth":
                video, depth_frames, fps = self._get_frames(self.video_paths[index], index)
                data["depth_frames"] = depth_frames
            if self.modality == "segm":
                video, segm_frames, fps = self._get_frames(self.video_paths[index], index)
                data["segm_frames"] = segm_frames
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
            video_path = self.video_paths[index]
            if not self.single_t5_embedding:
                t5_embedding_path = os.path.join(
                    self.t5_dir,
                    os.path.basename(video_path).replace(".mp4", ".pickle"),
                )
            else:
                t5_embedding_path = os.path.join(
                    self.t5_dir,
                    "general_prompt.pickle",
                )
            data["video"] = video
            data["video_name"] = {
                "video_path": video_path,
                "t5_embedding_path": t5_embedding_path,
            }

            _, _, h, w = video.shape

            # Just add these to fit the interface
            with open(t5_embedding_path, "rb") as f:
                t5_embedding_raw = pickle.load(f)
                assert isinstance(t5_embedding_raw, list)
                assert len(t5_embedding_raw) == 1
                t5_embedding = t5_embedding_raw[0]  # [n_tokens, CosmosTextEncoderConfig.EMBED_DIM]
                assert isinstance(t5_embedding, np.ndarray)
                assert len(t5_embedding.shape) == 2
            n_tokens = t5_embedding.shape[0]
            if n_tokens < CosmosTextEncoderConfig.NUM_TOKENS:
                t5_embedding = np.concatenate(
                    [
                        t5_embedding,
                        np.zeros(
                            (CosmosTextEncoderConfig.NUM_TOKENS - n_tokens, CosmosTextEncoderConfig.EMBED_DIM),
                            dtype=np.float32,
                        ),
                    ],
                    axis=0,
                )
            t5_text_mask = torch.zeros(CosmosTextEncoderConfig.NUM_TOKENS, dtype=torch.int64)
            t5_text_mask[:n_tokens] = 1

            data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)
            data["t5_text_mask"] = t5_text_mask
            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception:
            warnings.warn(  # noqa: B028
                f"Invalid data encountered: {self.video_paths[index]}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")  # noqa: B028
            warnings.warn(traceback.format_exc())  # noqa: B028
            self.wrong_number += 1
            log.info(self.wrong_number, rank0_only=False)
            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    dataset = Dataset(
    dataset_dir="/home/ubuntu/hieracosmos2/datasets/cityscapes/val",
    num_frames=25,
    video_size=(432, 768),
    single_t5_embedding=True,
    modality="segm",
    )

    indices = range(10)
    for idx in indices:
        data = dataset[idx]
        # dfeats = data.get("dino_frames", None)
        segmaps = data.get("segm_frames", None)

        log.info(
            f"{idx=} "
            f"{data['video'].sum()=}\n"
            f"{data['video'].shape=}\n"
            f"{data['video_name']=}\n"
            f"{data['t5_text_embeddings'].shape=}\n"
            f"{segmaps.shape=}\n"
            # f"{data['dino_frames'].shape=}\n"
            "---"
        )
        # Plot Depths (They are 7 frames)
        # print(data['depth_frames'].shape)
        # fig, ax = plt.subplots(1, 7, figsize=(20, 5))
        # for i in range(7):
        #     depth_frame = data['depth_frames'][i].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        #     ax[i].imshow((depth_frame*255).astype(np.uint8))
        #     ax[i].set_title(f'Depth Map {i+1}')
        #     ax[i].axis('off')

        # Plot Segmentation Maps (They are 7 frames)
        # t, C, H, W = segmaps.shape
        # video_frames = data['video'].permute(1, 0, 2, 3)  # CTHW -> TCHW
        
        # # Sample every 4th frame starting from 0
        # sampled_indices = list(range(0, min(len(video_frames), t * 4), 4))
        # num_samples = len(sampled_indices)
        
        # fig, ax = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
        # if num_samples == 1:
        #     ax = ax.reshape(2, 1)
        
        # for plot_idx, frame_idx in enumerate(sampled_indices):
        #     # Plot RGB frame on top row
        #     rgb_frame = video_frames[frame_idx].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        #     ax[0, plot_idx].imshow(rgb_frame)
        #     ax[0, plot_idx].set_title(f'RGB Frame {frame_idx}')
        #     ax[0, plot_idx].axis('off')
            
        #     # Plot corresponding segmentation map on bottom row
        #     seg_idx = frame_idx // 4
        #     if seg_idx < t:
        #         segmap = segmaps[seg_idx].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        #     ax[1, plot_idx].imshow(segmap.squeeze(), cmap='gray')
        #     ax[1, plot_idx].set_title(f'Segmentation Map {frame_idx}')
        #     ax[1, plot_idx].axis('off')

        # plt.tight_layout()
        # plt.savefig(f"rgb_seg_maps_{idx}.png")
        # plt.close()
        # breakpoint()
