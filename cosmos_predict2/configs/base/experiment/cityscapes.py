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

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_image import ImageDataset
from cosmos_predict2.data.dataset_video import Dataset
from cosmos_predict2.data.dataset_video_multi import MultisourceDataset
from imaginaire.lazy_config import LazyCall as L


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()


# Cityscapes video2world example
example_video_dataset_cityscapes_train = L(MultisourceDataset)(
    dataset_dir="datasets/cityscapes/train",
    num_frames=25,
    video_size=(432, 768),
    single_t5_embedding=True,
    modality='dino',
)

dataloader_train_cityscapes = L(DataLoader)(
    dataset=example_video_dataset_cityscapes_train,
    sampler=L(get_sampler)(dataset=example_video_dataset_cityscapes_train),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

example_video_dataset_cityscapes_val = L(MultisourceDataset)(
    dataset_dir="datasets/cityscapes/val",
    num_frames=25,
    video_size=(432, 768),
    single_t5_embedding=True,
    modality='dino',
)

dataloader_val_cityscapes = L(DataLoader)(
    dataset=example_video_dataset_cityscapes_val,
    sampler=L(get_sampler)(dataset=example_video_dataset_cityscapes_val),
    batch_size=1,
    drop_last=False,
    num_workers=8,
    pin_memory=True,
)


# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_cityscapes
predict2_video2world_training_2b_cityscapes = dict(
    defaults=[
        {"override /model": "predict2_video2world_dino_fsdp_2b_480p_16fps_scratch"}, 
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_cityscapes_16fps_480p_25frames_dino_foresight${model.config.foresight_prob}_layers14_dropcomp_scratch_40k",
    ),
    model=dict(
        config=dict(
            foresight_prob=0.1,
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
                min_num_conditional_frames=4,
                max_num_conditional_frames=4,
                state_t=7, # 7-1//4 +1 
                net=dict(num_blocks=14) # Training with 14 Layer, Erase it for default 28 layers
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_cityscapes,
    dataloader_val=dataloader_val_cityscapes,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=20000,
    ),
    checkpoint=dict(
        save_iter=10000,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        warm_up_steps=[2_000],
        cycle_lengths=[20000],
        f_max=[0.6],
        f_min=[0.0],
    ),
)

for _item in [
    # 2b, cityscapes
    predict2_video2world_training_2b_cityscapes,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
