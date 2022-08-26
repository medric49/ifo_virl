import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch.utils.data
from torch import nn
from torch.utils.data.dataset import T_co
from torch.nn import functional as F
from PIL import Image
import utils


class VideoDataset(torch.utils.data.IterableDataset):

    def __init__(self, root, episode_len, im_w=64, im_h=64):
        self._root = Path(root)
        self._files = []
        self.im_w = im_w
        self.im_h = im_h
        self._episode_len = episode_len + 1

    def update_files(self, max_num_video=None):
        files = list(sorted(self._root.iterdir()))
        if max_num_video is not None and len(files) > max_num_video:
            old_files = files[:-max_num_video]
            files = files[-max_num_video:]
            for f in old_files:
                os.remove(f)
        self._files = files

    def _sample(self):
        video_i_path, video_n_path = random.choices(self._files, k=2)
        video_i = np.load(video_i_path)[0, :self._episode_len]
        video_p = np.load(video_i_path)[1, :self._episode_len]
        video_n = np.load(video_n_path)[0, :self._episode_len]

        if tuple(video_i.shape[1:3]) != (self.im_h, self.im_w):
            video_i = VideoDataset.resize(video_i, self.im_w, self.im_h)
        if tuple(video_p.shape[1:3]) != (self.im_h, self.im_w):
            video_p = VideoDataset.resize(video_p, self.im_w, self.im_h)
        if tuple(video_n.shape[1:3]) != (self.im_h, self.im_w):
            video_n = VideoDataset.resize(video_n, self.im_w, self.im_h)

        video_i = video_i.transpose(0, 3, 1, 2).copy()
        video_p = video_p.transpose(0, 3, 1, 2).copy()
        video_n = video_n.transpose(0, 3, 1, 2).copy()

        return video_i, video_p, video_n

    @staticmethod
    def resize(video, im_w, im_h):
        frame_list = []
        for t in range(video.shape[0]):
            frame = Image.fromarray(video[t])
            frame = np.array(frame.resize((im_w, im_h), Image.BICUBIC), dtype=np.float32)
            frame_list.append(frame)
        frame_list = np.stack(frame_list)
        return frame_list

    @staticmethod
    def augment(video_i: torch.Tensor, video_n: torch.Tensor):
        T = video_i.shape[1]
        p_list = [0.05 for i in range(T)]
        indices = [i for i in range(T) if np.random.rand() > p_list[i]]
        video_i = video_i[:, indices, :, :, :]
        video_n = video_n[:, indices, :, :, :]
        return video_i, video_n

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()
