"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import re
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset
from lavis.datasets.data_utils import load_video
import pdb
import pickle

class MSVDVQADataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train'):
        self.vis_root = vis_root

        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.question_id_list = list(self.annotation.keys())
        self.question_id_list.sort()
        self.fps = 10

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt

        subtitles_path = "/home/zhouhao/lmy/hcrn-videoqa_feature_all_question_extraction/data_out/msrvtt-qa/msrvtt-qa_questions_all_summary.pkl"
        self.subtitles = pickle.load(open(subtitles_path, "rb"))


        self.prompt_context = "contextual prompts: {}\n Question: {} Short answer:"


        # self._add_instance_ids()
        # pdb.set_trace()

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]
        ann_video = ann['video']
        frame_list = []
        frame_length = len(os.listdir(os.path.join(self.vis_root, ann_video)))

        # if frame_length != ann["frame_length"]:
        #     print(1)

        # Divide the range into num_frames segments and select a random index from each segment
        segment_list = np.linspace(0, frame_length, self.num_frames + 1, dtype=int)
        segment_start_list = segment_list[:-1]
        segment_end_list = segment_list[1:]
        selected_frame_index = []
        for start, end in zip(segment_start_list, segment_end_list):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann_video, "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)
        # print(selected_frame_index, video.shape)
        caption = self.subtitles[int(ann_video.split("vid")[1])]
        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
            question_context = self.prompt_context.format(caption, question)
        answer = self.text_processor(ann["answer"])
        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
            "question_context": question_context,
            # "instance_id": ann["instance_id"],
        }
        
    def __len__(self):
        return len(self.question_id_list)

class MSVDVQAEvalDataset(MSVDVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test')

        subtitles_path = "/home/zhouhao/lmy/hcrn-videoqa_feature_all_question_extraction/data_out/msrvtt-qa/msrvtt-qa_questions_all_summary.pkl"
        self.subtitles = pickle.load(open(subtitles_path, "rb"))


    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]
        ann_video = ann['video']

        frame_length = len(os.listdir(os.path.join(self.vis_root, ann_video)))

        selected_frame_index = np.rint(np.linspace(0, frame_length-1, self.num_frames)).astype(int).tolist()
        frame_list = []

        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann_video, "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        caption = self.subtitles[int(ann_video.split("vid")[1])]

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
            "caption": caption
            # "instance_id": ann["instance_id"],
        }