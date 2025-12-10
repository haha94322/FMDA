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

class ActivityNetVQADataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train'):
        self.vis_root = vis_root
        self.annotation = {}
        idx = 0
        records = pd.read_csv(ann_paths[0])
        self.fps = 10

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt

        subtitles_path = "/home/zhouhao/lmy/hcrn-videoqa_feature_all_question_extraction/data_out/nextqa/nextqa_questions_all_max.pkl"
        self.subtitles = pickle.load(open(subtitles_path, "rb"))

        self.prompt_context= "contextual prompts: {}\n Question: {}\n  Options:\n {} \n Watch the video and Answer with the option\'s letter from the given choices directly and only give the best option\'s letter. "

        for record in records.itertuples():
            answer = record.answer
            options = [record.a0, record.a1, record.a2, record.a3, record.a4]
            options = [opt for opt in options if not pd.isna(opt)]

            option_letters = []
            for option_idx, option in enumerate(options):
                option_letters.append((f"{chr(ord('A') + option_idx)}") + ': ' + option)
                if option == answer:
                    answer_idx = ((f"{chr(ord('A') + option_idx)}"))

            self.annotation[idx] = {
                "video": record.video_id,
                "ground_truth": answer_idx,
                "question": record.question,
                "options": options,
                "option_letters": option_letters,
                "question_id": record.qid,
                "question_index": str(idx)
            }
            idx += 1


        # self._add_instance_ids()
        # pdb.set_trace()

    def __getitem__(self, index):

        ann = self.annotation[index]
        ann_video = str(ann['video'])
        frame_list = []
        frame_length = len(os.listdir(os.path.join(self.vis_root, ann_video)))


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
        caption = self.subtitles[ann['video']]
        question_org = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question_org, ann["option_letters"])
            question_context = self.prompt_context.format(caption, question_org, ann["option_letters"])
        # answer = self.text_processor(ann["ground_truth"])
        answer = ann["ground_truth"]
        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann_video + "_" + str(ann["question_id"]),
            "question_context": question,
            # "instance_id": ann["instance_id"],
        }
        
    def __len__(self):
        return len(self.annotation)

class ActivityNetVQAEvalDataset(ActivityNetVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test')
        subtitles_path = "/home/zhouhao/lmy/hcrn-videoqa_feature_all_question_extraction/data_out/nextqa/nextqa_questions_all_max.pkl"
        self.subtitles = pickle.load(open(subtitles_path, "rb"))

    def __getitem__(self, index):
        # assert (
        #     self.class_labels
        # ), f"class_labels of {__class__.__name__} is not built yet."
        # question_id = self.question_id_list[index]
        ann = self.annotation[index]
        ann_video = str(ann['video'])

        frame_length = len(os.listdir(os.path.join(self.vis_root, ann_video)))

        selected_frame_index = np.rint(np.linspace(0, frame_length-1, self.num_frames)).astype(int).tolist()
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann_video, "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)
        self.prompt_org = "Question: {}\n  Options:\n {} \n Watch the video and Answer with the option\\'s letter from the given choices directly and only give the best option\\'s letter. "
        question_org = self.text_processor(ann["question"])
        question_org = self.prompt_org.format(question_org, ann["option_letters"])
        # if len(self.prompt) > 0:
        #     question = self.prompt.format(question_org, ann["option_letters"])
        # answer = self.text_processor(ann["ground_truth"])
        answer = ann["ground_truth"]

        caption = self.subtitles[ann['video']]
        options = "{}".format(ann["option_letters"])
        return {
            "image": video,
            "text_input": question_org,
            "text_output": answer,
            "question_id": ann_video + "_" + str(ann["question_id"]),
            "option": options,
            "caption": caption,
            "question_index": ann["question_index"],
            # "instance_id": ann["instance_id"],
        }
