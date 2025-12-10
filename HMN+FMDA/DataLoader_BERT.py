# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.
import os
import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
import utils


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab

def nozero_row(A):
    i = 0
    for row in A:
        if row.sum()==0:
            break
        i += 1

    return i

def transform_bb(roi_bbox):
    dshape = list(roi_bbox.shape)
    tmp_bbox = roi_bbox.reshape([-1, 4])
    relative_bbox = tmp_bbox
    relative_area = (tmp_bbox[:, 2] - tmp_bbox[:, 0] + 1) * \
                    (tmp_bbox[:, 3] - tmp_bbox[:, 1] + 1)
    relative_area = relative_area.reshape(-1, 1)
    bbox_feat = np.hstack((relative_bbox, relative_area))
    dshape[-1] += 1
    bbox_feat = bbox_feat.reshape(dshape)

    return bbox_feat


class VideoQADataset(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_len, video_ids, q_ids,
                 app_feature_h5, app_feat_id_to_index, motion_feature_h5, motion_feat_id_to_index
                 ,  question_bert_features, question_bert_indexs, questions_raw_id):
        # convert data to tensor
        self.bbox_num = 10
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))

        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.bert_feature_h5 = question_bert_features
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.question_bert_indexs = question_bert_indexs
        self.questions_raw_id = questions_raw_id


        if not np.any(ans_candidates):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        question_raw_idx = self.questions_raw_id[index]


        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]


        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)
        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)

        candidate_id = self.question_bert_indexs.index(int(question_raw_idx))
        with h5py.File(self.bert_feature_h5, 'r') as f_bert:
            bert_feat = f_bert['bert_features'][candidate_id] # (20, 768)
            question_bert = torch.from_numpy(bert_feat).type(torch.float32)
        qa_lengths = nozero_row(question_bert)

        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)

        return (
            video_idx, question_idx, answer, ans_candidates, ans_candidates_len,
            appearance_feat.float(), motion_feat.float(), question,
            question_len, question_bert, qa_lengths)

    def __len__(self):
        return len(self.all_questions)


def load_question_text_data(question_path):

    with open(question_path) as f:
        df = json.load(f)
    return df



class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)
        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')
        cfg = kwargs.pop('setting')
        split = kwargs.pop('split')
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            questions_raw_id = obj['question_raw_ids']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        question_path = kwargs.pop('question_path')

        question_bert_features = os.path.join(question_path,
                         '%s_branch_embeddings_%s_%s_%s.h5' % (cfg.dataset.name, split, 395, cfg.index))

        question_bert_indexs = utils.load_obj(
                os.path.join(question_path,
                             '%s_branch_labels_%s_%s_%s.pckl' % (cfg.dataset.name, split, 395, cfg.index))).tolist()


        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')



        self.dataset = VideoQADataset(answers, ans_candidates, ans_candidates_len, questions, questions_len,
                                      video_ids, q_ids,
                                      self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5,
                                      motion_feat_id_to_index,
                                      question_bert_features, question_bert_indexs, questions_raw_id)

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
