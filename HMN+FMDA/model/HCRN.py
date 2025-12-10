import numpy as np
from torch.nn import functional as F

from .PyTransformer import PyTransformer
from .utils import *
from .modules.TransformerEncoders import *
from .Embedding import visualembedding

from .OutLayers import OutOpenEnded, OutMultiChoices, OutCount


class InputUnitVisual(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(InputUnitVisual, self).__init__()

        self.feat_proj = visualembedding(level=spl_resolution, embed_dim=module_dim, v_inDim=vision_dim)
        self.block = PyTransformer(level=spl_resolution, embed_dim=module_dim)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding, question_bert, qa_lengths):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        app_mot = self.feat_proj(appearance_video_feat, motion_video_feat)

        out = self.block(app_mot, question_bert.permute(1, 0, 2), question_embedding, qa_lengths)



        return out

class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 3, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, video_mot_level_decoder, video_app_level_decoder):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([question_embedding, video_mot_level_decoder, video_app_level_decoder], 1)
        # out_app = torch.cat([question_embedding, video_app_level_decoder], 1)
        # out_motion = torch.cat([question_embedding, video_mot_level_decoder], 1)
        # out_object = torch.cat([question_embedding, video_object_level_decoder], 1)
        # out_app = self.classifier1(out_app)
        # out_motion = self.classifier2(out_motion)
        # out_object = self.classifier3(out_object)
        out = self.classifier(out)

        return out
        # return out_motion



class HCRNNetwork(nn.Module):
    def __init__(self, bert_model, vision_dim, module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, question_type):
        super(HCRNNetwork, self).__init__()
        self.bert = True
        self.question_type = question_type
        self.bert_pool = BertPooler(768)

        self.num_classes = len(vocab['answer_token_to_idx'])

        self.linguistic_input_proj = nn.Linear(768, module_dim)

        self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level,
                                                 spl_resolution=spl_resolution, vision_dim=vision_dim,
                                                 module_dim=module_dim)
        self.output_unit = OutOpenEnded(num_answers=self.num_classes)



        init_modules(self.modules(), w_init="xavier_uniform")
        if not self.bert:
            nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)


    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,
                question_len, question_bert, qa_lengths, _):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        Choices_num = 0
        if self.bert:
            question_embedding = self.bert_pool(question_bert)
            question_embedding = self.linguistic_input_proj(question_embedding)
            question_bert = self.linguistic_input_proj(question_bert)
            # question_embedding = self.linguistic_input_unit(question_embedding, qa_lengths.cpu())
        else:
            question_embedding = self.linguistic_input_unit(question, question_len.cpu())



        video = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding, question_bert, qa_lengths)

        out = self.output_unit(question_embedding,
                               video)

        return out

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
