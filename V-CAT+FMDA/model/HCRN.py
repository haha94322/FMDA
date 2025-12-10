import numpy as np
from torch.nn import functional as F

from .utils import *
from model.cross_fly_v2 import fly_1d as CRN


class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill


class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        output, (hidden, _) = self.encoder(embed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=None)
        text_embedding = output

        question_embedding = torch.cat([hidden[0], hidden[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding, text_embedding

class att(nn.Module):
    def __init__(self, module_dim=512):
        super(att, self).__init__()

        self.att = nn.MultiheadAttention(embed_dim=module_dim, num_heads=8, dropout=0.1)
        self.liner1 = nn.Linear(module_dim, module_dim)
        self.liner2 = nn.Linear(module_dim, module_dim)

    def forward(self, q, k, v):

        video_feat = self.att(q, k, v)[0]
        video_feat = self.liner2(self.liner1(video_feat))

        return video_feat





class InputUnitVisual(nn.Module):
    def __init__(self, vision_dim, module_dim=512):
        super(InputUnitVisual, self).__init__()

        self.context_encoder_layer_num = 8
        self.video_encoder_layer_num = 1
        self.question_encoder_layer_num = 1
        self.video_decoder_layer_num = 7
        self.question_decoder_layer_num = 4

        self.question = nn.Linear(module_dim, module_dim)
        self.video = nn.Linear(vision_dim, module_dim)
        self.context = nn.Linear(module_dim, module_dim)

        self.video_encoder_layers = nn.ModuleList([])
        for layer in range(self.video_encoder_layer_num):
            self.video_encoder_layers.append(att(module_dim))

        self.question_encoder_layers = nn.ModuleList([])
        for layer in range(self.question_encoder_layer_num):
            self.question_encoder_layers.append(att(module_dim))

        self.context_encoder_layers = nn.ModuleList([])
        for layer in range(self.context_encoder_layer_num):
            self.context_encoder_layers.append(att(module_dim))


        self.video_decoder_layers = nn.ModuleList([])
        for layer in range(self.video_decoder_layer_num):
            self.video_decoder_layers.append(att(module_dim))

        self.question_decoder_layers = nn.ModuleList([])
        for layer in range(self.question_decoder_layer_num):
            self.question_decoder_layers.append(att(module_dim))




    def forward(self, video_feat, question_embedding, question_bert):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        video_feat = self.video(video_feat)
        question_feat = self.question(question_embedding)
        context_feat = self.context(question_bert)

        for layer in range(self.video_encoder_layer_num):
            video_feat = video_feat.permute(1, 0, 2)
            video_feat = self.video_encoder_layers[layer](video_feat, video_feat, video_feat).permute(1, 0, 2)

        for layer in range(self.context_encoder_layer_num):
            context_feat = context_feat.permute(1, 0, 2)
            context_feat = self.context_encoder_layers[layer](context_feat, context_feat, context_feat).permute(1, 0, 2)

        for layer in range(self.question_encoder_layer_num):
            question_feat = question_feat.permute(1, 0, 2)
            question_feat = self.question_encoder_layers[layer](question_feat, question_feat, question_feat).permute(1, 0, 2)

        for layer in range(self.video_decoder_layer_num):
            video_feat = video_feat.permute(1, 0, 2)
            context_feat = context_feat.permute(1, 0, 2)
            video_feat = self.video_decoder_layers[layer](video_feat, context_feat, context_feat).permute(1, 0, 2)

        for layer in range(self.question_encoder_layer_num):
            question_feat = question_feat.permute(1, 0, 2)
            video_feat = video_feat.permute(1, 0, 2)
            out = self.question_encoder_layers[layer](question_feat, video_feat, video_feat).permute(1, 0, 2)

        return out


class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out




class HCRNNetwork(nn.Module):
    def __init__(self, vision_dim, module_dim, word_dim, k_max_frame_level,
                 k_max_clip_level, spl_resolution, vocab, question_type, calculate_loss = False):
        super(HCRNNetwork, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        if self.question_type in ['action', 'transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.linguistic_input_video_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(vision_dim=vision_dim,
                                                     module_dim=module_dim)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes)

        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,
                question_len, video_questions, video_questions_len, _):
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
        loss = 0.0
        batch_size = question.size(0)

        question_video_embedding, text_video_embedding = self.linguistic_input_video_unit(video_questions,
                                                                                          video_questions_len.cpu())
        question_embedding, text_embedding = self.linguistic_input_unit(question, question_len.cpu())

        video = torch.cat((video_appearance_feat.sum(-2), video_motion_feat), dim=1)

        visual_embedding = self.visual_input_unit(video, text_embedding,
                                                  text_video_embedding)

        visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

        out = self.output_unit(question_embedding, visual_embedding)

        return out
