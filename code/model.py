import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_candidates
import math

class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.gru.flatten_parameters()

    def forward(self, x, seq_len, max_num_frames):
        sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
        _, original_idx = torch.sort(sorted_idx, dim=0, descending=False)
        if self.batch_first:
            sorted_x = x.index_select(0, sorted_idx)
        else:
            sorted_x = x.index_select(1, sorted_idx)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            sorted_x, sorted_seq_len.cpu().data.numpy(), batch_first=self.batch_first)

        out, state = self.gru(packed_x)

        unpacked_x, unpacked_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)

        if self.batch_first:
            out = unpacked_x.index_select(0, original_idx)
            if out.shape[1] < max_num_frames:
                out = F.pad(out, [0, 0, 0, max_num_frames - out.shape[1]])
        else:
            out = unpacked_x.index_select(1, original_idx)
            if out.shape[0] < max_num_frames:
                out = F.pad(out, [0, 0, 0, 0, 0, max_num_frames - out.shape[0]])

        return out


class NodeInitializer(nn.Module):
    def __init__(self, node_num, input_dim, node_dim, dropout):
        super().__init__()
        self.node_num = node_num
        self.dropout = dropout
        self.rnn = DynamicGRU(input_dim, node_dim >> 1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(node_dim, node_dim)

    def forward(self, x, mask):
        length = mask.sum(dim=-1)
        x = self.rnn(x, length, self.node_num)
        # x_trans = F.leaky_relu(self.fc(x))
        # x_dropout = F.dropout(x, self.dropout, self.training)
        return x, x


class GraphConvolution(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.node_dim = node_dim
        self.wvv = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.wss = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.wvs = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.wsv = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.wgatev = nn.Linear(self.node_dim << 1, self.node_dim)
        self.wgates = nn.Linear(self.node_dim << 1, self.node_dim)

    def forward(self, v, avv, s, ass):
        vs = torch.matmul(v, s.transpose(2, 1))
        avs = torch.softmax(vs, -1)
        asv = torch.softmax(vs.transpose(2, 1), -1)
        v = F.leaky_relu(self.wvv(torch.matmul(avv, v)))
        s = F.leaky_relu(self.wss(torch.matmul(ass, s)))

        hv = self.wsv(torch.matmul(avs, s))
        zv = torch.sigmoid(self.wgatev(torch.cat([v, hv], dim=-1)))
        v = zv * v + (1 - zv) * hv
        v = F.leaky_relu(v)

        hs = self.wvs(torch.matmul(asv, v))
        zs = torch.sigmoid(self.wgates(torch.cat([s, hs], dim=-1)))
        s = zs * s + (1 - zs) * hs
        s = F.leaky_relu(s)
        return v, s


class ResNet_GRU(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dropout = opt.dropout
        self.fusion_fc_0 = nn.Linear(opt.node_dim*2, opt.node_dim*2)

        self.fusion_gru_1 = NodeInitializer(
            node_num = opt.max_frames_num,
            input_dim = opt.node_dim*2,
            node_dim = opt.node_dim*2,
            dropout = self.dropout
        )
        self.fusion_fc_1 = nn.Linear(opt.node_dim*2, opt.node_dim*2)

        self.fusion_gru_2 = NodeInitializer(
            node_num = opt.max_frames_num,
            input_dim = opt.node_dim*2,
            node_dim = opt.node_dim*2,
            dropout = self.dropout
        )
        self.fusion_fc_2 = nn.Linear(opt.node_dim*2, opt.node_dim*2)

        self.fusion_gru_3 = NodeInitializer(
            node_num = opt.max_frames_num,
            input_dim = opt.node_dim*2,
            node_dim = opt.node_dim*2,
            dropout = self.dropout
        )
        self.fusion_fc_3 = nn.Linear(opt.node_dim*2, opt.node_dim*2)

    def forward(self, muti, frame_mask, identity = None):
        if identity is not None:
            identity_1 = identity
        else:
            identity_1 = muti
        _, muti = self.fusion_gru_1(muti, frame_mask)
        muti = self.fusion_fc_1(muti)
        muti = muti + identity_1
        muti = F.relu(muti)

        identity_2 = muti
        _, muti = self.fusion_gru_2(muti, frame_mask)
        muti = self.fusion_fc_2(muti)
        muti = muti + identity_2
        muti = F.relu(muti)

        identity_3 = muti
        _, muti = self.fusion_gru_3(muti, frame_mask)
        muti = self.fusion_fc_3(muti)
        muti = muti + identity_3
        muti = F.relu(muti)

        return muti


class SelfAttention(nn.Module):
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):   
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:  
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)   
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask.type(torch.cuda.FloatTensor)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class mutiLayer_SemanticNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.v_actions_mat_1 = nn.Linear(opt.node_dim, opt.node_dim)
        self.s_selfAttention = SelfAttention(opt.node_dim, 1, opt.dropout)

    def forward(self, actions, s, v):
        # s_actions_first_semantic_1 = actions.unsqueeze(-1).expand(s.shape[0],s.shape[1],s.shape[2]).type(torch.cuda.FloatTensor)
        # s_actions_1 = s * s_actions_first_semantic_1
        # s_actions_1 = torch.sum(s_actions_1,1)
        s_actions_1 = self.s_selfAttention(s, actions)
        s_actions_1 = torch.sum(s_actions_1, dim=1)
        s_actions_1 = s_actions_1.unsqueeze(1).expand(s.shape[0], v.shape[1], s.shape[2])
        v_actions_1 = self.v_actions_mat_1(v)
        muti_actions_1 = torch.cat((v_actions_1, s_actions_1), 2)

        return muti_actions_1


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.alpha = opt.alpha
        self.dropout = opt.dropout
        self.max_frames_num = opt.max_frames_num
        self.max_words_num = opt.max_words_num
        self.max_srl_num = opt.max_srl_num
        self.vnode_initializer = NodeInitializer(node_num=opt.max_frames_num,
                                                 input_dim=opt.frame_feature_dim,
                                                 node_dim=opt.node_dim, dropout=self.dropout)
        self.wnode_initializer = NodeInitializer(node_num=opt.max_words_num,
                                                 input_dim=opt.word_feature_dim,
                                                 node_dim=opt.node_dim, dropout=self.dropout)
        
        self.s_selfAttention = SelfAttention(opt.node_dim, 1, opt.dropout)

        self.candidates, self.window_widths = generate_candidates(opt.max_frames_num, opt.window_widths)
        self.candidates = torch.from_numpy(self.candidates).float().cuda()

        self.conv_cls = nn.ModuleList([
            nn.Conv1d(opt.node_dim << 1, 1, w * 2, padding=w//2)
            for w in self.window_widths
        ])
        self.conv_reg = nn.ModuleList([
            nn.Conv1d(opt.node_dim << 1, 2, w * 2, padding=w//2)
            for w in self.window_widths
        ])
        self.conv_cls_actions_1 = nn.ModuleList([
            nn.Conv1d(opt.node_dim << 1, 1, w * 2, padding=w//2)
            for w in self.window_widths
        ])
        # self.conv_reg_actions_1 = nn.ModuleList([
        #     nn.Conv1d(opt.node_dim << 1, 2, w * 2, padding=w//2)
        #     for w in self.window_widths
        # ])
        self.conv_cls_actions_2 = nn.ModuleList([
            nn.Conv1d(opt.node_dim << 1, 1, w * 2, padding=w//2)
            for w in self.window_widths
        ])
        # self.conv_reg_actions_2 = nn.ModuleList([
        #     nn.Conv1d(opt.node_dim << 1, 2, w * 2, padding=w//2)
        #     for w in self.window_widths
        # ])
        # self.conv_cls_actions_3 = nn.ModuleList([
        #     nn.Conv1d(opt.node_dim << 1, 1, w * 2, padding=w//2)
        #     for w in self.window_widths
        # ])
        # self.conv_reg_actions_3 = nn.ModuleList([
        #     nn.Conv1d(opt.node_dim << 1, 2, w * 2, padding=w//2)
        #     for w in self.window_widths
        # ])
        self.conv_cls_objects_1 = nn.ModuleList([
            nn.Conv1d(opt.node_dim << 1, 1, w * 2, padding=w//2)
            for w in self.window_widths
        ])
        # self.conv_reg_objects_1 = nn.ModuleList([
        #     nn.Conv1d(opt.node_dim << 1, 2, w * 2, padding=w//2)
        #     for w in self.window_widths
        # ])
        self.conv_cls_objects_2 = nn.ModuleList([
            nn.Conv1d(opt.node_dim << 1, 1, w * 2, padding=w//2)
            for w in self.window_widths
        ])
        # self.conv_reg_objects_2 = nn.ModuleList([
        #     nn.Conv1d(opt.node_dim << 1, 2, w * 2, padding=w//2)
        #     for w in self.window_widths
        # ])

        self.criterion_score = nn.BCEWithLogitsLoss()
        # self.criterion_score = nn.MSELoss()
        self.criterion_reg = nn.SmoothL1Loss()

        self.v_fc = nn.Linear(opt.node_dim, opt.node_dim)
        self.s_fc = nn.Linear(opt.node_dim, opt.node_dim)

        self.mutiLayer_action_1 = mutiLayer_SemanticNet(opt)
        self.mutiLayer_action_2 = mutiLayer_SemanticNet(opt)
        # self.mutiLayer_action_3 = mutiLayer_SemanticNet(opt)
        self.mutiLayer_object_1 = mutiLayer_SemanticNet(opt)
        self.mutiLayer_object_2 = mutiLayer_SemanticNet(opt)
        
        self.resGRU_layers_global = ResNet_GRU(opt)
        self.resGRU_layers_actions_1 = ResNet_GRU(opt)
        self.resGRU_layers_actions_2 = ResNet_GRU(opt)
        # self.resGRU_layers_actions_3 = ResNet_GRU(opt)
        self.resGRU_layers_objects_1 = ResNet_GRU(opt)
        self.resGRU_layers_objects_2 = ResNet_GRU(opt)
        # self.s_gloval_attention = nn.Linear(opt.node_dim, 1)
        
        self.weighted_predict_scores = nn.Linear(5,1)
        self.weighted_offset = nn.Linear(6,2)


    def forward(self, vfeats, frame_mask, frame_mat, wfeats, word_mask, word_mats, label, scores, objects, actions, srl_num): #label:[bs,2]  scores:[bs,218] objects/actions:[bs,max_srl_num,max_words_num] srl_num:[bs]
        v_mask, v = self.vnode_initializer(vfeats, frame_mask) #[bs,max_frames_num,node_dim]
        s_mask, s = self.wnode_initializer(wfeats, word_mask) #[bs,max_words_num,node_dim] #s_mask[1,1,1,1,...,0,0,0,0] ; s with fc,relu,dropout

        v_identity = v #[bs,max_frames_num,node_dim]
        s_identity = self.s_selfAttention(s,word_mask)
        s_identity = torch.sum(s_identity, dim=1) #[bs,node_dim(sum)]
        s_identity = torch.unsqueeze(s_identity,1).expand(s.shape[0], v.shape[1], s.shape[2]) #[bs,max_frames_num,node_dim]
        identity = torch.cat((v_identity, s_identity), 2) #[bs,max_frames_num,node_dim*2]

        v = self.v_fc(v)
        s = self.s_fc(s)

        muti_actions_1 = self.mutiLayer_action_1(actions[:,0,:], s, v)
        muti_actions_2 = self.mutiLayer_action_2(actions[:,1,:], s, v)
        # muti_actions_3 = self.mutiLayer_action_3(actions[:,2,:], s, v)
        muti_actions_1 = self.resGRU_layers_actions_1(muti_actions_1, frame_mask)
        muti_actions_2 = self.resGRU_layers_actions_2(muti_actions_2, frame_mask)
        # muti_actions_3 = self.resGRU_layers_actions_3(muti_actions_3, frame_mask)
        muti_objects_1 = self.mutiLayer_object_1(objects[:,0,:], s, v)
        muti_objects_2 = self.mutiLayer_object_2(objects[:,1,:], s, v)
        muti_objects_1 = self.resGRU_layers_objects_1(muti_objects_1, frame_mask)
        muti_objects_2 = self.resGRU_layers_objects_2(muti_objects_2, frame_mask)

        # s = F.leaky_relu(self.s_embed(s.permute(0, 2, 1)).permute(0, 2, 1)) #[bs,1,node_dim]
        # s = s.expand(s.shape[0], v.shape[1], s.shape[2]) #[bs,max_frames_num,node_dim]
        # muti_global = torch.cat((v, s), 2) #[bs,max_frames_num,node_dim*2]
        # muti_global = F.relu(torch.cat((v, s), 2) + identity) #[bs,max_frames_num,node_dim*2]
        muti_global = self.resGRU_layers_global(identity, frame_mask)

        predict_scores = torch.cat([
            self.conv_cls[i](muti_global.permute(0,2,1)).permute(0,2,1)
            for i in range(len(self.conv_cls))
        ], dim=1)
        offset = torch.cat([
            self.conv_reg[i](muti_global.permute(0,2,1)).permute(0,2,1)
            for i in range(len(self.conv_reg))
        ], dim=1)
        actions_predict_scores_1 = torch.cat([
            self.conv_cls_actions_1[i](muti_actions_1.permute(0,2,1)).permute(0,2,1)
            for i in range(len(self.conv_cls))
        ], dim=1)
        # actions_offset_1 = torch.cat([
        #     self.conv_reg_actions_1[i](muti_actions_1.permute(0,2,1)).permute(0,2,1)
        #     for i in range(len(self.conv_reg))
        # ], dim=1)
        actions_predict_scores_2 = torch.cat([
            self.conv_cls_actions_2[i](muti_actions_2.permute(0,2,1)).permute(0,2,1)
            for i in range(len(self.conv_cls))
        ], dim=1)
        # actions_offset_2 = torch.cat([
        #     self.conv_reg_actions_2[i](muti_actions_2.permute(0,2,1)).permute(0,2,1)
        #     for i in range(len(self.conv_reg))
        # ], dim=1)
        # actions_predict_scores_3 = torch.cat([
        #     self.conv_cls_actions_3[i](muti_actions_3.permute(0,2,1)).permute(0,2,1)
        #     for i in range(len(self.conv_cls))
        # ], dim=1)
        # actions_offset_3 = torch.cat([
        #     self.conv_reg_actions_3[i](muti_actions_3.permute(0,2,1)).permute(0,2,1)
        #     for i in range(len(self.conv_reg))
        # ], dim=1)
        objects_predict_scores_1 = torch.cat([
            self.conv_cls_objects_1[i](muti_objects_1.permute(0,2,1)).permute(0,2,1)
            for i in range(len(self.conv_cls))
        ], dim=1)
        # objects_offset_1 = torch.cat([
        #     self.conv_reg_objects_1[i](muti_objects_1.permute(0,2,1)).permute(0,2,1)
        #     for i in range(len(self.conv_reg))
        # ], dim=1)
        objects_predict_scores_2 = torch.cat([
            self.conv_cls_objects_2[i](muti_objects_2.permute(0,2,1)).permute(0,2,1)
            for i in range(len(self.conv_cls))
        ], dim=1)
        # objects_offset_2 = torch.cat([
        #     self.conv_reg_objects_2[i](muti_objects_2.permute(0,2,1)).permute(0,2,1)
        #     for i in range(len(self.conv_reg))
        # ], dim=1)

        # predict_scores = 0.5*predict_scores.squeeze() + 0.25*actions_predict_scores_1.squeeze() + 0.125*actions_predict_scores_2.squeeze() + 0.0625*actions_predict_scores_2.squeeze()
        predict_scores = self.weighted_predict_scores(torch.cat((predict_scores, actions_predict_scores_1, actions_predict_scores_2, objects_predict_scores_1, objects_predict_scores_2), dim=-1)).squeeze()
        # predict_scores = self.weighted_predict_scores(torch.cat((predict_scores, actions_predict_scores_1, actions_predict_scores_2), dim=-1)).squeeze()

        # predict_scores = self.weighted_predict_scores(torch.cat((predict_scores, actions_predict_scores_1), dim=-1)).squeeze()
        # offset = (offset + actions_offset + objects_offset)/3
        # offset = self.weighted_offset(torch.cat((torch.cat((offset, actions_offset_1), dim=-1),
        #                                                     objects_offset_1), dim=-1)).squeeze()

        if self.training:
            indices = scores.max(dim=1)[1]
        else:
            indices = predict_scores.max(dim=1)[1]
        predict_box = self.candidates[indices]  # [bs,2]
        predict_reg = offset[range(offset.shape[0]), indices]  # [bs,2]
        refined_box = predict_box + predict_reg  # [bs,2]

        cls_loss = self.criterion_score(predict_scores, scores)
        reg_loss = self.criterion_reg(refined_box, label.float())
        loss = cls_loss + self.alpha * reg_loss
        return refined_box, loss
