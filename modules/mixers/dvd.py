import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. 定义多头图注意力层 (Multi-Head GAT) ---
# 对应论文公式 (8), (9), (10)
# 这里的图是全连接的 (Fully Connected)，所以我们不需要邻接矩阵，直接做 Attention
class MultiHeadGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads):
        super(MultiHeadGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # W 矩阵: 将输入的 hidden_state 映射到 GAT 的特征空间
        # 输出维度: n_heads * hidden_dim
        self.W = nn.Linear(input_dim, n_heads * hidden_dim, bias=False)
        
        # Attention 向量 a: 用于计算节点间的注意力权重
        # 输入是拼接的两个节点特征 [Wh_i || Wh_j]，所以是 2 * hidden_dim
        self.att_a = nn.Parameter(th.Tensor(1, n_heads, 2 * hidden_dim)) # 创建一个矩阵并且将其变成可学习的
        nn.init.xavier_uniform_(self.att_a.data, gain=1.414) # 初始化
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, return_attention=False):
        # h shape: (batch_size, n_agents, input_dim)
        bs, n_agents, _ = h.size()

        h_prime = self.W(h)
        h_prime = h_prime.view(bs, n_agents, self.n_heads, self.hidden_dim)
        h_prime = h_prime.permute(0, 2, 1, 3)

        h_i = h_prime.unsqueeze(3)
        h_j = h_prime.unsqueeze(2)

        h_cat = th.cat([h_i.repeat(1, 1, 1, n_agents, 1),
                        h_j.repeat(1, 1, n_agents, 1, 1)], dim=-1)

        e = (h_cat * self.att_a.unsqueeze(2).unsqueeze(3)).sum(dim=-1)
        e = self.leaky_relu(e)

        attention = F.softmax(e, dim=-1)

        h_new = th.matmul(attention, h_prime)
        h_new = F.elu(h_new)

        if return_attention:
            return h_new, attention
        return h_new


# --- 2. 定义 DVD Mixer ---
class DVDMixer(nn.Module):
    def __init__(self, args):
        super(DVDMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        # 强制 abs=False，或者依赖 args
        self.abs = getattr(self.args, 'abs', True) 
        
        self.rnn_hidden_dim = args.rnn_hidden_dim 
        self.n_heads = getattr(args, 'dvd_heads', 4)
        self.gat_dim = getattr(args, 'gat_embed_dim', 32)

        # 组件 1: GAT
        self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.n_heads)

        self.combined_dim = self.gat_dim

        # 创新 1：渐进式 abs 退火
        self.use_abs_anneal = getattr(args, 'use_abs_anneal', False)
        self.abs_anneal_steps = getattr(args, 'abs_anneal_steps', 2000000)

        # 创新 2：因果感知探索（输出 attention 熵）
        self.use_causal_explore = getattr(args, 'use_causal_explore', False)

        # 创新 3：多头竞争加权
        self.use_head_competition = getattr(args, 'use_head_competition', False)
        if self.use_head_competition:
            self.head_logits = nn.Parameter(th.zeros(self.n_heads))

        # 组件 2: 状态超网络 (生成 W1)
        self.hyper_w_1_state = nn.Linear(self.state_dim, self.n_heads * self.embed_dim * self.combined_dim)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # 组件 3: 第二层混合 (W_final)
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        else:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(hypernet_embed, self.embed_dim))
            
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

        # [修正 2] 针对 Non-Monotonic 的特殊初始化
        # 如果 abs=False，我们将最后一层的权重初始化得非常小，避免初始 Q 值震荡
        self.init_weights()

        # 归一化
        # self.layernorm = nn.LayerNorm(self.embed_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 默认使用正交初始化或 Xavier
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        
        # 如果是非单调模式，特别处理超网络的输出层
        if not self.abs:
            # 使 W1 初始值很小
            self.hyper_w_1_state.weight.data.mul_(0.01)
            # 使 W_final 初始值很小
            if isinstance(self.hyper_w_final, nn.Linear):
                self.hyper_w_final.weight.data.mul_(0.01)
            else:
                # 如果是 Sequential，处理最后一层
                self.hyper_w_final[-1].weight.data.mul_(0.01)

    def forward(self, agent_qs, states, hidden_states, t_env=0):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        hidden_states = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

        # Step 1: GAT 采样（可选返回 attention 用于计算熵）
        need_attention = self.use_causal_explore
        if need_attention:
            graphs_out, attention = self.gat(hidden_states, return_attention=True)
        else:
            graphs_out = self.gat(hidden_states)

        graphs_final = graphs_out

        # Step 2: 计算 W1
        w1_state = self.hyper_w_1_state(states)
        w1_state = w1_state.view(-1, self.n_heads, self.embed_dim, self.combined_dim)

        graphs_T = graphs_final.permute(0, 1, 3, 2)

        w1_heads = th.matmul(w1_state, graphs_T)

        # 创新 1：渐进式 abs 退火
        if self.use_abs_anneal:
            abs_weight = max(0.0, 1.0 - t_env / self.abs_anneal_steps)
            w1_heads = abs_weight * th.abs(w1_heads) + (1.0 - abs_weight) * w1_heads
        elif self.abs:
            w1_heads = th.abs(w1_heads)

        # 创新 3：多头竞争加权 vs 简单均值
        if self.use_head_competition:
            head_weights = F.softmax(self.head_logits, dim=0)
            w1 = (w1_heads * head_weights.view(1, self.n_heads, 1, 1)).sum(dim=1)
        else:
            w1 = w1_heads.mean(dim=1)
        w1 = w1.permute(0, 2, 1)

        # Step 3: 混合
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        w_final = self.hyper_w_final(states)
        if self.use_abs_anneal:
            abs_weight = max(0.0, 1.0 - t_env / self.abs_anneal_steps)
            w_final_raw = w_final
            w_final = abs_weight * th.abs(w_final_raw) + (1.0 - abs_weight) * w_final_raw
        elif self.abs:
            w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.embed_dim, 1)

        v = self.V(states).view(-1, 1, 1)

        y = th.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1, 1)

        # 创新 2：因果感知探索 — 返回 attention 熵
        if need_attention:
            # attention: (bs*T, heads, agents, agents)
            log_attn = th.log(attention + 1e-8)
            entropy = -(attention * log_attn).sum(dim=-1)  # (bs*T, heads, agents)
            attn_entropy = entropy.mean(dim=-1).mean(dim=-1)  # (bs*T,)
            attn_entropy = attn_entropy.view(bs, -1, 1)  # (bs, T, 1)
            return q_tot, attn_entropy

        return q_tot

###########################################
# 把自己的hidden_state加入了
###########################################
# class DVDMixer(nn.Module):
#     def __init__(self, args):
#         super(DVDMixer, self).__init__()
#         self.args = args
#         self.n_agents = args.n_agents
#         self.state_dim = int(np.prod(args.state_shape))
#         self.embed_dim = args.mixing_embed_dim
#         self.abs = getattr(self.args, 'abs', True)
        
#         # DVD 特有参数
#         self.rnn_hidden_dim = args.rnn_hidden_dim 
#         self.n_heads = getattr(args, 'dvd_heads', 4)
#         self.gat_dim = getattr(args, 'gat_embed_dim', 32)

#         # --- 组件 1: 轨迹图生成器 (GAT) ---
#         self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.n_heads)

#         # --- [修改点 1] ---
#         # 计算拼接后的维度: GAT输出维度 + 原始Hidden维度
#         self.combined_dim = self.gat_dim + self.rnn_hidden_dim

#         # 组件 2: 状态超网络 (用于生成 W1)
#         # 输出维度变大，因为我们要处理 (GAT特征 + 原始特征)
#         # Old: ... * self.gat_dim
#         # New: ... * self.combined_dim
#         self.hyper_w_1_state = nn.Linear(self.state_dim, self.n_heads * self.embed_dim * self.combined_dim)
        
#         self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

#         # --- 组件 3: 第二层混合 (W_final) ---
#         if getattr(args, "hypernet_layers", 1) == 1:
#             self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
#         else:
#             hypernet_embed = self.args.hypernet_embed
#             self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
#                                                nn.ReLU(inplace=True),
#                                                nn.Linear(hypernet_embed, self.embed_dim))
            
#         self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
#                                nn.ReLU(inplace=True),
#                                nn.Linear(self.embed_dim, 1))

#         # 初始化
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.fill_(0)

#     def forward(self, agent_qs, states, hidden_states):
#         bs = agent_qs.size(0) 
#         states = states.reshape(-1, self.state_dim)
#         agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
#         hidden_states = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

#         # -----------------------------------------------------------
#         # Step 1: GAT 采样
#         # -----------------------------------------------------------
#         # graphs_out: (bs*T, n_heads, n_agents, gat_dim)
#         graphs_out = self.gat(hidden_states)

#         # --- [修改点 2: 拼接残差连接] ---
#         # 目标: 将原始 hidden_states 拼接到 graphs_out 后面
        
#         # 1. 扩展 hidden_states 维度以匹配 Multi-Head
#         # (bs*T, n_agents, rnn_dim) -> (bs*T, 1, n_agents, rnn_dim) -> (bs*T, n_heads, n_agents, rnn_dim)
#         h_expanded = hidden_states.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

#         # 2. 在最后一个维度拼接
#         # 结果维度: (bs*T, n_heads, n_agents, gat_dim + rnn_dim)
#         graphs_combined = th.cat([graphs_out, h_expanded], dim=-1)

#         # -----------------------------------------------------------
#         # Step 2: 计算 W1
#         # -----------------------------------------------------------
#         # 2.1 生成状态表示 f_s(s)
#         # 注意这里维度是 combined_dim
#         w1_state = self.hyper_w_1_state(states)
#         w1_state = w1_state.view(-1, self.n_heads, self.embed_dim, self.combined_dim)
        
#         # 2.2 调整维度准备矩阵乘法
#         # (bs*T, n_heads, agents, combined_dim) -> (bs*T, n_heads, combined_dim, agents)
#         graphs_T = graphs_combined.permute(0, 1, 3, 2)
        
#         # 2.3 MatMul
#         # (bs*T, heads, embed, combined) @ (bs*T, heads, combined, agents) 
#         # -> (bs*T, heads, embed, agents)
#         w1_heads = th.matmul(w1_state, graphs_T)
        
#         if self.abs:
#             w1_heads = th.abs(w1_heads)
            
#         w1 = w1_heads.mean(dim=1)
#         w1 = w1.permute(0, 2, 1)

#         # -----------------------------------------------------------
#         # Step 3: QMIX 标准流程
#         # -----------------------------------------------------------
#         b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
#         hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
#         w_final = self.hyper_w_final(states)
#         if self.abs:
#             w_final = th.abs(w_final)
#         w_final = w_final.view(-1, self.embed_dim, 1)
        
#         v = self.V(states).view(-1, 1, 1)
        
#         y = th.bmm(hidden, w_final) + v
#         q_tot = y.view(bs, -1, 1)
        
#         return q_tot