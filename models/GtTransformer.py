import numpy as np

import torch
from torch import nn
import random

class GtTransformer(nn.Module):
    """
    The one from the assignment...
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        super(GtTransformer, self).__init__()
        assert hidden_dim % num_heads == 0

        self.flatten = nn.Flatten(start_dim=1)
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        seed_torch(0)

        self.word_embeddings = nn.Embedding(self.input_size, self.word_embedding_dim)
        self.position_embeddings = nn.Embedding(self.max_length, self.word_embedding_dim)

        self.heads = dict()
        for head in range(num_heads):
            self.heads[head] = dict()
            self.heads[head]['k'] = nn.Linear(self.hidden_dim, self.dim_k).to(self.device)
            self.heads[head]['v'] = nn.Linear(self.hidden_dim, self.dim_v).to(self.device)
            self.heads[head]['q'] = nn.Linear(self.hidden_dim, self.dim_q).to(self.device)

        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)


        self.ffn_l1 = nn.Linear(input_size, self.dim_feedforward)
        self.ffn_relu = nn.ReLU()
        self.ffn_l2 = nn.Linear(self.dim_feedforward, input_size)
        self.ffn_norm = nn.LayerNorm(self.input_size)

        self.final_linear = nn.Linear(input_size, self.output_size)

    def forward(self, inputs):
#         embeds = self.embed(inputs)

        hidden_states = self.multi_head_attention(inputs)
        x = self.flatten(hidden_states)
        outputs = self.feedforward_layer(x)
        scores = self.final_layer(outputs)
        return scores


    def embed(self, inputs):
        word_embeddings = self.word_embeddings(inputs)

        pos_embeddings = self.position_embeddings(torch.arange(inputs.shape[1]).to(self.device))
        return word_embeddings + pos_embeddings

    def multi_head_attention(self, inputs):
        attentions = None
        for index in range(self.num_heads):
            head = self.heads[index]
            k = head['k'](inputs)
            q = head['q'](inputs)
            v = head['v'](inputs)

            score = torch.bmm(q,  k.permute(0, 2, 1))
            d_k = torch.zeros(score.shape).to(self.device) + self.dim_k

            norm_score = score / torch.sqrt(d_k)
            sm_score = self.softmax(norm_score)
            attention = torch.bmm(sm_score, v)

            attentions = attention if attentions is None else torch.cat([attentions, attention], dim=2).to(self.device)

        projection = self.attention_head_projection(attentions)

        outputs = self.norm_mh(inputs + projection)

        return outputs


    def feedforward_layer(self, inputs):
        x = self.ffn_l1(inputs)
        x = self.ffn_relu(x)
        x = self.ffn_l2(x)

        x = self.ffn_norm(x + inputs)
        return x


    def final_layer(self, inputs):
        outputs = self.final_linear(inputs)
        return outputs


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True