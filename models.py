"""
This file contains the pytorch model definitions for the dataset using
the top 1000 select tags.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, Dropout, ReLU, Sigmoid, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d, LeakyReLU


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 3, 3)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class Conv_2d(nn.Module):
	def __init__(self, input_channels, output_channels, shape=3, pooling=2):
		super(Conv_2d, self).__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape//2)
		self.bn = nn.BatchNorm2d(output_channels)
		self.relu = nn.ReLU()
		self.mp = nn.MaxPool2d(pooling)

	def forward(self, x):
		out = self.mp(self.relu(self.bn(self.conv(x))))
		return out


class Conv_emb(nn.Module):
	def __init__(self, input_channels, output_channels):
		super(Conv_emb, self).__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, 1)
		self.bn = nn.BatchNorm2d(output_channels)
		self.relu = nn.ReLU()

	def forward(self, x):
		out = self.relu(self.bn(self.conv(x)))
		return out

class AudioEncoder(nn.Module):
    def __init__(self, size_w_rep=128):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = Sequential(
            nn.BatchNorm2d(1), #256x48
            Conv_2d(1, 128, pooling=2),#128x24
            Conv_2d(128, 128, pooling=2), #64x12
            Conv_2d(128, 256, pooling=2), #32x6
            Conv_2d(256, 256, pooling=2), #16x3
            Conv_2d(256, 256, pooling=(1,2)), #8x3
            Conv_2d(256, 256, pooling=(1,2)), #4x3
            Conv_2d(256, 512, pooling=2), #2x1
            Flatten()
        )
        self.fc_audio = Sequential(
            Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            Dropout(0.5),
            Linear(512, size_w_rep),
            nn.LayerNorm(size_w_rep, eps=1e-6)
            )

    def forward(self, x):
        z = self.audio_encoder(x)
        z_d = self.fc_audio(z)
        return z, z_d


class CFEncoder(nn.Module):
    def __init__(self, embedding_dim, d_model, dropout=0.1):
        super(CFEncoder, self).__init__()
        self.d_model = d_model


        self.fc = Sequential(
                Linear(embedding_dim, 128),
                nn.ReLU(),
                Dropout(0.3),
                Linear(128, d_model),
                nn.LayerNorm(d_model, eps=1e-6)
            )

    def forward(self, x, mask=None):
        return self.fc(x)


class FCFusion(nn.Module):
    def __init__(self, d_model):
        super(FCFusion, self).__init__()

        self.fc = Sequential(
                Linear(256, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model, eps=1e-6)
            )

    def forward(self, cf_x, gnr_y, mask=None):
        x = torch.cat((cf_x, gnr_y), -1)
        return self.fc(x)

class Multiobjective(nn.Module):
    def __init__(self, d_out_1, d_out_2):
        super(Multiobjective, self).__init__()

        self.out_1 = nn.Linear(128, d_out_1)
        self.out_2 = nn.Linear(128, d_out_2)

    def forward(self, audio_x):
        x = self.out_1(audio_x)
        y = self.out_2(audio_x)
        return x, y


class TagMeanEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim, d_model, emb_file, dropout=0.1):
        super(TagMeanEncoder, self).__init__()
        self.d_model = d_model

        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(np.load(emb_file)))

        self.fc = nn.Linear(word_embedding_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tags, z_audio, mask=None):
        tag_embeddings = self.embeddings(tags)
        q = self.fc(tag_embeddings)
        # get mean average over non-0 embeddings
        q = torch.mul(q.sum(1), torch.repeat_interleave(1/((mask !=0).sum(-1).float()), repeats=self.d_model, dim=-1))
        #q = torch.mean(q, dim=1)
        q = self.dropout(q)
        q = self.layer_norm(q)
        return q, None


class TagSelfAttentionEncoder(nn.Module):
    def __init__(self, max_num_tags, word_embedding_dim, n_head, d_model, d_k, d_v, emb_file, dropout=0.1):
        super(TagSelfAttentionEncoder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(np.load(emb_file)), freeze=False)

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tags, q=None, mask=None):
        # in self attention q comes from the same source
        tag_embeddings = self.embeddings(tags)
        k = tag_embeddings
        v = tag_embeddings
        q = tag_embeddings

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # No residual for now
        # residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # q += residual

        # we sum all the values that were multiplied with attention weights
        q = q.sum(1)
        q = self.layer_norm(q)
        q = q.view(-1, q.shape[-1])

        return q, attn


class TagEncoder(nn.Module):
    def __init__(self):
        super(TagEncoder, self).__init__()

        self.tag_encoder = Sequential(
            Linear(1000, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 1152),
            BatchNorm1d(1152),
            ReLU(),
            Dropout(.25),
        )

        self.fc_tag = Sequential(
            Linear(1152, 1152, bias=False),
            Dropout(.25),
        )

    def forward(self, tags):
        z = self.tag_encoder(tags)
        z_d = self.fc_tag(z)
        return z, z_d


class TagDecoder(nn.Module):
    def __init__(self):
        super(TagDecoder, self).__init__()

        self.tag_decoder = Sequential(
            Linear(128, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 1000),
            BatchNorm1d(1000),
            Sigmoid(),
        )

    def forward(self, z):
        return self.tag_decoder(z)
