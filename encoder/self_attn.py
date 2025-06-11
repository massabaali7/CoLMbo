import torch
import torch.nn as nn
from encoder.mha import MultiHeadAttention
from encoder.attentive_pooling import SelfAttentionPooling

class FlippedReLU(nn.Module):
    def __init__(self):
        super(FlippedReLU, self).__init__()

    def forward(self, x):
        return torch.where(x < 0, x, torch.zeros_like(x))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, prob_phn=None, mask=None, lambda_val=None):
        attn_output, attn_mask = self.self_attn(x, x, x, prob_phn=prob_phn, mask=mask, lambda_val=lambda_val)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn_mask


class TransformerSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, number_Of_spks, dropout=0.0):
        """EncoderBlock.

        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        # Attention layer
        self.self_mha_attn = EncoderLayer(input_dim, num_heads, dim_feedforward*8,dropout) 
        self.attn_pooling = SelfAttentionPooling(input_dim)
        self.emb1 = nn.Linear(input_dim*2, dim_feedforward*8) 
        self.emb2 = nn.Linear(input_dim*2, dim_feedforward*8) 
        self.emb2.weight.data = self.emb1.weight.data.clone()
        self.emb2.bias.data = self.emb1.bias.data.clone()
        self.bn = nn.BatchNorm1d(dim_feedforward*8)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim_feedforward*8, number_Of_spks)
        self.flipped_relu = FlippedReLU()


    def forward(self, x, prob_phn=None, mask=None, lambda_val=None):
        # Attention part
        attn_out, attn_mask = self.self_mha_attn(x,prob_phn=prob_phn, mask=mask, lambda_val=lambda_val)
        attn_mask= attn_mask.squeeze(1)
        attn_out_mean,attn_out_std = self.attn_pooling(attn_out,attn_mask)
        attn_concat = torch.cat((attn_out_mean, attn_out_std),dim=1).to(dtype=torch.float32)
        
        emb1 = self.emb1(attn_concat).to(dtype=torch.float32)
        emb1 = self.act(emb1)

        emb2 = self.emb2(attn_concat).to(dtype=torch.float32)
        emb2 = self.flipped_relu(emb2)

        emb = emb1 + emb2
        emb = self.bn(emb)
        x = self.classifier(emb)
        return x,emb
