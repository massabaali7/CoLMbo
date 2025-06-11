import torch
import torch.nn as nn
import math 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
    
    def scaled_dot_product_attention(self, Q, K, V, prob_phn=None, mask=None, lambda_val=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  
        # Add a singleton dimension to prob_phn at index 1
        prob_phn = prob_phn.unsqueeze(1)
        # Expand prob_phn to match the shape of attn_scores
        # This will not increase memory usage as expand returns a new view on the existing tensor
        prob_phn = prob_phn.expand(-1, self.num_heads, -1, -1)
        if lambda_val > 0:
            attn_scores = attn_scores - lambda_val * prob_phn.transpose(-2, -1)
        attn_mask = mask
        if mask is not None:
            # print(mask.shape)
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = attn_probs.float()
        output = torch.matmul(attn_probs, V)
        return output, attn_mask
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, prob_phn=None, mask=None, lambda_val=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output, attn_mask = self.scaled_dot_product_attention(Q, K, V, prob_phn, mask,lambda_val)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_mask