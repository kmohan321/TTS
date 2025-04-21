import torch
import torch.nn as nn
from modules import GQA, FFN, RMSE_NORM, Frequencies

# config = {
#     "model": {
#         "num_blocks": 2,
#         "hidden_dims": 256,
#         "num_heads_q": 8,
#         "num_heads_kv": 4,
#         "seq_length": 4096,
#         "ffn_multiplier": 4,
#         "vocab_size": 52000,
#         "eps": 1e-5,
#         "head_dim": 256//8
#     }
# }

class Attention_Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    
    self.atten = GQA(config["hidden_dims"], config["num_heads_q"], config["num_heads_kv"], config["seq_length"])
    self.norm1 = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.norm2 = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn = FFN(config["hidden_dims"], config["ffn_multiplier"])
    
  def forward(self,x):
    x = x + self.atten(self.norm1(x))
    x = self.ffn(self.norm2(x))
    return x
  
class Transformer(nn.Module):
  def __init__(self,config):
    super().__init__()    
    
    self.blocks = nn.ModuleList([Attention_Block(config) for _ in range(config["num_blocks"])])
    self.conditional_embedding = nn.Embedding(config["vocab_size"],config["hidden_dims"]) #for conditional_input
    self.codebook_embedding = nn.Embedding(config["num_cd_vectors"],config["hidden_dims"]) #for codebook
    self.rmse_norm = RMSE_NORM(config["hidden_dims"],config["eps"])
    
    self.final_heads = nn.ModuleList([nn.Linear(config["hidden_dims"], config["num_cd_vectors"], bias=False) for _ in range(config["num_codebooks"])])
    self.freq = Frequencies(config["seq_length"], config["head_dim"])
    
  def forward(self, conditional_input :torch.Tensor, audio_tokens: torch.Tensor):
    #(b,s) -> (b,s,d)
    cond_input = self.conditional_embedding(conditional_input)
    #(b,s,num_codebook) -> (b,s,num_codebook,d) #for discrete tokenization
    audio_input = self.codebook_embedding(audio_tokens)
    
    #cond_input -> prefix
    audio_input = torch.sum(audio_tokens, dim=2)
    x = torch.cat([cond_input, audio_input], dim=1) #(b,s_concat,d)
    
    for block in self.blocks:
      x = block(x, self.freq)
    
    x = self.rmse_norm(x)
    x = torch.stack([head(x).unsqueeze(2) for head in self.final_heads],dim=2)
    return x #(b,s,num_codebook,d)
  
#remember -> have to give some fix sequence length 
