import torch
import torch.nn as nn
from modules import GQA, FFN, RMSE_NORM, Frequencies

class Attention_Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    
    self.atten = GQA(config["hidden_dims"], config["num_heads_q"], config["num_heads_kv"], config["seq_length"])
    self.norm1 = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.norm2 = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn = FFN(config["hidden_dims"], config["ffn_multiplier"])
    
  def forward(self,x,audio_freq,cond_freq):
    x = x + self.atten(self.norm1(x),audio_freq,cond_freq)
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
    self.audio_freq = Frequencies(config["max_audio_seq_length"], config["head_dim"])
    self.conditional_freq = Frequencies(config["max_cond_seq_length"], config["head_dim"])
    self.register_buffer("audio_rope_freq", self.audio_freq)
    self.register_buffer("conditional_rope_freq", self.conditional_freq)
    
  def forward(self, audio_tokens: torch.Tensor, conditional_input :torch.Tensor):
    #(b,s) -> (b,s,d)
    cond_input = self.conditional_embedding(conditional_input) #considering text only
    #(b,s,num_codebook) -> (b,s,num_codebook,d) #for discrete tokenization
    audio_input = self.codebook_embedding(audio_tokens)
    
    cond_len = cond_input.shape[1]
    audio_len = audio_input.shape[1]
    
    #cond_input -> prefix
    audio_input = torch.sum(audio_input, dim=2)
    x = torch.cat([cond_input, audio_input], dim=1) #(b,s_concat,d)
    
    for block in self.blocks:
      x = block(x, self.audio_rope_freq[:,:audio_len], self.conditional_rope_freq[:,:cond_len])
    
    x = self.rmse_norm(x)
    x = torch.stack([head(x).unsqueeze(2) for head in self.final_heads],dim=2)
    return x #(b,s,num_codebook,d)