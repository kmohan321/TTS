import torch
import torch.nn as nn
from modules import MHA, DiT_FFN, Positional_Encoding

class timestep_MLP(nn.Module):
  def __init__(self,
               hidden_dim: int
               ):
    super().__init__()
    
    self.mlp = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, 6*hidden_dim)
    )
    
  def forward(self, t):
    return self.mlp(t)
  
  
class DiT_Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    self.mha = MHA(config["hidden_dim"], config["heads"])
    self.ffn = DiT_FFN(config["hidden_dim"], config["ffn_multiplier"])
    self.ln = nn.LayerNorm(config["hidden_dim"], elementwise_affine=False)
    
  def forward(self, x: torch.Tensor, conditonal: torch.Tensor):
    alpha1, beta1, gamma1, alpha2, beta2, gamma2 = torch.split(conditonal[:,None,:], 6, dim=-1)
    
    x = x + self.mha(self.ln(x) * gamma1 + beta1) * alpha1
    x = x + self.ffn(self.ln(x) * gamma2 + beta2) * alpha2
    return x
  
class DiT(nn.Module):
  def __init__(self,config):
    super().__init__()
    
    self.pos = Positional_Encoding(config["hidden_dim"], config["seq_length"])
    self.dit_blocks = nn.ModuleList([DiT_Block(config) for _ in range(config["num_blocks"])])
    self.codebook_embedding = nn.Embedding(config["num_cd_vectors"], config["hidden_dims"])
    self.conditional_mlp = timestep_MLP(config["hidden_dim"])
    
  def forward(self, x: torch.Tensor, t: torch.Tensor):
    
    #(b,s,num_codebook) -> (b,s,num_codebook,d)->(b,s,d)
    x = torch.sum(self.codebook_embedding(x),dim=2) 
    
    
    
    

    
    
    
    
    
