import torch
import torch.nn as nn

def Frequencies(seq_length: int, head_dim: int):
    assert head_dim%2==0, 'head_dim should be even'
    
    # m values for different positions in sequence
    m = torch.arange(0,seq_length)
    
    # theta values for different index in token vector
    theta = 1/(10000**(2*torch.arange(0,head_dim//2)/head_dim))
    
    #all possible combinations for m and theta
    freq = torch.outer(m,theta) #shape->(m,d/2)
    
    #converting freq to polar
    complex_freq = torch.polar(torch.ones_like(freq),freq)
    
    return complex_freq.unsqueeze(0).unsqueeze(2)


def apply_rope(self, x:torch.Tensor, freq:torch.Tensor):
    b ,s, h, d = x.shape
    x = x.view(b, s, h, -1, 2)
    x = torch.view_as_complex(x)
    x = x * freq[:,:s,:,:]
    x = torch.view_as_real(x)
    x = x.view(b,s,h,d)
    return x

class GQA(nn.Module):
  def __init__(self,
               hidden_dims :int,
               num_heads_q: int,
               num_heads_kv :int,
               seq_length :int):
    super().__init__()
    
    self.num_heads_q = num_heads_q
    self.num_heads_kv = num_heads_kv
    
    assert hidden_dims % self.num_heads_q==0, 'hidden_dim must be divisible by num_heads'
    assert self.num_heads_q % self.num_heads_kv == 0, "num_heads_q must be divisible by num_heads_kv"
    
    self.head_dim = hidden_dims//self.num_heads_q
    self.groups = self.num_heads_q//self.num_heads_kv
    
    #head_dim remains constant
    self.wq = nn.Linear(hidden_dims,num_heads_q*self.head_dim,bias=False)
    self.wk = nn.Linear(hidden_dims,num_heads_kv*self.head_dim,bias=False) 
    self.wv = nn.Linear(hidden_dims,num_heads_kv*self.head_dim,bias=False)
    
    #buffer for mask
    self.register_buffer(
      "mask",
      torch.tril(torch.ones(seq_length,seq_length,dtype=torch.bool))
    )
    #output layer
    self.out_layer = nn.Linear(self.num_heads_q*self.head_dim,hidden_dims,bias=False)
  
  def forward(self, x: torch.Tensor, freq: torch.Tensor, is_causal = True):
    
    b, s, d = x.shape
    query = self.wq(x).view(b,s,self.num_heads_q,self.head_dim)
    key = self.wk(x).view(b,s,self.num_heads_kv,self.head_dim)
    value = self.wv(x).view(b,s,self.num_heads_kv,self.head_dim).transpose(1,2)
    
    #(b,s,h,d) -> (b,h,s,d) 
    rotated_query = apply_rope(query, freq).transpose(1,2) #applying the rope here
    rotated_key = apply_rope(key, freq).transpose(1,2)

    # Group query heads to match key heads
    #(B,Hq,S,D) -> (B,Hkv,G,S,D)
    rotated_query = rotated_query.view(b,self.num_heads_kv,self.groups,s,self.head_dim)

    attention_score = rotated_query @ rotated_key.transpose(2,3).unsqueeze(2)
    if is_causal:
      attention_score = torch.masked_fill(attention_score, mask=self.mask[:s,:s]==0,value = -torch.inf)
    attention_weights = torch.softmax(attention_score/self.head_dim**0.5,dim=-1)
    
    out = attention_weights @ value.unsqueeze(2)
    #(b,hkv,G,S,D) -> (b,hq,S,D) ->(b,S,hq,D)
    out = out.view(b,self.num_heads_kv*self.groups,s,self.head_dim).transpose(1,2)
    out = out.contiguous().view(b,s,d)
    
    return self.out_layer(out)
  
class FFN(nn.Module):
  def __init__(self,
               hidden_dim:int,
               ffn_multiplier:int
               ):
    super().__init__()
    
    self.w1 = nn.Linear(hidden_dim,ffn_multiplier*hidden_dim,bias=False)
    self.w2 = nn.Linear(ffn_multiplier*hidden_dim,hidden_dim,bias=False)
    self.w3 = nn.Linear(hidden_dim,ffn_multiplier*hidden_dim,bias=False)
    self.act = nn.SiLU()
    
  def forward(self,x):
    x1 = self.act(self.w1(x)) 
    gated_value = self.w3(x)
    x = x1 * gated_value #swiglu
    return self.w2(x)
  
class RMSE_NORM(nn.Module):
  def __init__(self,
               hidden_dim:int,
               eps:float = 1e-5
               ):
    super().__init__()
    self.eps = eps
    self.shift = nn.Parameter(torch.ones(hidden_dim))

  def forward(self,x):
    x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
    return x  * self.shift
  
class Positional_Encoding(nn.Module):
    def __init__(self, d: int, seq_length: int):
        super().__init__()
        self.s = seq_length
        self.d_half = d//2
        self.register_buffer("freq", 10000 ** (-2 * torch.arange(0, self.d_half, 2) / self.d_half))

    def forward(self):
        positions = torch.arange(self.s, device=self.freq.device).unsqueeze(1)  # (s, 1)
        sin_embedding = torch.sin(positions * self.freq)  # (s, d//2)
        cos_embedding = torch.cos(positions * self.freq)  # (s, d//2)
        pe = torch.cat([sin_embedding, cos_embedding], dim=-1)  # (s, d)
        return pe.unsqueeze(0)  # (1, s, d)
      
class MHA(nn.Module):
  def __init__(self,
               hidden_dim :int,
               heads: int
               ):
     super().__init__()
     
     assert hidden_dim%heads == 0, "hidden_dim must be divisible by heads"
     
     self.head_dim = hidden_dim // heads
     self.heads = heads
     self.w = nn.Linear(hidden_dim, 3*self.head_dim*heads, bias=False)
     self.w_out = nn.Linear(heads * self.head_dim, hidden_dim, bias=False)
     
     self.scale = self.head_dim ** -0.5
     
  def forward(self, x: torch.Tensor):
    b, s, d = x.shape
    q, k, v = torch.split(self.w(x), 3, dim=-1)
    q, k, v = q.view(b,s,self.heads, self.head_dim), k.view(b,s,self.heads, self.head_dim), v.view(b,s,self.heads, self.head_dim)
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    
    attn_weights = torch.softmax((q @ k.transpose(2,3))*self.scale , dim=-1)
    out = (attn_weights @ v).transpose(1,2).contiguous().view(b,s,d)
    return self.w_out(out)
  
class DiT_FFN(nn.Module):
  def __init__(self,
               hidden_dim:int,
               ffn_multiplier:int
               ):
    super().__init__()
    
    self.w1 = nn.Linear(hidden_dim,ffn_multiplier*hidden_dim,bias=False)
    self.w2 = nn.Linear(ffn_multiplier*hidden_dim,hidden_dim,bias=False)
    self.act = nn.GELU()
    
  def forward(self,x):
    return self.w2(self.act(self.w1(x)))
     
