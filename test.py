import torch
import torch.nn as nn
import librosa

#let's work on compression ratio first that is how much to compress
# class Conv_Compression(nn.Module):
#   def __init__(self,
#                strides: list[int],
#                input_channels: int,
#                hidden_channels: int,
#                output_channels: int,
#                kernel: int
#                ):
#     super().__init__()
    
#     self.strides = strides
#     self.cin = input_channels
#     self.cout = output_channels
#     self.kernel = kernel
#     self.hidden_channels = hidden_channels
#     self.input_layer = nn.Conv1d(in_channels = self.cin, out_channels = self.hidden_channels,kernel_size=self.kernel,stride=self.strides[0])
#     self.conv1d_layers = nn.ModuleList([nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=self.kernel, stride=self.strides[i+1])
#                                         for i in range(len(self.strides)-2)])
#     self.output_layer = nn.Conv1d(self.hidden_channels, self.cout, kernel_size=kernel, stride=strides[-1])
  # def forward(self, x :torch.Tensor):
  #   #x -> shape(b,1,s)
  #   x = self.input_layer(x)
  #   for layer in self.conv1d_layers:
  #     x = layer(x)
  #   #shape -> (b,cout,s_)
  #   x = self.output_layer(x)
  #   return x 

class residualblock(nn.Module):
  def __init__(self,
               cin,
               cout,
               stride,
               kernel
               ):
    super().__init__()
    
    self.residual_block = nn.Sequential(
      nn.Conv1d(in_channels=cin, out_channels=cin, stride=stride, kernel_size=kernel),
      nn.BatchNorm1d(cin),
      nn.ReLU(),
      nn.Conv1d(in_channels=cin, out_channels=cout, stride=1, kernel_size=kernel, padding=1),
      nn.BatchNorm1d(cout)
    )
    self.residual_layer = nn.Conv1d(cin, cout, kernel_size=1, stride=stride)
    
  def forward(self, x: torch.Tensor):
    return self.residual_layer(x) + self.residual_block(x)


class Conv_Compression(nn.Module):
  def __init__(self,
               strides: list[int],
               input_channels: int,
               hidden_channels: int,
               output_channels: int,
               kernel: int
               ):
    super().__init__()
    
    self.strides = strides
    self.cin = input_channels
    self.cout = output_channels
    self.kernel = kernel
    self.hidden_channels = hidden_channels
    self.input_layer = residualblock(cin = input_channels, cout= hidden_channels, stride=strides[0], kernel=kernel)
    self.conv1d_layers = nn.ModuleList([residualblock(cin = hidden_channels, cout= hidden_channels, stride=strides[i+1], kernel=kernel)
                                        for i in range(len(self.strides)-2)])
    self.output_layer = residualblock(cin = hidden_channels, cout= output_channels, stride=strides[-1], kernel=kernel)
    
  def forward(self, x :torch.Tensor):
    #x -> shape(b,1,s)
    x = self.input_layer(x)
    for layer in self.conv1d_layers:
      x = layer(x)
    #shape -> (b,cout,s_)
    x = self.output_layer(x)
    return x 

class CodeBook(nn.Module):
  def __init__(self,
               cd_size: int,
               cd_vector_dim: int
               ):
    super().__init__()
    
    self.codebook = nn.Embedding(cd_size, cd_vector_dim)
    
  def forward(self, x: torch.Tensor):
    #shape ->(b*s,d)
    x = x.view(-1, x.shape[-1])
    #shape ->(b*s,cd_size) #for every position get the euclidean distance
    distance = torch.cdist(x, self.codebook.weight, p=2) 
    encoding = torch.argmin(distance,dim=-1) #(b*s,)
    nearest_codebook = self.codebook.weight[encoding]
    codebook_loss = nn.MSELoss()(x.detach(), nearest_codebook)
    encoder_loss = nn.MSELoss()(x, nearest_codebook.detach())
    return encoding, codebook_loss, encoder_loss, nearest_codebook 

 
class RVQ(nn.Module):
  def __init__(self,
               num_codebooks: int,
               cd_size: int,
               cd_vector_dim: int,
               ):
    super().__init__()
    
    self.num_codebooks = num_codebooks
    self.cd_size = cd_size
    self.cd_vector_sim = cd_vector_dim
    self.codebooks = nn.ModuleList([CodeBook(cd_size, cd_vector_dim) for _ in range(num_codebooks)])
    
  def forward(self, x: torch.Tensor):
    
    b,s,d = x.shape
    discrete_enc = torch.zeros(size=(b*s, self.num_codebooks),dtype=torch.long)
    quantised_vectors = torch.zeros_like(x)
    
    loss_enc = 0
    loss_cd = 0
    for idx, codebook in enumerate(self.codebooks):
       encoding, codebook_loss, encoder_loss, nearest_codebook = codebook(x)
       x = x - nearest_codebook #simply the residual value
       quantised_vectors += nearest_codebook #for the decoding
       loss_cd += codebook_loss
       loss_enc += encoder_loss
       discrete_enc[:,idx] = encoding
       
    discrete_enc = discrete_enc.reshape(b,s,-1)
    quantised_vectors = x + (quantised_vectors - x).detach() 
    return loss_cd, loss_enc, discrete_enc, quantised_vectors
  
# class residualblock_transpose(nn.Module):
#   def __init__(self,
#                cin,
#                cout,
#                stride,
#                kernel
#                ):
#     super().__init__()
    
#     self.residual_block = nn.Sequential(
#       nn.ConvTranspose1d(in_channels=cin, out_channels=cout, stride=1, kernel_size=kernel,padding=1),
#       nn.BatchNorm1d(cout),
#       nn.ReLU(),
#       nn.ConvTranspose1d(in_channels=cout, out_channels=cout, stride=stride, kernel_size=kernel,padding=2),
#       nn.BatchNorm1d(cout)
#     )
#     self.residual_layer = nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=stride, output_padding=4)
    
#   def forward(self, x: torch.Tensor):
#     return self.residual_layer(x) + self.residual_block(x) 

# class Conv_Decoder(nn.Module):
#   def __init__(self,
#                strides: list[int],
#                input_channels: int,
#                hidden_channels: int,
#                output_channels: int,
#                kernel: int
#                ):
#     super().__init__()
    
#     self.strides = strides[::-1]
#     self.cin = input_channels
#     self.cout = output_channels
#     self.kernel = kernel
#     self.hidden_channels = hidden_channels
#     self.input_layer = residualblock_transpose(cin = input_channels, cout= hidden_channels, stride=strides[0], kernel=kernel)
#     self.conv1d_layers = nn.ModuleList([residualblock_transpose(cin = hidden_channels, cout= hidden_channels, stride=strides[i+1], kernel=kernel)
#                                         for i in range(len(self.strides)-2)])
#     self.output_layer = residualblock_transpose(cin = hidden_channels, cout= output_channels, stride=strides[-1], kernel=kernel)
    
#   def forward(self, x :torch.Tensor):
#     #x -> shape(b,1,s)
#     x = self.input_layer(x)
#     for layer in self.conv1d_layers:
#       x = layer(x)
#     #shape -> (b,cout,s_)
#     x = self.output_layer(x)
#     return x 


waveform, sr = librosa.load("sample-12s.wav", sr=None)
waveform = waveform[:441000]
print(f" waveform_shape: {waveform.shape}, sampling_rate: {sr}")
waveform = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(1)
layers = Conv_Compression(strides=[4,6,6,8],input_channels=1,hidden_channels=64,output_channels=250,kernel=3)
x = layers(waveform)
print(f"output shape: {x.shape}")
rvq = RVQ(3,4,250)
l1, l2, out, quantised = rvq(x.transpose(1,2))
print(out.shape)

vocab_layer = nn.Embedding(4,512)
print(vocab_layer(out).shape)
# decoder = Conv_Decoder(strides=[4,6,6,8],input_channels=250,hidden_channels=64,output_channels=1,kernel=4)
# print(decoder(quantised.permute(0,2,1)).shape)
# breakpoint()
