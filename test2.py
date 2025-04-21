import torch
import torch.nn as nn

# c = 250
# s = 489
# x = torch.randn(size=(1,c,s))

# layer = nn.Sequential(
#   nn.ConvTranspose1d(250,64,kernel_size=4,stride=8, output_padding=2),
#   nn.ConvTranspose1d(64,64,kernel_size=4,stride=6, output_padding=2),
#   nn.ConvTranspose1d(64,64,kernel_size=4,stride=6,output_padding=2),
#   nn.ConvTranspose1d(64,1,kernel_size=4,stride=4,output_padding=2),
# )

# print(layer(x).shape)

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
      nn.Conv1d(in_channels=cin, out_channels=cout, stride=1, kernel_size=kernel,padding=1),
      nn.BatchNorm1d(cout)
    )
    self.residual_layer = nn.Conv1d(cin, cout, kernel_size=1, stride=stride)
    
  def forward(self, x: torch.Tensor):
    return self.residual_layer(x) + self.residual_block(x)

class residualblock_transpose(nn.Module):
  def __init__(self,
               cin,
               cout,
               stride,
               kernel
               ):
    super().__init__()
    
    self.residual_block = nn.Sequential(
      nn.ConvTranspose1d(in_channels=cout, out_channels=cin, stride=1, kernel_size=kernel,padding=1),
      nn.BatchNorm1d(cin),
      nn.ReLU(),
      nn.ConvTranspose1d(in_channels=cin, out_channels=cin, stride=stride, kernel_size=kernel,padding=2),
      nn.BatchNorm1d(cin)
    )
    self.residual_layer = nn.ConvTranspose1d(cout, cin, kernel_size=4, stride=stride, output_padding=4)
    
  def forward(self, x: torch.Tensor):
    return self.residual_layer(x) + self.residual_block(x)
 
# x = torch.randn(size=(1,1,441000))
# print(f"input_shape: {x.shape}")
# block1 = residualblock(1,40,8,3)
# y = block1(x)
# print(f"encoder_shape: {y.shape}")
# block2 = residualblock_transpose(1,40,8,4)
# print(f"decoder_shape: {block2(y).shape}") 
    