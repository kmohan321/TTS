import torch
import torch.nn as nn

# def shape_checker(x):
#   return print(f"shape: {x.shape}")

class ResidualBlock_Down(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, kernel: int):
        super().__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.kernel = kernel

        self.residual_block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=kernel, stride=1, padding='same'),
            nn.BatchNorm1d(cout),
            nn.ReLU(),
            nn.Conv1d(cout, cout, kernel_size=kernel, stride=stride, padding=kernel//2),
            nn.BatchNorm1d(cout)
        )

        self.residual_layer = nn.Conv1d(cin, cout, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.residual_layer(x) + self.residual_block(x)
      
class ResidualBlock_Up(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, kernel: int, output_padding=0):
        super().__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.kernel = kernel

        self.residual_block = nn.Sequential(
            nn.ConvTranspose1d(cin, cout, kernel_size=kernel, stride=1, padding=kernel//2),
            nn.BatchNorm1d(cout),
            nn.ReLU(),
            nn.ConvTranspose1d(cout, cout, kernel_size=kernel, stride=stride, padding=kernel//2, output_padding=output_padding),
            nn.BatchNorm1d(cout)
        )

        self.residual_layer = nn.ConvTranspose1d(cin, cout, kernel_size=1, stride=stride, output_padding=output_padding)

    def forward(self, x):
        return self.residual_layer(x) + self.residual_block(x)
      
class Conv_Downsampling(nn.Module):
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
    assert kernel % 2!=0, "kernel must be odd"
    self.input_layer = ResidualBlock_Down(cin = input_channels, cout = hidden_channels, stride=strides[0], kernel=kernel)
    self.conv1d_layers = nn.ModuleList([ResidualBlock_Down(cin = hidden_channels, cout= hidden_channels, stride=strides[i+1], kernel=kernel)
                                        for i in range(len(self.strides)-2)])
    self.output_layer = ResidualBlock_Down(cin = hidden_channels, cout = output_channels, stride = strides[-1], kernel=3)
    
  def forward(self, x :torch.Tensor):
    #x -> shape(b,1,s)
    x = self.input_layer(x)
    
    for layer in self.conv1d_layers:
      x = layer(x)
      
    #shape -> (b,cout,s_)
    x = self.output_layer(x)
    return x.permute(0,2,1)

class Conv_Upsampling(nn.Module):
  def __init__(self,
               strides: list[int],
               input_channels: int,
               hidden_channels: int,
               output_channels: int,
               kernel: int
               ):
    super().__init__()
    
    self.strides = strides[::-1]
    self.cin = input_channels
    self.cout = output_channels
    self.kernel = kernel
    self.hidden_channels = hidden_channels
    assert kernel % 2!=0, "kernel must be odd"
    
    #need to be caculated
    self.output_paddings = [0, 1, 2, 3]
    
    self.input_layer = ResidualBlock_Up(
        cin=input_channels, 
        cout=hidden_channels, 
        stride=self.strides[0], 
        kernel=3, 
        output_padding=self.output_paddings[0]
    )
    
    self.conv1d_layers = nn.ModuleList([
        ResidualBlock_Up(
            cin=hidden_channels, 
            cout=hidden_channels, 
            stride=self.strides[i+1], 
            kernel=kernel,
            output_padding=self.output_paddings[i+1]
        ) for i in range(len(self.strides)-2)
    ])
    
    self.output_layer = ResidualBlock_Up(
        cin=hidden_channels, 
        cout=output_channels, 
        stride=self.strides[-1], 
        kernel=kernel,
        output_padding=self.output_paddings[-1]
    )
  
  def forward(self, x: torch.Tensor):
    #x -> shape(b,s,c) -> permute to (b,c,s)
    x = x.permute(0,2,1)
    
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
    b,s,d = x.shape
    x = x.reshape(-1, d)
    #shape ->(b*s,cd_size) #for every position get the euclidean distance
    distance = torch.cdist(x, self.codebook.weight, p=2) 
    encoding = torch.argmin(distance,dim=-1) #(b*s,)
    nearest_codebook = self.codebook.weight[encoding]
    codebook_loss = nn.MSELoss()(x.detach(), nearest_codebook)
    encoder_loss = nn.MSELoss()(x, nearest_codebook.detach())
    return encoding.view(b,s), codebook_loss, encoder_loss, nearest_codebook.view(b,s,d) 

 
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
    discrete_enc = torch.zeros(size=(b, s, self.num_codebooks),dtype=torch.long)
    quantised_vectors = torch.zeros_like(x)
    
    loss_enc = 0
    loss_cd = 0
    for idx, codebook in enumerate(self.codebooks):
       encoding, codebook_loss, encoder_loss, nearest_codebook = codebook(x)
       x = x - nearest_codebook #simply the residual value
       quantised_vectors += nearest_codebook #for the decoding
       loss_cd += codebook_loss
       loss_enc += encoder_loss
       discrete_enc[:,:,idx] = encoding
       
    discrete_enc = discrete_enc.reshape(b,s,-1)
    quantised_vectors = x + (quantised_vectors - x).detach() 
    return loss_cd, loss_enc, discrete_enc, quantised_vectors


class AudioEncoder(nn.Module):
    def __init__(self, 
                strides: list[int],
                input_channels: int,
                hidden_channels: int,
                latent_channels: int,
                kernel: int,
                num_codebook: int,
                codebook_size: int,
                codebook_dim: int
                ):
        super().__init__()
        
        assert latent_channels==codebook_dim, "codebook dim must be equal to the latent channels"
        
        self.encoder = Conv_Downsampling(
            strides=strides,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=latent_channels,
            kernel=kernel
        )
        
        self.rvq = RVQ(num_codebook, codebook_size, codebook_dim)
        
        self.decoder = Conv_Upsampling(
            strides=strides,
            input_channels=latent_channels,
            hidden_channels=hidden_channels,
            output_channels=input_channels,
            kernel=kernel
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        loss_cd, loss_enc, discrete_enc, quantised_vectors = self.rvq(encoded)
        decoded = self.decoder(quantised_vectors)
        return decoded, loss_cd, loss_enc, discrete_enc

if __name__ == "__main__":
    x = torch.randn(size=(3,1,25000))
    print(f"input_shape: {x.shape}")
    
    # block1 = Conv_Downsampling([8,6,4,4,2], 1, 64, 40, 7)
    # y, shapes = block1(x)
    # print(f"down_out: {y.shape}")
    
    # block2 = Conv_Upsampling([8,6,4,4,2], 40, 64, 1, 7)
    # out = block2(y, shapes)
    # print(f"up_out: {out.shape}")
    
    model = AudioEncoder(strides = [6,4,4,2], input_channels=1, hidden_channels= 64, latent_channels=512, kernel=7,
                           num_codebook=4,codebook_dim=512,codebook_size=2048)
    decoded, loss_cd, loss_enc, discrete_enc = model(x)
    print(f"reconstructed: {decoded.shape}")
    # breakpoint()
    
    assert x.shape == decoded.shape, "Input and output shapes don't match!"
    print("Success! Input and output shapes match exactly.")