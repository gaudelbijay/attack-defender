import torch
from attend import Attend 
from utils import *
import torch.nn as nn 
import torch.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
'''
refrence: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
'''

def Upsample(dim, dim_out = None):
    '''
    dim: The number of input channels for the 2D convolution operation.
    dim_out: (Optional) The number of output channels for the 2D convolution operation. 

    Sample Input shape: torch.Size([1, 3, 8, 8])
    Output shape after Upsample and Convolution: torch.Size([1, 3, 16, 16])
    '''
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'), # height and width will be multiplied by 2
        nn.Conv2d(in_channels = dim, out_channels = default(dim_out, dim), kernel_size=3, padding=1)
    )

def Downsample(dim, dim_out = None):
    '''
    for Rearrange() function refers to the following links
    https://einops.rocks/api/rearrange/
    https://stackoverflow.com/questions/74223784/rearrange-a-5d-tensor-using-einops


    Sample Input shape: torch.Size([1, 3, 8, 8])
    Output shape after Downsample and Convolution: torch.Size([1, 3, 4, 4])
    '''
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2), # rearrannge each tensor by dividing height and weidth by 2 and multiplying channels by 2 X 2. 
        nn.Conv2d(in_channels = dim*4, out_channels = default(dim_out, dim), kernel_size = 1, padding = 1)
    )

class RMSNorm(nn.Module):
    '''
    Normalize the data based on 2nd dimension (channel dimension).
    RMSNorm operation is applied independently to each channel in the input tensor. 


    Example: Sample Input:
    tensor([[[1., 2.],
            [3., 4.]],

            [[5., 6.],
            [7., 8.]]])

    Output after RMSNorm:
    tensor([[[0.1826, 0.3651],
            [0.5488, 0.7314]],

            [[0.3790, 0.4548],
            [0.5286, 0.5044]]])
    '''

    def __init__(self, dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.register_parameter("g", self.g) # registering self.g to learnable parameters of the model

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class SinusoidalPosEmb(nn.Module):

    '''
    Example: Original embeddings:
    tensor([[ 0.3441,  0.1543,  0.5645, -1.5631],
            [ 0.4515,  0.4640,  0.4615, -0.5615],
            [-0.4298, -0.1267,  0.8482,  1.0303],
            [ 1.0126, -0.2076, -0.7711, -0.6452],
            [ 0.1487,  0.7580,  1.7112, -1.1173]])
    Sine components:
    tensor([[ 0.3401,  0.1537,  0.5441, -0.9980],
            [ 0.4286,  0.4509,  0.4451, -0.5290],
            [-0.4152, -0.1261,  0.7530,  0.8647],
            [ 0.8480, -0.2067, -0.6865, -0.5948],
            [ 0.1481,  0.6790,  1.1533, -0.8912]])
    Cosine components:
    tensor([[ 0.9404,  0.9880,  0.8396,  0.0621],
            [ 0.9035,  0.8926,  0.8955,  0.8486],
            [ 0.9092,  0.9920,  0.6588,  0.5029],
            [ 0.5307,  0.9784,  0.7271,  0.8038],
            [ 0.9889,  0.7347,  0.9886,  0.4532]])
    Concatenated embeddings:
    tensor([[ 0.3401,  0.1537,  0.5441, -0.9980,  0.9404,  0.9880,  0.8396,  0.0621],
            [ 0.4286,  0.4509,  0.4451, -0.5290,  0.9035,  0.8926,  0.8955,  0.8486],
            [-0.4152, -0.1261,  0.7530,  0.8647,  0.9092,  0.9920,  0.6588,  0.5029],
            [ 0.8480, -0.2067, -0.6865, -0.5948,  0.5307,  0.9784,  0.7271,  0.8038],
            [ 0.1481,  0.6790,  1.1533, -0.8912,  0.9889,  0.7347,  0.9886,  0.4532]])

    '''
    def __init__(self, dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim 
    
    def forward(self, x):
        device = x.device # cpu or cuda
        half_dim = self.dim // 2 # //2 -> to stack sine and cosine component later. 
        emb = math.log(10000) / (half_dim - 1) # half_dim - 1 => (counting from 0)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # (0.000, .... ,  n.xxx)
        emb = x[:, None] * emb[None, :] # expand x in last dim e.g. -> x = [1, 2, 3], x[:, None] = [[1], [2], [3]]. emb = [1, 2, 3], emb[None, :] = [[1, 2, 3]]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # [[sin(emb)..., cos(emb)], [sin(emb)..., cos(emb)]]]
        return emb 


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    '''
    example: rearrange(x, 'd -> 1 d') 
    input: x = array([-0.43589004, -1.48200149, -0.33785058, -1.57400589,  0.32930165])

    output: 
    array([[-0.43589004],
       [-1.48200149],
       [-0.33785058],
       [-1.57400589],
       [ 0.32930165]])

    '''
    def __init__(self, dim, is_random=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert(divisible_by(dim, 2)) # check if it is divisible exactly into half to stack sin and cosine component
        half_dim = dim // 2 
        self.weight = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random) # for random positional embedding we don't need gradient 
        self.weight = rearrange(self.weight, 'd -> 1 d') # changing to row vector to column
        self.register_parameter("weight", self.weight)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1') 
        frequency = x * self.weight * 2 * math.pi 
        fouriered = torch.cat((frequency.sin(), frequency.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered 


class Block(nn.Module):
    ''' 
    Example: 

    block = Block(dim=3, dim_out=16)
    batch_size = 1
    channels = 3
    height, width = 32, 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    print(input_tensor.shape)
    -> [1, 3, 32, 32]

    output = block(input_tensor)
    print("Output tensor shape:", output.shape) 
    -> [1, 16, 32, 32]

    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d (for convolution output size)
    '''
    def __init__(self, dim, dim_out, groups=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.projection = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU() # SiLU(x) = x * sigmoid(x); sigmoid(x) = 1 / (1 + exp(-x))

    def forward(self, x, scale_shift = None):
        x = self.projection(x) 
        x = self.norm(x) 

        if exists(scale_shift):
            scale, shfit = scale_shift 
            x = x * (scale + 1) + shfit 
        
        x = self.act(x) 
        return x 

class ResnetBlock(nn.Module):
    '''
    batch_size = 1
    embedding_dim = 5
    time_emb = torch.randn(batch_size, 2 * embedding_dim)
    print("Time emb :", time_emb)
    -> Time emb: tensor([[ 0.5875,  0.9348,  0.0872, -0.1188, -0.6787, -0.0931,  0.4430,  1.2456, 0.3441, -0.5510]])

    scale, shift = time_emb.chunk(2, dim=1)
    print("Scale tensor shape:", scale)
    print("Shift tensor shape:", shift)

    -> Scale tensor : tensor([[ 0.5875,  0.9348,  0.0872, -0.1188, -0.6787]])
    -> Shift tensor : tensor([[-0.0931,  0.4430,  1.2456,  0.3441, -0.5510]])
    '''

    def __init__(self, dim, dim_out, time_emb_dim = None, groups = 8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mlp = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None 

        self.block1 = Block(dim=dim, dim_out=dim_out, groups=groups) 
        self.block2 = Block(dim=dim, dim_out=dim_out, groups=groups)

        # Either 1 x 1 convolution or Identity for skip connection
        self.res_conv = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=1) if dim !=dim_out else nn.Identity() 

    def forward(self, x, time_emb = None):
        scale_shift = None 
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c-> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    '''

    '''

    def __init__(self, dim, heads = 4, dim_head = 32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = dim_head ** -0.5 # the scaling factor is often set to inverse sqrt of dim_head
        self.heads = heads 
        hidden_dim = dim_head * heads # intermediate dimension used to represent the connected outputs of multiple attention heads 

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size = 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape 

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale 

        context = torch.einsum('b h d n, b h e n -> b h c (x y)', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)