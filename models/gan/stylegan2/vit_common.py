import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import Parameter

from models.gan.stylegan2.layers import EqualLinear

def convert_to_coord_format(b, h, w, integer_values=False):
    if integer_values:
        x_channel = torch.range(w, dtype=torch.float32)
        x_channel = torch.reshape(x_channel, (1, 1, -1, 1))
        x_channel = torch.tile(x_channel, (b, w, 1, 1))


        y_channel = torch.range(h, dtype=torch.float32)
        y_channel = torch.reshape(y_channel, (1, -1, 1, 1))
        y_channel = torch.tile(y_channel, (b, 1, h, 1))
    else:
        x_channel = torch.linspace(-1, 1, w)
        x_channel = torch.reshape(x_channel, (1, 1, -1, 1))
        x_channel = torch.tile(x_channel, (b, w, 1, 1))


        y_channel = torch.linspace(-1, 1, h)
        y_channel = torch.reshape(y_channel, (1, -1, 1, 1))
        y_channel = torch.tile(y_channel, (b, 1, h, 1))

    ret = torch.cat((x_channel, y_channel), 3)
    return ret

def l2normalize(v, eps=1e-4):
  return v / (v.norm() + eps)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SpectralNorm(nn.Module):
  def __init__(self, module, name='weight', spectral_multiplier=True, power_iterations=1):
    super(SpectralNorm, self).__init__()
    self.module = module
    self.name = name
    self.power_iterations = power_iterations
    self.spectral_multiplier = spectral_multiplier

    if not self._made_params():
      self._make_params()

  def _update_u_v(self):
    u = getattr(self.module, self.name + "_u")
    v = getattr(self.module, self.name + "_v")
    w = getattr(self.module, self.name + "_bar")

    height = w.data.shape[0]
    _w = w.view(height, -1)
    for _ in range(self.power_iterations):
      v = l2normalize(torch.matmul(_w.t(), u))
      u = l2normalize(torch.matmul(_w, v))

    sigma = u.dot((_w).mv(v))
    
    if self.name in self.module._parameters.keys():
      del self.module._parameters[self.name]
    setattr(self.module, self.name, self.multiplier * w / sigma.expand_as(w))
    #setattr(self.module, self.name, w / sigma.expand_as(w))

  def _made_params(self):
    try:
      getattr(self.module, self.name + "_u")
      getattr(self.module, self.name + "_v")
      getattr(self.module, self.name + "_bar")
      return True
    except AttributeError:
      return False

  def _make_params(self):
    w = getattr(self.module, self.name)

    height = w.data.shape[0]
    width = w.view(height, -1).data.shape[1]

    u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
    v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
    u.data = l2normalize(u.data)
    v.data = l2normalize(v.data)
    w_bar = Parameter(w.data)

    #del self.module._parameters[self.name]
    self.module.register_parameter(self.name + "_u", u)
    self.module.register_parameter(self.name + "_v", v)
    self.module.register_parameter(self.name + "_bar", w_bar)

    _w = w.view(height, -1)

    if self.spectral_multiplier:
      _, s, _ = torch.svd(_w, compute_uv=False)
      self.multiplier = s[0].detach()
      
    else:
      self.multiplier = 1.0

  def forward(self, *args):
    self._update_u_v()
    return self.module.forward(*args)


class Attention(nn.Module):
    def __init__(self, dim = 384, heads = 6, dim_head = 64, \
        l2_attn = True, spectral_norm = True, dropout = 0.):

        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.temperature = nn.Parameter(torch.FloatTensor([1.0]))

        self.attend = nn.Softmax(dim = -1)

        self.l2_attn = l2_attn

        if spectral_norm:
            self.to_qkv = SpectralNorm(EqualLinear(dim, inner_dim * 3, use_bias=False), spectral_multiplier=True)
            self.to_out = nn.Sequential(
                SpectralNorm(EqualLinear(inner_dim, dim), spectral_multiplier=True),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

        else:
            self.to_qkv = EqualLinear(dim, inner_dim * 3, use_bias=False)

            self.to_out = nn.Sequential(
                EqualLinear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if self.l2_attn:
            AB = torch.matmul(q, k.transpose(-1, -2))
            AA = torch.mean(q * q, -1, keepdim=True)
            BB = AA    # Since query and key are tied.
            BB = BB.transpose(-1, -2)
            dots = AA - 2 * AB + BB
            dots = dots * self.scale
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots * self.temperature

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, \
        spectral_norm = True, dropout = 0.):

        super().__init__()

        if spectral_norm:
            self.net = nn.Sequential(
                SpectralNorm(EqualLinear(dim, hidden_dim), spectral_multiplier=True),
                nn.SiLU(),
                nn.Dropout(dropout),
                SpectralNorm(EqualLinear(hidden_dim, dim), spectral_multiplier=True),
                nn.Dropout(dropout)
            )
        else:
            self.net = nn.Sequential(
                EqualLinear(dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                EqualLinear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
    def forward(self, x):
        return self.net(x)
