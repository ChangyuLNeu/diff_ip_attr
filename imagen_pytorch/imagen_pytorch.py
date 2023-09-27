import math
import copy
from random import random
from typing import List, Union
from beartype import beartype
from tqdm.auto import tqdm
from functools import partial, wraps
from contextlib import contextmanager, nullcontext
from collections import namedtuple
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import nn, einsum
from torch.cuda.amp import autocast
from torch.special import expm1
import torchvision.transforms as T
import numpy as np
import kornia.augmentation as K

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from einops_exts import rearrange_many, repeat_many, check_shape

from imagen_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

from imagen_pytorch.imagen_video.imagen_video import Unet3D, resize_video_to
from utils import ltplot
import ops
# helper functions

# from arch.models import Querier
# from arch.models import Querier128
# from arch.models import Querier128_with_mask
# from arch.models import Querier32
from arch.modules import SAUNet, Scorer, QuerierImageAttr, QuerierLinearImageAttr, QuerierConvImageAttr, S_AE, S_ConvAE, S_PoolMLP, QuerierFactorizedImageAttr, QuerierAttn, QuerierCelebAttr
from arch.vector_querier_models import QueryEncoder, QueryLinearEncoder, answer_queries, answer_single_query
from arch.cifar10 import QuerierAE
# from arch.models import Querier128Flat


def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

def is_float_dtype(dtype):
    return any([dtype == float_dtype for float_dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)])

def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255

def module_device(module):
    return next(module.parameters()).device

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def pad_tuple_to_length(t, length, fillvalue = None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))

# helper classes

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

# tensor helpers

def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, dim = -1)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)

def resize_image_to(
    image,
    target_image_size,
    clamp_range = None
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = 'nearest')

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out

# image normalization functions
# ddpms expect images to be in the range of -1 to 1

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# gaussian diffusion with continuous time helper functions and classes
# large part of this was thanks to @crowsonkb at https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

@torch.jit.script
def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # not sure if this accounts for beta being clipped to 0.999 in discrete version

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class GaussianDiffusionContinuousTimes(nn.Module):
    def __init__(self, *, noise_schedule, timesteps = 1000):
        super().__init__()

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        # elif noise_schedule.startswith("constant"):
        #     self.log_snr = partial(constant_log_snr(constant=float(noise_schedule.split('-')[-1])))
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level, *, device):
        return torch.full((batch_size,), noise_level, device = device, dtype = torch.float32)

    def sample_random_times(self, batch_size, *, device):
        return torch.zeros((batch_size,), device = device).float().uniform_(0, 1)

    def sample_constant_times(self, value, batch_size, *, device):
        return torch.ones((batch_size,), device = device).float() * value

    def get_condition(self, times):
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.num_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times


    def q_posterior(self, x_start, x_t, t, *, t_next = None):
        t_next = default(t_next, lambda: (t - 1. / self.num_timesteps).clamp(min = 0.))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        dtype = x_start.dtype

        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device = x_start.device, dtype = dtype)

        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(t).type(dtype)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype
        batch = shape[0]

        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device = device, dtype = dtype)

        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device = device, dtype = dtype)

        noise = default(noise, lambda: torch.randn_like(x_from))

        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to =  log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma * alpha_to) / alpha

    def predict_start_from_v(self, x_t, t, v):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-8)

# norms and residuals

class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)

ChanLayerNorm = partial(LayerNorm, dim = -3)

class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)

# attention pooling

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        cosine_sim_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask = None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.cosine_sim_scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_latents_mean_pooled = 4, # number of latents derived from mean pooled representation of the sequence
        max_seq_len = 512,
        ff_mult = 4,
        cosine_sim_attn = False
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled)
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads, cosine_sim_attn = cosine_sim_attn),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, mask = None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device = device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(x, dim = 1, mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim = -2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask = mask) + latents
            latents = ff(latents) + latents

        return latents

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        context_dim = None,
        cosine_sim_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context = None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 d', b = b)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)
        feat_kv_dim = v.shape[-2]
        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)
        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.cosine_sim_scale

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (0, feat_kv_dim), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# decoder

def Upsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1)

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        norm = True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8,
        linear_attn = False,
        use_gca = False,
        squeeze_excite = False,
        **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim = dim_out,
                context_dim = cond_dim,
                **attn_kwargs
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)

        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()


    def forward(self, x, time_emb = None, cond = None, cond_mask = None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = rearrange(h, 'b c h w -> b h w c')
            h, ps = pack([h], 'b * c')
            h = self.cross_attn(h, context = cond, mask = cond_mask) + h
            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b h w c -> b c h w')

        h = self.block2(h, scale_shift = scale_shift)

        h = h * self.gca(h)

        return h + self.res_conv(x)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False,
        cosine_sim_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b h 1 d', h = self.heads,  b = b)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q = q * self.scale

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.cosine_sim_scale

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> (b h) n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> (b h) 1 d', h = self.heads,  b = b)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context = None):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many((q, k, v), 'b (h c) x y -> (b h) (x y) c', h = h)

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = rearrange_many((ck, cv), 'b n (h d) -> (b h) n d', h = h)
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        out = rearrange(out, '... -> ... 1')
        return self.net(out)

def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )

def ChanFeedForward(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias = False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias = False)
    )

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None,
        cosine_sim_attn = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim, cosine_sim_attn = cosine_sim_attn),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None, mask = None):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        for attn, ff in self.layers:
            x = attn(x, context = context, mask = mask) + x
            x = ff(x) + x

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')
        return x

class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),
                ChanFeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    ):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps = None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim = 1)


class Unet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        patch_size = 1,
        image_size = (128, 128),
        sampling_mode = 'random',
        max_num_queries = 100,
        max_num_attributes = 10,
        max_num_objects = 5,
        image_embed_dim = 1024,
        text_embed_dim = get_encoded_dim(DEFAULT_T5_NAME),
        num_resnet_blocks = 1,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        learned_sinu_pos_emb_dim = 16,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        cond_images_channels = 0,
        channels = 3,
        channels_out = None,
        attn_dim_head = 64,
        attn_heads = 8,
        ff_mult = 2.,
        lowres_cond = False,                # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns = True,
        layer_attns_depth = 1,
        layer_mid_attns_depth = 1, # Note: last change
        layer_attns_add_text_cond = True,   # whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
        attend_at_middle = True,            # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        layer_cross_attns = True,
        use_linear_attn = False,
        use_linear_cross_attn = False,
        cond_on_text = True,
        max_text_len = 256,
        init_dim = None,
        resnet_groups = 8,
        init_conv_kernel_size = 7,          # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        attn_pool_text = True,
        attn_pool_num_latents = 32,
        dropout = 0.,
        memory_efficient = False,
        init_conv_to_final_conv_residual = False,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        cosine_sim_attn = False,
        self_cond = False,
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = True,        # may address checkboard artifacts
        FLAGS = None,
    ):
        super().__init__()

        self.args = FLAGS
        self.cmi = self.args.cmi

        self.image_size = image_size
        self.patch_size = patch_size
        self.sampling = sampling_mode
        self.null_val = self.args.null_val

        # guide researchers
        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        if dim < 128:
            print_once('The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/')

        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)
        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))
        init_dim = default(init_dim, dim)

        self.self_cond = self_cond

        # optional image conditioning

        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels


        self.max_num_queries = max_num_queries
        self.max_num_attributes = max_num_attributes
        self.max_num_objects = max_num_objects

        self.query_decoder = True if self.args.query_mode == 'encoder-decoder' else False
        self.object_encoding = True
        self.attention_querier = True
        add_object_embedding = True if self.args.experiment_type == 'attributes' else False
        latent_dim = 64 if self.args.experiment_type == 'attributes' else 64

        self.encode_clean_features, self.encode_query_features = False, self.args.encode_query_features

        if cond_dim is not None:

            embed_dim = self.args.embed_dim #cond_dim
            #self.cond_embedding = nn.Embedding(self.max_num_attributes * 2 + 1, embed_dim)
            self.cond_embedding = nn.Embedding(self.max_num_attributes * 2, text_embed_dim)

            self.querier = QuerierCelebAttr(only_image=self.args.only_image)

        if self.has_cond_image and cond_dim is None:     
            self.include_gt = self.args.include_gt
            if self.include_gt:
                input_chans = init_channels + cond_images_channels
            else: input_chans = cond_images_channels
            self.cond_img_size = image_size[0]

            # self.querier = SAUNet(c_in=input_chans,
            #                       c_out=1, size=self.cond_img_size, patch_size=self.patch_size, multi_resolution=False)

            self.querier = QuerierAE(c_in=input_chans,
                                  c_out=1, size=self.cond_img_size, patch_size=self.patch_size)

            # self.querier = Scorer(c_in=input_chans,
            #                       c_out=1, size=self.cond_img_size, patch_size=self.patch_size)
            init_channels += cond_images_channels # Previously 1 for mask only

        # initial convolution
        self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1) if init_cross_embed else nn.Conv2d(init_channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        cond_dim = default(cond_dim, dim) # TODO: I removed this. Because I dont understand it and it bugs me.
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        # embedding time for log(snr) noise from continuous version

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        # low res aug noise conditioning

        self.lowres_cond = lowres_cond

        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU()
            )

            self.to_lowres_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r = num_time_tokens)
            )

        # normalizations

        self.norm_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text:
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on text encodings

        self.cond_on_text = cond_on_text

        # attention pooling

        self.attn_pool = PerceiverResampler(dim = cond_dim, depth = 2, dim_head = attn_dim_head, heads = attn_heads, num_latents = attn_pool_num_latents, cosine_sim_attn = cosine_sim_attn) if attn_pool_text else None

        # for classifier free guidance

        self.max_text_len = max_text_len

        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

        #self.null_attr_hidden = nn.Parameter(torch.randn(1, time_cond_dim))    #diff
        #self.null_attr_embed = nn.Parameter(torch.randn(1, cond_dim))          #diff
        #self.cls_token = nn.Parameter(torch.randn(1, 1, cond_dim))

        # self.null_attr_embed = nn.Parameter(torch.zeros((1, max_text_len)), requires_grad=False)

        # for non-attention based text conditioning at all points in the network where time is also conditioned

        self.to_text_non_attn_cond = None

        if cond_on_text:
            print('removed initial layer norm')
            # exit()
            # To sample from last trained model add LayerNorm and remove process_tokens layer
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),                                #
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim),
                # nn.SiLU(),
                # nn.Linear(time_cond_dim, time_cond_dim),
            )
            
            self.process_tokens = nn.Sequential(
                # nn.Identity(),
                nn.Linear(cond_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim)
            )

        self.cond_dim = cond_dim
        # attention related params

        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head, cosine_sim_attn = cosine_sim_attn)

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        resnet_klass = partial(ResnetBlock, **attn_kwargs)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        use_linear_attn = cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        # downsample klass

        downsample_klass = Downsample

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # initial resnet block (for memory efficient unet)

        self.init_resnet_block = resnet_klass(init_dim, init_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[0], use_gca = use_global_context_attn) if memory_efficient else None

        # scale for resnet skip connections

        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_attns_depth, layer_cross_attns, use_linear_attn, use_linear_cross_attn]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out) if not is_last else Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), nn.Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, current_dim, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(current_dim, current_dim, time_cond_dim = time_cond_dim, groups = groups, use_gca = use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
                post_downsample
            ]))

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])
        self.mid_attn = TransformerBlock(mid_dim, depth = layer_mid_attns_depth, **attn_kwargs) if attend_at_middle else None # , context_dim = cond_dim # Note: last change
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])

        # upsample klass

        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # upsampling layers

        upsample_fmap_dims = []

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_klass(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(dim_out + skip_connect_dim, dim_out, time_cond_dim = time_cond_dim, groups = groups, use_gca = use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = dim_out, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out

        self.upsample_combiner = UpsampleCombiner(
            dim = dim,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = dim
        )

        # whether to do a final residual from initial conv to the final resnet block out

        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (dim if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out

        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim = time_cond_dim, groups = resnet_groups[0], use_gca = True) if final_resnet_block else None

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        final_conv_dim_in += (channels if lowres_cond else 0)

        self.final_conv = nn.Conv2d(final_conv_dim_in, self.channels_out, final_conv_kernel_size, padding = final_conv_kernel_size // 2)

        zero_init_(self.final_conv)

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embedding.transpose(0, 1))

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    # methods for returning the full unet config as well as its parameter state

    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # class method for rehydrating the unet from its config and state dict

    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # methods for persisting unet to disk

    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok = True, parents = True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config = config, state_dict = state_dict)
        torch.save(pkg, str(path))

    # class method for rehydrating the unet from file saved with `persist_to_file`

    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        assert 'config' in pkg and 'state_dict' in pkg
        config, state_dict = pkg['config'], pkg['state_dict']

        return Unet.from_config_and_state_dict(config, state_dict)

    # forward with classifier free guidance

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits_with_q, _, _, logits, _, _, _, _= self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits_with_q, _, _, null_logits, _, _, _, _= self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def biased_sampling(self, gt_input, cond_masks, num_queries, masked_x, mask, patch_size=1):
        N, device = gt_input.shape[0], gt_input.device

        for _ in range(num_queries):
            if self.include_gt:
                querier_inputs = torch.cat([masked_x, gt_input], dim=1).to(device) # TODO: if including_gt
            else:
                querier_inputs = masked_x.to(device)
            query_vec, _, _, _ = self.querier(querier_inputs, mask)
            mask[torch.arange(N), query_vec.argmax(dim=1)] = 1.0
            masked_x = ops.update_masked_image(masked_x, cond_masks, query_vec, patch_size=patch_size)

        return masked_x, mask

    def biased_sampling_attr(self, x_start, num_queries, attr_embed):
        batch_size, device = x_start.shape[0], x_start.device
        mask = torch.zeros(batch_size, self.max_num_attributes).to(device)      #initial masked_attr
        mask_3d = rearrange(mask, 'b a -> b a 1') 
        attr_embeds_masked = attr_embed * mask_3d                      #initial masked_attr
        for _ in range(num_queries):
            query_vec, _, _ = self.querier(image = x_start, mask = mask, query_features = attr_embeds_masked) 
            #mask[torch.arange(N), query_vec.argmax(dim=1)] = 1.0
            mask = mask + query_vec
            mask_3d = rearrange(mask, 'b a -> b a 1')
            attr_embeds_masked = attr_embed * mask_3d

        return mask

    def text_to_embed_ind(self, text_one_hot, max_num_attributes):
        #text_one_hot values are 1, -1   text_one_hot is origianal text attribute
        #change 1, -1 to index for embedding

        index = torch.linspace(0, max_num_attributes-1, max_num_attributes, device=text_one_hot.device,
                                                   dtype=text_one_hot.dtype)
        text_one_hot_new = (text_one_hot+1) / 2
        text_embed_index = index * 2 + text_one_hot_new

        return text_embed_index.type_as(text_one_hot)


    def forward(
        self,
        x,                                                  #x_t
        time,
        *,
        lowres_cond_img = None,
        lowres_noise_times = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,                                  #usage?
        self_cond = None,
        # attr_embeds = None, # Note: Added attributes
        cond_drop_prob = 0.,
        x_start = None                                       #
    ): # UNET
        batch_size, device = x.shape[0], x.device
        aux_losses = {}          #after modifying, it is in inner_forward
        aux_outputs = {}
        # condition on self
        # print(x)
        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim = 1)

        # add low resolution conditioning, if present

        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)), 'low resolution conditioning noise time must be present'

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # condition on input image

        assert not (self.has_cond_image ^ exists(cond_images)), 'you either requested to condition on an image on the unet, but the conditioning image is not supplied, or vice versa'


        query_soft = None
        if exists(cond_images) and not exists(text_embeds):
            N = cond_images.shape[0]
            PATCH_SIZE = self.patch_size
            QUERY_ALL = (self.image_size[0] - PATCH_SIZE + 1) ** 2
            MAX_RAND_QUERIES = self.args.max_queries_random
            NULL_VAL = self.null_val
            if self.cond_images_channels == 1:

                # assert cond_images.shape[1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'
                cond_images = resize_image_to(cond_images, self.cond_img_size)

                cond_masks, cond_rgb = cond_images[:, :1], cond_images[:, 1:] # We use only the mask to condition the diffusion process

                # initial random sampling
                if self.args.train_querier and not self.args.all_queries:
                    rand, th, init_zeros = random(), 0.2, True
                    # gt_input = cond_images
                    gt_input = cond_images
                    gt_input = cond_rgb if self.channels == 3 else cond_masks
                    if self.sampling == 'biased' and rand > th:
                        num_queries = torch.randint(low=0, high=self.args.max_queries_biased, size=(1,))
                        # # mask, masked_x = ops.adaptive_sampling(x, num_queries, self.querier, PATCH_SIZE, QUERY_ALL)
                        if init_zeros:
                            masked_x = torch.zeros_like(cond_masks) + NULL_VAL
                            mask = torch.zeros(N, QUERY_ALL).to(cond_images.device)
                        else:
                            mask = ops.random_sampling(MAX_RAND_QUERIES, QUERY_ALL, x.size(0)).to(device)
                            masked_x, _, _, _ = ops.get_patch_mask(mask, cond_masks, patch_size=PATCH_SIZE, null_val=NULL_VAL)
                        #
                        # num_queries = torch.randint(low=0, high=self.args.max_queries_biased, size=(masked_x.shape[0],))
                        # if self.include_gt:
                        #     querier_inputs = torch.cat([masked_x, gt_input], dim=1).to(device)
                        # else:
                        #     querier_inputs = masked_x
                        # mask, masked_x = ops.adaptive_sampling(querier_inputs, num_queries, self.querier,
                        #                                        patch_size=self.patch_size,
                        #                                        max_queries=self.args.max_queries_biased)
                        with torch.no_grad():
                            masked_x, mask = self.biased_sampling(gt_input, cond_masks, num_queries, masked_x, mask, patch_size=PATCH_SIZE)

                        # two backwards
                        # masked_x, mask = self.biased_sampling(gt_input, cond_masks, 1, masked_x, mask, patch_size=PATCH_SIZE)
                    elif self.sampling == 'random' or rand <= th:
                        empty = random() < 0.01
                        mask = ops.random_sampling(MAX_RAND_QUERIES, QUERY_ALL, x.size(0), empty=empty).to(device)
                        masked_x, S_v, S_ij, split = ops.get_patch_mask(mask, cond_masks, patch_size=PATCH_SIZE, null_val=NULL_VAL)

                    if self.include_gt:
                        querier_inputs = torch.cat([masked_x, gt_input], dim=1).to(device)
                    else: querier_inputs = masked_x
                    query_vec, query_soft = self.querier(querier_inputs.contiguous(), mask.contiguous())
                    # # NOTE I JUST SELECTED PATCH WITH QUERY_SOFT
                    # query_vec = query_soft
                    masked_x = ops.update_masked_image(masked_x, cond_masks, query_vec, patch_size=PATCH_SIZE)
                else:
                    masked_x = cond_masks
                aux_outputs['masked_x'] = masked_x
                # masked_x = resize_image_to(masked_x, x.shape[-1])

            elif self.cond_images_channels == 3:
                # TODO:
                #  write sampling code,
                #  implement zeroing out of the image with drop prob.
                #  Check out for overfitting with patches new implementation.
                assert cond_images.shape[
                           1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'
                cond_images = resize_image_to(cond_images, self.cond_img_size)
                cond_rgb = cond_images  # We use only the mask to condition the diffusion process

                N = cond_images.shape[0]
                num_queries = (self.image_size[0] - self.patch_size + 1) ** 2

                # initial random sampling
                if self.args.train_querier and not self.args.all_queries:
                    rand, th, init_zeros = random(), 0.1, True
                    # gt_input = cond_images
                    gt_input = cond_rgb
                    if self.sampling == 'biased' and rand > th:
                        max_queries_biased = self.args.max_queries_biased
                        num_queries = torch.randint(low=0, high=max_queries_biased, size=(1,))
                        # # mask, masked_x = ops.adaptive_sampling(x, num_queries, self.querier, PATCH_SIZE, QUERY_ALL)
                        if init_zeros:
                            masked_x = torch.zeros_like(cond_rgb) + NULL_VAL
                            mask = torch.zeros(N, QUERY_ALL).to(cond_images.device)
                        else:
                            mask = ops.random_sampling(MAX_RAND_QUERIES, QUERY_ALL, x.size(0)).to(device)
                            masked_x, _, _, _ = ops.get_patch_mask(mask, cond_rgb, patch_size=PATCH_SIZE, null_val=NULL_VAL)
                        # num_queries = torch.randint(low=0, high=self.args.max_queries_biased, size=(masked_x.shape[0],))
                        # if self.include_gt:
                        #     querier_inputs = torch.cat([masked_x, gt_input], dim=1).to(device)
                        # else:
                        #     querier_inputs = masked_x
                        # mask, masked_x = ops.adaptive_sampling(querier_inputs, num_queries, self.querier, patch_size=self.patch_size, max_queries=self.args.max_queries_biased)

                        with torch.no_grad():
                            masked_x, mask = self.biased_sampling(gt_input, cond_rgb, num_queries, masked_x, mask,
                                                                  patch_size=self.patch_size)

                    elif self.sampling == 'random' or rand <= th:
                        empty = random() < 0.01
                        mask = ops.random_sampling(self.args.max_queries_random, num_queries, x.size(0), empty).to(device)
                        masked_x, S_v, S_ij, split = ops.get_patch_mask(mask, cond_rgb, patch_size=self.patch_size,   
                                                                        null_val=self.null_val)             #cond_rgb is gt_input
                    if self.include_gt:
                        querier_inputs = torch.cat([masked_x, gt_input], dim=1).to(device)
                    else: querier_inputs = masked_x.to(device)
                    query_vec, query_soft, query_logits, attn = self.querier(querier_inputs.contiguous(), mask.contiguous())     #generate a query        #
                    masked_x_new = ops.update_masked_image(masked_x, cond_rgb, query_vec, patch_size=self.patch_size) #masked_x -> masked_x_new
                    #masked_x = ops.update_masked_image(masked_x, cond_rgb, query_vec, patch_size=self.patch_size)
                else:
                    masked_x_new = cond_rgb # If training is false, this will be directly the masked_x      #masked_x -> masked_x_new
                    #masked_x = cond_rgb
                    #beacause we need to return query_logits and query_vec, so we need tp give None values to it
                    query_logits, query_vec = None, None

                masked_x_new = resize_image_to(masked_x_new, x.shape[-1])              #size 64
            masked_x_new_x = torch.cat((masked_x_new, x), dim=1)                     #6 channels

            # # ORTH LOSS - Cross Entropy
            # if exists(query_soft):
            #     loss_orth = loss_orth_fn(query_soft)
            #     alpha = 1e-3
            #     aux_losses = {'loss_orth': (alpha, loss_orth)}

        '''
        # initial convolution
        # self.encode_clean_features = False
        if self.encode_clean_features and x_start is not None:
            x = torch.cat([x, x_start], dim=0) # use clean data too to guide querier
            time = torch.cat([time, torch.zeros_like(time)], dim=0)

        x = self.init_conv(x) # [16, 32, 128, 128]
        # init conv residual

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone() # doesnt get here

        # time conditioning
        time_hiddens = self.to_time_hiddens(time) # TODO: for cond take last time

        # derive time tokens
        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)


        # Pretrained Feature extractor block
        out_clean_features = None
        if self.encode_clean_features and x_start is not None:
            with torch.no_grad():
                c_= torch.zeros(batch_size, 1, self.cond_dim).to(device)
                c_mask_ = c_[..., 0] != 0
                x, x_ = torch.chunk(x, 2, dim=0)
                t, t_ = torch.chunk(t, 2, dim=0)
                if exists(self.init_resnet_block):
                    x_ = self.init_resnet_block(x_, t_) # Not in here
                # go through the layers of the unet, down and up
                hiddens = []
                for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs[:-1]:

                    if exists(pre_downsample):
                        x_ = pre_downsample(x_)
                    x_ = init_block(x_, t_, c_, c_mask_)
                    for resnet_block in resnet_blocks:
                        x_ = resnet_block(x_, t_)
                        # hiddens.append(x_)

                    x_ = attn_block(x_, c_, c_mask_)
                    # hiddens.append(x_)
                    if exists(post_downsample):
                        x_ = post_downsample(x_)
                out_clean_features = x_ # 256, 16, 16


        # add lowres time conditioning to time hiddens
        # and add lowres time tokens along sequence dimension for attention

        if self.lowres_cond:
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)

            t = t + lowres_t
            time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim = -2)

        text_tokens = None
        c_mask = None

        # Note: Attribute embeddings
        if exists(text_embeds):
            histories = None
            cond_att, ans_att = None, None
            # cond_images = cond_images
            if len(text_embeds.shape) == 4:
                sampling = True
                cond_pos, cond_neg = text_embeds[..., :-1, 0], text_embeds[..., :-1, 1]
                ans_pos, ans_neg = text_embeds[..., -1:, 0], text_embeds[..., -1:, 1]
                if text_embeds.shape[-1] == 3:
                    cond_unasked, ans_unasked = text_embeds[..., :-1, 2], text_embeds[..., -1:, 2]
                cond_att, ans_att = None, None
            else:
                sampling = False
                attrs = text_embeds

                max_prob = 0.99
                if self.args.all_queries or (not self.args.train_querier and np.random.random() > 1-0.01): # All queries in 10% of the cases
                    prob_ask = 1 # Select for test.
                    exact = True
                    # print('a', prob_ask)
                else:
                    prob_ask = np.random.uniform(0, max_prob)
                    exact = False
                    # print('b', prob_ask)
                # prob_ask = 1 # Select for test.
                q_mask = torch.ones((batch_size, self.max_num_attributes, self.max_num_objects),
                                    device=text_embeds.device, dtype=attrs.dtype)
                q_mask_bool = prob_mask_like((batch_size, self.max_num_attributes, self.max_num_objects), prob_ask,
                                             device=device)
                q_mask = q_mask * q_mask_bool
                q_all = torch.linspace(1, self.max_num_attributes, self.max_num_attributes, device=text_embeds.device,
                                       dtype=attrs.dtype)[None, :, None]
                q = (q_mask * q_all).reshape(batch_size, -1)

                # Get embeddings for all queries
                attr_embeds = self.cond_embedding(q)
                attr_embeds_neg = self.cond_embedding_neg(q)
                zero_embed = self.cond_embedding(torch.zeros_like(q))
                obj_embed = self.query_encoder.object_embedding(batch_size, device)


                if self.args.experiment_type == 'attributes':
                    ans_all = answer_queries(q_mask.clone(), # Note: Q is not really used
                                             attrs.clone())  # q: [q x num_attr x max_obj], binary - gt_attrs: [b x max_obj]
                else:
                    ans_all = attrs

                ans = ans_all * q_mask.reshape(*ans_all.shape)
                # Select the asked queries in their embeddings according to the answers.
                ans_neg = torch.zeros_like(ans)
                ans_pos = torch.zeros_like(ans)
                ans_unasked = torch.zeros_like(ans)

                ans_neg[ans == -1] = 1
                ans_pos[ans == 1] = 1
                ans_unasked[ans == 0] = 1

                cond_pos_all, cond_neg_all, cond_unasked_all = (attr_embeds), \
                    (attr_embeds_neg ), (zero_embed )
                cond_pos_all_o, cond_neg_all_o = [torch.cat([c, obj_embed], dim=-1) * m for c, m in
                                                                zip((cond_pos_all, cond_neg_all),
                                                                    (1, 1) # Identity
                                                                    # (ans_pos, ans_neg)
                                                                    )]

                rand, th = random(), 0.2
                if self.args.all_queries:
                    cond_pos, cond_neg, cond_unasked = cond_pos_all * ans_pos, cond_neg_all * ans_neg, cond_unasked_all * ans_unasked

                elif self.sampling == 'random' or rand < th: #Random 20% of the times

                    # Randomly sample queries with probability prob_ask: [either 1 or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                    # q_mask_bool = prob_mask_like((batch_size, self.max_num_attributes, self.max_num_objects), prob_ask, device=device)
                    # q_mask = q_mask * q_mask_bool

                    empty = random() < 0.1
                    q_mask = ops.random_sampling(self.max_num_attributes, self.args.max_queries_random, batch_size, exact=exact, empty=empty).to(device)\
                        .reshape(batch_size, self.max_num_attributes, self.max_num_objects)

                    # q_mask maps all non asked questions
                    cond_pos, cond_neg = [c * q_mask.reshape(*ans_all.shape) * a for c, a in
                                                        zip((cond_pos_all, cond_neg_all),
                                                            (ans_pos, ans_neg))]

                elif self.sampling == 'biased':

                    # Randomly sample queries with probability prob_ask: [either 1 or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                    q_mask_bool = 0
                    q_mask = q_mask * q_mask_bool

                    # Select the asked queries in their embeddings according to the answers.
                    ans_neg = torch.zeros_like(q)[..., None]
                    ans_pos = ans_neg.clone()
                    ans_unasked = ans_neg.clone()

                    cond_pos, cond_neg, cond_unasked = (attr_embeds * ans_pos), \
                        (attr_embeds_neg * ans_neg), (zero_embed * ans_unasked)

                    num_steps = np.random.randint(1, self.args.max_queries_biased)
                    for step in range(num_steps):
                        in_embeds = torch.stack([cond_pos, cond_neg], dim=-1) # , cond_unasked
                        in_embeds_all = torch.stack([cond_pos_all_o, cond_neg_all_o], dim=-1) # , cond_unasked
                        in_ans = torch.stack([ans_pos, ans_neg], dim=-1) #, ans_unasked

                        with torch.no_grad():

                            out_query_features = None
                            if self.encode_query_features:
                                attr_embeds_ = self.query_encoder(cond=(cond_pos, cond_neg),
                                                                  ans=(ans_pos, ans_neg),
                                                                  cond_att=cond_att,
                                                                  ans_att=ans_att)
                                # attr_tokens_ = self.to_text_non_attn_cond(attr_embeds_)
                                out_query_features = self.process_tokens(attr_embeds_)

                            q_new, q_soft, attn = self.querier(cond=in_embeds, ans=in_ans, image=cond_images, image_features=out_clean_features,
                                                       query_features=out_query_features, mask=q_mask.reshape(batch_size, -1), cond_all=in_embeds_all,
                                                       return_attn=True)

                            # answer new query
                            # ans_new, chosen_attr, gt_attrs_rem  = \
                            #     answer_single_query(q_new.reshape(N, max_num_attributes, max_num_objects),
                            #                         gt_attrs_rem)

                            if self.args.experiment_type == 'attributes':
                                ans_all = \
                                    answer_queries(q_new.reshape(batch_size, self.max_num_attributes, self.max_num_objects),
                                                   attrs, ans_all)
                            else:
                                ans_all = attrs

                            ans_new = ans_all * q_new
                            # select the asked queries in their embeddings according to the answers.

                            bool_pos, bool_neg = (ans_new > 0), (ans_new < 0)


                            q_mask = torch.clamp(q_new + q_mask.reshape(*q_new.shape), 0, 1)

                            # select the asked queries in their embeddings according to the answers.
                            cond_pos = torch.where(bool_pos, attr_embeds, cond_pos) * q_mask
                            cond_neg = torch.where(bool_neg, attr_embeds_neg, cond_neg) * q_mask
                            cond_unasked = torch.where(ans_new != 0, torch.zeros_like(cond_unasked), cond_unasked)

                            ans_pos = torch.where(bool_pos, torch.ones_like(ans_pos), ans_pos) * q_mask
                            ans_neg = torch.where(bool_neg, torch.ones_like(ans_pos), ans_neg) * q_mask
                            ans_unasked = torch.where(ans_new != 0, torch.zeros_like(ans_unasked), ans_unasked)


                else: raise NotImplementedError



                # Get new query with gradients.
                if prob_ask < max_prob and self.args.train_querier: # TODO: Set back to true to train querier.

                    # Note: code to get encodings for the querier
                    out_query_features = None
                    if self.encode_query_features:
                        with torch.no_grad():
                            attr_embeds_ = self.query_encoder(cond=(cond_pos, cond_neg),
                                                                 ans=(ans_pos, ans_neg),
                                                                 cond_att=cond_att,
                                                                 ans_att=ans_att)
                            # attr_tokens_ = self.to_text_non_attn_cond(attr_embeds_)
                            out_query_features = self.process_tokens(attr_embeds_)

                    in_embeds = torch.stack([cond_pos, cond_neg], dim=-1)
                    in_embeds_all = torch.stack([cond_pos_all_o, cond_neg_all_o], dim=-1)
                    in_ans = torch.stack([ans_pos, ans_neg], dim=-1)
                    q_new_flat, q_soft_flat = self.querier(cond=in_embeds, cond_all=in_embeds_all, ans=in_ans, image=cond_images, image_features=out_clean_features,
                                                       query_features=out_query_features, mask=q_mask.reshape(batch_size, -1))

                    soft = False
                    if soft:
                        q_new = q_soft_flat
                    else:
                        q_new = q_new_flat
                    # answer new query
                    # ans_new, _, _ = answer_single_query(q_new.reshape(batch_size, self.max_num_attributes, self.max_num_objects),
                    #                               gt_attrs_rem)
                    if self.args.experiment_type == 'attributes':
                        ans_all = answer_queries(
                            q_new.reshape(batch_size, self.max_num_attributes, self.max_num_objects),
                            attrs, ans_all)
                    else:
                        ans_all = attrs

                    # k = 4
                    # if soft and k > 1:
                    #     q_new_k = torch.topk(q_new, k, dim=1, largest=True, sorted=True)
                    #     q_new_k_mask = torch.zeros_like(q_new).scatter(1, q_new_k.indices, 1)
                    #     top_k = (q_new * q_new_k_mask)
                    #     q_new = top_k.detach() - q_new.detach() + q_new
                        # q_new = q_new / (q_new.sum(dim=1, keepdim=True) + 1e-10) # normalize
                    ans_new = ans_all * q_new

                    # Apply orthogonality loss


                    loss_orth = loss_orth_fn(q_soft_flat)
                    alpha = 0.000001 # 000
                    aux_losses = {'loss_orth': (alpha, loss_orth)}

                    bool_pos, bool_neg = (ans_new > 0), (ans_new < 0)

                    if self.attention_querier:
                        ans_new_pos = torch.where(bool_pos, ans_new, torch.zeros_like(ans_pos)).sum(1, keepdims=True)
                        ans_new_neg = torch.where(bool_neg, ans_new, torch.zeros_like(ans_neg)).sum(1, keepdims=True)

                        # obj_embed = self.query_encoder.object_embedding(batch_size, device)
                        # cond_pos_obj = torch.cat([cond_pos, obj_embed], dim=-1)
                        # cond_neg_obj = torch.cat([cond_neg, obj_embed], dim=-1)
                        cond_new_pos = torch.where(bool_pos, cond_pos_all_o * q_new,
                                                   torch.zeros_like(cond_pos_all_o)).sum(1, keepdims=True)
                        cond_new_neg = torch.where(bool_neg, cond_neg_all_o * q_new,
                                                   torch.zeros_like(cond_neg_all_o)).sum(1, keepdims=True)

                        # q_mask = torch.cat([q_mask, torch.ones_like(q_mask[:, 0, 0][:, None])], dim=-1)
                        # ans_pos = torch.cat([ans_pos, ans_new_pos], dim=1)
                        # ans_neg = torch.cat([ans_neg, ans_new_neg], dim=1)
                        # cond_pos = torch.cat([cond_pos, cond_new_pos], dim=1)
                        # cond_neg = torch.cat([cond_neg, cond_new_neg], dim=1)
                        # select the asked queries in their embeddings according to the answers.

                        cond_att = (cond_new_pos, cond_new_neg)
                        ans_att = (ans_new_pos, ans_new_neg)
                    else:

                        q_mask = torch.clamp(q_new + q_mask.reshape(*q_new.shape), 0, 1)

                        # select the asked queries in their embeddings according to the answers.
                        cond_pos = torch.where(bool_pos, attr_embeds, cond_pos) * q_mask
                        cond_neg = torch.where(bool_neg, attr_embeds_neg, cond_neg) * q_mask
                        cond_unasked = torch.where(ans_new != 0, torch.zeros_like(cond_unasked), cond_unasked)

                        ans_pos = torch.where(bool_pos, torch.ones_like(ans_pos), ans_pos) * q_mask
                        ans_neg = torch.where(bool_neg, torch.ones_like(ans_pos), ans_neg) * q_mask
                        ans_unasked = torch.where(ans_new != 0, torch.zeros_like(ans_unasked), ans_unasked)

                        cond_att = None
                        ans_att = None

            # 0-out the attributes with certain probability (for class-free guidance)
            # TODO: check that it's nulled if indicated as unconditional.
            if (self.args.train_querier and self.args.freeze_unet) or self.args.all_queries:
                cond_drop_prob = 0
            attr_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
            attr_keep_mask_hidden = rearrange(attr_keep_mask, 'b -> b 1 1')
            #null_attr_hidden = self.null_attr_hidden.to(t.dtype)
            # For Guidance
            # null_attr_hidden = self.cond_embedding(torch.zeros_like(q))

            cond_pos = torch.where(
                attr_keep_mask_hidden,
                cond_pos,
                torch.zeros_like(cond_pos)
            )
            cond_neg = torch.where(
                attr_keep_mask_hidden,
                cond_neg,
                torch.zeros_like(cond_neg)
            )
            # cond_unasked = torch.where(
            #     attr_keep_mask_hidden,
            #     cond_unasked,
            #     zero_embed,
            # )

            # For Guidance
            ans_pos_enc = torch.where(
                attr_keep_mask_hidden,
                ans_pos,
                torch.zeros_like(ans_pos)
            )
            ans_neg_enc = torch.where(
                attr_keep_mask_hidden,
                - ans_neg,
                torch.zeros_like(ans_pos)
            )
            # ans_unasked_enc = torch.where(
            #     attr_keep_mask_hidden,
            #     self.null_val * ans_unasked,
            #     self.null_val * torch.ones_like(ans_unasked),
            # )


            # Get the final condition vector from encoding the embeddings
            # print(ans_pos_enc[0] - ans_neg_enc[0])
            t_cond = (cond_pos, cond_neg)
            t_ans = (ans_pos_enc, ans_neg_enc)
            if self.query_decoder:
                S, S_dec, attr_embeds_enc = self.query_encoder(cond=t_cond,
                                                            ans=t_ans)
                loss_rec = self.loss_rec(S_dec, S)
                alpha = 0.01
                aux_losses = {'loss_rec': (alpha, loss_rec)}
            else:
                attr_embeds_enc = self.query_encoder(cond=t_cond,
                                                     ans=t_ans,
                                                     cond_att=cond_att,
                                                     ans_att=ans_att)



            attr_tokens = attr_embeds_enc

            c_mask = ((ans_neg_enc != 0).logical_or(ans_pos_enc != 0))[..., 0]


            # print(c_mask.sum())
            if self.attention_querier and not self.args.all_queries and self.args.train_querier:
                c_mask = F.pad(c_mask, (0, 1), value=True)

            # TODO: restore and check.
            c_mask = torch.where(
                attr_keep_mask_hidden[..., 0],
                c_mask,
                False
            )
            attr_hiddens = self.to_text_non_attn_cond(attr_tokens)
            if len(attr_tokens.shape) == 3:
                attr_hiddens_out = attr_hiddens.sum(1) #.sum(1) # Note: Uncomment this and switch above for flatten_attr_process_token
                text_tokens = self.process_tokens(attr_tokens)
            else:
                print('we must have more than one attribute token')
                exit()
            t = t #+ attr_hiddens_out
            # text conditioning

        # c = time_tokens if not exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim = -2)
        c = time_tokens if not exists(text_tokens) else text_tokens

        # normalize conditioning tokens

        # c = self.norm_cond(c) # TODO: check again

        # initial resnet block (for memory efficient unet)

        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t) # Not in here
        # go through the layers of the unet, down and up

        hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c, c_mask)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c, c_mask)
            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, t, c, c_mask) # [16, 256, 16, 16] # self.noise_schedulers[unet_index](torch.zeros((x.shape[0]), device = x.device)), argument: cond_dim = None

        if exists(self.mid_attn):
            x = self.mid_attn(x) # [16, 256, 16, 16]
            # x = self.mid_attn(x, c) # [16, 256, 16, 16] # Note: last change

        x = self.mid_block2(x, t, c, c_mask) # [16, 256, 16, 16]

        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)

        up_hiddens = []

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            up_hiddens.append(x.contiguous())
            x = upsample(x)

        # whether to combine all feature maps from upsample blocks

        x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim = 1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)
        
        return self.final_conv(x), aux_losses, aux_outputs
        '''
         
        
        def inner_forward(x, time, text_hiddens, text_tokens):
            aux_losses = {}
            aux_outputs = {}        #
            aux_outputs['masked_x'] = x[:,:3]    #first 3 channel is  masked_x
            # initial convolution
            # self.encode_clean_features = False
            if self.encode_clean_features and x_start is not None:
                x = torch.cat([x, x_start], dim=0) # use clean data too to guide querier
                time = torch.cat([time, torch.zeros_like(time)], dim=0)
            
            x = self.init_conv(x) # [16, 32, 128, 128]
            # init conv residual
            
            if self.init_conv_to_final_conv_residual:
                init_conv_residual = x.clone() # doesnt get here
            
            # time conditioning
            time_hiddens = self.to_time_hiddens(time) # TODO: for cond take last time
            
            # derive time tokens
            time_tokens = self.to_time_tokens(time_hiddens)
            t = self.to_time_cond(time_hiddens)
            
            
            # Pretrained Feature extractor block
            out_clean_features = None
            if self.encode_clean_features and x_start is not None:
                with torch.no_grad():
                    c_= torch.zeros(batch_size, 1, self.cond_dim).to(device)
                    c_mask_ = c_[..., 0] != 0
                    x, x_ = torch.chunk(x, 2, dim=0)
                    t, t_ = torch.chunk(t, 2, dim=0)
                    if exists(self.init_resnet_block):
                        x_ = self.init_resnet_block(x_, t_) # Not in here
                    # go through the layers of the unet, down and up
                    hiddens = []
                    for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs[:-1]:
            
                        if exists(pre_downsample):
                            x_ = pre_downsample(x_)
                        x_ = init_block(x_, t_, c_, c_mask_)
                        for resnet_block in resnet_blocks:
                            x_ = resnet_block(x_, t_)
                            # hiddens.append(x_)
            
                        x_ = attn_block(x_, c_, c_mask_)
                        # hiddens.append(x_)
                        if exists(post_downsample):
                            x_ = post_downsample(x_)
                    out_clean_features = x_ # 256, 16, 16
            
            
            # add lowres time conditioning to time hiddens
            # and add lowres time tokens along sequence dimension for attention
            
            if self.lowres_cond:
                lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
                lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
                lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)
            
                t = t + lowres_t
                time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim = -2)

            # Note: Attribute embeddings
            if exists(text_embeds):
                t = t + text_hiddens

            c = time_tokens if not exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim = -2)
            #c = time_tokens if not exists(text_tokens) else text_tokens          #use text_token
            
            # normalize conditioning tokens
            
            c = self.norm_cond(c) 
            
            # initial resnet block (for memory efficient unet)
            
            if exists(self.init_resnet_block):
                x = self.init_resnet_block(x, t) # Not in here
            # go through the layers of the unet, down and up
            
            hiddens = []
            
            for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
                if exists(pre_downsample):
                    x = pre_downsample(x)
            
                x = init_block(x, t, c)    #remove c_mask
            
                for resnet_block in resnet_blocks:
                    x = resnet_block(x, t)
                    hiddens.append(x)
            
                x = attn_block(x, c)
                hiddens.append(x)
            
                if exists(post_downsample):
                    x = post_downsample(x)
            
            x = self.mid_block1(x, t, c) # [16, 256, 16, 16] # self.noise_schedulers[unet_index](torch.zeros((x.shape[0]), device = x.device)), argument: cond_dim = None
            
            if exists(self.mid_attn):
                x = self.mid_attn(x) # [16, 256, 16, 16]
                # x = self.mid_attn(x, c) # [16, 256, 16, 16] # Note: last change
            
            x = self.mid_block2(x, t, c) # [16, 256, 16, 16]
            
            add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)
            
            up_hiddens = []
            
            for init_block, resnet_blocks, attn_block, upsample in self.ups:
                x = add_skip_connection(x)
                x = init_block(x, t, c)
            
                for resnet_block in resnet_blocks:
                    x = add_skip_connection(x)
                    x = resnet_block(x, t)
            
                x = attn_block(x, c)
                up_hiddens.append(x.contiguous())
                x = upsample(x)
            
            # whether to combine all feature maps from upsample blocks
            
            x = self.upsample_combiner(x, up_hiddens)
            
            # final top-most residual if needed
            
            if self.init_conv_to_final_conv_residual:
                x = torch.cat((x, init_conv_residual), dim = 1)
            
            if exists(self.final_res_block):
                x = self.final_res_block(x, t)
            
            if exists(lowres_cond_img):
                x = torch.cat((x, lowres_cond_img), dim = 1)
            x_output = self.final_conv(x)

            return x_output, aux_losses, aux_outputs
        
               
        def get_text_tokens(attr_embeds_masked, text_mask, text_keep_mask_embed, text_keep_mask_hidden):
            #only use diff attr_embeds_masked value.
            text_tokens = self.text_to_cond(attr_embeds_masked)      #4 * 256 * 768 -> 4 * 256 * 512
            text_tokens = text_tokens[:, :self.max_text_len]
            
            #remove text_mask, we just use text_keep_mask_embed and text_keep_mask_hidden
            if exists(text_mask):                                                                           #4*256
                text_mask = text_mask[:, :self.max_text_len]
            
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len
            
            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
            
            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value = False)
            
                text_mask = rearrange(text_mask, 'b n -> b n 1')
                text_keep_mask_embed = text_mask & text_keep_mask_embed
            
            null_text_embed = self.null_text_embed.to(text_tokens.dtype) # for some reason pytorch AMP not working       #1 * 256 * 512
            
            text_tokens = torch.where(
                text_keep_mask_embed==1,
                text_tokens,
                null_text_embed
            )                                                                  #use  text_keep_mask_embed to mask and use null_text_embed to replace null
            
            if exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)                             #4*36*512
            
            # extra non-attention conditioning by projecting and then summing text embeddings to time
            # termed as text hiddens
            
            mean_pooled_text_tokens = text_tokens.mean(dim = -2)                      #4*512
            
            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)        #4*128
            
            null_text_hidden = self.null_text_hidden.to(time.dtype)
            
            text_hiddens = torch.where(                                               #text_hiddens      added to t
                text_keep_mask_hidden,
                text_hiddens,
                null_text_hidden
            )
            return text_hiddens, text_tokens

        if exists(text_embeds):
            text_embed_index = self.text_to_embed_ind(text_embeds, self.max_num_attributes)
            attr_embeds = self.cond_embedding(text_embed_index) 

            text_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device = device)        #4
            
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')                             #4*1*1      for batch_size level mask
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')                               #4*1       for batch_size level mask
            if self.training:
                if self.sampling == 'random':
                    empty = random() < 0.1
                    attr_mask_wo_query = ops.random_sampling(self.max_num_attributes, self.args.max_queries_random, batch_size, empty=empty).to(device)
                elif self.sampling == 'biased':
                    #TODO need to modify for bias sampling
                    num_queries = torch.randint(low=0, high=self.args.max_queries_biased-1, size=(1,))

                    with torch.no_grad():
                        attr_mask_wo_query = self.biased_sampling_attr(x_start, num_queries, attr_embeds) 

                else: raise NotImplementedError
            else:
                attr_mask_wo_query = text_mask     #use mask form outside

            attr_mask_wo_query = rearrange(attr_mask_wo_query, 'b a -> b a 1')
            attr_embeds_masked_wo_query = attr_embeds * attr_mask_wo_query
            text_hiddens_wo_q, text_tokens_wo_q = get_text_tokens(attr_embeds_masked_wo_query, text_mask, text_keep_mask_embed, text_keep_mask_hidden)

            #TODO: add querier
            if self.args.train_querier and self.training:
                #TODO 
                attr_mask_wo_query = rearrange(attr_mask_wo_query, 'b a 1 -> b a')
                query_vec, query_logits, attn = self.querier(image = x_start, mask = attr_mask_wo_query, query_features = attr_embeds_masked_wo_query)     #query_features are masked attr_embed
                attr_mask_with_query = attr_mask_wo_query + query_vec
                attr_mask_with_query = rearrange(attr_mask_with_query, 'b a -> b a 1')
                attr_embeds_masked_with_query = attr_embeds * attr_mask_with_query
                text_hiddens_with_q, text_tokens_w_q = get_text_tokens(attr_embeds_masked_with_query, text_mask, text_keep_mask_embed, text_keep_mask_hidden)
            else:
                query_logits, query_vec = None, None
        else:
            text_hiddens_with_q, text_tokens_w_q = None, None
            text_hiddens_wo_q, text_tokens_wo_q = None, None
        #x_output_with_query, aux_losses_with_query, aux_outputs_with_query  = inner_forward(masked_x_new_x, time)  
        if self.args.train_querier and self.training :
            x_output_with_query, aux_losses_with_query, aux_outputs_with_query  = inner_forward(x, time, text_hiddens_with_q, text_tokens_w_q)
        else:
            x_output_with_query, aux_losses_with_query, aux_outputs_with_query = None,None,None


        
        x_output_without_query = None
        aux_losses_without_query = None
        aux_outputs_without_query = None
        
        x_output_without_query, aux_losses_without_query, aux_outputs_without_query  = inner_forward(x, time, text_hiddens_wo_q, text_tokens_wo_q)
        

        return x_output_with_query, aux_losses_with_query, aux_outputs_with_query, x_output_without_query, aux_losses_without_query, aux_outputs_without_query, query_logits, query_vec    #query_logits not assgined

# null unet
class NullUnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x

# predefined unets, with configs lining up with hyperparameters in appendix of paper

class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 512,
            dim_mults = (1, 2, 3, 4),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
            layer_cross_attns = (False, True, True, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = False
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = (False, False, False, True),
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = False,
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

# main imagen ddpm class, which is a cascading DDPM from Ho et al.

@beartype
class Imagen(nn.Module):
    def __init__(
        self,
        unets,
        *,
        image_sizes,                                # for cascading ddpm, image size at each stage
        text_encoder_name = DEFAULT_T5_NAME,
        text_embed_dim = None,
        channels = 3,
        timesteps = 1000,
        cond_drop_prob = 0.1,
        loss_type = 'l2',
        noise_schedules = 'cosine',
        pred_objectives = 'noise',
        random_crop_sizes = None,
        lowres_noise_schedule = 'linear',
        lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        per_sample_random_aug_noise_level = False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        condition_on_text = True,
        auto_normalize_img = False,                  # Note: modified, # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        p2_loss_weight_gamma = 0.5,                 # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time
        p2_loss_weight_k = 1,
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.95,     # unsure what this was based on perusal of paper
        only_train_unet_number = None
    ):
        super().__init__()

        # loss

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # conditioning hparams

        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text

        # channels

        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # determine noise schedules per unet

        timesteps = cast_tuple(timesteps, num_unets)

        # make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets

        noise_schedules = cast_tuple(noise_schedules)
        noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'cosine')
        noise_schedules = pad_tuple_to_length(noise_schedules, num_unets, 'linear')

        # construct noise schedulers

        noise_scheduler_klass = GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.ModuleList([])

        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = noise_scheduler_klass(noise_schedule = noise_schedule, timesteps = timestep)
            self.noise_schedulers.append(noise_scheduler)

        # randomly cropping for upsampler training

        self.random_crop_sizes = cast_tuple(random_crop_sizes, num_unets)
        assert not exists(first(self.random_crop_sizes)), 'you should not need to randomly crop image during training for base unet, only for upsamplers - so pass in `random_crop_sizes = (None, 128, 256)` as example'

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule = lowres_noise_schedule)

        # ddpm objectives - predicting noise by default

        self.pred_objectives = cast_tuple(pred_objectives, num_unets)

        # get text encoder

        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))

        self.encode_text = partial(t5_encode_text, name = text_encoder_name)

        # construct unets

        self.unets = nn.ModuleList([])

        self.unet_being_trained_index = -1 # keeps track of which unet is being trained at the moment
        self.only_train_unet_number = only_train_unet_number

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, (Unet, Unet3D, NullUnet))
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                cond_on_text = self.condition_on_text,
                text_embed_dim = self.text_embed_dim if self.condition_on_text else None,
                channels = self.channels,
                channels_out = self.channels
            )

            self.unets.append(one_unet)

        # unet image sizes

        image_sizes = cast_tuple(image_sizes)
        self.image_sizes = image_sizes

        assert num_unets == len(image_sizes), f'you did not supply the correct number of u-nets ({len(unets)}) for resolutions {image_sizes}'

        self.sample_channels = cast_tuple(self.channels, num_unets)

        # determine whether we are training on images or video

        is_video = any([isinstance(unet, Unet3D) for unet in self.unets])
        self.is_video = is_video

        self.right_pad_dims_to_datatype = partial(rearrange, pattern = ('b -> b 1 1 1' if not is_video else 'b -> b 1 1 1 1'))
        self.resize_to = resize_video_to if is_video else resize_image_to

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # p2 loss weight

        self.p2_loss_weight_k = p2_loss_weight_k
        self.p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        assert all([(gamma_value <= 2) for gamma_value in self.p2_loss_weight_gamma]), 'in paper, they noticed any gamma greater than 2 is harmful'

        # one temp parameter for keeping track of device

        self.register_buffer('_temp', torch.tensor([0.]), persistent = False)

        # default to device of unets passed in

        self.to(next(self.unets.parameters()).device)

    def force_unconditional_(self):
        self.condition_on_text = False
        self.unconditional = True

        for unet in self.unets:
            unet.cond_on_text = False

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    def reset_unets_all_one_device(self, device = None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # gaussian diffusion methods

    def p_mean_variance(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        lowres_cond_img = None,
        self_cond = None,
        lowres_noise_times = None,
        cond_scale = 1.,
        model_output = None,
        t_next = None,
        pred_objective = 'noise',
        dynamic_threshold = True
    ):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, noise_scheduler.get_condition(t), text_embeds = text_embeds, text_mask = text_mask, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, self_cond = self_cond, lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_noise_times)))

        if pred_objective == 'noise':
            x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)
        elif pred_objective == 'x_start':
            x_start = pred
        elif pred_objective == 'v':
            x_start = noise_scheduler.predict_start_from_v(x, t = t, v = pred)
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            s = torch.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.dynamic_thresholding_percentile,
                dim = -1
            )

            s.clamp_(min = 1.)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s
        else:
            x_start.clamp_(-1., 1.)

        mean_and_variance = noise_scheduler.q_posterior(x_start = x_start, x_t = x, t = t, t_next = t_next)
        return mean_and_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        t_next = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        cond_scale = 1.,
        self_cond = None,
        lowres_cond_img = None,
        lowres_noise_times = None,
        pred_objective = 'noise',
        dynamic_threshold = True
    ):
        b, *_, device = *x.shape, x.device
        (model_mean, _, model_log_variance), x_start = self.p_mean_variance(unet, x = x, t = t, t_next = t_next, noise_scheduler = noise_scheduler, text_embeds = text_embeds, text_mask = text_mask, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, self_cond = self_cond, lowres_noise_times = lowres_noise_times, pred_objective = pred_objective, dynamic_threshold = dynamic_threshold)
        noise = torch.randn_like(x)
        # no noise when t == 0
        is_last_sampling_timestep = (t_next == 0) if isinstance(noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        unet,
        shape,
        *,
        noise_scheduler,
        lowres_cond_img = None,
        lowres_noise_times = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        init_images = None,
        skip_steps = None,
        cond_scale = 1,
        pred_objective = 'noise',
        dynamic_threshold = True,
        use_tqdm = True
    ):
        device = self.device

        batch = shape[0]
        img = torch.randn(shape, device = device)

        # for initialization with an image or video

        if exists(init_images):
            img += init_images

        # keep track of x0, for self conditioning

        x_start = None

        # prepare inpainting

        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        resample_times = inpaint_resample_times if has_inpainting else 1

        if has_inpainting:
            inpaint_images = self.normalize_img(inpaint_images)
            inpaint_images = self.resize_to(inpaint_images, shape[-1])
            inpaint_masks = self.resize_to(rearrange(inpaint_masks, 'b ... -> b 1 ...').float(), shape[-1]).bool()

        # time

        timesteps = noise_scheduler.get_sampling_timesteps(batch, device = device)

        # whether to skip any steps

        skip_steps = default(skip_steps, 0)
        timesteps = timesteps[skip_steps:]

        for times, times_next in tqdm(timesteps, desc = 'sampling loop time step', total = len(timesteps), disable = not use_tqdm):
            is_last_timestep = times_next == 0

            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                if has_inpainting:
                    noised_inpaint_images, *_ = noise_scheduler.q_sexample(inpaint_images, t = times)
                    img = img * ~inpaint_masks + noised_inpaint_images * inpaint_masks

                self_cond = x_start if unet.self_cond else None

                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    t_next = times_next,
                    text_embeds = text_embeds,
                    text_mask = text_mask,
                    cond_images = cond_images,
                    cond_scale = cond_scale,
                    self_cond = self_cond,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold
                )

                if has_inpainting and not (is_last_resample_step or torch.all(is_last_timestep)):
                    renoised_img = noise_scheduler.q_sample_from_to(img, times_next, times)

                    img = torch.where(
                        self.right_pad_dims_to_datatype(is_last_timestep),
                        img,
                        renoised_img
                    )

        img.clamp_(-1., 1.)

        # final inpainting

        if has_inpainting:
            img = img * ~inpaint_masks + inpaint_images * inpaint_masks

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        texts: List[str] = None,
        text_masks = None,
        text_embeds = None,
        video_frames = None,
        cond_images = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        init_images = None,
        skip_steps = None,
        batch_size = 1,
        cond_scale = 1.,
        lowres_sample_noise_level = None,
        start_at_unet_number = 1,
        start_image_or_video = None,
        stop_at_unet_number = None,
        return_all_unet_outputs = False,
        return_pil_images = False,
        device = None,
        use_tqdm = True,
        num_samples = 1
    ):
        device = default(device, self.device)
        self.reset_unets_all_one_device(device = device)

        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        #
        print(f"Repeating samples {num_samples} times.")
        if text_embeds is not None and num_samples > 1:
            text_embeds = torch.repeat_interleave(text_embeds[:, None], num_samples, 1).flatten(0,1)
        if cond_images is not None and num_samples > 1:
            cond_images = torch.repeat_interleave(cond_images[:, None], num_samples, 1).flatten(0,1)
        if num_samples > 1:
            batch_size *= num_samples

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert all([*map(len, texts)]), 'text cannot be empty'

            with autocast(enabled = False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask = True)

            text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))

        if not self.unconditional:
            assert exists(text_embeds), 'text must be passed in if the network was not trained without text `condition_on_text` must be set to `False` when training'

            #text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))   #text_embeds dim is N * attr_num     we need masks has the same dimension, set them as true all
            text_masks = default(text_masks, lambda: torch.ones_like(text_embeds).bool())

            batch_size = text_embeds.shape[0]

        if exists(inpaint_images):
            if self.unconditional:
                if batch_size == 1: # assume researcher wants to broadcast along inpainted images
                    batch_size = inpaint_images.shape[0]

            assert inpaint_images.shape[0] == batch_size, 'number of inpainting images must be equal to the specified batch size on sample `sample(batch_size=<int>)``'
            assert not (self.condition_on_text and inpaint_images.shape[0] != text_embeds.shape[0]), 'number of inpainting images must be equal to the number of text to be conditioned on'

        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into imagen if specified'
        assert not (not self.condition_on_text and exists(text_embeds)), 'imagen specified not to be conditioned on text, yet it is presented'
        # assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        assert not (exists(inpaint_images) ^ exists(inpaint_masks)),  'inpaint images and masks must be both passed in to do inpainting'

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        num_unets = len(self.unets)

        # condition scaling

        cond_scale = cast_tuple(cond_scale, num_unets)

        # add frame dimension for video

        assert not (self.is_video and not exists(video_frames)), 'video_frames must be passed in on sample time if training on video'

        frame_dims = (video_frames,) if self.is_video else tuple()

        # for initial image and skipping steps

        init_images = cast_tuple(init_images, num_unets)
        init_images = [maybe(self.normalize_img)(init_image) for init_image in init_images]

        skip_steps = cast_tuple(skip_steps, num_unets)

        # handle starting at a unet greater than 1, for training only-upscaler training

        if start_at_unet_number > 1:
            assert start_at_unet_number <= num_unets, 'must start a unet that is less than the total number of unets'
            assert not exists(stop_at_unet_number) or start_at_unet_number <= stop_at_unet_number
            assert exists(start_image_or_video), 'starting image or video must be supplied if only doing upscaling'

            prev_image_size = self.image_sizes[start_at_unet_number - 2]
            img = self.resize_to(start_image_or_video, prev_image_size)

        # go through each unet in cascade

        for unet_number, unet, channel, image_size, noise_scheduler, pred_objective, dynamic_threshold, unet_cond_scale, unet_init_images, unet_skip_steps in tqdm(zip(range(1, num_unets + 1), self.unets, self.sample_channels, self.image_sizes, self.noise_schedulers, self.pred_objectives, self.dynamic_thresholding, cond_scale, init_images, skip_steps), disable = not use_tqdm):

            if unet_number < start_at_unet_number:
                continue

            assert not isinstance(unet, NullUnet), 'one cannot sample from null / placeholder unets'

            context = self.one_unet_in_gpu(unet = unet) if is_cuda else nullcontext()

            with context:
                lowres_cond_img = lowres_noise_times = None
                shape = (batch_size, channel, *frame_dims, image_size, image_size)

                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule.get_times(batch_size, lowres_sample_noise_level, device = device)

                    lowres_cond_img = self.resize_to(img, image_size)

                    lowres_cond_img = self.normalize_img(lowres_cond_img)
                    lowres_cond_img, *_ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_noise_times, noise = torch.randn_like(lowres_cond_img))

                if exists(unet_init_images):
                    unet_init_images = self.resize_to(unet_init_images, image_size)

                shape = (batch_size, self.channels, *frame_dims, image_size, image_size)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    text_embeds = text_embeds,
                    text_mask = text_masks,
                    cond_images = cond_images,
                    inpaint_images = inpaint_images,
                    inpaint_masks = inpaint_masks,
                    inpaint_resample_times = inpaint_resample_times,
                    init_images = unet_init_images,
                    skip_steps = unet_skip_steps,
                    cond_scale = unet_cond_scale,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold,
                    use_tqdm = use_tqdm
                )

                outputs.append(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        output_index = -1 if not return_all_unet_outputs else slice(None) # either return last unet output or all unet outputs

        if not return_pil_images:
            return outputs[output_index]

        if not return_all_unet_outputs:
            outputs = outputs[-1:]

        assert not self.is_video, 'converting sampled video tensor to video file is not supported yet'

        pil_images = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim = 0))), outputs))

        return pil_images[output_index] # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)

    def p_losses(
        self,
        unet: Union[Unet, Unet3D, NullUnet, DistributedDataParallel],
        x_start,
        times,                       
        *,
        noise_scheduler,
        lowres_cond_img = None,
        lowres_aug_times = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        noise = None,
        times_next = None,
        pred_objective = 'noise',
        p2_loss_weight_gamma = 0.,
        random_crop_size = None
    ):
        is_video = x_start.ndim == 5

        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # random cropping during training
        # for upsamplers

        if exists(random_crop_size):
            if is_video:
                frames = x_start.shape[2]
                x_start, lowres_cond_img, noise = rearrange_many((x_start, lowres_cond_img, noise), 'b c f h w -> (b f) c h w')

            aug = K.RandomCrop((random_crop_size, random_crop_size), p = 1.)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            x_start = aug(x_start)
            lowres_cond_img = aug(lowres_cond_img, params = aug._params)
            noise = aug(noise, params = aug._params)

            if is_video:
                x_start, lowres_cond_img, noise = rearrange_many((x_start, lowres_cond_img, noise), '(b f) c h w -> b c f h w', f = frames)

        # get x_t

        x_noisy, log_snr, alpha, sigma = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)   

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, *_ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_aug_times, noise = torch.randn_like(lowres_cond_img))

        # time condition
        noise_cond = noise_scheduler.get_condition(times)

        # unet kwargs

        unet_kwargs = dict(
            text_embeds = text_embeds,
            text_mask = text_mask,
            cond_images = cond_images,
            lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_aug_times),
            lowres_cond_img = lowres_cond_img_noisy,
            cond_drop_prob = self.cond_drop_prob,
        )

        # self condition if needed

        # Because 'unet' can be an instance of DistributedDataParallel coming from the
        # ImagenTrainer.unet_being_trained when invoking ImagenTrainer.forward(), we need to
        # access the member 'module' of the wrapped unet instance.
        self_cond = unet.module.self_cond if isinstance(unet, DistributedDataParallel) else unet.self_cond

        if self_cond and random() < 0.5:
            with torch.no_grad():
                pred, aux_losses, _, pred_wo_q, _, _, query_logits, query_vec  = unet.forward(
                    x_noisy,
                    noise_cond,
                    **unet_kwargs
                ).detach()

                x_start = noise_scheduler.predict_start_from_noise(x_noisy, t = times, noise = pred) if pred_objective == 'noise' else pred

                unet_kwargs = {**unet_kwargs, 'self_cond': x_start}
        
        # get prediction
        pred, aux_losses, _, pred_wo_q, aux_losses_wo_q, _, query_logits, query_vec  = unet.forward(       #aux_losses and aux_outputs we don't use later   TODO add model output here
            x_noisy,                        #x_t
            noise_cond,
            x_start = x_start,              #x_0
            **unet_kwargs
        )

        # prediction objective
        if pred_objective == 'noise':
            target = noise
        elif pred_objective == 'x_start':
            target = x_start
        elif pred_objective == 'v':
            # derivation detailed in Appendix D of Progressive Distillation paper
            # https://arxiv.org/abs/2202.00512
            # this makes distillation viable as well as solve an issue with color shifting in upresoluting unets, noted in imagen-video
            target = alpha * noise - sigma * x_start
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        # losses
        if unet.cmi:
            losses_w_q = self.loss_fn(pred, target, reduction = 'none')    #cal mse with query of diffussion
            losses_wo_q = self.loss_fn(pred_wo_q, target, reduction = 'none')    #cal mse without query of diffussion
            query_logits_gt = (losses_wo_q - losses_w_q).detach()
            query_logits_gt = query_logits_gt.sum(dim=[1,2,3])
            max_ind_query = query_vec.max(1)                     #get index of query
            query_score_second = query_logits[torch.arange(query_logits.size(0)), max_ind_query[1]]
            losses = torch.abs(query_logits_gt - query_score_second)   #l1 as loss
            #losses = torch.square(query_logits_gt - query_score_second)   #l2 as loss
        else:
            losses = self.loss_fn(pred_wo_q, target, reduction = 'none')
        # masked_mse = False
        # if masked_mse and 'masked_x' in aux_outputs:
        #     mask = aux_outputs['masked_x'].clone()
        #     mask[ mask != -10 ] = 1; mask[ mask == -10 ] = 0
        #     mse_mask = (1-mask).detach()
        #     losses = reduce(losses * mse_mask, 'b ... -> b', 'sum') / reduce(mse_mask, 'b ... -> b', 'sum')
        # else:



        losses = reduce(losses, 'b ... -> b', 'mean')

        # p2 loss reweighting

        if p2_loss_weight_gamma > 0:
            loss_weight = (self.p2_loss_weight_k + log_snr.exp()) ** -p2_loss_weight_gamma
            losses = losses * loss_weight

        # adding auxiliary losses

        for k, v in aux_losses_wo_q.items():
            losses = losses + v[0] * reduce(v[1], 'b ... -> b', 'mean')

        return losses.mean()

    def forward(
        self,
        images,                                                                   #origianl images
        unet: Union[Unet, Unet3D, NullUnet, DistributedDataParallel] = None,
        texts: List[str] = None,
        text_embeds = None,
        text_masks = None,
        unet_number = None,
        cond_images = None
    ):    #imagen
        assert images.shape[-1] == images.shape[-2], f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you can only train on unet #{self.only_train_unet_number}'

        images = cast_uint8_images_to_float(images)
        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        assert is_float_dtype(images.dtype), f'images tensor needs to be floats but {images.dtype} dtype found instead'

        unet_index = unet_number - 1

        unet = default(unet, lambda: self.get_unet(unet_number))

        assert not isinstance(unet, NullUnet), 'null unet cannot and should not be trained'

        noise_scheduler      = self.noise_schedulers[unet_index]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[unet_index]
        pred_objective       = self.pred_objectives[unet_index]
        target_image_size    = self.image_sizes[unet_index]
        random_crop_size     = self.random_crop_sizes[unet_index]
        prev_image_size      = self.image_sizes[unet_index - 1] if unet_index > 0 else None

        b, c, *_, h, w, device, is_video = *images.shape, images.device, images.ndim == 5

        check_shape(images, 'b c ...', c = self.channels)
        assert h >= target_image_size and w >= target_image_size

        frames = images.shape[2] if is_video else None

        ## For times = 1 we have pure noise.
        if self.unets[0].args.train_querier and self.unets[0].args.freeze_unet and self.unets[0].args.constant_t > 0:
            # print('Warning we are sampling a constant time')
            # exit()
            times = noise_scheduler.sample_constant_times(self.unets[0].args.constant_t, b, device = device)
        else:
            times = noise_scheduler.sample_random_times(b, device = device)
        # times = noise_scheduler.sample_random_times(b, device = device)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert all([*map(len, texts)]), 'text cannot be empty'
            assert len(texts) == len(images), 'number of text captions does not match up with the number of images given'

            with autocast(enabled = False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask = True)

            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        if not self.unconditional:
            #text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))     #text_embeds dim is N * attr_num     we need masks has the same dimension, set them as true all
            text_masks = default(text_masks, lambda: torch.ones_like(text_embeds).bool())

        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text and exists(text_embeds)), 'decoder specified not to be conditioned on text, yet it is presented'

        # Note: Changed
        # assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = self.resize_to(images, prev_image_size, clamp_range = self.input_image_range)
            lowres_cond_img = self.resize_to(lowres_cond_img, target_image_size, clamp_range = self.input_image_range)

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(b, device = device)
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(1, device = device)
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b = b)

        images = self.resize_to(images, target_image_size)

        # Note: added contiguous
        return self.p_losses(unet, images, times, text_embeds = text_embeds, text_mask = text_masks, cond_images = cond_images, noise_scheduler = noise_scheduler, lowres_cond_img = lowres_cond_img, lowres_aug_times = lowres_aug_times, pred_objective = pred_objective, p2_loss_weight_gamma = p2_loss_weight_gamma, random_crop_size = random_crop_size)


def loss_orth_fn(q):
    q = torch.clamp(q, min=0.00001, max=0.999)
    # print(q.max(), q.min())
    a, b = q[None], q[:, None]
    # e = a * b
    # e = (torch.log(a) + torch.log(b))
    cross_entropy = nn.BCELoss(reduction='none')
    e = -cross_entropy(torch.repeat_interleave(a, q.shape[0], 0),
                       torch.repeat_interleave(b, q.shape[0], 1))
    return e